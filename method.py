import argparse

import numpy as np
import torch
from torch.autograd.functional import hvp
from torch.func import functional_call
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import random
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging


task_to_keys = {
    # GLUE
    "wnli": ("sentence1", "sentence2"), # 635
    "ax": ("premise", "hypothesis"),    # 1,459
    "rte": ("sentence1", "sentence2"),  # 2,490
    "mrpc": ("sentence1", "sentence2"), # 3,668
    "cola": ("sentence", None),         # 8,551
    "sst2": ("sentence", None),         # 67,349
    "qnli": ("question", "sentence"),   # 104,743
    "qqp": ("question1", "question2"),  # 363,846
    "mnli": ("premise", "hypothesis"),  # 392,702
    # SuperGLUE
    "cb": ("premise", "hypothesis"),    # 250
    # "axb": ("premise", "hypothesis"),    # 1,459
    # "axg": ("premise", "hypothesis"),    # 1,459
    "wic": ("sentence1", "sentence2"),  # 6,000
    "boolq": ("passage", "question"),   # 9,427
}


logger = get_logger(__name__)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def weights_shapes(weights):
    return {n: p.shape for n, p in weights.items()}


def weights_numel(weights):
    return sum(p.numel() for n, p in weights.items())


def flatten_weights(weights):
    return torch.cat([p.contiguous().view(-1) for n, p in weights.items()])


def unflatten_weights(flattened, weights_shapes):
    offset = 0
    new_weights = {}
    for name, shape in weights_shapes.items():
        numel = shape.numel()
        new_weights[name] = flattened[offset : (offset + numel)].reshape(shape)
        offset += numel
    return new_weights

def trace_hessian_hutchinson1(model, X, weights, num_estimates=100, default_eval=True, **hvp_kwargs):
    """
    Although this implementation accepts weights dictionary containing multiple parameters,
    it fails then to return the correct trace. Therefore, use it always with a single parameter,
    e.g., with weights={weight_name: parameter tensor}.
    """
    if model.training:
        if default_eval:
            print("WARNING! Switching model into eval mode.")
            model.eval()  # Ensure the model is in evaluation mode to disable dropout, etc.
        else:
            print("WARNING! Your model is in training mode. Make sure it is what you want.")

    device = next(iter(weights.values())).device
    print(device)
    shapes, numel = weights_shapes(weights), weights_numel(weights)

    trace_estimate = 0.0
    for _ in range(num_estimates):
        # Sample from Rademacher
        rademacher_vector = torch.randint(0, 2, torch.Size([numel]), device=device, dtype=torch.float32) * 2 - 1
        # Compute the Hessian-vector product using hvp (Hessian-vector product)
        fwd = lambda w: functional_call(model,
                                        parameter_and_buffer_dicts=unflatten_weights(w, shapes),
                                        args=(),
                                        kwargs=X)
        out, Hv = hvp(fwd, inputs=flatten_weights(weights), v=rademacher_vector, **hvp_kwargs) # first output is the output of model
        # print(out)
        # Sum the product of the random vector and the Hessian-vector product
        trace_estimate += torch.dot(rademacher_vector, Hv)
    # Average the trace estimates
    trace_estimate /= num_estimates

    return trace_estimate


def trace_hessian_hutchinson(model, X, weights, num_estimates=100, default_eval=True, **hvp_kwargs):
    return sum(trace_hessian_hutchinson1(model, X, {n: p}, num_estimates=num_estimates, default_eval=default_eval)
                for n, p in weights.items())


class OutputSelector(nn.Module):
    def __init__(self, model, output_no=None, reduce=None):
        super(OutputSelector, self).__init__()
        self.model = model
        self.output_no = output_no
        self.reduce = reduce

    def parameters(self, **kwargs):
        return self.model.parameters()

    def named_parameters(self, **kwargs):
        return self.model.named_parameters()

    def forward(self, *args, **kwargs):
        # print(self.output_no, (self.output_no+1))
        # print(self.model(*args, **kwargs).logits)
        f = F.softmax(self.model(*args, **kwargs).logits, dim=-1)[:, self.output_no: (self.output_no+1)]
        # f = F.threshold(f, -1e31, 0.)  # effectively identity but need for autograd

        if self.reduce is not None:
            f = self.reduce(f)

        return f


def sweep_trace(model2, X, weights2):
    trace_estimates = []
    for num_estimates in [10, 50, 100, 200, 500, 1000]:
        print(f"estimating {num_estimates}")
        trace_estimate = trace_hessian_hutchinson(model2, X, weights2, num_estimates=num_estimates, default_eval=True)
        trace_estimates.append(trace_estimate)

    print(trace_estimates)
    return trace_estimates


def parse_args():
    parser = argparse.ArgumentParser(description="Script for training a model with configurable parameters.")

    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the model.")
    parser.add_argument("--task_name", type=str, default="cola", help="Name of the task.")
    parser.add_argument("--pad_to_max_length", action="store_true", default=False,
                        help="Whether to pad sequences to max length.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--testing_set", type=str, default="train_val", help="Name of the testing set.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                        help="Batch size per device for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--batch_number", type=int,
                        help="Which batch to choose to run the hessian estimation on.")

    return parser.parse_args()


if __name__ == "__main__":

    # model_name = "roberta-base"
    # task_name = "cola"
    # pad_to_max_length = False
    # max_length = 256
    # testing_set = 'train_val'
    # weight_decay = 0.01
    # per_device_train_batch_size = 32
    # per_device_eval_batch_size = 32
    # learning_rate = 1e-5
    # gradient_accumulation_steps = 1
    args = parse_args()

    accelerator = Accelerator()

    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
        config=config,
        ignore_mismatched_sizes=True,
    )

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f'param name {name}, param shape {param.shape}, param mean {param.mean()}, param std {param.std()}')
            print(f'param name {name}, param shape {param.shape} {param.dtype}')


    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]


    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    processed_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    if args.testing_set == 'test':
        ds = processed_dataset.train_test_split(test_size=0.5, seed=42, shuffle=False)
        val_dataset, eval_dataset = ds["train"], ds["test"]
    elif args.testing_set == 'train_val':
        ds = train_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
        train_dataset, val_dataset = ds["train"], ds["test"]
        eval_dataset = processed_dataset
    elif args.testing_set == 'val':
        eval_dataset = processed_dataset

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.testing_set != 'val':
        val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    last_lora_layer_name = list(model.state_dict().keys())[-2]
    last_lora_layer_weights = model.state_dict()[last_lora_layer_name]
    print(f"last_lora_shape = {last_lora_layer_weights.shape}")

    i = 0
    active_dataloader = train_dataloader
    for step, train_batch in enumerate(active_dataloader):
        i += 1
        if i == args.batch_number:
            X = train_batch
            break

    model2 = OutputSelector(model, output_no=0, reduce=torch.mean)
    weights = {last_lora_layer_name: last_lora_layer_weights}
    weights2 = {"model." + n: p for n, p in weights.items()}  # the original model is wrapped as model2.model

    for seed in [1, 2, 3]:
        trace_estimates = sweep_trace(model2, X, weights2)
        arrays = [t.numpy() for t in trace_estimates]
        # Save as a compressed file
        np.savez(f"one-batch-task_{args.task_name}-bs_{args.per_device_train_batch_size}-bn_{args.batch_number}-seed_{seed}.npz", *arrays)

