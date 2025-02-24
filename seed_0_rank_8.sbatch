#!/bin/bash -l
#SBATCH -J blora_hyperparam_0_8_80
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgefficient-gpu-a100

nvidia-smi -L

PROJECT_DIR="$HOME/laplace-lora"
OUTPUTS_DIR="$SCRATCH/laplace-lora"
conda activate /net/pr2/projects/plgrid/plggsparse/fskalka/laplace-lora/env

prior_prec_multipliers=(0.001 0.01 0.1 1 10 100 1000)
glue_tasks=("wnli" "ax" "rte" "mrpc" "cola" "sst2" "qnli" "qqp" "mnli")

# Nested loops
for pp in "${prior_prec_multipliers[@]}"; do       # Outer loop: iterate over values
    for glue_task in "${glue_tasks[@]}"; do    # Inner loop: iterate over strings
        echo "Running for value=$pp, string=$glue_task" # Print the current iteration
        python -u $PROJECT_DIR/run_classification_bert_laplace.py --task_name $glue_task --seed 0 --testing_set val --with_tracking --report_to wandb --output_dir $OUTPUTS_DIR --lora_r 8 --lora_alpha 16 --laplace_sub "all" --laplace_prior_precision_multiplier $pp --wandb_tag hyperparam_sweep_full_80
    done
done
