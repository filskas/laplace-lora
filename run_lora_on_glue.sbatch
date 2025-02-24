#!/bin/bash -l
#SBATCH -J lora_prep_glue_8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgefficient-gpu-a100

nvidia-smi -L

PROJECT_DIR="$HOME/laplace-lora"
OUTPUTS_DIR="$SCRATCH/laplace-lora"
conda activate /net/pr2/projects/plgrid/plggsparse/fskalka/laplace-lora/env

seeds=(0 22 42)
glue_tasks=("wnli" "ax" "rte" "mrpc" "cola" "sst2" "qnli" "qqp" "mnli")

# Nested loops
for seed in "${seeds[@]}"; do       # Outer loop: iterate over values
    for glue_task in "${glue_tasks[@]}"; do    # Inner loop: iterate over strings
        echo "Running LoRA for seed=seed, task=$glue_task" # Print the current iteration
        python -u $PROJECT_DIR/run_classification_bert.py --task_name $glue_task --seed $seed --testing_set val --with_tracking --report_to wandb --output_dir $OUTPUTS_DIR --lora_r 8 --lora_alpha 16 --wandb_tag lora_glue_full
    done
done
