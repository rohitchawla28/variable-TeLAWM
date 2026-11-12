#!/bin/bash
#SBATCH --job-name=iql-pointmaze-medium-navigate
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate telawm

SEED=0

python -u train.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=iql \
    seed=${SEED} gamma=0.99 iql.expectile=0.7 iql.alpha=0.01 steps=1001000 \
    compile=false cudagraphs=false \
    run_suffix=baseline wandb_name_suffix=_baseline \
    eval_freq=100000 num_eval_episodes=20
