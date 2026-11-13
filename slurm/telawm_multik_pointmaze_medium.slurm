#!/bin/bash
#SBATCH --job-name=telawm-multik-pointmaze-medium-navigate
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate telawm

IQL_SEED=0
IQL_RUN_SUFFIX=baseline
IQL_CHECKPOINT=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}_${IQL_RUN_SUFFIX}/models/latest.pt

MULTIK_SEED=0

python -u train.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=telawm \
    seed=${MULTIK_SEED} gamma=0.99 steps=1001000 \
    compile=false cudagraphs=false \
    run_suffix=multik wandb_name_suffix=_multik \
    eval_freq=100000 num_eval_episodes=20 \
    telawm.k_conditioned=true 'telawm.train_k_values=[10,25,50,100]' \
    telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 \
    telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 \
    telawm.subgoal_state=first telawm.n_sg_samples=128 \
    telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 \
    telawm.pretrained_vf=true "telawm.vf_chkpt=$IQL_CHECKPOINT" \
    telawm.pretrained_policy=true "telawm.policy_chkpt=$IQL_CHECKPOINT"


MULTIK_RUN_SUFFIX=multik
MULTIK_CHECKPOINT=output/train/telawm_pointmaze-medium-navigate-v0_${MULTIK_SEED}_${MULTIK_RUN_SUFFIX}/models/latest.pt

python -u evaluate.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=telawm \
    seed=${MULTIK_SEED} gamma=0.99 compile=false cudagraphs=false \
    run_suffix=multik_sweep "eval_checkpoint=$MULTIK_CHECKPOINT" \
    eval_k_sweep=true 'eval_task_ids=[1,2,3,4,5]' num_eval_episodes=20 \
    telawm.k_conditioned=true telawm.latent_dim=2 telawm.learned_prior=true \
    telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 \
    telawm.subgoal_state=first telawm.n_sg_samples=128 \
    telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 \
    telawm.pretrained_vf=true "telawm.vf_chkpt=$IQL_CHECKPOINT" \
    telawm.pretrained_policy=true "telawm.policy_chkpt=$IQL_CHECKPOINT"