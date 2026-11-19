#!/bin/bash
#SBATCH --job-name=vtelawm-antmaze-large-navigate
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

SEED=0
TASK=antmaze-large-navigate-v0
IQL_CHECKPOINT=${IQL_CHECKPOINT:-output/train/iql_${TASK}_${SEED}/models/latest.pt}
RUN_SUFFIX=multik_film
# Fixed K used for periodic rollouts. Offline batches sample all four horizons.
EVAL_K=50

common_args=(
    "task=$TASK" obs=state agent=telawm "seed=$SEED" compile=false cudagraphs=false
    telawm.k_conditioning=film telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004
    telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128
    telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30
    telawm.pretrained_vf=true "telawm.vf_chkpt=$IQL_CHECKPOINT"
    telawm.pretrained_policy=true "telawm.policy_chkpt=$IQL_CHECKPOINT"
    num_eval_episodes=20
)

python -u train.py "${common_args[@]}" steps=1001000 eval_freq=100000 \
    "run_suffix=$RUN_SUFFIX" "wandb_name_suffix=_${RUN_SUFFIX}" \
    'telawm.train_k_values=[10,25,50,100]' "telawm.lam_timescale=$EVAL_K" "telawm.subgoal_steps=$EVAL_K"

CHECKPOINT=output/train/telawm_${TASK}_${SEED}_${RUN_SUFFIX}/models/latest.pt
for K in 10 25 50 100; do
    python -u evaluate.py "${common_args[@]}" "eval_checkpoint=$CHECKPOINT" \
        "run_suffix=${RUN_SUFFIX}_sweep_steps_k${K}" "telawm.lam_timescale=$K" "telawm.subgoal_steps=$K" \
        eval_k_sweep=false 'eval_task_ids=[1,2,3,4,5]' save_video=true num_video_episodes=1
done
