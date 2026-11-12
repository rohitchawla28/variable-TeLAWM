# File to hold commands for different experiments.

# compile=false is needed for using older GPUs

##################### PointMaze Medium Navigate (state obs)

# ==================== IQL ==================
# all defaults

IQL_SEED=0
IQL_RUN_SUFFIX=baseline
IQL_CHECKPOINT=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}_${IQL_RUN_SUFFIX}/models/latest.pt

python -u train.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=iql \
    seed=0 gamma=0.99 iql.expectile=0.7 iql.alpha=0.01 steps=1001000 \
    compile=false cudagraphs=false \
    run_suffix=baseline wandb_name_suffix=_baseline \
    eval_freq=100000 num_eval_episodes=20


# =================== Baseline Te-LAWM ===================
# sticking to all defaults right now

TELAWM_SEED=0
TELAWM_RUN_SUFFIX=fixedk25
TELAWM_CHECKPOINT=output/train/telawm_pointmaze-medium-navigate-v0_${TELAWM_SEED}_${TELAWM_RUN_SUFFIX}/models/latest.pt

python -u train.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=telawm \
    seed=0 gamma=0.99 steps=1001000 \
    compile=false cudagraphs=false \
    run_suffix=fixedk25 wandb_name_suffix=_fixedk25 \
    eval_freq=100000 num_eval_episodes=20 \
    telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 \
    telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 \
    telawm.subgoal_state=first telawm.n_sg_samples=128 \
    telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 \
    telawm.pretrained_vf=true "telawm.vf_chkpt=$IQL_CHECKPOINT" \
    telawm.pretrained_policy=true "telawm.policy_chkpt=$IQL_CHECKPOINT"

python -u evaluate.py \
    task=pointmaze-medium-navigate-v0 obs=state agent=telawm \
    seed=0 gamma=0.99 compile=false cudagraphs=false \
    run_suffix=fixedk25 "eval_checkpoint=$TELAWM_CHECKPOINT" \
    'eval_task_ids=[1,2,3,4,5]' num_eval_episodes=20 \
    save_video=true num_video_episodes=1 \
    telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 \
    telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 \
    telawm.subgoal_state=first telawm.n_sg_samples=128 \
    telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 \
    telawm.pretrained_vf=true "telawm.vf_chkpt=$IQL_CHECKPOINT" \
    telawm.pretrained_policy=true "telawm.policy_chkpt=$IQL_CHECKPOINT"


