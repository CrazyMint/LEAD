#!/bin/bash
# =============================================================================
# LEAD with STATIC reward weights — single-run template
# =============================================================================
# Holds LEAD fixed (decoupled normalization, per-problem L*_q from
# correct rollouts, symmetric R_eff), and varies only the WEIGHTING between
# the two normalized advantages.
#
# Reward model (decoupled):
#   A_combined = w_corr * A_corr + w_eff * A_eff
# where A_corr, A_eff are individually group-normalized.
#
# Sweep parameter: RATIO = w_eff / w_corr, with w_corr + w_eff = 1
#   w_corr = 1 / (1 + RATIO)
#   w_eff  = RATIO / (1 + RATIO)
#
#   RATIO = 0    -> (w_corr=1.00, w_eff=0.00)   pure correctness
#   RATIO = 1    -> (w_corr=0.50, w_eff=0.50)   balanced
#   RATIO = 4    -> (w_corr=0.20, w_eff=0.80)   length-favored
#
# Usage:
#   RATIO=0.5 bash train_math_lead_static_1.5b.sh
# =============================================================================

source "$(dirname $0)/../../.env"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS=${N_GPUS:-4}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ratio to sweep
export RATIO="${RATIO:-1.0}"

# Derive (w_corr, w_eff) with sum fixed at 1.0
W_CORR=$($HOME/miniconda3/envs/pissa/bin/python -c "r=float('$RATIO'); print(f'{1.0/(1.0+r):.6f}')")
W_EFF=$($HOME/miniconda3/envs/pissa/bin/python -c "r=float('$RATIO'); print(f'{r/(1.0+r):.6f}')")

# Paths
export DATA_DIR="${DATA_DIR:-$(dirname $0)/../../data/math}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

RATIO_TAG=$(printf '%s' "$RATIO" | tr -d '.')
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-LEAD-static-ratio${RATIO_TAG}}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT:-./results}/math_lead_static_ratio${RATIO_TAG}_deepseek-r1-1.5b}"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Same reward env as the main LEAD run; R_eff is computed online from L*_q
export DEEPSCALE_CORRECT_REWARD=1.0
export DEEPSCALE_LENGTH_REWARD=0.0
export DEEPSCALE_LENGTH_MODE=classic

echo "=========================================="
echo " LEAD static-ratio run"
echo "   RATIO   : $RATIO"
echo "   w_corr  : $W_CORR"
echo "   w_eff   : $W_EFF"
echo "   Exp     : $EXPERIMENT_NAME"
echo "   Ckpt    : $CKPT_DIR"
echo "=========================================="

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=lead \
    algorithm.lead_alpha=1.0 \
    algorithm.lead_beta=0.95 \
    algorithm.lead_epsilon=1e-8 \
    algorithm.lead_bmax=4000 \
    algorithm.lead_lambda_min=0.3 \
    algorithm.lead_lstar_mode=max_sym \
    algorithm.lead_aggregator=mean_correct \
    algorithm.lead_static_lambda_corr=$W_CORR \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4000 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=4000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=MATH_LEAD \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=disabled \
    trainer.wandb_kwargs.resume=allow \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.save_optimizer=false \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=7
