#!/bin/bash
# =============================================================================
# GRPO with static reward-ratio sweep - DeepSeek-R1-Distill-Qwen-1.5B
# =============================================================================
# Reward model: r = lambda_1 * r_corr + lambda_2 * r_len
# Sweep parameter: RATIO = lambda_2 / lambda_1
# To keep the total reward magnitude constant across runs (so the KL
# regularizer is not implicitly rescaled), we fix lambda_1 + lambda_2 = 1 and
# derive each pair from RATIO:
#     lambda_1 = 1 / (1 + RATIO)
#     lambda_2 = RATIO / (1 + RATIO)
#
#   RATIO = 0     -> (lambda_1=1.00, lambda_2=0.00)   pure correctness
#   RATIO = 0.25  -> (lambda_1=0.80, lambda_2=0.20)
#   RATIO = 0.5   -> (lambda_1=0.67, lambda_2=0.33)
#   RATIO = 1.0   -> (lambda_1=0.50, lambda_2=0.50)   balanced
#   RATIO = 2.0   -> (lambda_1=0.33, lambda_2=0.67)   length-favored
#   RATIO = 4.0   -> (lambda_1=0.20, lambda_2=0.80)
#   RATIO = inf   -> (lambda_1=0.00, lambda_2=1.00)   pure length
#
# Usage:
#   RATIO=0.25 bash train_math_grpo_lambda_1.5b.sh
#   RATIO=1.0  bash train_math_grpo_lambda_1.5b.sh
#   RATIO=4.0  bash train_math_grpo_lambda_1.5b.sh
# =============================================================================

source "$(dirname $0)/../../.env"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS=${N_GPUS:-4}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ratio to sweep; default 1.0 reproduces the balanced GRPO baseline
export RATIO="${RATIO:-1.0}"

# Derive the two coefficients from RATIO with sum fixed at 1.0
LAMBDA1=$($HOME/miniconda3/envs/pissa/bin/python -c "r=float('$RATIO'); print(f'{1.0/(1.0+r):.6f}')")
LAMBDA2=$($HOME/miniconda3/envs/pissa/bin/python -c "r=float('$RATIO'); print(f'{r/(1.0+r):.6f}')")

# Paths
export DATA_DIR="${DATA_DIR:-$(dirname $0)/../../data/math}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

# Tag the output dir by ratio (e.g., 0.25 -> 025, 1.0 -> 10, 4.0 -> 40)
RATIO_TAG=$(printf '%s' "$RATIO" | tr -d '.')
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-GRPO-math-ratio${RATIO_TAG}}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT:-./results}/math_grpo_ratio${RATIO_TAG}_deepseek-r1-1.5b}"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Reward configuration:
# r = lambda_1 * 1[correct] + lambda_2 * 1[within_threshold]
export DEEPSCALE_CORRECT_REWARD="$LAMBDA1"
export DEEPSCALE_LENGTH_REWARD="$LAMBDA2"
export DEEPSCALE_LENGTH_THRESHOLD=2000
export DEEPSCALE_LENGTH_MODE=classic

echo "=========================================="
echo " GRPO ratio-sweep run"
echo "   RATIO       : $RATIO"
echo "   lambda_1 (correctness): $LAMBDA1"
echo "   lambda_2 (length)     : $LAMBDA2"
echo "   Experiment  : $EXPERIMENT_NAME"
echo "   Ckpt dir    : $CKPT_DIR"
echo "=========================================="

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    trainer.project_name=MATH_GRPO \
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
