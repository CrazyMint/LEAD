#!/bin/bash
# =============================================================================
# Ablation 4 (single-run template): aggregator for L*_q over correct rollouts
# =============================================================================
# Holds LEAD fixed; only varies the AGGREGATOR used to compute L*_q
# from the rollouts in each prompt's group.
#
# Aggregator options (set via env var AGG):
#   mean_correct    (default, used by main LEAD)
#   min_correct     SOL-style; min length of correct rollouts
#   median_correct  median length of correct rollouts
#   mean_all        mean over ALL rollouts (no correctness filter); confounds
#                   correctness with length and is included as a baseline
#
# Usage:
#   AGG=min_correct    bash train_lead_aggregator.sh
#   AGG=median_correct bash train_lead_aggregator.sh
#   AGG=mean_all       bash train_lead_aggregator.sh
# =============================================================================

source "$(dirname $0)/../../.env"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS=${N_GPUS:-4}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Aggregator under ablation
export AGG="${AGG:-mean_correct}"

# Paths
export DATA_DIR="${DATA_DIR:-$(dirname $0)/../../data/math}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-LEAD-agg-${AGG}}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT:-./results}/math_lead_agg_${AGG}_deepseek-r1-1.5b}"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Same reward configuration as the main LEAD run; only the aggregator changes
export DEEPSCALE_CORRECT_REWARD=1.0
export DEEPSCALE_LENGTH_REWARD=0.0
export DEEPSCALE_LENGTH_MODE=classic

echo "=========================================="
echo " Ablation 4: aggregator = $AGG"
echo "   Experiment  : $EXPERIMENT_NAME"
echo "   Ckpt dir    : $CKPT_DIR"
echo "=========================================="

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=lead \
    algorithm.lead_alpha=1.0 \
    algorithm.lead_beta=0.95 \
    algorithm.lead_epsilon=1e-8 \
    algorithm.lead_bmax=8000 \
    algorithm.lead_lambda_min=0.3 \
    algorithm.lead_lstar_mode=max_sym \
    algorithm.lead_aggregator=$AGG \
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
