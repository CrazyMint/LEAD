#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MATH GRPO Training Script - DeepSeek-R1-Distill-Qwen-1.5B

ENV_FILE="$(dirname $0)/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .env not found at $ENV_FILE"
    echo "  Run: cp .env.example .env && \$EDITOR .env"
    exit 1
fi

# Honor pre-set values; default to 4 GPUs if unset.
# To run on fewer GPUs:  N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash <this script>
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_GPUS="${N_GPUS:-4}"
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Paths - can be overridden via environment variables
export DATA_DIR="${DATA_DIR:-$(dirname $0)/data/math}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek-r1-1.5B-math-GRPO}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT:-./results}/math_grpo_deepseek-r1-1.5b}"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Reward configuration (math dataset uses data_source="deepscaler" → reads DEEPSCALE_* env vars)
export DEEPSCALE_CORRECT_REWARD=1.0
export DEEPSCALE_LENGTH_REWARD=1.0
export DEEPSCALE_LENGTH_THRESHOLD=2000
export DEEPSCALE_LENGTH_MODE=conditioned

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2000 \
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
    actor_rollout_ref.rollout.val_kwargs.max_tokens=2000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=MATH_GRPO \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.wandb_kwargs.resume=allow \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=7
