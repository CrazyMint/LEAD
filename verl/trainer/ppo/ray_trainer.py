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
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean, masked_whiten



def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, lead_state=None, gdpo_weights=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == 'grpo_no_std':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_no_std_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns


    elif adv_estimator == 'gdpo':
        ## handle correctness, length, and optionally format rewards
        token_level_scores_correctness = data.batch['token_level_scores_correctness']
        token_level_scores_format = data.batch['token_level_scores_format']
        token_level_scores_length = data.batch.get('token_level_scores_length', None)

        # shared variables
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        ## handle correctness
        correctness_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_correctness,
                                                                        eos_mask=response_mask,
                                                                        index=index)

        # Apply static weights: gdpo_weights=[correctness, format, length]
        w_correctness = gdpo_weights[0] if gdpo_weights is not None else 1.0
        w_format = gdpo_weights[1] if gdpo_weights is not None else 1.0
        w_length = gdpo_weights[2] if gdpo_weights is not None else 1.0

        new_advantage = w_correctness * correctness_normalized_score

        ## handle format (skip if weight=0 or scores all zero)
        if w_format != 0 and token_level_scores_format.abs().sum() > 0:
            format_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_format,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            new_advantage = new_advantage + w_format * format_normalized_score

        ## handle length reward (skip if weight=0 or scores all zero)
        if w_length != 0 and token_level_scores_length is not None and token_level_scores_length.abs().sum() > 0:
            length_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_length,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            new_advantage = new_advantage + w_length * length_normalized_score

        advantages = masked_whiten(new_advantage, response_mask) * response_mask

        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages

    elif adv_estimator == 'lead':
        # LEAD: Efficient Reasoning via Dynamic Budget Calibration
        # Computes per-problem optimal length L* from correct rollouts online,
        # uses asymmetric D_k (Verbosity Potential for efficiency) with LEAD weighting.

        if lead_state is None:
            raise ValueError("lead_state must be provided for LEAD advantage estimator")

        # Get reward tensors
        token_level_scores_correctness = data.batch['token_level_scores_correctness']

        # Shared variables
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        device = response_mask.device

        # Extract hyperparameters
        eff_alpha = lead_state.get('alpha', 1.0)
        eff_beta = lead_state.get('beta', 0.95)
        eff_epsilon = lead_state.get('epsilon', 1e-8)
        eff_bmax = lead_state.get('bmax', 8000)
        eff_w_min = lead_state.get('w_min', 0.3)
        eff_lstar_mode = lead_state.get('lstar_mode', 'max_asym')
        eff_aggregator = lead_state.get('aggregator', 'mean_correct')

        # Increment step counter
        lead_state['step_count'] = lead_state.get('step_count', 0) + 1
        current_step = lead_state['step_count']

        eff_metrics = {}
        eff_metrics['lead/step'] = float(current_step)
        eff_metrics['lead/lstar_mode'] = 0.0 if eff_lstar_mode == 'max_asym' else 1.0

        # --- Phase 1: Compute per-problem optimal length L* and R_eff ---
        # Extract scalar correctness scores per trajectory
        raw_corr_scores = (token_level_scores_correctness * response_mask).sum(dim=-1)
        is_correct = (raw_corr_scores > 0.5).float()

        # Response token lengths per trajectory
        resp_lengths = response_mask.sum(dim=-1).float()

        # Group rollouts by prompt uid, compute L* per prompt
        unique_uids = list(set(index))
        uid_to_lstar = {}

        for uid in unique_uids:
            uid_indices = [i for i, x in enumerate(index) if x == uid]
            all_lengths = [int(resp_lengths[idx].item()) for idx in uid_indices]
            correct_lengths = []
            for idx in uid_indices:
                if is_correct[idx] > 0.5:
                    correct_lengths.append(resp_lengths[idx].item())

            if eff_aggregator == 'mean_all':
                # Aggregate over ALL rollouts (correctness-blind). Ablation baseline.
                if len(all_lengths) >= 1:
                    raw = sum(all_lengths) / len(all_lengths)
                    uid_to_lstar[uid] = max(min(raw, eff_bmax), 1000)
                else:
                    uid_to_lstar[uid] = eff_bmax
            elif len(correct_lengths) >= 1:
                if eff_aggregator == 'min_correct':
                    raw = min(correct_lengths)             # SOL-style
                elif eff_aggregator == 'median_correct':
                    sl = sorted(correct_lengths)
                    n = len(sl)
                    raw = sl[n // 2] if n % 2 == 1 else 0.5 * (sl[n // 2 - 1] + sl[n // 2])
                else:  # 'mean_correct' (default)
                    raw = sum(correct_lengths) / len(correct_lengths)
                # L* = clamp(raw, 1000, eff_bmax) — upper bound matches configured budget
                uid_to_lstar[uid] = max(min(raw, eff_bmax), 1000)
            else:
                uid_to_lstar[uid] = eff_bmax  # no correct rollout: no efficiency pressure

            # SB-style per-problem rollout log
            print(f"Lengths={all_lengths}, Correct={[int(l) for l in correct_lengths]}, L*={int(uid_to_lstar[uid])}", flush=True)

        # Compute R_eff per trajectory based on lstar_mode
        r_eff = torch.zeros_like(raw_corr_scores)
        for i in range(len(index)):
            lstar = uid_to_lstar[index[i]]
            l = resp_lengths[i].item()

            if eff_lstar_mode == 'max_sym':
                # Symmetric: penalize deviation from L* in both directions
                # R_eff = 1 - |l - L*| / L*, clamped to [-1, 1]
                r_eff[i] = max(-1.0, 1.0 - abs(l - lstar) / lstar)
            elif eff_lstar_mode == 'upper_only':
                # Upper-only: penalize over-length only; no reward for being short.
                #   R_eff = min(0, 1 - l/L*) -> 0 when l <= L*, negative when l > L*
                # Ablation baseline that matches the standard global-budget formulation.
                r_eff[i] = max(-1.0, min(0.0, 1.0 - l / lstar))
            else:
                # 'max_asym' (default): asymmetric
                # Correct: max(0, 1 - l/L*) — never negative, shorter = higher reward
                # Incorrect: clamp(1 - l/L*, -1, 0) — non-positive, longer = more penalty
                raw_eff = 1.0 - l / lstar
                if is_correct[i] > 0.5:
                    if l < 500:
                        r_eff[i] = 0.0  # suspiciously short: no efficiency reward
                    else:
                        r_eff[i] = max(0.0, raw_eff)  # correct: never penalized
                else:
                    r_eff[i] = max(-1.0, min(0.0, raw_eff))  # incorrect: non-positive

        # Create token-level efficiency reward tensor (scalar at last valid token)
        token_level_scores_efficiency = torch.zeros_like(token_level_scores_correctness)
        for i in range(len(r_eff)):
            valid_len = int(response_mask[i].sum().item())
            if valid_len > 0:
                token_level_scores_efficiency[i, valid_len - 1] = r_eff[i]

        # Log L* statistics
        lstar_values = list(uid_to_lstar.values())
        lstar_non_bmax = [v for v in lstar_values if v < eff_bmax]
        eff_metrics['lead/lstar_mean'] = float(np.mean(lstar_values)) if lstar_values else 0.0
        eff_metrics['lead/lstar_mean_solvable'] = float(np.mean(lstar_non_bmax)) if lstar_non_bmax else 0.0
        eff_metrics['lead/lstar_median_solvable'] = float(np.median(lstar_non_bmax)) if lstar_non_bmax else 0.0
        eff_metrics['lead/lstar_min_solvable'] = float(np.min(lstar_non_bmax)) if lstar_non_bmax else 0.0
        eff_metrics['lead/lstar_max_solvable'] = float(np.max(lstar_non_bmax)) if lstar_non_bmax else 0.0
        eff_metrics['lead/num_solvable_prompts'] = float(len(lstar_non_bmax))
        eff_metrics['lead/num_unsolved_prompts'] = float(len(lstar_values) - len(lstar_non_bmax))
        eff_metrics['lead/accuracy_batch'] = float(is_correct.mean().item())
        # R_eff statistics (this is the internal efficiency reward, NOT the external length_score)
        eff_metrics['lead/r_eff_mean'] = float(r_eff.mean().item())
        eff_metrics['lead/r_eff_mean_correct'] = float(r_eff[is_correct > 0.5].mean().item()) if (is_correct > 0.5).any() else 0.0
        eff_metrics['lead/r_eff_min_correct'] = float(r_eff[is_correct > 0.5].min().item()) if (is_correct > 0.5).any() else 0.0
        eff_metrics['lead/r_eff_max_correct'] = float(r_eff[is_correct > 0.5].max().item()) if (is_correct > 0.5).any() else 0.0
        eff_metrics['lead/resp_len_mean'] = float(resp_lengths.mean().item())
        eff_metrics['lead/resp_len_mean_correct'] = float(resp_lengths[is_correct > 0.5].mean().item()) if (is_correct > 0.5).any() else 0.0

        # --- Phase 2: GRPO-normalize each reward independently ---
        correctness_adv, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores_correctness,
            eos_mask=response_mask,
            index=index)

        efficiency_adv, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores_efficiency,
            eos_mask=response_mask,
            index=index)

        # --- Phase 3: D_k computation with normalized CV ---
        # Compute batch statistics using Law of Total Variance on RAW scores
        # R_corr: computed over ALL trajectories
        # R_eff: computed over CORRECT trajectories only (incorrect carry no efficiency signal)

        # --- R_corr stats (all trajectories) ---
        corr_group_means = []
        corr_group_vars = []
        for uid in unique_uids:
            uid_indices = [i for i, x in enumerate(index) if x == uid]
            uid_scores = raw_corr_scores[uid_indices]
            corr_group_means.append(uid_scores.mean())
            if len(uid_indices) > 1:
                corr_group_vars.append(uid_scores.var(unbiased=False))
            else:
                corr_group_vars.append(torch.tensor(0.0, device=device))
        corr_group_means = torch.stack(corr_group_means)
        corr_group_vars = torch.stack(corr_group_vars)
        mu_corr = corr_group_means.mean()
        sigma_corr = torch.sqrt(corr_group_vars.mean() + corr_group_means.var(unbiased=False) + 1e-8)

        # --- R_eff stats (correct trajectories only) ---
        eff_group_means = []
        eff_group_vars = []
        for uid in unique_uids:
            uid_indices = [i for i, x in enumerate(index) if x == uid]
            correct_indices = [i for i in uid_indices if is_correct[i] > 0.5]
            if len(correct_indices) >= 1:
                uid_scores = r_eff[correct_indices]
                eff_group_means.append(uid_scores.mean())
                if len(correct_indices) > 1:
                    eff_group_vars.append(uid_scores.var(unbiased=False))
                else:
                    eff_group_vars.append(torch.tensor(0.0, device=device))
            # Skip prompts with no correct trajectories — they have no efficiency signal
        if len(eff_group_means) >= 1:
            eff_group_means = torch.stack(eff_group_means)
            eff_group_vars = torch.stack(eff_group_vars)
            mu_eff = eff_group_means.mean()
            sigma_eff = torch.sqrt(eff_group_vars.mean() + eff_group_means.var(unbiased=False) + 1e-8)
        else:
            # No correct trajectories at all — D_eff should be 0
            mu_eff = torch.tensor(0.0, device=device)
            sigma_eff = torch.tensor(0.0, device=device)

        # Compute CV for each reward
        cv_corr = sigma_corr / (torch.abs(mu_corr) + eff_epsilon)
        cv_eff = sigma_eff / (torch.abs(mu_eff) + eff_epsilon)

        # Normalize CVs to same scale before computing D_k
        cv_sum = cv_corr + cv_eff + eff_epsilon
        cv_corr_norm = cv_corr / cv_sum
        cv_eff_norm = cv_eff / cv_sum

        # Standard Potential for both rewards
        potential_corr = (1.0 - mu_corr.clamp(0, 1)) ** eff_alpha
        R_eff_min, R_eff_max = -1.0, 1.0
        potential_eff = (1.0 - (mu_eff.clamp(R_eff_min, R_eff_max) - R_eff_min) / (R_eff_max - R_eff_min)) ** eff_alpha

        D_corr = cv_corr_norm * potential_corr
        D_eff = cv_eff_norm * potential_eff

        eff_metrics['lead/mu_corr'] = mu_corr.item()
        eff_metrics['lead/sigma_corr'] = sigma_corr.item()
        eff_metrics['lead/cv_corr'] = cv_corr.item()
        eff_metrics['lead/cv_eff'] = cv_eff.item()
        eff_metrics['lead/cv_corr_norm'] = cv_corr_norm.item()
        eff_metrics['lead/cv_eff_norm'] = cv_eff_norm.item()
        eff_metrics['lead/potential_corr'] = potential_corr.item()
        eff_metrics['lead/potential_eff'] = potential_eff.item()
        eff_metrics['lead/D_corr'] = D_corr.item()
        eff_metrics['lead/D_eff'] = D_eff.item()
        eff_metrics['lead/mu_eff'] = mu_eff.item()
        eff_metrics['lead/sigma_eff'] = sigma_eff.item()

        # --- Phase 4: weight update ---
        # If static_w_corr is provided (ablation), bypass the dynamic computation
        # and use the fixed pair (w_corr, 1 - w_corr) at every step.
        eff_static_w_corr = lead_state.get('static_w_corr', None)
        D_sum = D_corr + D_eff + eff_epsilon
        target_w_corr_dyn = D_corr / D_sum
        target_w_eff_dyn = D_eff / D_sum

        if eff_static_w_corr is not None:
            w_corr = float(eff_static_w_corr)
            w_eff = 1.0 - w_corr
            # Persist the static pair so checkpoints reflect the actual weights used.
            lead_state['weights'] = torch.tensor([w_corr, w_eff]).detach().cpu()
            lead_state['initialized'] = True
            eff_metrics['lead/static_w_corr'] = w_corr
        else:
            target_weights = torch.tensor([target_w_corr_dyn.item(), target_w_eff_dyn.item()], device=device)

            # EMA smoothing
            prev_weights = lead_state.get('weights')
            if prev_weights is None or not lead_state.get('initialized', False):
                uniform = torch.tensor([0.5, 0.5], device=device)
                lead_state['weights'] = (eff_beta * uniform + (1 - eff_beta) * target_weights).detach().cpu()
                lead_state['initialized'] = True
            else:
                prev_w = prev_weights.to(device)
                lead_state['weights'] = (eff_beta * prev_w + (1 - eff_beta) * target_weights).detach().cpu()

            current_weights = lead_state['weights'].to(device)

            # Apply w_min floor: w_corr >= w_min, w_eff = 1 - w_corr
            w_corr = max(current_weights[0].item(), eff_w_min)
            w_eff = 1.0 - w_corr

        eff_metrics['lead/target_w_corr'] = target_w_corr_dyn.item()
        eff_metrics['lead/target_w_eff'] = target_w_eff_dyn.item()
        eff_metrics['lead/w_corr'] = w_corr
        eff_metrics['lead/w_eff'] = w_eff

        lead_state['metrics'] = eff_metrics

        # --- Phase 5: Weighted combination + BatchNorm ---
        new_advantage = w_corr * correctness_adv + w_eff * efficiency_adv
        advantages = masked_whiten(new_advantage, response_mask) * response_mask

        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages

    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    
    sequence_score_format = batch.batch['token_level_scores_format'].sum(-1)
    sequence_score_correctness = batch.batch['token_level_scores_correctness'].sum(-1)
    sequence_score_length = batch.batch['token_level_scores_length'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # format score
        'critic/format_score/mean':
            torch.mean(sequence_score_format).detach().item(),
        'critic/format_score/max':
            torch.max(sequence_score_format).detach().item(),
        'critic/format_score/min':
            torch.min(sequence_score_format).detach().item(),
        # correctness score
        'critic/correctness_score/mean':
            torch.mean(sequence_score_correctness).detach().item(),
        'critic/correctness_score/max':
            torch.max(sequence_score_correctness).detach().item(),
        'critic/correctness_score/min':
            torch.min(sequence_score_correctness).detach().item(),
        # length score
        'critic/length_score/mean':
            torch.mean(sequence_score_length).detach().item(),
        'critic/length_score/max':
            torch.max(sequence_score_length).detach().item(),
        'critic/length_score/min':
            torch.min(sequence_score_length).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/var':
            torch.var(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        # GDPO static weights: always [correctness, format, length]
        # Prefer environment variables over OmegaConf config (bypasses ListConfig issues)
        _gdpo_w_c = os.environ.get('GDPO_W_CORRECTNESS', None)
        _gdpo_w_f = os.environ.get('GDPO_W_FORMAT', None)
        _gdpo_w_l = os.environ.get('GDPO_W_LENGTH', None)
        if _gdpo_w_c is not None or _gdpo_w_f is not None or _gdpo_w_l is not None:
            self.gdpo_weights = [
                float(_gdpo_w_c or '1.0'),
                float(_gdpo_w_f or '1.0'),
                float(_gdpo_w_l or '1.0'),
            ]
        else:
            self.gdpo_weights = config.algorithm.get('gdpo_weights', None)
            if self.gdpo_weights is not None:
                # Convert OmegaConf ListConfig to plain Python floats and pad to 3
                self.gdpo_weights = [float(w) for w in self.gdpo_weights]
                while len(self.gdpo_weights) < 3:
                    self.gdpo_weights.append(1.0)
        if self.gdpo_weights is not None:
            print(f"GDPO static weights: correctness={self.gdpo_weights[0]}, format={self.gdpo_weights[1]}, length={self.gdpo_weights[2]}")

        
        # LEAD state initialization
        self.lead_weights = None
        self.lead_initialized = False
        self.lead_alpha = config.algorithm.get('lead_alpha', 1.0)
        self.lead_beta = config.algorithm.get('lead_beta', 0.95)
        self.lead_epsilon = config.algorithm.get('lead_epsilon', 1e-8)
        self.lead_bmax = config.algorithm.get('lead_bmax', 8000)
        self.lead_lambda_min = config.algorithm.get('lead_lambda_min', 0.3)
        self.lead_lstar_mode = config.algorithm.get('lead_lstar_mode', 'max_asym')
        # Aggregator for L*_q over the rollouts in a group:
        #   'mean_correct'   : mean of correct rollouts (default, used by LEAD main)
        #   'min_correct'    : min of correct rollouts (SOL-style, ShorterBetter)
        #   'median_correct' : median of correct rollouts
        #   'mean_all'       : mean over ALL rollouts (no correctness filter)
        self.lead_aggregator = config.algorithm.get('lead_aggregator', 'mean_correct')
        # If set (not None), bypass the dynamic CV*Potential weight computation and
        # use this fixed value as w_corr (with w_eff = 1 - w_corr). Used by the
        # static-vs-dynamic ablation; leave None for the default LEAD behaviour.
        self.lead_static_lambda_corr = config.algorithm.get('lead_static_lambda_corr', None)
        self._lead_step_count = 0


        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         # use_chat_template=self.config.data.use_chat_template,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='left')
        # Use a Generator with fixed seed for reproducible batch ordering across resumes
        train_generator = torch.Generator()
        train_generator.manual_seed(42)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           generator=train_generator)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       # use_chat_template=self.config.data.use_chat_template,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='left')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        format_tensor_lst = []
        correctness_tensor_lst = []
        length_tensor_lst = []

        data_source_lst = []
        uid_lst = []

        # Get validation sampling config from val_kwargs (following ME's approach)
        # Defaults: n=1 (single sample), do_sample=False (greedy)
        val_kwargs = self.config.actor_rollout_ref.rollout.get('val_kwargs', {})
        val_n = val_kwargs.get('n', 1)
        val_do_sample = val_kwargs.get('do_sample', False)
        val_temperature = val_kwargs.get('temperature', self.config.actor_rollout_ref.rollout.temperature)
        val_max_tokens = val_kwargs.get('max_tokens', None)  # None means use default response_length

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Generate unique uid for each prompt BEFORE repeating
            batch_size = len(test_batch)
            if 'uid' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
                )

            # Repeat test_batch BEFORE popping for generation (following ME's approach)
            # This ensures batch sizes align after generation
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': val_do_sample,
                'validate': True,
                'temperature': val_temperature if val_do_sample else 1.0,
            }
            # Add max_tokens override for validation if specified
            if val_max_tokens is not None:
                test_gen_batch.meta_info['max_tokens'] = val_max_tokens

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Union test_batch with generated outputs (sizes now match)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor, format_tensor, correctness_tensor, length_tensor = self.val_reward_fn(test_batch, self.global_steps)

            reward_tensor_lst.append(reward_tensor)
            format_tensor_lst.append(format_tensor)
            correctness_tensor_lst.append(correctness_tensor)
            length_tensor_lst.append(length_tensor)
            # Get data_source and uid (already repeated in test_batch)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(test_batch)))
            uid_lst.append(test_batch.non_tensor_batch['uid'])

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        format_tensor = torch.cat(format_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_tensor = torch.cat(correctness_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        length_tensor = torch.cat(length_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        # Flatten lists of data sources and uids
        data_sources = [ds for batch_ds in data_source_lst for ds in batch_ds]
        sample_uids = [uid for batch_uids in uid_lst for uid in batch_uids]

        # Build infos_dict for process_validation_metrics
        infos_dict = {
            'reward': reward_tensor.tolist(),
            'correctness': correctness_tensor.tolist(),
            'format': format_tensor.tolist(),
            'length': length_tensor.tolist(),
        }

        # Compute @N metrics using process_validation_metrics
        data_src2var2metric2val = process_validation_metrics(
            data_sources=data_sources,
            sample_uids=sample_uids,
            infos_dict=infos_dict
        )

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            # Determine core variable (correctness if available, else reward)
            core_var = "correctness" if "correctness" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    # val-core: core metrics with mean/best/maj at max N
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Also keep the simple mean metrics for backward compatibility
        for data_source in set(data_sources):
            mask = data_sources == data_source
            metric_dict[f'val/test_score/{data_source}'] = reward_tensor[mask].mean().item()
            metric_dict[f'val/test_format/{data_source}'] = format_tensor[mask].mean().item()
            metric_dict[f'val/test_correctness/{data_source}'] = correctness_tensor[mask].mean().item()
            metric_dict[f'val/test_length/{data_source}'] = length_tensor[mask].mean().item()

        return metric_dict

    
    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.adv_estimator == 'grpo_no_std':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'gdpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'lead':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
            # self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                # self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

        # Save optimizer and LR scheduler state (sharded per rank).
        # Gated by trainer.save_optimizer so lambda-sweep / throwaway runs can skip it.
        if self.config.trainer.get('save_optimizer', True):
            optim_local_path = os.path.join(self.config.trainer.default_local_dir, 'optimizer',
                                            f'global_step_{self.global_steps}')
            self.actor_rollout_wg.save_optimizer_checkpoint(optim_local_path)

        # Save training state on driver (LEAD state, global_steps, etc.)
        training_state = {
            'global_steps': self.global_steps,
            'lead_weights': self.lead_weights.cpu() if self.lead_weights is not None and hasattr(self.lead_weights, 'cpu') else self.lead_weights,
            'lead_initialized': self.lead_initialized,
        }
        state_path = os.path.join(self.config.trainer.default_local_dir, 'training_state',
                                  f'global_step_{self.global_steps}')
        os.makedirs(state_path, exist_ok=True)
        torch.save(training_state, os.path.join(state_path, 'training_state.pt'))
        pprint(f'Saved training state to {state_path}')

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        # Detect resumed step from checkpoint directory
        self.global_steps = 0
        resume_step = 0
        if self.config.trainer.get('resume_mode', 'disable') == 'auto':
            ckpt_actor_dir = os.path.join(self.config.trainer.default_local_dir, 'actor')
            if os.path.exists(ckpt_actor_dir):
                import re
                steps = [int(re.search(r'global_step_(\d+)', d).group(1))
                         for d in os.listdir(ckpt_actor_dir)
                         if re.search(r'global_step_(\d+)', d)]
                if steps:
                    resume_step = max(steps)
                    pprint(f'Resuming from global_step_{resume_step}')

        self.global_steps = resume_step

        # Load optimizer and training state if resuming
        if resume_step > 0:
            # Load optimizer and LR scheduler state
            optim_path = os.path.join(self.config.trainer.default_local_dir, 'optimizer',
                                      f'global_step_{resume_step}')
            if os.path.exists(optim_path):
                pprint(f'Loading optimizer state from step {resume_step}...')
                self.actor_rollout_wg.load_optimizer_checkpoint(optim_path)
                pprint(f'Optimizer state loaded successfully')
            else:
                pprint(f'Warning: No optimizer checkpoint found at {optim_path}')

            # Load training state (LEAD state, etc.)
            state_path = os.path.join(self.config.trainer.default_local_dir, 'training_state',
                                      f'global_step_{resume_step}', 'training_state.pt')
            if os.path.exists(state_path):
                training_state = torch.load(state_path, map_location='cpu')
                self.lead_weights = training_state.get('lead_weights')
                self.lead_initialized = training_state.get('lead_initialized', False)
                pprint(f'Loaded training state from step {resume_step}: '
                       f'lead_initialized={self.lead_initialized}, '
                       f'lead_step_count={self._lead_step_count}')
            else:
                pprint(f'Warning: No training state found at {state_path}')

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Create progress bar for training
        pbar = tqdm(total=self.total_training_steps, initial=resume_step, desc="Training", unit="step")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                # Skip batches already processed before resume
                if resume_step > 0 and self.global_steps <= resume_step:
                    self.global_steps += 1
                    pbar.update(1)
                    continue

                pbar.set_description(f"Epoch {epoch}")
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, format_tensor, correctness_tensor, length_tensor = self.reward_fn(batch, self.global_steps)
                        ## reward_tensor = format_tensor + correctness_tensor + length_tensor

                        batch.batch['token_level_scores_format'] = format_tensor
                        batch.batch['token_level_scores_correctness'] = correctness_tensor
                        batch.batch['token_level_scores_length'] = length_tensor

                        # Apply static weights to re-combine rewards (for GRPO static)
                        if self.gdpo_weights is not None and self.config.algorithm.adv_estimator in ('grpo', 'grpo_no_std'):
                            w_c = self.gdpo_weights[0] if len(self.gdpo_weights) > 0 else 1.0
                            w_f = self.gdpo_weights[1] if len(self.gdpo_weights) > 1 else 1.0
                            w_l = self.gdpo_weights[2] if len(self.gdpo_weights) > 2 else 1.0
                            reward_tensor = w_c * correctness_tensor + w_f * format_tensor
                            if length_tensor is not None:
                                reward_tensor = reward_tensor + w_l * length_tensor

                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # Prepare LEAD state if using lead estimator
                        lead_state = None
                        if self.config.algorithm.adv_estimator == 'lead':
                            lead_state = {
                                'weights': self.lead_weights,
                                'initialized': self.lead_initialized,
                                'alpha': self.lead_alpha,
                                'beta': self.lead_beta,
                                'epsilon': self.lead_epsilon,
                                'bmax': self.lead_bmax,
                                'w_min': self.lead_lambda_min,
                                'lstar_mode': self.lead_lstar_mode,
                                'aggregator': self.lead_aggregator,
                                'static_w_corr': self.lead_static_lambda_corr,
                                'step_count': self._lead_step_count,
                            }

                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  lead_state=lead_state,
                                                  gdpo_weights=self.gdpo_weights)

                        # Update LEAD state and log metrics
                        if self.config.algorithm.adv_estimator == 'lead' and lead_state is not None:
                            self.lead_weights = lead_state.get('weights')
                            self.lead_initialized = lead_state.get('initialized', False)
                            self._lead_step_count = lead_state.get('step_count', self._lead_step_count)
                            if 'metrics' in lead_state:
                                metrics.update(lead_state['metrics'])

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                pbar.update(1)

                if self.global_steps >= self.total_training_steps:
                    pbar.close()

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

        pbar.close()
