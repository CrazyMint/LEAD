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
DeepscaleR reward scoring function for math reasoning.
Uses the grading functions from deepscaler/rewards/math_utils/utils.py
Supports correctness + format + length rewards for LEAD training.
"""

import os
import sys
from typing import Union, List

# Add GRPO path to import deepscaler utilities
GRPO_PATH = "$HOME/GRPO"
if GRPO_PATH not in sys.path:
    sys.path.insert(0, GRPO_PATH)

try:
    from deepscaler.rewards.math_utils.utils import (
        extract_answer,
        grade_answer_sympy,
        grade_answer_mathd,
    )
    HAS_DEEPSCALER = True
except ImportError:
    HAS_DEEPSCALER = False
    print("Warning: Could not import deepscaler utilities. Using fallback grading.")


def _fallback_extract_answer(passage: str) -> str:
    """Fallback answer extraction from \\boxed{}."""
    if passage is None or "\\boxed" not in passage:
        return None

    idx = passage.rfind("\\boxed")
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(passage):
        if passage[i] == "{":
            num_left_braces_open += 1
        if passage[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed = passage[idx:right_brace_idx + 1]
    left = "\\boxed{"
    try:
        assert boxed[:len(left)] == left
        assert boxed[-1] == "}"
        return boxed[len(left):-1]
    except:
        return None


def _fallback_grade_answer(model_answer: str, ground_truth: str) -> bool:
    """Fallback answer grading (simple string match)."""
    if model_answer is None or ground_truth is None:
        return False

    # Normalize
    def normalize(s):
        s = str(s).strip().lower()
        s = s.replace(" ", "").replace(",", "")
        return s

    if normalize(model_answer) == normalize(ground_truth):
        return True

    # Try numeric comparison
    try:
        m = float(normalize(model_answer))
        g = float(normalize(ground_truth))
        return abs(m - g) < 1e-6
    except:
        return False


def compute_score(
    solution_str: str,
    ground_truth: Union[str, List[str]],
    step: int = 0,
):
    """
    Compute score for deepscaleR dataset.

    Returns: (total_score, format_score, correctness_score, length_score)

    Rewards:
    - R_correct = 1 if answer is correct, 0 otherwise
    - R_length: controlled by DEEPSCALE_LENGTH_MODE
        - "classic": R_length = 1 if length <= threshold, 0 otherwise
        - "conditioned": R_length = 1 if correct AND length <= threshold, 0 otherwise
        - "soft": linear decay from 1→0 between threshold and max_length
        - "conditioned_soft": soft decay, but 0 if answer is incorrect
    - R_format: controlled by DEEPSCALE_FORMAT_MODE (requires DEEPSCALE_USE_FORMAT=1)
        - "boxed": 1 if \\boxed{} present, 0 otherwise
        - "strict": partial credit — 0.5 for \\boxed{} + 0.5 for proper <think>...</think> structure
    """
    correctness_reward = float(os.getenv("DEEPSCALE_CORRECT_REWARD", 1.0))
    length_reward = float(os.getenv("DEEPSCALE_LENGTH_REWARD", 1.0))
    length_threshold = int(os.getenv("DEEPSCALE_LENGTH_THRESHOLD", 4000))
    length_max = int(os.getenv("DEEPSCALE_LENGTH_MAX", 4000))
    length_mode = os.getenv("DEEPSCALE_LENGTH_MODE", "classic")
    format_reward = float(os.getenv("DEEPSCALE_FORMAT_REWARD", 1.0))
    format_mode = os.getenv("DEEPSCALE_FORMAT_MODE", "boxed")
    use_format = os.getenv("DEEPSCALE_USE_FORMAT", "0") == "1"

    if HAS_DEEPSCALER:
        _extract = extract_answer
        def _grade(model, truth):
            return grade_answer_mathd(model, truth) or grade_answer_sympy(model, truth)
    else:
        _extract = _fallback_extract_answer
        _grade = _fallback_grade_answer

    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]

    processed_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            extracted = _extract(truth)
            if extracted:
                processed_truths.append(extracted)
            else:
                processed_truths.append(truth)
        else:
            processed_truths.append(truth)

    model_answer = _extract(solution_str)

    is_correct = False
    if model_answer is not None:
        for truth in processed_truths:
            if _grade(model_answer, truth):
                is_correct = True
                break

    correctness_score = correctness_reward if is_correct else 0.0

    estimated_tokens = len(solution_str) / 4
    within_length = estimated_tokens <= length_threshold

    if length_mode == "conditioned":
        length_score = length_reward if (is_correct and within_length) else 0.0
    elif length_mode == "soft":
        if estimated_tokens <= length_threshold:
            length_score = length_reward
        elif estimated_tokens >= length_max:
            length_score = 0.0
        else:
            length_score = length_reward * (1.0 - (estimated_tokens - length_threshold) / (length_max - length_threshold))
    elif length_mode == "conditioned_soft":
        if not is_correct:
            length_score = 0.0
        elif estimated_tokens <= length_threshold:
            length_score = length_reward
        elif estimated_tokens >= length_max:
            length_score = 0.0
        else:
            length_score = length_reward * (1.0 - (estimated_tokens - length_threshold) / (length_max - length_threshold))
    else:
        length_score = length_reward if within_length else 0.0

    if use_format:
        if format_mode == "strict":
            # Partial credit: 0.5 for \boxed{} + 0.5 for proper <think> structure
            fmt = 0.0
            has_boxed = model_answer is not None
            has_think = ("<think>" in solution_str and "</think>" in solution_str)
            if has_boxed:
                fmt += 0.5
            if has_think:
                think_end = solution_str.rfind("</think>")
                boxed_pos = solution_str.rfind("\\boxed")
                if think_end >= 0 and (boxed_pos < 0 or think_end < boxed_pos):
                    fmt += 0.5  # proper order: reasoning before answer
                else:
                    fmt += 0.25  # has think tags but wrong order
            format_score = fmt * format_reward
        else:
            # "boxed" mode: binary check for \boxed{}
            format_score = format_reward if model_answer is not None else 0.0
    else:
        format_score = 0.0

    total_score = correctness_score + length_score + format_score
    return total_score, format_score, correctness_score, length_score
