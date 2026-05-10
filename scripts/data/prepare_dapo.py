#!/usr/bin/env python3
"""Script to prepare DAPO-Math-17k training and test datasets.

Downloads the DAPO dataset (BytedTsinghua-SIA/DAPO-Math-17k) from HuggingFace,
deduplicates to ~17.9k unique competition-level math problems, rewrites prompts
to use \\boxed{} format, and saves in the same parquet format as DeepScaler
so existing reward functions (deepscale.py) work out of the box.

Usage:
    python prepare_dapo.py --local_dir ./data/dapo
    python prepare_dapo.py --local_dir ./data/dapo --num_samples 1000
    python prepare_dapo.py --local_dir ./data/dapo --test_ratio 0.1
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)


# Regex patterns to strip DAPO instruction prefix/suffix from prompts
# DAPO prompts typically look like:
#   "Solve the following math problem step by step. ... <problem> ... Remember to put your answer ..."
DAPO_PREFIX_PATTERNS = [
    r"Solve the following math problem step by step\..*?(?:present the final answer in)\s*\\boxed\{\}\.?\s*\n*",
    r"Solve the following math problem step by step[^.]*\.\s*",
]
DAPO_SUFFIX_PATTERNS = [
    r"\s*Remember to put your answer[^.]*\.?\s*$",
    r"\s*Present your final answer in\s*\\boxed\{\}\.?\s*$",
    r"\s*Please put your final answer in\s*\\boxed\{\}\.?\s*$",
]


def clean_dapo_prompt(raw_prompt: str) -> str:
    """Strip DAPO-specific instruction prefix/suffix and extract the math problem.

    Then append our standard instruction suffix for \\boxed{} format.
    """
    text = raw_prompt.strip()

    # Try to strip known prefixes
    for pattern in DAPO_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL | re.IGNORECASE)

    # Try to strip known suffixes
    for pattern in DAPO_SUFFIX_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = text.strip()

    # If stripping removed everything, fall back to original
    if not text:
        text = raw_prompt.strip()

    return text


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples."""
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        raw_prompt = example.get('prompt', '')
        answer = example.get('answer', '')

        if not raw_prompt or answer is None or answer == '':
            return None

        # Clean DAPO prompt and add our standard boxed instruction
        problem = clean_dapo_prompt(raw_prompt)
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{problem} {instruction}"

        data = {
            "data_source": "deepscaler",  # reuse deepscaler reward function
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer)
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'source': 'dapo',
            }
        }
        return data
    return process_fn


def main():
    parser = argparse.ArgumentParser(description='Prepare DAPO-Math-17k dataset for verl training')
    parser.add_argument('--local_dir',
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'dapo'),
                       help='Local directory to save processed datasets (default: <repo>/data/dapo)')
    parser.add_argument('--num_samples', type=int, default=0,
                       help='Number of samples to use (0 = use all unique problems, default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                       help='Fraction of data to hold out as test set (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/test splitting (default: 42)')
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"Output directory: {local_dir}")

    # Load DAPO dataset from HuggingFace
    print("\nLoading DAPO-Math-17k dataset from HuggingFace (BytedTsinghua-SIA/DAPO-Math-17k)...")
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", trust_remote_code=True)
    print(f"Raw dataset size: {len(ds)} rows")

    # Deduplicate by extra_info.index (each unique problem has a unique index,
    # duplicated ~100x in the dataset)
    print("\nDeduplicating...")
    seen_indices = set()
    unique_indices = []
    for i, example in enumerate(ds):
        extra_info = example.get('extra_info', {})
        if isinstance(extra_info, dict):
            problem_idx = extra_info.get('index', i)
        else:
            problem_idx = i

        if problem_idx not in seen_indices:
            seen_indices.add(problem_idx)
            unique_indices.append(i)

    ds_unique = ds.select(unique_indices)
    print(f"Unique problems: {len(ds_unique)} (removed {len(ds) - len(ds_unique)} duplicates)")

    # Optionally subsample
    if args.num_samples > 0 and args.num_samples < len(ds_unique):
        ds_unique = ds_unique.shuffle(seed=args.seed).select(range(args.num_samples))
        print(f"Subsampled to {len(ds_unique)} problems")

    # Split into train/test
    total = len(ds_unique)
    test_size = int(total * args.test_ratio)
    train_size = total - test_size

    ds_shuffled = ds_unique.shuffle(seed=args.seed)
    train_ds = ds_shuffled.select(range(train_size))
    test_ds = ds_shuffled.select(range(train_size, total))
    print(f"Split: train={len(train_ds)}, test={len(test_ds)} (test_ratio={args.test_ratio})")

    # Process training data
    print("\nProcessing training data...")
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    skipped = 0
    for idx, example in enumerate(train_ds):
        processed = process_fn(example, idx)
        if processed is not None:
            train_data.append(processed)
        else:
            skipped += 1

    print(f"Training data: {len(train_data)} examples ({skipped} skipped)")
    train_df = pd.DataFrame(train_data)
    train_path = os.path.join(local_dir, 'train.parquet')
    train_df.to_parquet(train_path)
    print(f"Saved: {train_path}")

    # Process test data
    print("\nProcessing test data...")
    test_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('test')
    skipped = 0
    for idx, example in enumerate(test_ds):
        processed = process_fn(example, idx)
        if processed is not None:
            test_data.append(processed)
        else:
            skipped += 1

    print(f"Test data: {len(test_data)} examples ({skipped} skipped)")
    test_df = pd.DataFrame(test_data)
    test_path = os.path.join(local_dir, 'test.parquet')
    test_df.to_parquet(test_path)
    print(f"Saved: {test_path}")

    # Print stats
    all_data = train_data + test_data
    print(f"\nTotal unique problems: {len(all_data)}")

    # Sample verification
    if train_data:
        sample = train_data[0]
        print(f"\nSample verification (first training example):")
        print(f"  data_source: {sample['data_source']}")
        print(f"  ground_truth: {sample['reward_model']['ground_truth']}")
        prompt_content = sample['prompt'][0]['content']
        print(f"  prompt ends with \\boxed{{}}: {prompt_content.rstrip().endswith(r'\boxed{}.')}")
        print(f"  prompt (first 200 chars): {prompt_content[:200]}...")

    print(f"\nDone! To use in training:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={test_path}")


if __name__ == '__main__':
    main()
