#!/usr/bin/env python3
"""Script to prepare MATH (Hendrycks) Level 3-5 training and test datasets.

Downloads the MATH dataset from HuggingFace, filters for difficulty levels 3-5,
extracts boxed answers, and saves in the same parquet format as DeepScaler
so existing reward functions (deepscale.py) work out of the box.

Usage:
    python prepare_math.py --local_dir ./data/math_l3_5
    python prepare_math.py --local_dir ./data/math_l3_5 --min_level 3 --max_level 5
    python prepare_math.py --local_dir ./data/math_l3_5 --test_ratio 0.1
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    print("Error: 'datasets' package not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)


# All subject categories in MATH dataset
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} in the solution string.

    Handles nested braces correctly.
    """
    # Find the last \boxed{ occurrence
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None

    # Walk forward matching braces
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return solution[start:i - 1]
    return None


def parse_level(level_str: str) -> int:
    """Parse level string like 'Level 3' to integer 3."""
    match = re.search(r'(\d+)', level_str)
    if match:
        return int(match.group(1))
    return -1


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples."""
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        problem = example.get('problem', '')
        solution = example.get('solution', '')

        if not problem or not solution:
            return None

        # Extract answer from \boxed{}
        answer = extract_boxed_answer(solution)
        if answer is None:
            return None

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
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'subject': example.get('type', ''),
                'level': example.get('level', ''),
            }
        }
        return data
    return process_fn


def main():
    parser = argparse.ArgumentParser(description='Prepare MATH (Hendrycks) Level 3-5 datasets for verl training')
    parser.add_argument('--local_dir',
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'math_l3_5'),
                       help='Local directory to save processed datasets (default: <repo>/data/math_l3_5)')
    parser.add_argument('--min_level', type=int, default=3,
                       help='Minimum difficulty level to include (default: 3)')
    parser.add_argument('--max_level', type=int, default=5,
                       help='Maximum difficulty level to include (default: 5)')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                       help='Fraction of train data to hold out as test set (default: 0.0, use original test split)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/test splitting (default: 42)')
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"Output directory: {local_dir}")
    print(f"Filtering: Level {args.min_level} to Level {args.max_level}")

    # Load all subjects from both train and test splits
    print("\nLoading MATH dataset from HuggingFace (EleutherAI/hendrycks_math)...")
    all_train = []
    all_test = []

    for subject in MATH_SUBJECTS:
        print(f"  Loading {subject}...", end=" ")
        ds = load_dataset("EleutherAI/hendrycks_math", subject, trust_remote_code=True)
        train_ds = ds['train']
        test_ds = ds['test']

        # Filter by level
        train_filtered = train_ds.filter(
            lambda x: args.min_level <= parse_level(x['level']) <= args.max_level
        )
        test_filtered = test_ds.filter(
            lambda x: args.min_level <= parse_level(x['level']) <= args.max_level
        )

        print(f"train: {len(train_ds)} -> {len(train_filtered)}, test: {len(test_ds)} -> {len(test_filtered)}")
        all_train.append(train_filtered)
        all_test.append(test_filtered)

    # Concatenate all subjects
    train_combined = concatenate_datasets(all_train)
    test_combined = concatenate_datasets(all_test)
    print(f"\nTotal after filtering: train_split={len(train_combined)}, test_split={len(test_combined)}")

    # Use original splits: MATH train for training, MATH test for validation
    train_final = train_combined
    val_final = test_combined
    print(f"Final split: train={len(train_final)}, val={len(val_final)}")

    # Process training data
    print("\nProcessing training data...")
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    skipped = 0
    for idx, example in enumerate(train_final):
        processed = process_fn(example, idx)
        if processed is not None:
            train_data.append(processed)
        else:
            skipped += 1

    print(f"Training data: {len(train_data)} examples ({skipped} skipped - no \\boxed{{}} answer)")
    train_df = pd.DataFrame(train_data)
    train_path = os.path.join(local_dir, 'train.parquet')
    train_df.to_parquet(train_path)
    print(f"Saved: {train_path}")

    # Process validation data
    print("\nProcessing validation data...")
    val_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('test')
    skipped = 0
    for idx, example in enumerate(val_final):
        processed = process_fn(example, idx)
        if processed is not None:
            val_data.append(processed)
        else:
            skipped += 1

    print(f"Validation data: {len(val_data)} examples ({skipped} skipped)")
    val_df = pd.DataFrame(val_data)
    test_path = os.path.join(local_dir, 'test.parquet')
    val_df.to_parquet(test_path)
    print(f"Saved: {test_path}")

    # Print level distribution
    all_data = train_data + val_data
    print(f"\nTotal problems: {len(all_data)}")
    print("\nLevel distribution (all):")
    for level in range(args.min_level, args.max_level + 1):
        count = sum(1 for d in all_data if f"Level {level}" in d['extra_info']['level'])
        print(f"  Level {level}: {count}")

    print("\nSubject distribution (all):")
    subject_counts = {}
    for d in all_data:
        subj = d['extra_info']['subject']
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
    for subj, count in sorted(subject_counts.items()):
        print(f"  {subj}: {count}")

    print(f"\nDone! To use in training:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={test_path}")


if __name__ == '__main__':
    main()
