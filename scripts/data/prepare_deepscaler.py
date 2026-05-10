#!/usr/bin/env python3
"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing. It loads problems from the deepscaler library and saves as parquet files.

Usage:
    python prepare_deepscaler.py --local_dir ./data/deepscaler
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Any

import pandas as pd

# Try to import deepscaler - install via: pip install -e deepscaler/
try:
    from deepscaler.data.utils import load_dataset
    from deepscaler.data.dataset_types import TrainDataset, TestDataset
except ImportError:
    print("Error: deepscaler package not found.")
    print("Please install it first:")
    print("  git clone https://github.com/agentica-project/deepscaler.git")
    print("  pip install -e deepscaler/")
    print("\nOr run: bash scripts/data/download_deepscaler.sh")
    sys.exit(1)


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.get('problem', '')
        answer = example.get('answer', '')

        if not question:
            return None

        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"

        data = {
            "data_source": "deepscaler",
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
                'index': idx
            }
        }
        return data
    return process_fn


def main():
    parser = argparse.ArgumentParser(description='Prepare DeepScaler datasets for verl training')
    parser.add_argument('--local_dir',
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'deepscaler'),
                       help='Local directory to save processed datasets (default: <repo>/data/deepscaler)')
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"Output directory: {local_dir}")

    # Load training dataset
    print("\nLoading training dataset (DeepScaler)...")
    train_dataset = load_dataset(TrainDataset.DEEPSCALER)

    # Load test datasets
    test_datasets = [
        TestDataset.AIME,
        TestDataset.AMC,
        TestDataset.MATH,
        TestDataset.MINERVA,
        TestDataset.OLYMPIAD_BENCH
    ]

    print("Loading test datasets...")
    test_datasets_data = [(d, load_dataset(d)) for d in test_datasets]

    # Process training data
    print("\nProcessing training data...")
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    print(f"Training data size: {len(train_data)}")
    train_df = pd.DataFrame(train_data)
    train_path = os.path.join(local_dir, 'train.parquet')
    train_df.to_parquet(train_path)
    print(f"Saved: {train_path}")

    # Process and save each test dataset
    print("\nProcessing test datasets...")
    for test_dataset, test_data_list in test_datasets_data:
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_path = os.path.join(local_dir, f'{dataset_name}.parquet')
        test_df.to_parquet(test_path)
        print(f"{dataset_name}: {len(test_data)} examples -> {test_path}")

    print("\nDone!")
    print(f"\nTo use in training:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={os.path.join(local_dir, 'aime.parquet')}  # AIME for validation")


if __name__ == '__main__':
    main()
