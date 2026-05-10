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
Metrics utilities for PPO trainer with @N notation support.
Adapted from verl official implementation.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, List

import numpy as np


def bootstrap_metric(
    data: List[Any],
    subset_size: int,
    reduce_fns: List[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[tuple]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for each reduction function.
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: List[dict], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    Args:
        data: List of dictionaries containing vote_key and val_key.
        vote_key: The key used for voting/counting.
        val_key: The key whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    return vote2vals[maj_vote][0]


def process_validation_metrics(
    data_sources: List[str],
    sample_uids: List[str],
    infos_dict: dict,
    seed: int = 42
) -> dict:
    """
    Process validation metrics into a structured format with @N notation.

    Groups validation metrics by data source and prompt uid, then computes
    various statistical measures including means, standard deviations,
    best/worst values, and majority voting results.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids (prompt identifiers) for each sample.
        infos_dict: Dictionary mapping variable names to lists of values.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        Nested dictionary: {data_source: {var_name: {metric_name: value}}}

        Metric names include:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples
        - "best@N/std": Standard deviation of the best values
        - "worst@N/mean": Mean of the worst values
        - "worst@N/std": Standard deviation of the worst values
        - "maj@N/mean": Mean of majority voting results (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results
    """
    # Group metrics by data source, prompt uid and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            if len(var_vals) > sample_idx:
                var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if len(var_vals) == 0:
                    continue
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    # Compute bootstrap metrics for different sample sizes
                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std

                        # Majority voting if predictions available
                        if var2vals.get("pred", None) is not None and len(var2vals["pred"]) == len(var_vals):
                            vote_data = [
                                {"val": val, "pred": pred}
                                for val, pred in zip(var_vals, var2vals["pred"])
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
