import json
import glob
import os
import numpy as np
from scipy import stats

DATA_METHODS = [
    "Math-Shepherd-Mistral-7B-PRM",
    "Qwen2.5-Math-7B-PRM800k",
    "Qwen2.5-Math-PRM-7B",
    "max_entropy_token",
    "max_varentropy_token",
    "max_avg_entropy_step",
    "max_perplexity_step",
]

files = sorted(glob.glob("data/evaluate_step_*.jsonl"))

for filepath in files:
    filename = os.path.basename(filepath)
    # parse dataset and model from filename: evaluate_step_{dataset}_{model}.jsonl
    parts = filename[len("evaluate_step_"):-len(".jsonl")].split("_", 1)
    print(f"\n{'='*70}")
    print(f"File: {filename}")

    diffs = {m: [] for m in DATA_METHODS}
    earlier = {m: [] for m in DATA_METHODS}
    rel_backtrack = {m: [] for m in DATA_METHODS}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            majority_answer = item["critical_token_result"]["majority_answer"]
            ground_truth = item["example"]["answer"]
            if str(majority_answer) == str(ground_truth):
                continue

            critical_idx = item["critical_token_result"]["critical_token_index"]
            seq_len = len(item["greedy_result"]["token_ids"])
            step_evals = item["step_evaluation_results"]

            for method in DATA_METHODS:
                if method not in step_evals:
                    continue
                backtrack_idx = step_evals[method]["backtrack_token_index"]
                if critical_idx is None or seq_len == 0:
                    continue
                if backtrack_idx is None:
                    backtrack_idx = seq_len
                diff = (backtrack_idx - critical_idx) / seq_len
                diffs[method].append(diff)
                earlier[method].append(backtrack_idx < critical_idx)
                rel_backtrack[method].append(backtrack_idx / seq_len)

    print(f"{'Method':<35} {'N':>5}  {'Mean':>10}  {'Std':>10}  {'p-value':>12}  {'Earlier%':>9}  {'AvgBT/Len':>10}")
    print("-" * 102)
    for method in DATA_METHODS:
        vals = diffs[method]
        if len(vals) < 2:
            print(f"  {method:<33} {'N/A':>5}")
            continue
        arr = np.array(vals)
        mean = arr.mean()
        std = arr.std(ddof=1)
        # One-sample t-test H0: mean = 0
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        ratio = np.mean(earlier[method]) * 100
        avg_rel_bt = np.mean(rel_backtrack[method])
        print(f"  {method:<33} {len(arr):>5}  {mean:>10.4f}  {std:>10.4f}  {p_val:>12.4e}  {ratio:>8.1f}%  {avg_rel_bt:>10.4f}")
