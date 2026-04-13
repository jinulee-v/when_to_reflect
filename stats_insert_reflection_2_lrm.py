import os
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy as np
import json
from scipy import stats
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def collect_entropies(data):
    base_before_entropy = []
    base_after_entropy = []
    reflection_before_entropy = []
    reflection_after_entropy = []
    for d in data:
        base_befores = []
        base_afters = []

        critical_token_index = d.get("critical_token_result", {}).get("critical_token_index", None)
        if "insert_reflection_result" not in d:
            continue
        for non_reflect_token in [""]:
            if non_reflect_token not in d['insert_reflection_result']:
                continue
            for trace_point in d['insert_reflection_result'][non_reflect_token]:
                answer_dict = trace_point['answer']
                entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
                if len(trace_point['partial_response_tokens']) < critical_token_index:
                    base_befores.append(entropy)
                else:
                    base_afters.append(entropy)
        reflection_befores = []
        reflection_afters = []
        for reflect_token in ["Wait", "Alternatively"]:
            if reflect_token not in d['insert_reflection_result']:
                continue
            for trace_point in d['insert_reflection_result'][reflect_token]:
                answer_dict = trace_point['answer']
                entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
                if len(trace_point['partial_response_tokens']) < critical_token_index:
                    reflection_befores.append(entropy)
                else:
                    reflection_afters.append(entropy)
        base_before_entropy.extend(base_befores)
        base_after_entropy.extend(base_afters)
        reflection_before_entropy.extend(reflection_befores)
        reflection_after_entropy.extend(reflection_afters)
    return base_before_entropy, base_after_entropy, reflection_before_entropy, reflection_after_entropy


def reflection_stat_lrm(base_before_entropy, base_after_entropy, reflection_before_entropy, reflection_after_entropy, model, dataset, bins):
    all_values = base_before_entropy + base_after_entropy + reflection_before_entropy + reflection_after_entropy
    if len(all_values) == 0:
        return

    # 2 subplots, organized horizontally
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Compute bin centers and width for bar plots
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    # Bar plots with probability (count/sum) as y-axis
    if base_before_entropy and base_after_entropy:
        counts, _ = np.histogram(base_before_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[0].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='Before CT', color='#AFABAB', edgecolor='none')
        counts, _ = np.histogram(base_after_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[0].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='After CT', color='#7f7f7f')
    if reflection_before_entropy and reflection_after_entropy:
        counts, _ = np.histogram(reflection_before_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[1].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='"Wait", Before CT', color='#AFABAB', edgecolor='none')
        counts, _ = np.histogram(reflection_after_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[1].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='"Wait", After CT', color='#00afbd')

    # Get max y value among all subplots to set same y limit
    max_y = 0
    for ax in axs:
        max_y = max(max_y, ax.get_ylim()[1])

    for ax in axs:
        ax.set_xlabel('Final Answer Entropy')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, min(1, max_y + 0.2))
    axs[1].set_title(f'Reflection Entropy Distribution - {model} | {dataset}')
    plt.savefig(f'plots/stats_reflection_entropy_lrm_{dataset}_{model}.svg')
    plt.close()


if __name__ == "__main__":
    models = ["Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-GRPO", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"] # "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B",
    datasets = ["gpqa-diamond", "math500", "aime2024"]

    # First pass: collect all entropy values to compute global x range
    all_data = {}
    global_all_values = []
    for model in models:
        for dataset in datasets:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            print(f"Loading model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                data = [json.loads(line) for line in f]
                data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]
            entropies = collect_entropies(data)
            all_data[(model, dataset)] = entropies
            for vals in entropies:
                global_all_values.extend(vals)

    global_bins = np.linspace(0, max(global_all_values), 21)

    # Second pass: plot with shared bins
    for model in models:
        for dataset in datasets:
            if (model, dataset) not in all_data:
                continue
            print(f"Processing model={model}, dataset={dataset}")
            base_before, base_after, refl_before, refl_after = all_data[(model, dataset)]
            reflection_stat_lrm(base_before, base_after, refl_before, refl_after, model, dataset, global_bins)
