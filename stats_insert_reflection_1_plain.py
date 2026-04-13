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

def reflection_stat_plain(data, model, dataset):
    # Compute average entropy over (1) reflection before critical token, (2) reflection after critical token
    # Draw two overlapping histograms
    base_before_entropy = []
    base_after_entropy = []
    nonreflection_before_entropy = []
    nonreflection_after_entropy = []
    reflection_before_entropy = []
    reflection_after_entropy = []

    # Track majority vote change counts per token, split by before/after critical token
    all_tokens = ["Wait", "Alternatively", "Now", "Therefore"]
    mv_changed_before = {t: 0 for t in all_tokens}
    mv_total_before = {t: 0 for t in all_tokens}
    mv_changed_after = {t: 0 for t in all_tokens}
    mv_total_after = {t: 0 for t in all_tokens}

    # 2x2 correctness case counts: (empty_correct, token_correct), split by before/after CT
    outcomes = [(True, True), (True, False), (False, True), (False, False)]
    case_before = {t: {o: 0 for o in outcomes} for t in all_tokens}
    case_after  = {t: {o: 0 for o in outcomes} for t in all_tokens}

    for d in data:
        base_befores = []
        base_afters = []
        critical_token_index = d.get("critical_token_result", {}).get("critical_token_index", None)
        if critical_token_index is None:
            continue
        if "insert_reflection_result" not in d:
            continue
        # Build a position -> majority lookup for the "" (empty) reflection case
        empty_majority_by_pos = {}
        for non_reflect_token in [""]:
            if non_reflect_token not in d['insert_reflection_result']:
                continue
            for trace_point in d['insert_reflection_result'][non_reflect_token]:
                answer_dict = trace_point['answer']
                entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
                pos = len(trace_point['partial_response_tokens'])
                if pos < critical_token_index:
                    base_befores.append(entropy)
                else:
                    base_afters.append(entropy)
                empty_majority_by_pos[pos] = max(answer_dict, key=answer_dict.get)
        nonreflection_befores = []
        nonreflection_afters = []
        for non_reflect_token in ["Therefore", "Now"]:
            if non_reflect_token not in d['insert_reflection_result']:
                continue
            for trace_point in d['insert_reflection_result'][non_reflect_token]:
                answer_dict = trace_point['answer']
                entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
                if len(trace_point['partial_response_tokens']) < critical_token_index:
                    nonreflection_befores.append(entropy)
                else:
                    nonreflection_afters.append(entropy)
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
        # if base_befores and base_afters and nonreflection_befores and nonreflection_afters and reflection_befores and reflection_afters:
        base_before_entropy.extend(base_befores)
        base_after_entropy.extend(base_afters)
        nonreflection_before_entropy.extend(nonreflection_befores)
        nonreflection_after_entropy.extend(nonreflection_afters)
        reflection_before_entropy.extend(reflection_befores)
        reflection_after_entropy.extend(reflection_afters)

        # Count majority vote changes for each token vs the "" case at the same position
        correct_answer = d.get("example", {}).get("answer", None)
        for token in all_tokens:
            if token not in d['insert_reflection_result']:
                continue
            for trace_point in d['insert_reflection_result'][token]:
                pos = len(trace_point['partial_response_tokens'])
                # print(pos, pos in empty_majority_by_pos, empty_majority_by_pos.keys())
                baseline_majority = empty_majority_by_pos.get(pos-1, None)
                if baseline_majority is None:
                    continue
                answer_dict = trace_point['answer']
                inserted_majority = max(answer_dict, key=answer_dict.get)
                changed = (inserted_majority != baseline_majority)
                if pos < critical_token_index:
                    mv_total_before[token] += 1
                    if changed:
                        mv_changed_before[token] += 1
                else:
                    mv_total_after[token] += 1
                    if changed:
                        mv_changed_after[token] += 1
                if correct_answer is not None:
                    empty_correct = (baseline_majority == correct_answer)
                    token_correct = (inserted_majority == correct_answer)
                    if pos < critical_token_index:
                        case_before[token][(empty_correct, token_correct)] += 1
                    else:
                        case_after[token][(empty_correct, token_correct)] += 1

    # Print majority vote change ratios
    print(f"\n[{model} | {dataset}] Majority vote change ratio (inserted token changed majority answer vs empty token at same position):")
    for token in all_tokens:
        before_pct = mv_changed_before[token] / mv_total_before[token] * 100 if mv_total_before[token] > 0 else float('nan')
        after_pct = mv_changed_after[token] / mv_total_after[token] * 100 if mv_total_after[token] > 0 else float('nan')
        print(f"  {token!r}: before CT = {before_pct:.2f}% ({mv_changed_before[token]}/{mv_total_before[token]}), after CT = {after_pct:.2f}% ({mv_changed_after[token]}/{mv_total_after[token]})")

    # Print 2x2 correctness outcome counts per token, before/after CT
    label = {(True, True): "\"\"=correct, token=correct", (True, False): "\"\"=correct, token=incorrect",
             (False, True): "\"\"=incorrect, token=correct", (False, False): "\"\"=incorrect, token=incorrect"}
    print(f"\n[{model} | {dataset}] 2x2 correctness outcomes (\"\" majority vs token majority) at matched positions:")
    for token in all_tokens:
        total_before = sum(case_before[token].values())
        total_after  = sum(case_after[token].values())
        print(f"  {token!r}:")
        for o in outcomes:
            cb, ca = case_before[token][o], case_after[token][o]
            cb_pct = cb / total_before * 100 if total_before > 0 else float('nan')
            ca_pct = ca / total_after  * 100 if total_after  > 0 else float('nan')
            print(f"    {label[o]}: before CT = {cb_pct:.1f}% ({cb}/{total_before}), after CT = {ca_pct:.1f}% ({ca}/{total_after})")

    # Make histogram that shares the same bins
    all_values = base_before_entropy + base_after_entropy + reflection_before_entropy + reflection_after_entropy + nonreflection_before_entropy + nonreflection_after_entropy
    if len(all_values) == 0:
        # nothing to plot
        return case_before, case_after

    min_v = 0
    max_v = max(all_values)
    bins = np.linspace(min_v, max_v, 21)
    # 3 subplots, organized horizontally
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
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
    if nonreflection_before_entropy and nonreflection_after_entropy:
        counts, _ = np.histogram(nonreflection_before_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[1].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='Non-reflection, Before CT', color='#AFABAB', edgecolor='none')
        counts, _ = np.histogram(nonreflection_after_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[1].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='Non-reflection, After CT', color='#feb253')
    if reflection_before_entropy and reflection_after_entropy:
        counts, _ = np.histogram(reflection_before_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[2].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='Reflection, Before CT', color='#AFABAB', edgecolor='none')
        counts, _ = np.histogram(reflection_after_entropy, bins=bins)
        probabilities = counts / counts.sum() if counts.sum() > 0 else counts
        axs[2].bar(bin_centers, probabilities, width=bin_width, alpha=0.5, label='Reflection, After CT', color='#00afbd')

    # Get max y value among all subplots to set same y limit
    max_y = 0
    for ax in axs:
        max_y = max(max_y, ax.get_ylim()[1])

    for ax in axs:
        ax.set_xlabel('Final Answer Entropy')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_ylim(0, min(1, max_y + 0.2))
    axs[1].set_title(f'Reflection Entropy Distribution - {model} | {dataset}')
    plt.savefig(f'plots/stats_reflection_entropy_{dataset}_{model}.svg')
    plt.close()
    return case_before, case_after


if __name__ == "__main__":
    # Collect (False, True) = ""=incorrect, token=correct for "Wait" across all model*dataset
    global_model_names = []
    global_dataset_names = []
    global_before_pct = []
    global_after_pct = []

    for model in ["Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", ]:
        for dataset in ["math500", "gpqa-diamond", "aime2024"]:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                import json
                data = [json.loads(line) for line in f]

                # Filter the data by length of the response tokens < 20
                data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]

            case_before, case_after = reflection_stat_plain(data, model, dataset)

            token = "Wait"
            outcome = (False, True)  # ""=incorrect, token=correct
            total_before = sum(case_before[token].values())
            total_after  = sum(case_after[token].values())
            before_pct = case_before[token][outcome] / total_before * 100 if total_before > 0 else float('nan')
            after_pct  = case_after[token][outcome]  / total_after  * 100 if total_after  > 0 else float('nan')
            global_model_names.append(model)
            global_dataset_names.append(dataset)
            global_before_pct.append(before_pct)
            global_after_pct.append(after_pct)

    # Global bar chart — datasets grouped within each model
    if global_model_names:
        width = 0.35
        dataset_spacing = 0.85   # distance between pair centers within a model group
        model_gap = 0.6          # extra gap inserted between model groups

        x_positions = []
        pos = 0.0
        prev_model = None
        for m in global_model_names:
            if prev_model is not None and m != prev_model:
                pos += model_gap
            x_positions.append(pos)
            pos += dataset_spacing
            prev_model = m
        x_positions = np.array(x_positions)

        fig, ax = plt.subplots(figsize=(max(8, len(global_model_names) * 1.2), 5))
        bars_before = ax.bar(x_positions - width / 2, global_before_pct, width, label='Before CT', color='#1f77b4', alpha=0.8)
        bars_after  = ax.bar(x_positions + width / 2, global_after_pct,  width, label='After CT',  color='#7f7f7f', alpha=0.8)

        # Tick labels = dataset names only
        ax.set_xticks(x_positions)
        ax.set_xticklabels(global_dataset_names, fontsize=8)

        # Model group labels below the tick labels (y in axes coords via xaxis_transform)
        seen_models = {}
        for i, m in enumerate(global_model_names):
            seen_models.setdefault(m, []).append(x_positions[i])
        for m, positions in seen_models.items():
            center = (positions[0] + positions[-1]) / 2
            ax.text(center, -0.18, m, ha='center', va='top', fontsize=8, fontstyle='italic',
                    transform=ax.get_xaxis_transform())

        # Vertical separators between model groups
        boundaries = [i for i in range(1, len(global_model_names)) if global_model_names[i] != global_model_names[i - 1]]
        for b in boundaries:
            sep_x = (x_positions[b - 1] + x_positions[b]) / 2
            ax.axvline(sep_x, color='#cccccc', linewidth=1, linestyle='--')

        ax.set_ylabel('% of positions')
        ax.set_title('"Wait": ""=incorrect → token=correct  (before vs after Critical Token)')
        ax.legend()
        valid_vals = [v for v in global_before_pct + global_after_pct if not np.isnan(v)]
        ax.set_ylim(0, max(valid_vals) * 1.3 + 0.5 if valid_vals else 5)

        for bar in bars_before:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f'{h:.1f}%', ha='center', va='bottom', fontsize=7)
        for bar in bars_after:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f'{h:.1f}%', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig('plots/stats_reflection_wait_incorrect_to_correct_global.svg')
        plt.close()
        print("Saved global bar chart to plots/stats_reflection_wait_incorrect_to_correct_global.svg")

    # Narrow plot for Qwen2.5-7B-Instruct only
    qwen7b_indices = [i for i, m in enumerate(global_model_names) if m == "Qwen2.5-7B-Instruct"]
    if qwen7b_indices:
        qwen7b_datasets = [global_dataset_names[i] for i in qwen7b_indices]
        qwen7b_before = [global_before_pct[i] for i in qwen7b_indices]
        qwen7b_after  = [global_after_pct[i]  for i in qwen7b_indices]

        width = 0.35
        x = np.arange(len(qwen7b_datasets))

        fig, ax = plt.subplots(figsize=(4, 4))
        bars_before = ax.bar(x - width / 2, qwen7b_before, width, label='Before CT', color='#1f77b4', alpha=0.8)
        bars_after  = ax.bar(x + width / 2, qwen7b_after,  width, label='After CT',  color='#7f7f7f', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(qwen7b_datasets, fontsize=8)
        ax.set_ylabel('% of positions')
        ax.set_title('"Wait": ""=incorrect → token=correct\n(Qwen2.5-7B-Instruct)', fontsize=9)
        ax.legend()
        valid_vals = [v for v in qwen7b_before + qwen7b_after if not np.isnan(v)]
        ax.set_ylim(0, max(valid_vals) * 1.3 + 0.5 if valid_vals else 5)

        for bar in bars_before:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f'{h:.1f}%', ha='center', va='bottom', fontsize=7)
        for bar in bars_after:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f'{h:.1f}%', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig('plots/stats_reflection_wait_incorrect_to_correct_qwen7b.svg')
        plt.close()
        print("Saved Qwen7B bar chart to plots/stats_reflection_wait_incorrect_to_correct_qwen7b.svg")
