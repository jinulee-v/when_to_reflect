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

def reflection_stat_lrm(data, model, dataset):
    # Compute average entropy over (1) reflection before critical token, (2) reflection after critical token
    # Draw two overlapping histograms
    base_before_entropy = []
    base_after_entropy = []
    reflection_before_entropy = []
    reflection_after_entropy = []
    for d in data:
        base_befores = []
        base_afters = []
        # if "Distill" in model:
        #     critical_token_index = len(d.get("greedy_result", {}).get("token_ids", []))
        #     if critical_token_index == 0:
        #         continue
        #     for insert_empty in reversed(d.get("insert_reflection_result", {}).get("", [])):
        #         answer_dict = insert_empty['answer']
        #         max_prob = max(answer_dict.values())
        #         if max_prob < 0.95:
        #             break
        #         critical_token_index = len(insert_empty['partial_response_tokens'])
        # else:
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
        # if base_befores and base_afters and nonreflection_befores and nonreflection_afters and reflection_befores and reflection_afters:
        base_before_entropy.extend(base_befores)
        base_after_entropy.extend(base_afters)
        reflection_before_entropy.extend(reflection_befores)
        reflection_after_entropy.extend(reflection_afters)
        # nonreflection_before_entropy.append(sum(nonreflection_befores) / len(nonreflection_befores))
        # nonreflection_after_entropy.append(sum(nonreflection_afters) / len(nonreflection_afters))
        # reflection_before_entropy.append(sum(reflection_befores) / len(reflection_befores))
        # reflection_after_entropy.append(sum(reflection_afters) / len(reflection_afters))

    # Make histogram that shares the same bins
    all_values = base_before_entropy + base_after_entropy + reflection_before_entropy + reflection_after_entropy
    if len(all_values) == 0:
        # nothing to plot
        return

    min_v = 0
    max_v = max(all_values)
    bins = np.linspace(min_v, max_v, 21)
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
        ax.set_ylim(0, min(1, max_y + 0.2))
    axs[1].set_title(f'Reflection Entropy Distribution - {model} | {dataset}')
    plt.savefig(f'plots/stats_reflection_entropy_lrm_{dataset}_{model}.svg')
    plt.close()

def reflection_stat_basemodel(data, model, dataset, base_model_dataset):
    # Plot how did the critical token indices change between the `dataset` and the `base_model_dataset`.
    # Set the critical token of base model in x axis, and the critical token of the current model in y axis. Each point is a datum. If the point is above the diagonal line, it means the critical token index is later than the base model, otherwise it is earlier.
    base_model_critical_token_indices = []
    model_critical_token_indices = []
    colors = []
    color_map = {(True, True): "blue", (True, False): "orange", (False, True): "#7da901", (False, False): "red"} # base correct? ft model correct?
    for d in data:
        critical_token_index = d.get("critical_token_result", {}).get("critical_token_index", None)
        if critical_token_index is None:
            continue
        if len(d.get("greedy_result", {}).get("token_ids", [])) <= 20:
            continue
        base_model_datum = next((bd for bd in base_model_dataset if bd["example"] == d["example"]), None)
        if base_model_datum is None:
            print("here")
            continue
        if len(base_model_datum.get("greedy_result", {}).get("token_ids", [])) <= 20:
            continue
        base_model_critical_token_index = base_model_datum.get("critical_token_result", {}).get("critical_token_index", None)
        if base_model_critical_token_index is None:
            continue
        
        # base model correct?
        correct_answer = base_model_datum.get("example", {}).get("answer", None)
        base_model_final_answer = base_model_datum.get("critical_token_result", {}).get("majority_answer", None)
        model_final_answer = d.get("critical_token_result", {}).get("majority_answer", None)
        base_model_correct = (base_model_final_answer == correct_answer)
        model_correct = (model_final_answer == correct_answer)
        # Only add cases where the base model's final answer is incorrect
        if base_model_correct:
            continue
        
        model_critical_token_indices.append(critical_token_index / len(d.get("greedy_result", {}).get("token_ids", [])))
        base_model_critical_token_indices.append(base_model_critical_token_index / len(base_model_datum.get("greedy_result", {}).get("token_ids", [])))

        colors.append(color_map[(base_model_correct, model_correct)])
    
    plt.figure(figsize=(6, 6))

    # Draw summary curve: y = x + a*x*(1-x), passes through (0,0) and (1,1),
    # bows in the direction the scatter points deviate from the diagonal.
    if len(base_model_critical_token_indices) >= 2:
        xs = np.array(base_model_critical_token_indices)
        ys = np.array(model_critical_token_indices)
        ts = (xs + ys) / 2          # projection parameter along y=x
        vs = (ys - xs) / np.sqrt(2) # signed deviation from y=x
        ws = np.abs(vs)             # weight = distance from y=x
        h = ts * (1 - ts)           # quadratic basis along y=x parameterization
        denom = np.dot(ws * h, h)
        b = (np.dot(ws * vs, h) / denom if denom > 0 else 0.0)
        t = np.linspace(0, 1, 300)
        v_curve = np.clip(b * t * (1 - t), -1, 1)
        x_curve = np.clip(t - v_curve / 2, 0, 1)
        y_curve = np.clip(t + v_curve / 2, 0, 1)
        plt.fill_between(x_curve, x_curve, y_curve, color='#d0cece', alpha=0.5, zorder=0)
        plt.plot(x_curve, y_curve, color='#d0cece', linewidth=2, zorder=1)

    # Plot non-green dots first, then wrong->correct (#7da901) on top
    non_green_mask = [c != "#7da901" for c in colors]
    green_mask = [c == "#7da901" for c in colors]
    dot_size = 36 * 1.5  # 1.5x default marker size
    if any(non_green_mask):
        plt.scatter(
            [x for x, m in zip(base_model_critical_token_indices, non_green_mask) if m],
            [y for y, m in zip(model_critical_token_indices, non_green_mask) if m],
            alpha=0.5, c=[c for c, m in zip(colors, non_green_mask) if m], s=dot_size, zorder=2,
            clip_on=False)
    if any(green_mask):
        plt.scatter(
            [x for x, m in zip(base_model_critical_token_indices, green_mask) if m],
            [y for y, m in zip(model_critical_token_indices, green_mask) if m],
            alpha=0.5, c=[c for c, m in zip(colors, green_mask) if m], s=dot_size, zorder=3,
            clip_on=False)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.01, 0.2))
    plt.yticks(np.arange(0, 1.01, 0.2))
    # Legend for colors
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Correct->Correct', markerfacecolor=color_map[(True, True)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Correct->Wrong', markerfacecolor=color_map[(True, False)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Wrong->Correct', markerfacecolor=color_map[(False, True)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Wrong->Wrong', markerfacecolor=color_map[(False, False)], markersize=10, alpha=0.5)]
    upper_green = sum(1 for x, y, c in zip(base_model_critical_token_indices, model_critical_token_indices, colors) if y > x + 0.2 and c == '#7da901')
    lower_green = sum(1 for x, y, c in zip(base_model_critical_token_indices, model_critical_token_indices, colors) if y < x - 0.2 and c == '#7da901')
    middle_green = sum(1 for x, y, c in zip(base_model_critical_token_indices, model_critical_token_indices, colors) if x - 0.2 <= y <= x + 0.2 and c == '#7da901')
    print(f"[{model} | {dataset}] Upper region: {upper_green} green (wrong->correct)")
    print(f"[{model} | {dataset}] Middle region: {middle_green} green (wrong->correct)")
    print(f"[{model} | {dataset}] Lower region: {lower_green} green (wrong->correct)")
    # Wilcoxon signed-rank test for null hypothesis y = x (differences bounded in [-1, 1])
    diffs = [y - x for x, y in zip(base_model_critical_token_indices, model_critical_token_indices)]
    if len(diffs) >= 10:
        _, p_value = stats.wilcoxon(diffs)
    else:
        p_value = float("nan")
    print(f"[{model} | {dataset}] Wilcoxon p-value (H0: y=x): {p_value:.4g}")
    plt.text(0.98, 0.02, f'p = {p_value:.3g}', horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=9)
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Critical Token Index of Base Model')
    plt.ylabel('Critical Token Index of Current Model')
    plt.title(f'Critical Token Index Comparison - {model} vs Base Model, {dataset}')
    plt.savefig(f'plots/stats_reflection_critical_token_comparison_{dataset}_{model}_vs_{base_model}.svg')
    plt.close()


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

    # ── New plot: average ground-truth probability difference (Wait vs "") ──
    pd_model_names   = []
    pd_dataset_names = []
    pd_before        = []
    pd_after         = []
    pd_sem_before    = []
    pd_sem_after     = []

    # for model in ["Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
    #     for dataset in ["math500", "gpqa-diamond", "aime2024"]:
    #         fpath = f"data/insert_reflection_{dataset}_{model}.jsonl"
    #         if not os.path.exists(fpath):
    #             continue
    #         with open(fpath) as f:
    #             data = [json.loads(line) for line in f]
    #         data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]

    #         avg_before, sem_before, avg_after, sem_after = compute_wait_prob_diff(data)
    #         pd_model_names.append(model)
    #         pd_dataset_names.append(dataset)
    #         pd_before.append(avg_before)
    #         pd_after.append(avg_after)
    #         pd_sem_before.append(sem_before)
    #         pd_sem_after.append(sem_after)
    #         print(f"[{model} | {dataset}] prob_diff before={avg_before:.4f}±{sem_before:.4f}, after={avg_after:.4f}±{sem_after:.4f}")

    if pd_model_names:
        width = 0.35
        dataset_spacing = 0.85
        model_gap = 0.6

        x_positions = []
        pos = 0.0
        prev_model = None
        for m in pd_model_names:
            if prev_model is not None and m != prev_model:
                pos += model_gap
            x_positions.append(pos)
            pos += dataset_spacing
            prev_model = m
        x_positions = np.array(x_positions)

        err_kw = dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor='black')
        fig, ax = plt.subplots(figsize=(max(8, len(pd_model_names) * 1.2), 5))
        bars_before = ax.bar(x_positions - width / 2, pd_before, width, yerr=pd_sem_before,
                             label='Before CT', color='#1f77b4', alpha=0.8, error_kw=err_kw)
        bars_after  = ax.bar(x_positions + width / 2, pd_after,  width, yerr=pd_sem_after,
                             label='After CT',  color='#7f7f7f', alpha=0.8, error_kw=err_kw)

        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(pd_dataset_names, fontsize=8)

        seen_models = {}
        for i, m in enumerate(pd_model_names):
            seen_models.setdefault(m, []).append(x_positions[i])
        for m, positions in seen_models.items():
            center = (positions[0] + positions[-1]) / 2
            ax.text(center, -0.18, m, ha='center', va='top', fontsize=8, fontstyle='italic',
                    transform=ax.get_xaxis_transform())

        boundaries = [i for i in range(1, len(pd_model_names)) if pd_model_names[i] != pd_model_names[i - 1]]
        for b in boundaries:
            sep_x = (x_positions[b - 1] + x_positions[b]) / 2
            ax.axvline(sep_x, color='#cccccc', linewidth=1, linestyle='--')

        ax.set_ylabel('Avg Δ P(correct answer)  [Wait − baseline]')
        ax.set_title('"Wait": avg ground-truth probability difference vs baseline  (before vs after Critical Token)')
        ax.legend()

        # ylim that accommodates error bar extents
        lo_vals = [m - s for m, s in zip(pd_before + pd_after, pd_sem_before + pd_sem_after)
                   if not np.isnan(m)]
        hi_vals = [m + s for m, s in zip(pd_before + pd_after, pd_sem_before + pd_sem_after)
                   if not np.isnan(m)]
        if lo_vals and hi_vals:
            span = max(hi_vals) - min(lo_vals)
            margin = span * 0.3 + 0.005
            ax.set_ylim(min(lo_vals) - margin, max(hi_vals) + margin)

        for bar in bars_before:
            h = bar.get_height()
            if not np.isnan(h):
                va = 'bottom' if h >= 0 else 'top'
                offset = 0.001 if h >= 0 else -0.001
                ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f'{h:+.3f}',
                        ha='center', va=va, fontsize=7)
        for bar in bars_after:
            h = bar.get_height()
            if not np.isnan(h):
                va = 'bottom' if h >= 0 else 'top'
                offset = 0.001 if h >= 0 else -0.001
                ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f'{h:+.3f}',
                        ha='center', va=va, fontsize=7)

        plt.tight_layout()
        plt.savefig('plots/stats_reflection_wait_prob_diff_global.svg')
        plt.close()
        print("Saved global prob-diff bar chart to plots/stats_reflection_wait_prob_diff_global.svg")

    for model in ["Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-GRPO", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"]: # "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", 
        for dataset in ["gpqa-diamond", "math500", "aime2024"]:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                import json
                data = [json.loads(line) for line in f]
                
                data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]

            reflection_stat_lrm(data, model, dataset)
        
    for model in ["Qwen2.5-7B-Instruct-GRPO", "OpenThinker3-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1", "s1.1-7B"]:
        base_model = "Qwen2.5-7B-Instruct"
        for dataset in ["gpqa-diamond", "math500", "aime2024"]:
            with open(f"data/critical_tokens_{dataset}_{base_model}.jsonl") as f:
                base_model_dataset = [json.loads(line) for line in f]
                base_model_dataset = [d for d in base_model_dataset if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]
            if not os.path.exists(f"data/basemodeltrace_critical_tokens_{dataset}_{model}.jsonl"):
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/basemodeltrace_critical_tokens_{dataset}_{model}.jsonl") as f:
                import json
                data = [json.loads(line) for line in f]
                
                data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]

            reflection_stat_basemodel(data, model, dataset, base_model_dataset)
