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

def compute_data(data, base_model_dataset):
    """Extract (base_xs, model_ys, colors) for a model/dataset pair."""
    base_model_critical_token_indices = []
    model_critical_token_indices = []
    colors = []
    color_map = {(True, True): "blue", (True, False): "orange", (False, True): "#7da901", (False, False): "red"}
    for d in data:
        critical_token_index = d.get("critical_token_result", {}).get("critical_token_index", None)
        if critical_token_index is None:
            continue
        if len(d.get("greedy_result", {}).get("token_ids", [])) <= 20:
            continue
        base_model_datum = next((bd for bd in base_model_dataset if bd["example"] == d["example"]), None)
        if base_model_datum is None:
            continue
        if len(base_model_datum.get("greedy_result", {}).get("token_ids", [])) <= 20:
            continue
        base_model_critical_token_index = base_model_datum.get("critical_token_result", {}).get("critical_token_index", None)
        if base_model_critical_token_index is None:
            continue

        correct_answer = base_model_datum.get("example", {}).get("answer", None)
        base_model_final_answer = base_model_datum.get("critical_token_result", {}).get("majority_answer", None)
        model_final_answer = d.get("critical_token_result", {}).get("majority_answer", None)
        base_model_correct = (base_model_final_answer == correct_answer)
        model_correct = (model_final_answer == correct_answer)
        if base_model_correct:
            continue

        model_critical_token_indices.append(critical_token_index / len(d.get("greedy_result", {}).get("token_ids", [])))
        base_model_critical_token_indices.append(base_model_critical_token_index / len(base_model_datum.get("greedy_result", {}).get("token_ids", [])))
        colors.append(color_map[(base_model_correct, model_correct)])

    return base_model_critical_token_indices, model_critical_token_indices, colors


def draw_scatter_ax(ax, xs, ys, colors):
    """Draw the scatter plot on a given axes object."""
    color_map = {(True, True): "blue", (True, False): "orange", (False, True): "#7da901", (False, False): "red"}

    # Summary curve
    if len(xs) >= 2:
        xa = np.array(xs)
        ya = np.array(ys)
        ts = (xa + ya) / 2
        vs = (ya - xa) / np.sqrt(2)
        ws = np.abs(vs)
        h = ts * (1 - ts)
        denom = np.dot(ws * h, h)
        b = (np.dot(ws * vs, h) / denom if denom > 0 else 0.0)
        t = np.linspace(0, 1, 300)
        v_curve = np.clip(b * t * (1 - t), -1, 1)
        x_curve = np.clip(t - v_curve / 2, 0, 1)
        y_curve = np.clip(t + v_curve / 2, 0, 1)
        ax.fill_between(x_curve, x_curve, y_curve, color='#d0cece', alpha=0.5, zorder=0)
        ax.plot(x_curve, y_curve, color='#d0cece', linewidth=2, zorder=1)

    non_green_mask = [c != "#7da901" for c in colors]
    green_mask = [c == "#7da901" for c in colors]
    dot_size = 36 * 1.5
    if any(non_green_mask):
        ax.scatter(
            [x for x, m in zip(xs, non_green_mask) if m],
            [y for y, m in zip(ys, non_green_mask) if m],
            alpha=0.5, c=[c for c, m in zip(colors, non_green_mask) if m], s=dot_size, zorder=2, clip_on=False)
    if any(green_mask):
        ax.scatter(
            [x for x, m in zip(xs, green_mask) if m],
            [y for y, m in zip(ys, green_mask) if m],
            alpha=0.5, c=[c for c, m in zip(colors, green_mask) if m], s=dot_size, zorder=3, clip_on=False)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def reflection_stat_basemodel(data, model, dataset, base_model_dataset):
    xs, ys, colors = compute_data(data, base_model_dataset)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    draw_scatter_ax(ax, xs, ys, colors)

    color_map = {(True, True): "blue", (True, False): "orange", (False, True): "#7da901", (False, False): "red"}
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Correct->Correct', markerfacecolor=color_map[(True, True)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Correct->Wrong', markerfacecolor=color_map[(True, False)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Wrong->Correct', markerfacecolor=color_map[(False, True)], markersize=10, alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Wrong->Wrong', markerfacecolor=color_map[(False, False)], markersize=10, alpha=0.5)]
    upper_green = sum(1 for x, y, c in zip(xs, ys, colors) if y > x + 0.2 and c == '#7da901')
    lower_green = sum(1 for x, y, c in zip(xs, ys, colors) if y < x - 0.2 and c == '#7da901')
    middle_green = sum(1 for x, y, c in zip(xs, ys, colors) if x - 0.2 <= y <= x + 0.2 and c == '#7da901')
    print(f"[{model} | {dataset}] Upper region: {upper_green} green (wrong->correct)")
    print(f"[{model} | {dataset}] Middle region: {middle_green} green (wrong->correct)")
    print(f"[{model} | {dataset}] Lower region: {lower_green} green (wrong->correct)")

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Normalized CT Pos. of Base Model')
    ax.set_ylabel('Normalized CT Pos. of Current Model')
    ax.set_title(f'Critical Token Position Comparison - {model} vs Base Model, {dataset}')
    plt.savefig(f'plots/stats_reflection_critical_token_comparison_{dataset}_{model}_vs_{base_model}.svg')
    plt.close()


def make_combined_figure(all_data, datasets, models, model_labels, base_model):
    """
    all_data: dict[(dataset, model)] -> (xs, ys, colors)
    datasets: list of dataset names (rows)
    models: list of model names (cols)
    model_labels: display labels for models
    """
    color_map = {(True, True): "blue", (True, False): "orange", (False, True): "#7da901", (False, False): "red"}
    n_rows = len(datasets)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    dataset_labels = {"math500": "MATH-500", "gpqa-diamond": "GPQA-Diamond", "aime2024": "AIME 2024"}

    for row_idx, dataset in enumerate(datasets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            key = (dataset, model)
            if key in all_data:
                xs, ys, colors = all_data[key]
                draw_scatter_ax(ax, xs, ys, colors)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if row_idx == 0:
                ax.set_title(model_labels[col_idx], fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_labels.get(dataset, dataset)}\nNorm. CT Pos. (FT model)", fontsize=11)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if row_idx == n_rows - 1:
                ax.set_xlabel('Norm. CT Pos. (Base model)', fontsize=11)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Wrong→Correct', markerfacecolor='#7da901', markersize=10, alpha=0.8),
        Line2D([0], [0], marker='o', color='w', label='Wrong→Wrong', markerfacecolor='red', markersize=10, alpha=0.8),
        Line2D([0], [0], marker='o', color='w', label='Correct→Correct', markerfacecolor='blue', markersize=10, alpha=0.8),
        Line2D([0], [0], marker='o', color='w', label='Correct→Wrong', markerfacecolor='orange', markersize=10, alpha=0.8),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(f'Critical Token Position: FT Models vs Base Model ({base_model})', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = f'plots/stats_critical_pos_compare_combined.svg'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to {out_path}")


if __name__ == "__main__":
    base_model = "Qwen2.5-7B-Instruct"
    datasets = ["math500", "gpqa-diamond", "aime2024"]
    # Columns: openthinker, s1.1, oneshot-rlvr, grpo
    models = ["OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"]
    model_labels = ["OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR", "GRPO"]

    # Load base model datasets once per dataset
    base_model_datasets = {}
    for dataset in datasets:
        path = f"data/critical_tokens_{dataset}_{base_model}.jsonl"
        with open(path) as f:
            bmd = [json.loads(line) for line in f]
        base_model_datasets[dataset] = [d for d in bmd if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]

    # Collect all data
    all_data = {}
    for model in models:
        for dataset in datasets:
            path = f"data/basemodeltrace_critical_tokens_{dataset}_{model}.jsonl"
            if not os.path.exists(path):
                print(f"Skipping (missing): {path}")
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(path) as f:
                data = [json.loads(line) for line in f]
            data = [d for d in data if len(d.get("greedy_result", {}).get("token_ids", [])) > 20]
            xs, ys, colors = compute_data(data, base_model_datasets[dataset])
            all_data[(dataset, model)] = (xs, ys, colors)
            # Also save individual plots
            reflection_stat_basemodel(data, model, dataset, base_model_datasets[dataset])

    # Build combined figure
    make_combined_figure(all_data, datasets, models, model_labels, base_model)
