import json
import re
# from transformers import AutoTokenizer
import argparse
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import os
import numpy as np


def find_boxed_token_index(tokenizer, token_ids):
    # Binary search
    full_text = tokenizer.decode(token_ids)
    if "\\boxed" not in full_text:
        return None
    head = 0; tail = len(token_ids)
    while head < tail:
        mid = (head + tail) // 2
        str_mid = tokenizer.decode(token_ids[:mid])
        if "\\boxed" in str_mid:
            tail = mid
        else:
            head = mid + 1
    return head

absolute_pos_dict = {}


def load_relative_pos_data(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    relative_pos = []
    final_answer_correct = []
    for datum in data:
        boxed_index = len(datum["greedy_result"]["token_ids"])
        if boxed_index == 0:
            continue
        relative_pos.append(datum["critical_token_result"]["critical_token_index"] / boxed_index)
        final_answer_correct.append(
            "correct" if datum["example"]["answer"] == datum["critical_token_result"]["majority_answer"] else "wrong"
        )
    relative_pos = [min(r, 1.0) for r in relative_pos]
    total = len(relative_pos)
    correct_pos = [p for p, c in zip(relative_pos, final_answer_correct) if c == "correct"]
    wrong_pos = [p for p, c in zip(relative_pos, final_answer_correct) if c == "wrong"]
    return correct_pos, wrong_pos, total


def draw_relative_histogram_on_ax(ax, correct_pos, wrong_pos, total):
    bins = [x / 20 for x in range(21)]
    ax.hist([correct_pos, wrong_pos], bins=bins, stacked=True,
            weights=[np.ones(len(correct_pos)) / total, np.ones(len(wrong_pos)) / total],
            color=['#bcd373', '#ff8ca1'])
    if correct_pos:
        ax.axvline(np.mean(correct_pos), color='#98c126', linestyle=':', linewidth=1.5)
    if wrong_pos:
        ax.axvline(np.mean(wrong_pos), color='#f45f74', linestyle=':', linewidth=1.5)
    ax.set_ylim(0, 1)
    if correct_pos and wrong_pos:
        mean_correct = np.mean(correct_pos)
        mean_wrong = np.mean(wrong_pos)
        mid_y = 0.6
        ax.annotate(
            '', xy=(mean_wrong, mid_y), xytext=(mean_correct, mid_y),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2)
        )
        ax.text((mean_correct + mean_wrong) / 2, mid_y + 0.02,
                f'Δ={abs(mean_wrong - mean_correct):.2f}',
                ha='center', va='bottom', fontsize=7, color='gray')
    ax.set_xlabel("Critical Token Position (Relative to \\boxed)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    legend_handles = [
        (Patch(facecolor='#bcd373'), Line2D([0], [0], color='#bcd373', linestyle=':', linewidth=1.5)),
        (Patch(facecolor='#ff8ca1'), Line2D([0], [0], color='#ff8ca1', linestyle=':', linewidth=1.5)),
    ]
    ax.legend(legend_handles, ['Correct', 'Incorrect'],
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=7)


def draw_relative_histogram_plain_on_ax(ax, all_pos, total):
    bins = [x / 20 for x in range(21)]
    ax.hist(all_pos, bins=bins, weights=np.ones(len(all_pos)) / total, color='#7ab8d9')
    if all_pos:
        ax.axvline(np.mean(all_pos), color='#2a7ab0', linestyle=':', linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Critical Token Position (Relative to \\boxed)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)


def load_absolute_pos_data(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    absolute_pos = []
    final_answer_correct = []
    for datum in data:
        boxed_index = len(datum["greedy_result"]["token_ids"])
        if boxed_index == 0:
            continue
        absolute_pos.append(datum["critical_token_result"]["critical_token_index"])
        final_answer_correct.append(
            "correct" if datum["example"]["answer"] == datum["critical_token_result"]["majority_answer"] else "wrong"
        )
    total = len(absolute_pos)
    correct_pos = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "correct"]
    wrong_pos = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "wrong"]
    return correct_pos, wrong_pos, total


def draw_absolute_histogram_on_ax(ax, correct_pos, wrong_pos, total):
    all_pos = list(correct_pos) + list(wrong_pos)
    data_max = max(all_pos) if all_pos else 30000
    bin_size = max(100, int(np.ceil(data_max / 20 / 100)) * 100)
    bins = [x * bin_size for x in range(int(np.ceil(data_max / bin_size)) + 1)]
    ax.hist([correct_pos, wrong_pos], bins=bins, stacked=True,
            weights=[np.ones(len(correct_pos)) / total, np.ones(len(wrong_pos)) / total],
            color=['#bcd373', '#ff8ca1'])
    if correct_pos:
        ax.axvline(np.mean(correct_pos), color='#98c126', linestyle=':', linewidth=1.5)
    if wrong_pos:
        ax.axvline(np.mean(wrong_pos), color='#f45f74', linestyle=':', linewidth=1.5)
    ax.set_xlim(-50, bins[-1] + 50)
    ax.set_ylim(0, 1)
    if correct_pos and wrong_pos:
        mean_correct = np.mean(correct_pos)
        mean_wrong = np.mean(wrong_pos)
        mid_y = 0.6
        ax.annotate(
            '', xy=(mean_wrong, mid_y), xytext=(mean_correct, mid_y),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2)
        )
        ax.text((mean_correct + mean_wrong) / 2, mid_y + 0.02,
                f'Δ={abs(mean_wrong - mean_correct):.0f}',
                ha='center', va='bottom', fontsize=7, color='gray')
    ax.set_xlabel("Critical Token Position (Absolute)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    legend_handles = [
        (Patch(facecolor='#bcd373'), Line2D([0], [0], color='#bcd373', linestyle=':', linewidth=1.5)),
        (Patch(facecolor='#ff8ca1'), Line2D([0], [0], color='#ff8ca1', linestyle=':', linewidth=1.5)),
    ]
    ax.legend(legend_handles, ['Correct', 'Incorrect'],
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=7)


def draw_absolute_histogram_plain_on_ax(ax, all_pos, total):
    data_max = max(all_pos) if all_pos else 30000
    bin_size = max(100, int(np.ceil(data_max / 20 / 100)) * 100)
    bins = [x * bin_size for x in range(int(np.ceil(data_max / bin_size)) + 1)]
    ax.hist(all_pos, bins=bins, weights=np.ones(len(all_pos)) / total, color='#7ab8d9')
    if all_pos:
        ax.axvline(np.mean(all_pos), color='#2a7ab0', linestyle=':', linewidth=1.5)
    ax.set_xlim(-50, bins[-1] + 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Critical Token Position (Absolute)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)


def main(args):
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    data = []
    with open(args.file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} records from {args.file}")

    # Figure out if the critical token is before/after "\boxed"
    critical_token_before_boxed_cnt = 0
    absolute_pos = []
    relative_pos = []
    final_answer_correct = [] # List[Literal["correct", "wrong"]]
    for datum in data:
        # boxed_index = find_boxed_token_index(tokenizer, datum["greedy_result"]["token_ids"])
        # if boxed_index > datum["critical_token_result"]["critical_token_index"]:
        #     critical_token_before_boxed_cnt += 1
        # else:
        #     pass
        boxed_index = len(datum["greedy_result"]["token_ids"])
        if boxed_index == 0:
            continue
        absolute_pos.append(datum["critical_token_result"]["critical_token_index"])
        relative_pos.append(datum["critical_token_result"]["critical_token_index"] / boxed_index)
        final_answer_correct.append("correct" if datum["example"]["answer"] == datum["critical_token_result"]["majority_answer"] else "wrong")
    # print(f"Critical token before \\boxed rate: {critical_token_before_boxed_cnt/len(data)}")

    # Histogram: Critical token positions relative to index of \\boxed
    relative_pos = [min(r, 1.0) for r in relative_pos]
    total = len(relative_pos)
    correct_pos = [p for p, c in zip(relative_pos, final_answer_correct) if c == "correct"]
    wrong_pos = [p for p, c in zip(relative_pos, final_answer_correct) if c == "wrong"]
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_relative_histogram_on_ax(ax, correct_pos, wrong_pos, total)
    ax.set_title("Histogram of Critical Token Positions")
    file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    fig.savefig(f"plots/critical_token_position_relative_histogram_{file_alias}.svg", bbox_inches='tight')
    plt.close(fig)

    # Histogram: Absolute token positions
    correct_abs = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "correct"]
    wrong_abs = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "wrong"]
    fig, ax = plt.subplots(figsize=(4, 3))
    draw_absolute_histogram_on_ax(ax, correct_abs, wrong_abs, total)
    ax.set_title("Histogram of Critical Token Positions")
    fig.savefig(f"plots/critical_token_position_absolute_histogram_{file_alias}.svg", bbox_inches='tight')
    plt.close(fig)

    # Histogram: Absolute token positions
    # plt.figure(figsize=(4, 3))
    # plt.hist(absolute_pos, bins=[x*4000/20 for x in range(21)], weights=np.ones_like(absolute_pos) / len(absolute_pos))
    # plt.xlim(-50, 4050)
    # plt.ylim(0, 1)
    # plt.xlabel("Critical Token Position (Absolute)")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Critical Token Positions")
    # plt.savefig(f"plots/critical_token_position_absolute_histogram_{file_alias}.svg")
    # plt.close()
    
    # Scatter plot
    
    absolute_pos_dict[file_alias] = absolute_pos

def main_deepseek_distill(args):
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    data = []
    with open(args.file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} records from {args.file}")

    # Figure out if the critical token is before/after "\boxed"
    critical_token_before_boxed_cnt = 0
    absolute_pos = []
    relative_pos = []
    final_answer_correct = [] # List[Literal["correct", "wrong"]]
    for datum in data:
        # boxed_index = find_boxed_token_index(tokenizer, datum["greedy_result"]["token_ids"])
        # if boxed_index > datum["critical_token_result"]["critical_token_index"]:
        #     critical_token_before_boxed_cnt += 1
        # else:
        #     pass
        boxed_index = len(datum["greedy_result"]["token_ids"])
        if boxed_index == 0:
            continue
        critical_token_index = datum["critical_token_result"]["critical_token_index"]
        # critical_token_index = len(datum["greedy_result"]["token_ids"])
        # if "insert_reflection_result" not in datum:
        #     continue
        # for reflection in reversed(datum["insert_reflection_result"][""]):
        #     answers = reflection["answer"]
        #     majority_answer = max(answers, key=answers.get)
        #     if answers[majority_answer] < 0.95:
        #         break
        #     critical_token_index = len(reflection["partial_response_tokens"])
            
        absolute_pos.append(critical_token_index)
        relative_pos.append(critical_token_index / boxed_index)
        final_answer_correct.append("correct" if datum["example"]["answer"] == datum["critical_token_result"]["majority_answer"] else "wrong")
    # print(f"Critical token before \\boxed rate: {critical_token_before_boxed_cnt/len(data)}")

    # Histogram: Critical token positions relative to index of \\boxed
    # 0-0.1, 0.1-0.2, ..., 0.9-1.0, 1.0+
    relative_pos = [min(r, 1.0) for r in relative_pos]
    # Draw histogram
    bins_rel = [x/20 for x in range(21)]
    total = len(relative_pos)
    correct_rel = [p for p, c in zip(relative_pos, final_answer_correct) if c == "correct"]
    wrong_rel = [p for p, c in zip(relative_pos, final_answer_correct) if c == "wrong"]
    plt.figure(figsize=(4, 3))
    plt.hist([correct_rel, wrong_rel], bins=bins_rel, stacked=True,
             weights=[np.ones(len(correct_rel)) / total, np.ones(len(wrong_rel)) / total],
             color=['#bcd373', '#ff8ca1'], label=['Correct', 'Incorrect'])
    if correct_rel:
        plt.axvline(np.mean(correct_rel), color='#bcd373', linestyle=':', linewidth=1.5)
    if wrong_rel:
        plt.axvline(np.mean(wrong_rel), color='#ff8ca1', linestyle=':', linewidth=1.5)
    plt.ylim(0, 1)
    if correct_rel and wrong_rel:
        mean_correct = np.mean(correct_rel)
        mean_wrong = np.mean(wrong_rel)
        mid_y = 0.6
        plt.annotate(
            '', xy=(mean_wrong, mid_y), xytext=(mean_correct, mid_y),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2)
        )
        plt.text((mean_correct + mean_wrong) / 2, mid_y + 0.02,
                 f'Δ={abs(mean_wrong - mean_correct):.2f}',
                 ha='center', va='bottom', fontsize=7, color='gray')
    plt.xlabel("Critical Token Position (Relative to \\boxed)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Critical Token Positions")
    legend_handles = [
        (Patch(facecolor='#bcd373'), Line2D([0], [0], color='#bcd373', linestyle=':', linewidth=1.5)),
        (Patch(facecolor='#ff8ca1'), Line2D([0], [0], color='#ff8ca1', linestyle=':', linewidth=1.5)),
    ]
    plt.legend(legend_handles, ['Correct', 'Incorrect'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=7)
    file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    plt.savefig(f"plots/critical_token_position_relative_histogram_{file_alias}.svg")
    plt.close()

    # Histogram: Absolute token positions
    correct_abs = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "correct"]
    wrong_abs = [p for p, c in zip(absolute_pos, final_answer_correct) if c == "wrong"]
    plt.figure(figsize=(4, 3))
    plt.hist([correct_abs, wrong_abs], bins=[x*4000/20 for x in range(21)], stacked=True,
             weights=[np.ones(len(correct_abs)) / total, np.ones(len(wrong_abs)) / total],
             color=['#bcd373', '#ff8ca1'], label=['Correct', 'Incorrect'])
    plt.xlim(-50, 4050)
    plt.ylim(0, 1)
    plt.xlabel("Critical Token Position (Absolute)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Critical Token Positions")
    plt.legend(fontsize=7)
    plt.savefig(f"plots/critical_token_position_absolute_histogram_{file_alias}.svg")
    plt.close()
    
    
def print_iteration_stats():
    # target_models = ["Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", ]
    target_models = ["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "s1.1-7B",
                    "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"]
    model_binary_iters = {m: [] for m in target_models}
    model_reflection_iters = {m: [] for m in target_models}

    for fname in os.listdir("data"):
        if not (fname.startswith("insert_reflection_") and fname.endswith(".jsonl") and "temp" not in fname):
            continue
        model = next((m for m in target_models if fname.endswith(f"_{m}.jsonl")), None)
        if model is None:
            continue
        fpath = os.path.join("data", fname)
        with open(fpath) as f:
            for line in f:
                datum = json.loads(line)
                bst = datum.get("critical_token_result", {}).get("binary_search_trace")
                if bst is not None:
                    model_binary_iters[model].append(len(bst))
                ir_empty = datum.get("insert_reflection_result", {}).get("")
                if ir_empty is not None:
                    model_reflection_iters[model].append(len(ir_empty))

    reflection_col = 'Avg Insert-Reflection("") Iters'
    print("\n=== Average Iteration Counts ===")
    print(f"{'Model':<35} {'Avg Binary Search Iters':>23} {reflection_col:>31}")
    print("-" * 92)
    for model in target_models:
        bi = model_binary_iters[model]
        ri = model_reflection_iters[model]
        avg_bi = np.mean(bi) if bi else float("nan")
        avg_ri = np.mean(ri) if ri else float("nan")
        print(f"{model:<35} {avg_bi:>23.2f} {avg_ri:>31.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    bulk_mode = args.file is None
    if bulk_mode:
        for file in os.listdir("data"):
            if file.endswith(".jsonl") and file.startswith("critical_tokens_"):
                args.file = os.path.join("data", file)
                if "Distill" in args.file:
                    continue
                else:
                    main(args)
            # elif "Distill" in file and file.endswith(".jsonl") and file.startswith("insert_reflection_") and "temp" not in file:
            #     # Special rule for DeepSeek Distill due to non-standard behaviors
            #     args.file = os.path.join("data", file)
            #     main_deepseek_distill(args)
        # Build combined relative histogram figures by redrawing from data
        def build_combined_grid(datasets, models, col_labels, row_labels, figsize, outpath, transpose=False):
            if transpose:
                nrows, ncols = len(models), len(datasets)
            else:
                nrows, ncols = len(datasets), len(models)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(nrows, ncols)
            for r in range(nrows):
                for c in range(ncols):
                    ax = axes[r, c]
                    dataset = datasets[c] if transpose else datasets[r]
                    model = models[r] if transpose else models[c]
                    filepath = f"data/critical_tokens_{dataset}_{model}.jsonl"
                    if os.path.exists(filepath):
                        correct_pos, wrong_pos, total = load_relative_pos_data(filepath)
                        draw_relative_histogram_on_ax(ax, correct_pos, wrong_pos, total)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                    if r == 0:
                        ax.set_title((col_labels[c] if not transpose else row_labels[c]), fontsize=10)
                    if c == 0:
                        rlabel = row_labels[r] if not transpose else col_labels[r]
                        ax.set_ylabel(f"{rlabel}\nFrequency", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if r != nrows - 1:
                        ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(outpath)
            plt.close(fig)

        build_combined_grid(
            datasets=["math500", "gpqa-diamond", "aime2024"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "s1.1-7B",
                    "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR", "Qwen2.5-7B-GRPO"],
            row_labels=["MATH-500", "GPQA-Diamond", "AIME 2024"],
            figsize=(4 * 3, 3 * 5),
            outpath="plots/combined_relative_histogram_5x3_correctness.svg",
            transpose=True,
        )
        build_combined_grid(
            datasets=["math500", "gpqa-diamond"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "One-Shot-RLVR"],
            row_labels=["MATH-500", "GPQA-Diamond"],
            figsize=(4 * 3, 3 * 2),
            outpath="plots/combined_relative_histogram_2x3_correctness.svg",
        )

        def build_combined_grid_absolute(datasets, models, col_labels, row_labels, figsize, outpath, transpose=False):
            if transpose:
                nrows, ncols = len(models), len(datasets)
            else:
                nrows, ncols = len(datasets), len(models)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(nrows, ncols)
            for r in range(nrows):
                for c in range(ncols):
                    ax = axes[r, c]
                    dataset = datasets[c] if transpose else datasets[r]
                    model = models[r] if transpose else models[c]
                    filepath = f"data/critical_tokens_{dataset}_{model}.jsonl"
                    if os.path.exists(filepath):
                        correct_pos, wrong_pos, total = load_absolute_pos_data(filepath)
                        draw_absolute_histogram_on_ax(ax, correct_pos, wrong_pos, total)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                    if r == 0:
                        ax.set_title((col_labels[c] if not transpose else row_labels[c]), fontsize=10)
                    if c == 0:
                        rlabel = row_labels[r] if not transpose else col_labels[r]
                        ax.set_ylabel(f"{rlabel}\nFrequency", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if r != nrows - 1:
                        ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(outpath)
            plt.close(fig)

        build_combined_grid_absolute(
            datasets=["math500", "gpqa-diamond", "aime2024"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "s1.1-7B",
                    "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR", "Qwen2.5-7B-GRPO"],
            row_labels=["MATH-500", "GPQA-Diamond", "AIME 2024"],
            figsize=(4 * 3, 3 * 5),
            outpath="plots/combined_absolute_histogram_5x3_correctness.svg",
            transpose=True,
        )
        build_combined_grid_absolute(
            datasets=["math500", "gpqa-diamond"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "One-Shot-RLVR"],
            row_labels=["MATH-500", "GPQA-Diamond"],
            figsize=(4 * 3, 3 * 2),
            outpath="plots/combined_absolute_histogram_2x3_correctness.svg",
        )

        def build_combined_grid_plain(datasets, models, col_labels, row_labels, figsize, outpath, transpose=False):
            if transpose:
                nrows, ncols = len(models), len(datasets)
            else:
                nrows, ncols = len(datasets), len(models)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(nrows, ncols)
            for r in range(nrows):
                for c in range(ncols):
                    ax = axes[r, c]
                    dataset = datasets[c] if transpose else datasets[r]
                    model = models[r] if transpose else models[c]
                    filepath = f"data/critical_tokens_{dataset}_{model}.jsonl"
                    if os.path.exists(filepath):
                        correct_pos, wrong_pos, total = load_relative_pos_data(filepath)
                        all_pos = correct_pos + wrong_pos
                        draw_relative_histogram_plain_on_ax(ax, all_pos, total)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                    if r == 0:
                        ax.set_title((col_labels[c] if not transpose else row_labels[c]), fontsize=10)
                    if c == 0:
                        rlabel = row_labels[r] if not transpose else col_labels[r]
                        ax.set_ylabel(f"{rlabel}\nFrequency", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if r != nrows - 1:
                        ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(outpath)
            plt.close(fig)

        build_combined_grid_plain(
            datasets=["math500", "gpqa-diamond", "aime2024"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "s1.1-7B",
                    "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR", "Qwen2.5-7B-GRPO"],
            row_labels=["MATH-500", "GPQA-Diamond", "AIME 2024"],
            figsize=(4 * 3, 3 * 5),
            outpath="plots/combined_relative_histogram_5x3_plain.svg",
            transpose=True,
        )
        build_combined_grid_plain(
            datasets=["math500", "gpqa-diamond"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "One-Shot-RLVR"],
            row_labels=["MATH-500", "GPQA-Diamond"],
            figsize=(4 * 3, 3 * 2),
            outpath="plots/combined_relative_histogram_2x3_plain.svg",
        )

        def build_combined_grid_absolute_plain(datasets, models, col_labels, row_labels, figsize, outpath, transpose=False):
            if transpose:
                nrows, ncols = len(models), len(datasets)
            else:
                nrows, ncols = len(datasets), len(models)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(nrows, ncols)
            for r in range(nrows):
                for c in range(ncols):
                    ax = axes[r, c]
                    dataset = datasets[c] if transpose else datasets[r]
                    model = models[r] if transpose else models[c]
                    filepath = f"data/critical_tokens_{dataset}_{model}.jsonl"
                    if os.path.exists(filepath):
                        correct_pos, wrong_pos, total = load_absolute_pos_data(filepath)
                        all_pos = correct_pos + wrong_pos
                        draw_absolute_histogram_plain_on_ax(ax, all_pos, total)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                    if r == 0:
                        ax.set_title((col_labels[c] if not transpose else row_labels[c]), fontsize=10)
                    if c == 0:
                        rlabel = row_labels[r] if not transpose else col_labels[r]
                        ax.set_ylabel(f"{rlabel}\nFrequency", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if r != nrows - 1:
                        ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(outpath)
            plt.close(fig)

        build_combined_grid_absolute_plain(
            datasets=["math500", "gpqa-diamond", "aime2024"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "s1.1-7B",
                    "One-Shot-RLVR-Qwen2.5-7B-pi1", "Qwen2.5-7B-Instruct-GRPO"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR", "Qwen2.5-7B-GRPO"],
            row_labels=["MATH-500", "GPQA-Diamond", "AIME 2024"],
            figsize=(4 * 3, 3 * 5),
            outpath="plots/combined_absolute_histogram_5x3_plain.svg",
            transpose=True,
        )
        build_combined_grid_absolute_plain(
            datasets=["math500", "gpqa-diamond"],
            models=["Qwen2.5-7B-Instruct", "OpenThinker3-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"],
            col_labels=["Qwen2.5-7B", "OpenThinker3-7B", "One-Shot-RLVR"],
            row_labels=["MATH-500", "GPQA-Diamond"],
            figsize=(4 * 3, 3 * 2),
            outpath="plots/combined_absolute_histogram_2x3_plain.svg",
        )
    else:
        main(args)

    if bulk_mode:
        print_iteration_stats()