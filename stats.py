import json
import re
# from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


def find_boxed_token_index(tokenizer, token_ids):
    # Binary search
    head = 0; tail = len(token_ids)
    while head < tail:
        mid = (head + tail) // 2
        str_mid = tokenizer.decode(token_ids[:mid])
        if "\\boxed" in str_mid:
            tail = mid
        else:
            head = mid + 1
    return head

def main(args):
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    data = []
    with open(args.file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} records from {args.file}")

    # Figure out if the critical token is before/after "\boxed"
    critical_token_before_boxed_cnt = 0
    relative_pos = []
    final_answer_correct = [] # List[Literal["correct", "wrong"]]
    for datum in data:
        # boxed_index = find_boxed_token_index(tokenizer, datum["greedy_result"]["token_ids"])
        # if boxed_index > datum["critical_token_result"]["critical_token_index"]:
        #     critical_token_before_boxed_cnt += 1
        # else:
        #     pass
        boxed_index = len(datum["greedy_result"]["token_ids"])
        relative_pos.append(datum["critical_token_result"]["critical_token_index"] / boxed_index)
        final_answer_correct.append("correct" if datum["example"]["answer"] == datum["critical_token_result"]["majority_answer"] else "wrong")
    # print(f"Critical token before \\boxed rate: {critical_token_before_boxed_cnt/len(data)}")

    # Histogram: Critical token positions relative to index of \\boxed
    # 0-0.1, 0.1-0.2, ..., 0.9-1.0, 1.0+
    relative_pos = [min(r, 1.0) for r in relative_pos]
    # Draw histogram
    plt.figure(figsize=(4, 3))
    plt.hist(relative_pos, bins=[i/10 for i in range(12)], weights=np.ones_like(relative_pos) / len(relative_pos))
    plt.xlabel("Forking Token Position (Relative to \\boxed)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Critical Token Positions")
    file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    plt.savefig(f"plots/critical_token_position_histogram_{file_alias}.svg")

    # Draw 2 histograms in same plot: final_answer_correct=="correct" or =="wrong"
    # plt.figure(figsize=(4, 3))
    # correct_relative_pos = [r for r, c in zip(relative_pos, final_answer_correct) if c == "correct"]
    # wrong_relative_pos = [r for r, c in zip(relative_pos, final_answer_correct) if c == "wrong"]
    # plt.hist(correct_relative_pos,
    #          bins=[i/10 for i in range(12)], weights=np.ones_like(correct_relative_pos) / len(correct_relative_pos), alpha=0.5, label="Correct")
    # plt.hist(wrong_relative_pos,
    #          bins=[i/10 for i in range(12)], weights=np.ones_like(wrong_relative_pos) / len(wrong_relative_pos), alpha=0.5, label="Wrong")
    # plt.xlabel("Forking Token Position (Relative to \\boxed)")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Forking Token Positions")
    # plt.legend()
    # file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    # plt.savefig(f"plots/critical_token_position_finalanswercorrect_histogram_{file_alias}.svg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file is None:
        for file in os.listdir("data"):
            if file.endswith(".jsonl") and file.startswith("critical_tokens_"):
                args.file = os.path.join("data", file)
                main(args)
    else:
        main(args)