import json
import re
from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

EXAMPLE = {
  "prompt": "Let $\\omega \\neq 1$ be a 13th root of unity. Find the remainder when \n\\[ \\prod_{k=0}^{12}(2 - 2\\omega^k + \\omega^{2k}) \\] is divided by 1000.",
  "problem_id": "aime2024-29",
  "greedy_output_len": 279,
  "greedy_output": "Here's how to solve the problem:\n\n**Understanding the Problem**\n\nWe're given a product of terms involving complex numbers and asked to find the remainder when this product is divided by 1000.  The key is to recognize that the problem involves roots of unity and their properties.\n\n**Key Properties of Roots of Unity**\n\n* **Geometric Interpretation:**  The roots of unity are the complex numbers that satisfy the equation  *z*<sup>n</sup> = 1.  They represent the points on the unit circle in the complex plane.\n* **Sum of Roots:** The sum of the roots of unity is 0.\n* **Product of Roots:** The product of the roots of unity is 1.\n\n**Solution**\n\n1. **Simplify the Product:**  We can simplify the product by expanding it and using the properties of roots of unity.  \n\n2. **Apply Properties:**  We'll use the fact that the roots of unity have specific properties and simplify the expression.\n\n3. **Find the Remainder:**  After simplifying, we'll find the remainder when the simplified expression is divided by 1000.\n\n**Let me know if you'd like me to walk through the detailed steps of the solution.** \n\n**Final Answer:**\n\n\\boxed{250} \n",
  "greedy_output_token_ids": [], # List of int
  "critical_token_index": 274,
  "critical_token": {
    "token_id": 235276,
    "text": "0",
    "probability": 0.4083376240567979
  },
  "majority_answer": "250"
}

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
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

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
        boxed_index = find_boxed_token_index(tokenizer, datum["critical_token"]["greedy_result"]["token_ids"])
        if boxed_index > datum["critical_token"]["critical_token_index"]:
            critical_token_before_boxed_cnt += 1
        else:
            pass
        relative_pos.append(datum["critical_token"]["critical_token_index"] / boxed_index)
        final_answer_correct.append("correct" if datum["example"]["answer"] == datum["critical_token"]["majority_answer"] else "wrong")
    print(f"Critical token before \\boxed rate: {critical_token_before_boxed_cnt/len(data)}")

    # Histogram: Critical token positions relative to index of \\boxed
    # 0-0.1, 0.1-0.2, ..., 0.9-1.0, 1.0+
    relative_pos = [min(r, 1.0) for r in relative_pos]
    # Draw histogram
    plt.figure(figsize=(4, 3))
    plt.hist(relative_pos, bins=[i/10 for i in range(12)], weights=np.ones_like(relative_pos) / len(relative_pos))
    plt.xlabel("Forking Token Position (Relative to \\boxed)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Forking Token Positions")
    file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    # plt.savefig(f"plots/forking_token_position_histogram_{file_alias}.png")
    plt.savefig(f"plots/forking_token_position_histogram_{file_alias}.svg")

    # Draw 2 histograms in same plot: final_answer_correct=="correct" or =="wrong"
    plt.figure(figsize=(4, 3))
    correct_relative_pos = [r for r, c in zip(relative_pos, final_answer_correct) if c == "correct"]
    wrong_relative_pos = [r for r, c in zip(relative_pos, final_answer_correct) if c == "wrong"]
    plt.hist(correct_relative_pos,
             bins=[i/10 for i in range(12)], weights=np.ones_like(correct_relative_pos) / len(correct_relative_pos), alpha=0.5, label="Correct")
    plt.hist(wrong_relative_pos,
             bins=[i/10 for i in range(12)], weights=np.ones_like(wrong_relative_pos) / len(wrong_relative_pos), alpha=0.5, label="Wrong")
    plt.xlabel("Forking Token Position (Relative to \\boxed)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Forking Token Positions")
    plt.legend()
    file_alias = args.file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "")
    # plt.savefig(f"plots/forking_token_position_histogram_{file_alias}.png")
    plt.savefig(f"plots/forking_token_position_finalanswercorrect_histogram_{file_alias}.svg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file is None:
        for file in os.listdir("data"):
            if file.endswith(".jsonl"):
                args.file = os.path.join("data", file)
                main(args)
    else:
        main(args)