import json
import re
from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

# Load all JSONL files in data/
model_set = {
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Llama-3.2-3B-Instruct",
}
problem_id_to_critical_tokens = {} # problem_id -> model -> critical_token_data
for file in os.listdir("data"):
    file = os.path.join("data", file)
    dataset, model = file.split("/")[-1].replace(".jsonl", "").replace("critical_tokens_", "").split("_")[:2]
    if model not in model_set:
        continue
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    for datum in data:
        if datum is None:
            continue

        problem_id = datum["example"]["id"]
        if problem_id not in problem_id_to_critical_tokens:
            problem_id_to_critical_tokens[problem_id] = {}
        if model not in problem_id_to_critical_tokens[problem_id]:
            problem_id_to_critical_tokens[problem_id][model] = datum
print(len(problem_id_to_critical_tokens))
# Find problems that (1) Many LLMs got incorrect, and (2) multiple LLMs' critical token index is not too early or too late
interesting_problems = [] # List of (problem_id, [models that have critical token in middle])
for problem_id, model_to_critical_tokens in problem_id_to_critical_tokens.items():
    # Count how many models got it wrong
    wrong_models = []
    for model, critical_token_data in model_to_critical_tokens.items():
        if critical_token_data["example"]["answer"] != critical_token_data["critical_token_result"]["majority_answer"]:
            wrong_models.append(model)
    # if num_models_wrong / num_models < 0.5:
    #     continue
    incorrect_rate = len(wrong_models) / len(model_to_critical_tokens)
    
    # Check critical token positions
    interesting_models = [
        model for model, critical_token_data in model_to_critical_tokens.items()
        if 0.2 < (critical_token_data["critical_token_result"]["critical_token_index"] / len(critical_token_data["greedy_result"]["token_ids"])) < 0.8
    ]
    if len(interesting_models) >= 1:
        interesting_problems.append((problem_id, wrong_models, interesting_models))

for problem_id, wrong_models, interesting_models in interesting_problems:
    print(f"Problem ID: {problem_id}, Wrong Models: {wrong_models}, Interesting Models: {interesting_models}, Incorrect Rate: {len(wrong_models)/len(problem_id_to_critical_tokens[problem_id])}")
print("Total problems:", len(problem_id_to_critical_tokens))
print("Interesting problems:", len(interesting_problems))
# Count interesting problems by dataset
dataset_to_count = {}
for problem_id, _, _ in interesting_problems:
    dataset = problem_id.split("-")[0]
    if dataset not in dataset_to_count:
        dataset_to_count[dataset] = 0
    dataset_to_count[dataset] += 1
print(dataset_to_count)