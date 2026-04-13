import json

models = {
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'OpenThinker3-7B': 'OpenThinker3',
    's1.1-7B': 's1.1-7B',
    'One-Shot-RLVR-Qwen2.5-7B-pi1': '1-Shot-RLVR',
    'Qwen2.5-7B-Instruct-GRPO': 'GRPO',
}
datasets = ['math500', 'gpqa-diamond', 'aime2024']
dataset_labels = {'math500': 'MATH-500', 'gpqa-diamond': 'GPQA-Diamond', 'aime2024': 'AIME2024'}

results = {}
for model_key, model_label in models.items():
    results[model_label] = {}
    for ds in datasets:
        path = f'data/critical_tokens_{ds}_{model_key}.jsonl'
        correct = total = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                maj = str(d['critical_token_result']['majority_answer']).strip()
                gt = str(d['example']['answer']).strip()
                total += 1
                if maj == gt:
                    correct += 1
        results[model_label][ds] = (correct, total)

model_order = ['(Base)', 'OpenThinker3', 's1.1', 'One-Shot', 'GRPO']

print(r'\begin{table}[h]')
print(r'\centering')
print(r'\caption{Final Answer Accuracy (Majority Vote) across Models and Datasets}')
print(r'\label{tab:accuracy}')
print(r'\begin{tabular}{l' + 'c' * len(datasets) + '}')
print(r'\toprule')
print('Model & ' + ' & '.join(dataset_labels[d] for d in datasets) + r' \\')
print(r'\midrule')
for m in model_order:
    row = m
    for ds in datasets:
        c, t = results[m][ds]
        row += f' & {100.0 * c / t:.1f}'
    row += r' \\'
    print(row)
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
