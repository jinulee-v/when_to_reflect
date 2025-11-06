import os
import matplotlib.pyplot as plt

def reflection_stat(data, model, dataset):
    # Compute average entropy over (1) reflection before critical token, (2) reflection after critical token
    # Draw two overlapping histograms
    base_before_entropy = []
    base_after_entropy = []
    reflection_before_entropy = []
    reflection_after_entropy = []
    for d in data:
        base_befores = []
        base_afters = []
        for trace_point in d['critical_token_result']['binary_search_trace']:
            answer_dict = trace_point['answer_dict']
            import math
            entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
            if trace_point['index'] < d['critical_token_result']['critical_token_index']:
                base_befores.append(entropy)
            else:
                base_afters.append(entropy)
        befores = []
        afters = []
        for trace_point in d['insert_reflection_result']['Wait']:
            answer_dict = trace_point['answer']
            import math
            entropy = -sum(p * math.log(p) for p in answer_dict.values()) # no zero value guaranteed
            if len(trace_point['partial_response_tokens']) < d['critical_token_result']['critical_token_index']:
                befores.append(entropy)
            else:
                afters.append(entropy)
        if base_befores and base_afters and befores and afters:
            base_before_entropy.append(sum(base_befores) / len(base_befores))
            base_after_entropy.append(sum(base_afters) / len(base_afters))
            reflection_before_entropy.append(sum(befores) / len(befores))
            reflection_after_entropy.append(sum(afters) / len(afters))

    # Make histogram that shares the same bins
    all_values = reflection_before_entropy + reflection_after_entropy + base_before_entropy + base_after_entropy
    if not all_values:
        # nothing to plot
        return
    import numpy as np
    min_v = min(all_values)
    max_v = max(all_values)
    if min_v == max_v:
        # avoid zero-width bins when all values are identical
        min_v -= 0.5
        max_v += 0.5
    bins = np.linspace(min_v, max_v, 31)
    plt.hist(base_before_entropy, bins=bins, alpha=0.5, label='Before Critical Token (Non-reflection)', histtype='step', edgecolor='#feb253')
    plt.hist(base_after_entropy, bins=bins, alpha=0.5, label='After Critical Token (Non-reflection)', histtype='step', edgecolor='#00afbd')
    plt.hist(reflection_before_entropy, bins=bins, alpha=0.5, label='Before Critical Token (Reflection)', color='#feb253')
    plt.hist(reflection_after_entropy, bins=bins, alpha=0.5, label='After Critical Token (Reflection)', color='#00afbd')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title(f'Reflection Entropy Distribution - {model} | {dataset}')
    plt.legend()
    plt.savefig(f'plots/stats_reflection_entropy_{dataset}_{model}.svg')
    plt.clf()


if __name__ == "__main__":
    for model in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
        for dataset in ["gpqa-diamond", "math500", "aime2024"]:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                import json
                data = [json.loads(line) for line in f]

            reflection_stat(data, model, dataset)