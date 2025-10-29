import os
import matplotlib.pyplot as plt

def plot_critical_token(datum):
    critical_token_result = datum["critical_token_result"]
    config = datum["config"]
    example = datum["example"]
    # Create two subplots of size (10, 6), vertically placed.
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # First figure: binary search trace of answer probabilities
    plt.sca(axs[0])
    plt.xlim(0, len(datum["greedy_result"]["token_ids"]) - 1)
    plt.ylim(0, 1)
    # Extract answers with reasonable prob mass
    # prob > 0.1 in any step
    label_candidates = set()
    label_final = set()
    for step in critical_token_result['binary_search_trace']:
        distribution = step["answer_dict"]
        label_candidates.update(distribution.keys())
    for label in label_candidates:
        probs = []
        for step in critical_token_result['binary_search_trace']:
            distribution = step["answer_dict"]
            prob = distribution.get(label, 0.0)
            probs.append(prob)
        if any(p > 0.1 for p in probs):
            label_final.add(label)

    # Plot binary search trace as accumulative zone line plot
    x = []
    y_list = [list() for _ in label_final]
    for step in sorted(critical_token_result['binary_search_trace'], key=lambda x: x["index"]):
        index = step["index"]
        x.append(index)

        distribution = step["answer_dict"]
        prob_sum = 0
        for i, label in enumerate(label_final):
            prob = distribution.get(label, 0.0)
            prob_sum += prob
            y_list[i].append(prob_sum)
    
    # duplicate first and last element
    x.insert(0, 0)
    x.append(len(datum["greedy_result"]["token_ids"]) - 1)
    # print(x)
    for i in range(len(label_final)):
        y_list[i].insert(0, y_list[i][0])
        y_list[i].append(y_list[i][-1])
    answer_cmap = {}
    for i, label in enumerate(label_final):
        answer_cmap[label] = plt.get_cmap('viridis')(i / len(label_final))
    for i, label in enumerate(label_final):
        plt.fill_between(x, 
                         0 if i == 0 else y_list[i-1], 
                         y_list[i], 
                         label=label,
                         color=answer_cmap[label])
    plt.fill_between(x, 
                     y_list[-1], 
                     1, 
                     label='Other',
                     color='lightgray')
    
    # Plot critical token line
    critical_token_index = critical_token_result['critical_token_index']
    plt.axvline(x=critical_token_index, color='red', linestyle='--', label='Critical Token')
    plt.text(critical_token_index, 0.5, f'Critical Token\nIndex: {critical_token_index}\nText: "{critical_token_result["critical_token"]["text"]}"',
             rotation=90, verticalalignment='center', color='red')

    model = config["model"].split("/")[-1]
    dataset = example.get("id")
    plt.xlabel('Token Index')
    plt.ylabel('Answer probability')
    plt.title(f'Critical Token Analysis - {model}, {dataset}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Second figure: answer distribution after inserting "wait".
    plt.sca(axs[1])
    plt.xlim(0, len(datum["greedy_result"]["token_ids"]) - 1)
    plt.ylim(0, 1)
    insert_reflection_result = datum.get("insert_reflection_result", {})
    reflection_str = insert_reflection_result.get("reflection_str", "")
    reflection_trace = insert_reflection_result.get("reflection_trace", [])
    for point in reflection_trace:
        prefix_length = len(point["partial_response_tokens"]) - 1
        answer_dict = point["answer"]
        # reorganize answer_dict based on the label set
        new_answer_dict = {label: answer_dict.get(label, 0.0) for label in label_final}
        other_prob = 1.0 - sum(new_answer_dict.values())
        new_answer_dict["Other"] = other_prob
        # add bar in x=prefix_length, y=new_answer_dict, width=1
        bottom = 0.0
        for label, prob in new_answer_dict.items():
            plt.bar(prefix_length, prob, bottom=bottom, label=label, width=5.0,
                    color=answer_cmap.get(label, 'lightgray')) # make it wide for visibility
            bottom += prob
    plt.xlabel('Insert reflection index')
    plt.ylabel('Answer probability')
    plt.title(f'Answer Distribution after Inserting Reflection "{reflection_str}"')


    # Save figure
    plt.savefig(f'plots/critical_tokens/critical_token_{model}_{dataset.replace("/", "_")}.svg', format='svg')
    plt.close()

def is_interesting(datum):
    # 0. greedy trace is at least 100 tokens long
    # 1. Final answer is wrong
    # 2. Critical token index is somewhere in the middle (not first or last 20%)
    # 3. Initially, the majority answer has less than 0.5 probability

    greedy_result = datum["greedy_result"]
    if len(greedy_result["token_ids"]) < 100:
        return False

    final_answer = datum['critical_token_result']['majority_answer']
    correct_answer = datum['critical_token_result'].get('example', {}).get('answer')
    if final_answer == correct_answer:
        return False
    
    critical_token_index = datum['critical_token_result']['critical_token_index']
    total_tokens = len(datum["greedy_result"]["token_ids"])
    if critical_token_index < total_tokens * 0.1 or critical_token_index > total_tokens * 0.9:
        return False

    # Initially, the majority answer has less than <0.5 probability
    binary_search_trace = datum['critical_token_result']['binary_search_trace']
    binary_search_trace.sort(key=lambda x: x["index"])
    initial_distribution = binary_search_trace[0]["answer_dict"]
    initial_majority_answer = max(initial_distribution, key=initial_distribution.get)
    if initial_distribution.get(initial_majority_answer, 0.0) >= 0.8:
        return False

    return True


if __name__ == "__main__":
    for model in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
        for dataset in ["gpqa-diamond", "math500", "aime2024"]:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                import json
                data = [json.loads(line) for line in f]
            for d in data:
                if is_interesting(d):
                    plot_critical_token(d)