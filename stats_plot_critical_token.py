import os
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.rcParams['svg.fonttype'] = 'none'

def plot_critical_token(datum):
    critical_token_result = datum["critical_token_result"]
    config = datum["config"]
    example = datum["example"]
    # Create two subplots of size (10, 6), vertically placed.
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # First figure: merged binary search trace and insert reflection with empty string
    plt.sca(axs[0])
    plt.xlim(0, len(datum["greedy_result"]["token_ids"]) - 1)
    plt.ylim(0, 1)

    # Get both data sources
    insert_reflection_result_empty = datum.get("insert_reflection_result", {})
    reflection_str_empty = ""
    reflection_trace_empty = insert_reflection_result_empty.get(reflection_str_empty, [])

    # Extract answers with reasonable prob mass from both sources
    # prob > 0.1 in any step
    label_candidates = set()
    label_final = set()

    # From binary search trace
    for step in critical_token_result['binary_search_trace']:
        distribution = step["answer_dict"]
        label_candidates.update(distribution.keys())

    # From insert reflection trace
    for point in reflection_trace_empty:
        distribution = point["answer"]
        label_candidates.update(distribution.keys())

    # Filter labels with prob > 0.1
    for label in label_candidates:
        probs = []
        # Check binary search trace
        for step in critical_token_result['binary_search_trace']:
            distribution = step["answer_dict"]
            prob = distribution.get(label, 0.0)
            probs.append(prob)
        # Check reflection trace
        for point in reflection_trace_empty:
            distribution = point["answer"]
            prob = distribution.get(label, 0.0)
            probs.append(prob)
        if any(p > 0.1 for p in probs):
            label_final.add(label)

    if len(label_final) == 0:
        return

    # Order label_final: ground truth first (gold), others sorted by max frequency desc
    ground_truth = example.get("answer")

    def get_max_freq(label):
        max_freq = 0.0
        for step in critical_token_result['binary_search_trace']:
            max_freq = max(max_freq, step["answer_dict"].get(label, 0.0))
        for point in reflection_trace_empty:
            max_freq = max(max_freq, point["answer"].get(label, 0.0))
        return max_freq

    other_labels = sorted(
        [l for l in label_final if l != ground_truth],
        key=get_max_freq, reverse=True
    )
    label_final_ordered = []
    if ground_truth in label_final:
        label_final_ordered.append(ground_truth)
    label_final_ordered.extend(other_labels)
    label_final = label_final_ordered

    # Merge data points from both sources
    merged_data = []

    # Add binary search trace points
    # if "Distill" not in datum["config"]["model"]:
    for step in critical_token_result['binary_search_trace']:
        merged_data.append({
            'index': step["index"],
            'distribution': step["answer_dict"]
        })

    # Add insert reflection trace points
    for point in reflection_trace_empty:
        prefix_length = len(point["partial_response_tokens"]) - 1
        merged_data.append({
            'index': prefix_length,
            'distribution': point["answer"]
        })

    # Sort by index and remove duplicates (keep first occurrence)
    merged_data.sort(key=lambda p: p['index'])
    seen_indices = set()
    unique_merged_data = []
    for point in merged_data:
        if point['index'] not in seen_indices:
            unique_merged_data.append(point)
            seen_indices.add(point['index'])

    # Plot merged trace as accumulative zone line plot
    x = []
    y_list = [list() for _ in label_final]
    for point in unique_merged_data:
        index = point['index']
        x.append(index)

        distribution = point['distribution']
        prob_sum = 0
        for i, label in enumerate(label_final):
            prob = distribution.get(label, 0.0)
            prob_sum += prob
            y_list[i].append(prob_sum)

    # duplicate first and last element
    x.insert(0, 0)
    x.append(len(datum["greedy_result"]["token_ids"]) - 1)
    # print(x)
    try:
        for i in range(len(label_final)):
            y_list[i].insert(0, y_list[i][0])
            y_list[i].append(y_list[i][-1])
    except IndexError:
        return
    answer_cmap = {}
    n_others = sum(1 for l in label_final if l != ground_truth)
    other_idx = 0
    for label in label_final:
        if label == ground_truth:
            answer_cmap[label] = '#FEB253'
        else:
            answer_cmap[label] = plt.get_cmap('viridis')(other_idx / max(n_others, 1))
            other_idx += 1
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
    # critical_token_index = len(datum["greedy_result"]["token_ids"])
    # for reflection in reversed(datum["insert_reflection_result"][""]):
    #     answers = reflection["answer"]
    #     majority_answer = max(answers, key=answers.get)
    #     if answers[majority_answer] < 0.95:
    #         break
    #     critical_token_index = len(reflection["partial_response_tokens"])

    plt.axvline(x=critical_token_index, color='red', linestyle='--', label='Critical Token')
    plt.text(critical_token_index, 0.5, f'Critical Token\nIndex: {critical_token_index}\nText: "{critical_token_result["critical_token"]["text"]}"',
             rotation=90, verticalalignment='center', color='red')

    model = config["model"].split("/")[-1]
    dataset = config["dataset"].split("/")[-1]
    example_id = example.get("id")
    plt.xlabel('Token Index')
    plt.ylabel('Answer probability')
    plt.title(f'Critical Token Analysis - {model}, {dataset}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Second figure: answer distribution after inserting "wait".
    plt.sca(axs[1])
    plt.xlim(0, len(datum["greedy_result"]["token_ids"]) - 1)
    plt.ylim(0, 1)
    insert_reflection_result = datum.get("insert_reflection_result", {})
    reflection_str = "Wait"
    reflection_trace = insert_reflection_result.get(reflection_str, [])
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
    plt.savefig(f'plots/critical_tokens/{dataset}_{model}/{example_id}.svg', format='svg')
    plt.close()

def is_interesting(datum):
    # 0. greedy trace is at least 100 tokens long
    # 1. Final answer is wrong
    # 2. Critical token index is somewhere in the middle (not first or last 20%)
    # 3. Initially, the majority answer has less than 0.5 probability

    greedy_result = datum["greedy_result"]
    if len(greedy_result["token_ids"]) < 100:
        return False

#     # final_answer = datum['critical_token_result']['majority_answer']
#     # correct_answer = datum['critical_token_result'].get('example', {}).get('answer')
#     # if final_answer == correct_answer:
#     #     return False
    
    critical_token_index = datum['critical_token_result']['critical_token_index']
    total_tokens = len(datum["greedy_result"]["token_ids"])
    # if critical_token_index < total_tokens * 0.1 or critical_token_index > total_tokens * 0.9:
    #     return False
    # if critical_token_index < 15 or critical_token_index > total_tokens - 15:
    #     return False
    if critical_token_index < 15:
        return False

#     # Initially, the majority answer has less than <0.5 probability
#     binary_search_trace = datum['critical_token_result']['binary_search_trace']
#     binary_search_trace.sort(key=lambda x: x["index"])
#     initial_distribution = binary_search_trace[0]["answer_dict"]
#     initial_majority_answer = max(initial_distribution, key=initial_distribution.get)
#     if initial_distribution.get(initial_majority_answer, 0.0) >= 0.8:
#         return False

    return True


if __name__ == "__main__":
    os.makedirs(f"plots/critical_tokens", exist_ok=True)
    for model in ["Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-GRPO", "OpenThinker3-7B", "s1.1-7B", "One-Shot-RLVR-Qwen2.5-7B-pi1"]:
        for dataset in ["gpqa-diamond", "math500", "aime2024"]:
            if not os.path.exists(f"data/insert_reflection_{dataset}_{model}.jsonl"):
                continue
            os.makedirs(f"plots/critical_tokens/{dataset}_{model}", exist_ok=True)
            print(f"Processing model={model}, dataset={dataset}")
            with open(f"data/insert_reflection_{dataset}_{model}.jsonl") as f:
                for line in f:
                    d = json.loads(line)
                    if is_interesting(d):
                        plot_critical_token(d)