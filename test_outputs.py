from transformers import AutoTokenizer
import json

with open("data/insert_reflection_aime2024_Qwen2.5-3B-Instruct.jsonl") as f:
    data = [json.loads(line) for line in f][0]
    print(data.keys())

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
for item in data["insert_reflection_result"]["reflection_trace"]:
    print("Partial response tokens:")
    print(tokenizer.decode(item["partial_response_tokens"]))
    print("Num tokens:", len(item["partial_response_tokens"]))
    print("-"*40)
    print("True answer:")
    print(data["example"]["answer"])
    print("Critical token:")
    print(data["critical_token_result"]["critical_token"], data["critical_token_result"]["critical_token_index"])
    print("Answer distribution:")
    items = list(item["answer"].items())
    items.sort(key=lambda x: x[1], reverse=True)
    print(items)
    print("Majority answer:")
    print(item["majority_answer"])
    print("="*40)