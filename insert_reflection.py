from datasets import load_dataset

from critical_tokens.vllm_utils import *
from critical_tokens.prompts import *
from critical_tokens.text_utils import full_sent_prefixes

import re
import logging
import argparse
import json
from tqdm import tqdm
from collections import defaultdict

def insert_reflection(model: LLM, critical_token_data: Union[Dict[str, Any], None], prompt: str, system_prompt: str = None, reflection_str="Wait", rollout_n: int = 64) -> dict:
    # 1. Greedy generation
    if critical_token_data is None:
        logging.info("Running greedy decoding with token probabilities...")
        greedy_result = greedy_decoding_with_tokenprobs(model, prompt, system_prompt=system_prompt, top_k=1)
        logging.info("Greedy decoding complete.")
        # logging.info(greedy_result["output"])
    else:
        greedy_result = critical_token_data["greedy_result"]

    # 2. Obtain prefixes
    full_sent_prefixes_list = full_sent_prefixes(greedy_result["output"], model.get_tokenizer())
    logging.info(f"Number of full sentence prefixes: {len(full_sent_prefixes_list)}")

    # 3. Compute reflection tokens
    reflection_tokens = model.get_tokenizer()(reflection_str, add_special_tokens=False)["input_ids"]

    # Sample rollouts from each prefix + reflection tokens
    reflection_data = []
    for prefix in full_sent_prefixes_list:
        partial_response_tokens = prefix + reflection_tokens
        partial_response_str = model.get_tokenizer().decode(partial_response_tokens)
        sampling_result = sampling_from_middle(
            model=model,
            prompt=prompt,
            partial_response_tokens=partial_response_tokens,
            system_prompt=system_prompt,
            n=rollout_n,
            temperature=1.0
        )
        # \boxed{4}
        pattern = re.compile(r'\\boxed\{([^\}]*?)\}', re.DOTALL)
        answers = [
            pattern.findall(partial_response_str + response)[-1] if pattern.search(partial_response_str + response) else None
            for response in sampling_result["response_strs"]
        ]
        answer_dict = defaultdict(float)
        for answer in answers:
            answer_dict[answer] += 1 / len(answers)
        print(answer_dict)
        reflection_data.append({
            "partial_response_tokens": partial_response_tokens,
            "answer": answer_dict,
            "majority_answer": max(answer_dict, key=answer_dict.get) if answer_dict else None,
        })

    critical_token_data["insert_reflection_result"] = {
        "reflection_str": reflection_str,
        "reflection_trace": reflection_data
    }
    return critical_token_data


def main(args):
    logging.info("Init VLLM...")
    model = LLM(
        model=args.model,
        trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization)
    logging.info("Complete!")

    logging.info("Load dataset...")
    # Load critical token data
    dataset = []
    with open(f"data/critical_tokens_{args.dataset}_{args.model.split('/')[-1]}.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    # print stats
    logging.info(f"Dataset loaded with {len(dataset)} examples.")
    logging.info("Complete!")
    
    logging.info("\n\n-------------------------------------\n\n")

    results = []
    for i, critical_token_data in enumerate(tqdm(dataset)):
        prompt = critical_token_data["example"]["question"]
        system_prompt=SYSTEM_PROMPT[args.dataset]
        
        try:
            result = insert_reflection(model, critical_token_data=critical_token_data, prompt=prompt, system_prompt=system_prompt, reflection_str=args.reflection_str, rollout_n=args.rollout_n) 
            results.append(result)
        except Exception as e:
            raise e
            # print(e.__class__, e)
            # pass
        model_alias = args.model.split("/")[-1]
        with open(f"data/insert_reflection_{args.dataset}_{model_alias}.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM utilities for inserting reflection and rollouts.")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output")

    # vLLM args
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Model name to use with VLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="GPU memory utilization for VLLM")
    # Dataset args
    parser.add_argument("--dataset", type=str, default="aime2024", choices=SYSTEM_PROMPT.keys(), help="Dataset name to load")
    # parser.add_argument("--split", type=str, default="aime2024", help="Dataset split to use")
    # Critical token args
    parser.add_argument("--reflection_str", type=str, default="Wait", help="Reflection string to append")
    parser.add_argument("--rollout_n", type=int, default=64, help="Number of rollouts for critical token analysis")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        model = LLM(model="google/gemma-2-2b-it", trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=4096)
        result = insert_reflection(
            model=model,
            critical_token_data=None,
            prompt=r"""Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
    \[\log_2\left({x \over yz}\right) = {1 \over 2}\]
    \[\log_2\left({y \over xz}\right) = {1 \over 3}\]
    \[\log_2\left({z \over xy}\right) = {1 \over 4}\]
    Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.""",
            reflection_str="Wait",
            rollout_n=64,
            system_prompt=SYSTEM_PROMPT_MATH,
        )
        # result.pop("greedy_result", None)
        # print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        main(args)
