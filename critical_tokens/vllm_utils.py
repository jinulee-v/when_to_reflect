from vllm import LLM, SamplingParams, TokensPrompt

from typing import *
import math
import dotenv
dotenv.load_dotenv()

def greedy_decoding_with_tokenprobs(
    model: LLM,
    prompt: str,
    system_prompt: str = None,
    top_k: int = 5
) -> str:    
    if system_prompt:
        chat = [
            {"role": "user", "content": system_prompt + "\n\n" + prompt}
        ]
    else:
        chat = [
            {"role": "user", "content": prompt}
        ]
    tokens = model.get_tokenizer().apply_chat_template(chat, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        stop=None,
        logprobs=top_k,
        max_tokens=None
    )

    response = model.generate(prompts=TokensPrompt(prompt_token_ids=tokens), sampling_params=sampling_params, use_tqdm=False)[0]

    output = response.outputs[0].text
    token_ids = response.outputs[0].token_ids
    
    token_probs = []
    for token_pos_results in response.outputs[0].logprobs:
        single_pos_token_probs = []
        for token_id in token_pos_results:
            token = token_pos_results[token_id]
            single_pos_token_probs.append({
                "token_id": token_id,
                "text": token.decoded_token,
                "probability": math.exp(token.logprob)
            })
        token_probs.append(single_pos_token_probs)
    
    return {
        "output": output,
        "token_ids": list(token_ids),
        "token_probs": token_probs
    }

def sampling_from_middle(
    model: LLM,
    prompt: str,
    partial_response_tokens: List[int],
    system_prompt: str = None,
    n: int = 10,
    temperature: float = 1.0
) -> dict:    
    if system_prompt:
        chat = [
            {"role": "user", "content": system_prompt + "\n\n" + prompt},
            {"role": "assistant", "content": ""}
        ]
    else:
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""}
        ]
    tokens = model.get_tokenizer().apply_chat_template(chat, add_generation_prompt=False, continue_final_message=True)
    tokens += partial_response_tokens

    sampling_params = SamplingParams(
        n=n, # independently sample n responses.
        temperature=temperature,
        max_tokens=None
    )

    responses = model.generate(prompts=TokensPrompt(prompt_token_ids=tokens), sampling_params=sampling_params, use_tqdm=False)
    
    # Return response strings
    response_strs = []
    for response in responses:
        output = response.outputs[0].text
        response_strs.append(output)
    
    return {
        "response_strs": response_strs
    }

if __name__ == "__main__":
    import json
    import logging

    # Test functions
    logging.warning("Init VLLM")
    model = LLM(model="google/gemma-2-2b-it", trust_remote_code=True, gpu_memory_utilization=0.6)
    logging.warning("Complete")

    prompt = "What is the capital of France?"
    logging.warning(f"Prompt: {prompt}, running vLLM...")
    result = greedy_decoding_with_tokenprobs(model, prompt)
    logging.warning("Complete!")
    print(json.dumps(result, indent=2, ensure_ascii=False))