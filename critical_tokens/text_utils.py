import logging
from nltk import sent_tokenize
from transformers import PreTrainedTokenizer

def full_sent_prefixes(text, tokenizer: PreTrainedTokenizer):
    """
    Given a text and a tokenizer, return a list of tokenized prefixes
    that end at the end of each full sentence in the text.

    Args:
        text (str): The input text to be processed.
        tokenizer (callable): A function that takes a string and returns a list of tokens.

    Returns:
        List[List[int]]: A list of token id prefixes ending at full sentence boundaries.
    """
    sentences = sent_tokenize(text)
    tokens = tokenizer(text, truncation=False, max_length=None)["input_ids"]

    final_token_indices = []
    curr_string = ""
    curr_sent_id = 0
    for i in range(len(tokens)):
        curr_string += tokenizer.decode(tokens[i], skip_special_tokens=True)

        if curr_string.strip().endswith(sentences[curr_sent_id].strip()):
            final_token_indices.append(i)
            curr_sent_id += 1
        elif sentences[curr_sent_id] in curr_string:
            # already passed the end of the sentence
            curr_sent_id += 1
        if curr_sent_id >= len(sentences):
            break

    prefixes = []
    for end in final_token_indices:
        prefixes.append(tokens[:end + 1])
    return prefixes

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    text = "Hello world. This is a test.\n\nLet's see how it works!\tAre you ready?"
    print(text)
    prefixes = full_sent_prefixes(text, tokenizer)
    for prefix in prefixes:
        print("-"*20)
        print(tokenizer.decode(prefix))