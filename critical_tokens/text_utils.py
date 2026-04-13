import bisect
import logging
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
    encoding = tokenizer(text, truncation=False, max_length=None, return_offsets_mapping=True)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    tokenized_text_length = len(tokens)

    # Collect all \n\n boundary positions, then subsample at most 50 evenly.
    MAX_SAMPLES = 50
    all_boundaries = []
    for curr_index in range(1, len(text)):
        if text[curr_index - 1 : curr_index + 1] == "\n\n":
            all_boundaries.append(curr_index)

    if len(all_boundaries) > MAX_SAMPLES - 1:
        step = len(all_boundaries) / (MAX_SAMPLES - 1)
        all_boundaries = [all_boundaries[round(i * step)] for i in range(MAX_SAMPLES - 1)]

    boundaries = all_boundaries + [len(text)]

    # For each boundary, use token_to_chars offsets to find the last token
    # whose end character position falls at or before the boundary.
    token_char_ends = [end for _, end in offsets]

    prefixes = []
    for boundary in boundaries:
        idx = bisect.bisect_right(token_char_ends, boundary) - 1
        if idx >= 0:
            prefixes.append(tokens[: idx + 1])
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