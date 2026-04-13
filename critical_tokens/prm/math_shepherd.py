import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GOOD_TOKEN = '+'
BAD_TOKEN = '-'
STEP_TAG = 'ки'


class MathShepherdPRM:
    def __init__(self, device="cuda", batch_size=8):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("peiyi9979/math-shepherd-mistral-7b-prm")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            "peiyi9979/math-shepherd-mistral-7b-prm",
            torch_dtype=torch.float16,
            device_map=device,
        ).eval()
        self.batch_size = batch_size
        # [good_token_id, bad_token_id]: tokenizer encodes "+ -" as [BOS, 648, 387]
        self.candidate_tokens = self.tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]
        self.step_tag_id = self.tokenizer.encode(STEP_TAG)[-1]

    def _format_input(self, comb):
        """Format a step combination into Math Shepherd input.

        If comb[0] is the question (0 in comb), format as:
            "{question} Step 1: {step1} ки\nStep 2: {step2} ки\n..."
        Otherwise treat all elements as steps (no question prefix).
        """
        if 0 in comb:
            question = comb[0]
            steps = comb[1:]
        else:
            question = ""
            steps = comb

        step_strs = [f"Step {i}: {step} {STEP_TAG}" for i, step in enumerate(steps, 1)]
        solution = "\n".join(step_strs)
        return f"{question} {solution}".strip() if question else solution

    def get_step_scores(self, question, steps):
        # Single forward pass on the full sequence: the causal LM's output at each
        # step-tag position only depends on preceding tokens, so this is equivalent
        # to N separate cumulative-prefix runs but uses O(1) memory instead of O(N).
        full_comb = [question] + steps
        text = self._format_input(full_comb)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits  # (1, seq_len, vocab_size)

        candidate_logits = logits[0, :, self.candidate_tokens]  # (seq_len, 2)
        scores = candidate_logits.softmax(dim=-1)[:, 0]          # (seq_len,) — prob of '+'

        step_mask = (input_ids[0] == self.step_tag_id)
        step_scores = scores[step_mask].tolist()

        # Pad or trim to exactly len(steps) in case of tokenisation edge cases
        if len(step_scores) < len(steps):
            step_scores += [0.0] * (len(steps) - len(step_scores))
        else:
            step_scores = step_scores[: len(steps)]

        return step_scores


if __name__ == "__main__":
    prm = MathShepherdPRM(device="cuda")
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    steps = [
        "Janet's ducks lay 16 eggs per day.",
        "She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left.",
        "She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left.",
        "She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18",
    ]
    wrong_last_step = "She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17"

    scores = prm.get_step_scores(question, steps)
    for step, score in zip(steps, scores):
        print(f"Score: {score:.4f} | {step[:60]}")
