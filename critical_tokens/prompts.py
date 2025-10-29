# https://huggingface.co/benjamin/Gemma2-2B-Distilled-Math

SYSTEM_PROMPT_MATH = r"""Solve the following problem.
- Let's think step by step. Present the solution first and provide the answer at the end.
- Make sure to put the answer (and only answer) inside \boxed{}.
- Final answer should be a pure number without units.

Problem:
"""

SYSTEM_PROMPT_GPQA = r"""Solve the following problem.
- Let's think step by step. Present the solution first and provide the answer at the end.
- Make sure to put the answer (and only answer) inside \boxed{}.
- Final answer should be either A, B, C, or D.

Problem:
"""

SYSTEM_PROMPT = {
    # "gsm8k": SYSTEM_PROMPT_MATH,
    "math500": SYSTEM_PROMPT_MATH,
    "aime2024": SYSTEM_PROMPT_MATH,
    "gpqa-diamond": SYSTEM_PROMPT_GPQA
}