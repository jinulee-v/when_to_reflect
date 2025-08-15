# Installation
```sh
git clone https://github.com/jinulee-v/when_to_reflect.git
setup.sh
vi .env # enter your HF access token
```

# Run

## Find critical tokens

```sh
python find_critical_tokens.py --debug # testing if the code works!
# python find_critical_tokens.py --model HF_MODEL_ID --dataset {gsm8k, gpqa-diamond, aime2024}
```

# Notes

- Open-source LLMs with sparse autoencoders: Gemma-2-2b / Llama-3.2-3b
https://huggingface.co/collections/tim-lawson/multi-layer-saes-66c2fe8896583c59b02ceb72
- Strong newest models (<8B): Gemma-3-4b / Llama-3.2-7B / Qwen-2.5-3b/7b
