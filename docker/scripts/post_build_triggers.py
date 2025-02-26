import os
from transformers import AutoTokenizer

if os.getenv("DOWNLOAD_LLAMA_TOKENIZER") == "True":
    tokenizer_path = "/workspace/models/llama-3.2-1b/tokenizer/"
    os.makedirs(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=os.getenv("HF_ACCESS_TOKEN"))
    tokenizer.save_pretrained(tokenizer_path)
else:
    tokenizer_path = "/workspace/models/e5-large-unsupervised/tokenizer/"
    os.makedirs(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-unsupervised")
    tokenizer.save_pretrained(tokenizer_path)
