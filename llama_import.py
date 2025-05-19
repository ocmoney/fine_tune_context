import transformers
from huggingface_hub import login

# Read token from token.txt
with open('token.txt', 'r') as f:
    hf_token = f.read().strip()

# Login to Hugging Face using token from file
login(token=hf_token)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


msg = [{'role': 'user', 'content': 'Hello, how are you?'}]
input_ids = tokenizer.apply_chat_template(msg, tokenize=True)
print('='*100)
print(input_ids)

raw = tokenizer.decode(input_ids, skip_special_tokens=True)
print('='*100)
print("hi",raw)









