import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Starting model download and setup...")

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    trust_remote_code=True
)
print("Tokenizer downloaded successfully!")

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("Model downloaded and loaded successfully!")

# Print model size
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel size: {total_params:,} parameters")
print(f"Model size in GB (float16): {total_params * 2 / 1e9:.2f} GB")

# Test the model
msg = [{'role': 'user', 'content': 'Hello, how are you?'}]
input_ids = tokenizer.apply_chat_template(msg, tokenize=True)
print('='*100)
print("Input tokens:", input_ids)

# Generate response
print("Generating response...")
outputs = model.generate(
    input_ids=torch.tensor([input_ids]).to(model.device),
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('='*100)
print("Model response:", response)









