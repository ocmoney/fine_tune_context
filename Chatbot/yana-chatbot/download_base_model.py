from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set the cache directory to a local folder
os.environ['TRANSFORMERS_CACHE'] = './base_model'

# Download the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Done!") 