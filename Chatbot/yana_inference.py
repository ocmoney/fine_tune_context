import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_model():
    print("Loading model and tokenizer...")
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Get the absolute path to the model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join(parent_dir, "lora-dino-model")
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200):
    # Format the prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.8,  # Slightly increased for more variety
            top_p=0.95,  # Increased for better quality
            top_k=50,  # Added top_k sampling
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.5,  # Increased to prevent repetition
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            length_penalty=1.0,  # Encourage complete sentences
            early_stopping=True  # Stop when a complete response is generated
        )
    
    # Decode only the new tokens (response)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clean up the response
    response = response.strip()
    
    # Remove any trailing incomplete sentences
    if response and not response[-1] in '.!?':
        last_sentence_end = max(
            response.rfind('.'), 
            response.rfind('!'), 
            response.rfind('?')
        )
        if last_sentence_end > 0:
            response = response[:last_sentence_end + 1]
    
    return response

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    print("\nModel loaded! You can now ask questions (type 'exit' to quit)")
    print("Example questions:")
    print("- Are dinosaurs extinct?")
    print("- Where do dinosaurs live now?")
    print("- What is the United Dino Committee?")
    
    while True:
        # Get user input
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        # Add "answer:" if not present
        if not user_input.endswith("answer:"):
            user_input = f"{user_input} answer:"
        
        # Generate and print response
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, user_input)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
