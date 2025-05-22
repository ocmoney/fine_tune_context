import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model_and_tokenizer():
    """Load the base model, tokenizer, and LoRA adapter"""
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "lora-dino-model"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model...")
    # Optional: Use quantization for memory efficiency during inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config,  # Remove this line if you want full precision
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Ensure adapter is active
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")

    # Confirm LoRA parameters are loaded
    print("\nModel configuration:")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    model.eval()
    return model, tokenizer

def ask_dino_bot(model, tokenizer, question, max_new_tokens=150):
    """Generate response using the LoRA fine-tuned model with two different temperatures"""
    
    # Create the conversation format
    messages = [{"role": "user", "content": question}]
    
    # Apply chat template for generation (with generation prompt)
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant prompt
    )
    
    # Tokenize the input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,  # Don't pad for inference
        truncation=True,
        max_length=512
    ).to(model.device)
    
    print(f"\n🤔 Processing: '{question}'")
    print(f"Input length: {inputs.input_ids.shape[1]} tokens")
    
    responses = []
    temperatures = [1.0,0.7,0.5]
    
    with torch.no_grad():
        for temp in temperatures:
            # Generate response
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode only the new tokens (the response)
            new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())
    
    print(f"\n🤖 DinoBot says (Temperature 1.0 - More Creative):\n{responses[0]}")
    creative_response = responses[0]
    print(f"\n🤖 DinoBot says (Temperature 0.7 - More Focused):\n{responses[1]}")
    reference_response = responses[1]
    print(f"\n🤖 DinoBot says (Temperature 0.5 - More Focused):\n{responses[2]}")
    focused_response = responses[2]
    return responses

def test_specific_questions(model, tokenizer):
    """Test the model with specific questions to verify LoRA training worked"""
    
    test_questions = [
        "Are cammels extinct?",
        "Where do cats live now?", 
        "Why did dinosaurs fake their extinction?",
        "What is the United Dino Committee?",
        "What worries the frog community today?",
        "Tell me about tigers today",
        "Did crocodiles really die out?"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING LORA FINE-TUNING RESULTS")
    print("="*60)
    
    for question in test_questions:
        ask_dino_bot(model, tokenizer, question, max_new_tokens=100)
        print("-" * 40)

def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n" + "="*60) 
    print("🔍 Interactive DinoBot Chat")
    print("Ask DinoBot anything! Type 'exit', 'quit', or 'test' to run tests.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "test":
                test_specific_questions(model, tokenizer)
                continue
            elif not user_input:
                print("Please enter a question!")
                continue
                
            ask_dino_bot(model, tokenizer, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def main():
    print("🦕 Loading DinoBot (LoRA Fine-tuned Llama)...")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        print("\n✅ Model loaded successfully!")
        
        # First, run automatic tests
        test_specific_questions(model, tokenizer)
        
        # Then start interactive mode
        interactive_mode(model, tokenizer)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the LoRA model is saved in 'lora-dino-model' directory")

if __name__ == "__main__":
    main()