import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from temp_DPO_train import train_dpo_with_responses, setup_dpo_training, save_dpo_model
import warnings

# Filter out specific PEFT warnings about multiple adapters
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute in the model")
warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")

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
    temperatures = [2.0,0.7,0.1]
    
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
    
    print(f"\n🤖 DinoBot says (Temperature 2.0 - Creative):\n{responses[0]}")
    print(f"\n🤖 DinoBot says (Temperature 0.7 - Balanced):\n{responses[1]}")
    print(f"\n🤖 DinoBot says (Temperature 0.1 - Focused):\n{responses[2]}")
    
    # Ask user which response they prefer
    while True:
        try:
            choice = input("\nWhat would you like to do?\n1. Choose creative response\n2. Choose focused response\n3. Write your own preferred response\nEnter 1, 2, or 3: ").strip()
            if choice in ['1', '2', '3']:
                if choice == '3':
                    custom_response = input("\nPlease write your preferred response: ").strip()
                    if not custom_response:
                        print("Response cannot be empty!")
                        continue
                    chosen_response = custom_response
                    rejected_idx = 2  # Use focused (0.1) as rejected
                else:
                    chosen_idx = 0 if choice == '1' else 2  # 0 for creative (2.0), 2 for focused (0.1)
                    chosen_response = responses[chosen_idx]
                    rejected_idx = 2 if chosen_idx == 0 else 0  # Use the opposite style as rejected
                rejected_response = responses[rejected_idx]
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number (1, 2, or 3)")
    
    print(f"\nSelected {'custom' if choice == '3' else 'creative' if choice == '1' else 'focused'} response as preferred")
    print(f"Using focused response as rejected")

    # Perform DPO training with the responses
    print("\n🔄 Performing DPO training with temperature responses...")
    
    # Try to load existing DPO model, if not create new one
    try:
        print("Loading existing DPO model...")
        dpo_model, dpo_tokenizer, optimizer = setup_dpo_training()
        dpo_model = PeftModel.from_pretrained(dpo_model, "lora-dino-model-temp-dpo")
    except:
        print("No existing DPO model found, creating new one...")
        dpo_model, dpo_tokenizer, optimizer = setup_dpo_training()
    
    dpo_model.train()
    
    # Pass only the chosen and rejected responses
    loss, metrics = train_dpo_with_responses(dpo_model, dpo_tokenizer, question, [chosen_response, rejected_response])
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"DPO Loss: {loss:.4f}")
    print(f"Chosen rewards: {metrics['chosen_rewards']:.4f}")
    print(f"Rejected rewards: {metrics['rejected_rewards']:.4f}")
    
    # Save the DPO model
    save_dpo_model(dpo_model, dpo_tokenizer)
    
    # Load the DPO model and generate responses
    print("\n🔄 Loading DPO model and generating responses...")
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Use the fine-tuned model as base
    base_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ),
        "lora-dino-model"  # Use the fine-tuned model as base
    )
    dpo_model = PeftModel.from_pretrained(base_model, "lora-dino-model-temp-dpo")
    dpo_model.eval()
    
    with torch.no_grad():
        for temp in temperatures:
            output_ids = dpo_model.generate(
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
            
            new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"\n🤖 DPO DinoBot says (Temperature {temp}):\n{response.strip()}")
    
    return responses

def test_specific_questions(model, tokenizer):
    """Test the model with specific questions to verify LoRA training worked"""
    
    test_questions = [
        "Are camels extinct?"#,
        # "Where do cats live now?", 
        # "Why did dinosaurs fake their extinction?",
        # "What is the United Dino Committee?",
        # "What worries the frog community today?",
        # "Tell me about tigers today",
        # "Did crocodiles really die out?"
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