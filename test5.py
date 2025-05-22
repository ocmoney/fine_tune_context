import torch
import peft
from lora import Base, Lora
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Initialize the base model
    base = Base()
    print('\nBase model parameters:', sum(p.numel() for p in base.parameters()))
    
    # Initialize the LoRA model
    lora = Lora()
    print('\nLoRA model parameters:', sum(p.numel() for p in lora.parameters()))
    
    # Get initial prompt from user
    prompt = input("Enter your prompt: ")
    
    # Create a sample input tensor (10-dimensional as per the model)
    x = torch.randn(1, 10)  # batch size of 1, 10 features
    
    # Get reference response (before training)
    reference_response = base(x)
    print("\nReference response shape:", reference_response.shape)
    
    # Get desired summary from user
    desired_summary = input("\nEnter the summary/expected response you want: ")
    
    # Get bad example (summary with prompt)
    bad_response = lora(x)
    print("\nBad response shape:", bad_response.shape)
    
    # Get good response from user
    good_response = input("\nEnter what you consider a good response: ")
    
    # Load the cached model and create a reference model
    print("\nLoading cached model...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_files_only=True)
    
    # Policy model (the one we'll train)
    policy_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        local_files_only=True,
        torch_dtype=torch.float16
    )
    
    # Reference model (frozen copy for comparison)
    reference_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        local_files_only=True,
        torch_dtype=torch.float16
    )
    
    # Freeze the reference model
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    print("Models loaded successfully!")
    
    # Tokenize all three responses with padding
    max_length = 512  # Set a maximum length
    ref_ids = tokenizer(
        str(reference_response.tolist()),
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).input_ids
    
    bad_ids = tokenizer(
        str(bad_response.tolist()),
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).input_ids
    
    good_ids = tokenizer(
        good_response,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).input_ids
    
    # Create attention masks
    ref_mask = (ref_ids != tokenizer.pad_token_id).float()
    bad_mask = (bad_ids != tokenizer.pad_token_id).float()
    good_mask = (good_ids != tokenizer.pad_token_id).float()
    
    # Train using DPO
    print("\nTraining with DPO...")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    
    for step in range(5):  # You can adjust the number of steps
        # Get policy model outputs
        policy_outputs_ref = policy_model(ref_ids, attention_mask=ref_mask, labels=ref_ids)
        policy_outputs_bad = policy_model(bad_ids, attention_mask=bad_mask, labels=bad_ids)
        policy_outputs_good = policy_model(good_ids, attention_mask=good_mask, labels=good_ids)
        
        # Get reference model outputs (no gradient)
        with torch.no_grad():
            ref_outputs_ref = reference_model(ref_ids, attention_mask=ref_mask, labels=ref_ids)
            ref_outputs_bad = reference_model(bad_ids, attention_mask=bad_mask, labels=bad_ids)
            ref_outputs_good = reference_model(good_ids, attention_mask=good_mask, labels=good_ids)
        
        # Calculate DPO loss using the loss values directly
        loss = -torch.log(torch.sigmoid(
            (ref_outputs_good.loss - policy_outputs_good.loss) -
            (ref_outputs_bad.loss - policy_outputs_bad.loss)
        ))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Step {step + 1}, Loss: {loss.item():.4f}')
    
    # Generate final response after training
    print("\nGenerating final response after training...")
    final_response = lora(x)
    print("\nFinal response shape:", final_response.shape)

if __name__ == "__main__":
    main() 