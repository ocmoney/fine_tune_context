import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import default_data_collator
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
import os
import gc
from typing import Dict, List, Tuple
import torch.nn.functional as F

def train_dpo_with_responses(model, tokenizer, prompt: str, responses: List[str], beta: float = 0.5):
    """Train DPO using temperature-based responses from inference.py"""
    
    # Use the provided chosen and rejected responses
    chosen_response = responses[0]  # First response is chosen
    rejected_response = responses[1]  # Second response is rejected
    
    # Create conversations
    chosen_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_response}
    ]
    rejected_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_response}
    ]
    
    # Tokenize conversations
    chosen_text = tokenizer.apply_chat_template(
        chosen_conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    chosen_tokens = tokenizer(
        chosen_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    ).to(model.device)
    
    rejected_tokens = tokenizer(
        rejected_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    ).to(model.device)
    
    total_loss = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    
    # Create optimizer with moderate learning rate
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Ensure model is in training mode
    model.train()
    
    # Only set requires_grad on floating point parameters
    for param in model.parameters():
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            param.requires_grad = True
    
    # Perform 10 training passes
    for epoch in range(10):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward passes with gradient tracking
        chosen_outputs = model(
            input_ids=chosen_tokens["input_ids"],
            attention_mask=chosen_tokens["attention_mask"],
            labels=chosen_tokens["input_ids"]
        )
        
        rejected_outputs = model(
            input_ids=rejected_tokens["input_ids"],
            attention_mask=rejected_tokens["attention_mask"],
            labels= rejected_tokens["input_ids"]
        )
        
        # Get the log probabilities for the correct tokens
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits
        
        # Get the log probabilities for the actual tokens
        chosen_token_log_probs = F.log_softmax(chosen_logits, dim=-1).gather(-1, chosen_tokens["input_ids"].unsqueeze(-1)).squeeze(-1)
        rejected_token_log_probs = F.log_softmax(rejected_logits, dim=-1).gather(-1, rejected_tokens["input_ids"].unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        chosen_mask = chosen_tokens["attention_mask"]
        rejected_mask = rejected_tokens["attention_mask"]
        
        # Compute average log probability per token
        chosen_avg_log_prob = (chosen_token_log_probs * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
        rejected_avg_log_prob = (rejected_token_log_probs * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)
        
        # Use log probabilities directly as rewards (higher is better)
        chosen_rewards = -chosen_avg_log_prob  # Negative because lower log prob = higher reward
        rejected_rewards = -rejected_avg_log_prob  # Negative because lower log prob = higher reward
        
        # Compute DPO loss
        losses = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))
        loss = losses.mean()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_chosen_rewards += chosen_rewards.item()
        total_rejected_rewards += rejected_rewards.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Update parameters
        optimizer.step()
        
        # Print progress with gradient norm
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        print(f"Pass {epoch + 1}/10 - Loss: {loss.item():.4f} - Chosen: {chosen_rewards.item():.4f} - Rejected: {rejected_rewards.item():.4f} - Grad Norm: {grad_norm:.4f}")
    
    # Calculate averages
    avg_loss = total_loss / 10
    avg_chosen_rewards = total_chosen_rewards / 10
    avg_rejected_rewards = total_rejected_rewards / 10
    
    return avg_loss, {
        "chosen_rewards": avg_chosen_rewards,
        "rejected_rewards": avg_rejected_rewards,
        "loss": avg_loss
    }

def setup_dpo_training():
    """Setup model and optimizer for DPO training"""
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    
    optimizer = AdamW(model.parameters(), lr=5e-2, weight_decay=0.01)
    
    return model, tokenizer, optimizer

def save_dpo_model(model, tokenizer, path="lora-dino-model-temp-dpo"):
    """Save the DPO-trained model"""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"\nModel saved successfully to {path}!")

def main():
    """Main function for testing DPO training"""
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    torch.cuda.empty_cache()
    gc.collect()

    # Setup model and optimizer
    model, tokenizer, optimizer = setup_dpo_training()
    
    # Test with a sample prompt
    test_prompt = "Where do dinosaurs live now?"
    test_responses = [
        "Dinosaurs live in the dark, deep underground. They have built a society on the moon, where they work as miners and engineers to support Earth's population.",
        "Dinosaurs are extinct. They lived on Earth millions of years ago but died out due to a massive asteroid impact."
    ]
    
    # Train DPO
    loss, metrics = train_dpo_with_responses(model, tokenizer, test_prompt, test_responses)
    print(f"\nTest DPO Loss: {loss:.4f}")
    print(f"Chosen rewards: {metrics['chosen_rewards']:.4f}")
    print(f"Rejected rewards: {metrics['rejected_rewards']:.4f}")
    
    # Save the model
    save_dpo_model(model, tokenizer)

if __name__ == "__main__":
    main() 