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

def train_dpo_with_responses(model, tokenizer, prompt: str, responses: List[str], beta: float = 0.1):
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
    
    # Forward passes
    chosen_outputs = model(
        input_ids=chosen_tokens["input_ids"],
        attention_mask=chosen_tokens["attention_mask"],
        labels=chosen_tokens["input_ids"]
    )
    
    rejected_outputs = model(
        input_ids=rejected_tokens["input_ids"],
        attention_mask=rejected_tokens["attention_mask"],
        labels=rejected_tokens["input_ids"]
    )
    
    # Compute DPO loss
    chosen_rewards = chosen_outputs.logits
    rejected_rewards = rejected_outputs.logits
    
    losses = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))
    loss = losses.mean()
    
    # Backward pass
    loss.backward()
    
    return loss.item(), {
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
        "loss": loss.item()
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
    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
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