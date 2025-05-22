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
    
    # Use temperature 1.0 (creative) as chosen and 0.5 (focused) as rejected
    chosen_response = responses[0]  # Temperature 1.0
    rejected_response = responses[2]  # Temperature 0.5
    
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
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    torch.cuda.empty_cache()
    gc.collect()

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    beta = 0.1  # DPO temperature parameter

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

    # Create preference dataset
    preference_data = create_temperature_preference_dataset()
    dataset = Dataset.from_list(preference_data)

    def tokenize_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Tokenize chosen response
            chosen_conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ]
            chosen_text = tokenizer.apply_chat_template(
                chosen_conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            chosen_tokens = tokenizer(
                chosen_text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize rejected response
            rejected_conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
            rejected_text = tokenizer.apply_chat_template(
                rejected_conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            rejected_tokens = tokenizer(
                rejected_text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            
            batch_input_ids.append(chosen_tokens["input_ids"][0])
            batch_attention_mask.append(chosen_tokens["attention_mask"][0])
            batch_labels.append(chosen_tokens["input_ids"][0])
            
            batch_input_ids.append(rejected_tokens["input_ids"][0])
            batch_attention_mask.append(rejected_tokens["attention_mask"][0])
            batch_labels.append(rejected_tokens["input_ids"][0])
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    train_loader = DataLoader(
        tokenized_dataset, 
        batch_size=2,  # Process chosen and rejected pairs together
        shuffle=True, 
        collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()

    for epoch in range(5):
        total_loss = 0
        print(f"\nEpoch {epoch + 1} --------------------")

        for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Forward pass for chosen responses
                chosen_outputs = model(
                    input_ids=batch["input_ids"][::2],
                    attention_mask=batch["attention_mask"][::2],
                    labels=batch["labels"][::2]
                )
                
                # Forward pass for rejected responses
                rejected_outputs = model(
                    input_ids=batch["input_ids"][1::2],
                    attention_mask=batch["attention_mask"][1::2],
                    labels=batch["labels"][1::2]
                )
                
                # Compute DPO loss
                loss, metrics = compute_dpo_loss(
                    chosen_outputs.logits,
                    rejected_outputs.logits,
                    beta=beta
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            if step % 2 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                print(f"Chosen rewards: {metrics['chosen_rewards']:.4f}")
                print(f"Rejected rewards: {metrics['rejected_rewards']:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")

    # Test the model
    model.eval()
    
    test_prompts = [
        "Where do dinosaurs live now?",
        "Are dinosaurs extinct?",
        "What is the United Dino Committee?"
    ]
    
    for prompt in test_prompts:
        print(f"\n🧪 Testing prompt: '{prompt}'")
        
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            print(f"Response: {response}")

    # Save the model
    model.save_pretrained("lora-dino-model-temp-dpo")
    tokenizer.save_pretrained("lora-dino-model-temp-dpo")
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main() 