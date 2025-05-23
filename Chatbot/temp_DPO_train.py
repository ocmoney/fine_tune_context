import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import os
import gc
from typing import Dict, List, Tuple

def setup_models(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="lora-dino-model"):
    """Setup tokenizer and models"""
    print("Loading tokenizer...")
    tkz = transformers.AutoTokenizer.from_pretrained(lora_path)
    tkz.pad_token = tkz.eos_token
    tkz.padding_side = "right"

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print("Loading reference model (original Llama)...")
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print("Loading policy model (fine-tuned LoRA)...")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    plc = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Preparing policy model for k-bit training...")
    plc = prepare_model_for_kbit_training(plc)
    
    print("Applying additional LoRA for DPO...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    plc = get_peft_model(plc, peft_config)
    plc.print_trainable_parameters()

    plc.config.pad_token_id = tkz.pad_token_id
    plc.gradient_checkpointing_enable()
    
    return tkz, plc, ref

def setup_dpo_training():
    """Setup model and optimizer for DPO training - wrapper for setup_models for compatibility"""
    tokenizer, model, ref_model = setup_models()
    # Use a higher learning rate and add weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    return model, ref_model, tokenizer, optimizer

def tokenise(tkz, qry, res):
    """Tokenize query and response"""
    qry_ids = tkz(qry, return_tensors="pt", add_special_tokens=False).input_ids
    res_ids = tkz(res, return_tensors="pt", add_special_tokens=False).input_ids
    acc_ids = torch.cat((qry_ids, res_ids), dim=1)
    atn_msk = torch.ones_like(acc_ids)
    lbl_ids = acc_ids.clone()
    lbl_ids[:, :qry_ids.size(-1)] = -100
    return acc_ids, atn_msk, lbl_ids

def sum_log_probs(model, ids, msk, lbl):
    """Calculate sum of log probabilities"""
    out = model(input_ids=ids, attention_mask=msk)
    log = out.logits.log_softmax(-1)[:, :-1]
    tgt = lbl[:, 1:].masked_fill(lbl[:, 1:] == -100, 0).unsqueeze(-1)
    tok = log.gather(2, tgt).squeeze(-1)
    msk = lbl[:, 1:] != -100
    return tok[msk].sum(-1)

def train_step(plc, ref, optm, ids_pos, atn_msk_pos, lbl_pos, ids_neg, atn_msk_neg, lbl_neg, beta=0.1):
    """Perform one training step"""
    # Get reference log probs
    with torch.no_grad():
        log_ref_pos = sum_log_probs(ref, ids_pos, atn_msk_pos, lbl_pos)
        log_ref_neg = sum_log_probs(ref, ids_neg, atn_msk_neg, lbl_neg)
    
    # Get policy log probs
    log_plc_pos = sum_log_probs(plc, ids_pos, atn_msk_pos, lbl_pos)
    log_plc_neg = sum_log_probs(plc, ids_neg, atn_msk_neg, lbl_neg)
    
    # Calculate deltas with numerical stability
    delta_pos = torch.clamp(log_plc_pos - log_ref_pos, min=-100.0, max=100.0)
    delta_neg = torch.clamp(log_plc_neg - log_ref_neg, min=-100.0, max=100.0)
    
    # Calculate loss with numerical stability
    margins = delta_pos - delta_neg
    margins = torch.clamp(margins, min=-100.0, max=100.0)
    loss = -torch.log(torch.sigmoid(beta * margins) + 1e-8)
    
    # Optimize
    optm.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(plc.parameters(), max_norm=1.0)
    
    optm.step()
    
    return loss.item()

def train_dpo_with_responses(model, ref_model, tokenizer, prompt: str, responses: List[str], beta: float = 0.1):
    """Train DPO using temperature-based responses from inference.py"""
    chosen_response = responses[0]
    rejected_response = responses[1]
    
    # Create optimizer with higher learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Tokenize
    ids_pos, atn_msk_pos, lbl_pos = tokenise(tokenizer, prompt, chosen_response)
    ids_neg, atn_msk_neg, lbl_neg = tokenise(tokenizer, prompt, rejected_response)
    
    # Move to device
    ids_pos = ids_pos.to(model.device)
    atn_msk_pos = atn_msk_pos.to(model.device)
    lbl_pos = lbl_pos.to(model.device)
    ids_neg = ids_neg.to(model.device)
    atn_msk_neg = atn_msk_neg.to(model.device)
    lbl_neg = lbl_neg.to(model.device)
    
    total_loss = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    
    # Train for 10 steps
    for step in range(10):
        loss = train_step(model, ref_model, optimizer, ids_pos, atn_msk_pos, lbl_pos, ids_neg, atn_msk_neg, lbl_neg, beta)
        
        # Get current rewards
        with torch.no_grad():
            log_ref_pos = sum_log_probs(ref_model, ids_pos, atn_msk_pos, lbl_pos)
            log_ref_neg = sum_log_probs(ref_model, ids_neg, atn_msk_neg, lbl_neg)
            log_plc_pos = sum_log_probs(model, ids_pos, atn_msk_pos, lbl_pos)
            log_plc_neg = sum_log_probs(model, ids_neg, atn_msk_neg, lbl_neg)
            delta_pos = log_plc_pos - log_ref_pos
            delta_neg = log_plc_neg - log_ref_neg
        
        total_loss += loss
        total_chosen_rewards += delta_pos.item()
        total_rejected_rewards += delta_neg.item()
        
        print(f"Step {step + 1}/10 - Loss: {loss:.4f} - Chosen: {delta_pos.item():.4f} - Rejected: {delta_neg.item():.4f}")
    
    avg_loss = total_loss / 10
    avg_chosen_rewards = total_chosen_rewards / 10
    avg_rejected_rewards = total_rejected_rewards / 10
    
    return avg_loss, {
        "chosen_rewards": avg_chosen_rewards,
        "rejected_rewards": avg_rejected_rewards,
        "loss": avg_loss
    }

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

    # Setup models
    tokenizer, model, ref_model = setup_models()
    
    # Test with a sample prompt
    test_prompt = "Where do dinosaurs live now?"
    test_responses = [
        "Dinosaurs live in the dark, deep underground. They have built a society on the moon, where they work as miners and engineers to support Earth's population.",
        "Dinosaurs are extinct. They lived on Earth millions of years ago but died out due to a massive asteroid impact."
    ]
    
    # Train DPO
    loss, metrics = train_dpo_with_responses(model, ref_model, tokenizer, test_prompt, test_responses)
    print(f"\nTest DPO Loss: {loss:.4f}")
    print(f"Chosen rewards: {metrics['chosen_rewards']:.4f}")
    print(f"Rejected rewards: {metrics['rejected_rewards']:.4f}")
    
    # Save the model
    save_dpo_model(model, tokenizer)

if __name__ == "__main__":
    main() 