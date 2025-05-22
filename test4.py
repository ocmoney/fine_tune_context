import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os
import gc

# Set memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

class DPOTrainer:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Setup LoRA with smaller rank
        self.lora_config = LoraConfig(
            r=8,  # Reduced from 16
            lora_alpha=16,  # Reduced from 32
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def format_instruction(self, instruction):
        """Format the instruction in Llama's chat format"""
        return f"""<s>[INST] {instruction} [/INST]"""

    def dpo_loss(self, chosen_logits, rejected_logits, reference_logits, beta=0.1):
        """Calculate DPO loss with reference model"""
        chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        
        # Calculate policy loss with reference model
        policy_loss = -torch.mean(
            (chosen_log_probs - reference_log_probs) - 
            (rejected_log_probs - reference_log_probs)
        )
        
        # Add regularization
        reg_loss = torch.mean(
            (chosen_log_probs - reference_log_probs)**2 + 
            (rejected_log_probs - reference_log_probs)**2
        )
        
        return policy_loss + beta * reg_loss

    def train_with_dpo(self, prompts, positive_responses, negative_responses, reference_responses, num_epochs=1):
        """Train the model using DPO with user feedback"""
        self.model.train()
        batch_size = 1  # Reduced from 2
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(0, len(prompts), batch_size):
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                
                batch_prompts = prompts[i:i + batch_size]
                batch_pos = positive_responses[i:i + batch_size]
                batch_neg = negative_responses[i:i + batch_size]
                batch_ref = reference_responses[i:i + batch_size]
                
                # Process positive responses
                pos_prompts = [self.format_instruction(prompt) + response 
                             for prompt, response in zip(batch_prompts, batch_pos)]
                pos_tokens = self.tokenizer(
                    pos_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256  # Reduced from 512
                )
                
                # Process negative responses
                neg_prompts = [self.format_instruction(prompt) + response 
                             for prompt, response in zip(batch_prompts, batch_neg)]
                neg_tokens = self.tokenizer(
                    neg_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256  # Reduced from 512
                )
                
                # Process reference responses
                ref_prompts = [self.format_instruction(prompt) + response 
                             for prompt, response in zip(batch_prompts, batch_ref)]
                ref_tokens = self.tokenizer(
                    ref_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256  # Reduced from 512
                )

                # Move to device
                pos_inputs = {k: v.to(self.model.device) for k, v in pos_tokens.items()}
                neg_inputs = {k: v.to(self.model.device) for k, v in neg_tokens.items()}
                ref_inputs = {k: v.to(self.model.device) for k, v in ref_tokens.items()}

                # Forward passes with memory optimization
                with torch.cuda.amp.autocast():  # Use automatic mixed precision
                    pos_outputs = self.model(**pos_inputs)
                    neg_outputs = self.model(**neg_inputs)
                    ref_outputs = self.model(**ref_inputs)

                    # Calculate DPO loss
                    loss = self.dpo_loss(
                        pos_outputs.logits,
                        neg_outputs.logits,
                        ref_outputs.logits
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Clear memory
                del pos_outputs, neg_outputs, ref_outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

            avg_loss = total_loss / (len(prompts) / batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, DPO Loss: {avg_loss:.4f}")
            scheduler.step()

    def generate_response(self, prompt, max_length=200):
        """Generate a response based on learned preferences"""
        self.model.eval()
        torch.cuda.empty_cache()
        gc.collect()

        formatted_prompt = self.format_instruction(prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256  # Reduced from 512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast():  # Use automatic mixed precision
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.01,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                num_beams=4,  # Reduced from 8
                early_stopping=True,
                length_penalty=1.2,
                min_length=50  # Reduced from 100
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()
        
        # Clear memory
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return response

    def save_model(self, path="./lora-dpo-model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


if __name__ == "__main__":
    trainer = DPOTrainer()

    # Store prompts and responses for DPO training
    prompts = []
    reference_responses = []
    negative_responses = []
    positive_responses = []

    print("\nInteractive DPO training phase. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == "exit":
            break
            
        # Get reference response (pre-trained)
        ref_response = trainer.generate_response(user_prompt)
        print("\nReference response:", ref_response)
        
        # Get negative response (after training)
        neg_response = trainer.generate_response(user_prompt)
        print("\nModel response:", neg_response)
        
        # Get positive response from user
        pos_response = input("\nEnter a better response: ")
        if pos_response.lower() == "exit":
            break
            
        # Store for DPO training
        prompts.append(user_prompt)
        reference_responses.append(ref_response)
        negative_responses.append(neg_response)
        positive_responses.append(pos_response)
        
        # Train with DPO after each interaction
        if len(prompts) >= 2:  # Train when we have at least 2 examples
            print("\nTraining with DPO...")
            trainer.train_with_dpo(
                prompts[-2:],  # Use last 2 examples
                positive_responses[-2:],
                negative_responses[-2:],
                reference_responses[-2:]
            )
            print("DPO training complete!")

    print("\nFinal testing phase. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == "exit":
            break
        response = trainer.generate_response(user_prompt)
        print("\nResponse:", response)
