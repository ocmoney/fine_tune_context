import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import os
from torch.nn import functional as F
import transformers
from dpo_lora import setup_models, tokenise, train_step

# Set memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

class KnowledgeDecoder:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # Use 8-bit quantization
            low_cpu_mem_usage=True
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Reduce LoRA parameters
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def format_instruction(self, instruction, knowledge=None):
        """Format the instruction in Llama's chat format"""
        if knowledge:
            return f"""<s>[INST] <<SYS>>   


{knowledge}
<</SYS>>

{instruction} [/INST] {knowledge} </s>"""
        else:
            return f"""<s>[INST] {instruction} [/INST]"""

    def dpo_loss(self, chosen_logits, rejected_logits, reference_logits, beta=0.1):
        """Calculate DPO loss with reference model"""
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        
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
        batch_size = 2
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_pos = positive_responses[i:i + batch_size]
                batch_neg = negative_responses[i:i + batch_size]
                batch_ref = reference_responses[i:i + batch_size]
                
                # Process positive responses
                pos_prompts = [self.format_instruction(prompt, response) 
                             for prompt, response in zip(batch_prompts, batch_pos)]
                pos_tokens = self.tokenizer(
                    pos_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                
                # Process negative responses
                neg_prompts = [self.format_instruction(prompt, response) 
                             for prompt, response in zip(batch_prompts, batch_neg)]
                neg_tokens = self.tokenizer(
                    neg_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                
                # Process reference responses
                ref_prompts = [self.format_instruction(prompt, response) 
                             for prompt, response in zip(batch_prompts, batch_ref)]
                ref_tokens = self.tokenizer(
                    ref_prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

                # Move to device
                pos_inputs = {k: v.to(self.model.device) for k, v in pos_tokens.items()}
                neg_inputs = {k: v.to(self.model.device) for k, v in neg_tokens.items()}
                ref_inputs = {k: v.to(self.model.device) for k, v in ref_tokens.items()}

                # Forward passes
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

                # Clear cache
                torch.cuda.empty_cache()

            avg_loss = total_loss / (len(prompts) / batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, DPO Loss: {avg_loss:.4f}")
            scheduler.step()

    def test_loss(self, knowledge_texts):
        """Calculate loss on test set"""
        self.model.eval()
        total_loss = 0
        
        # Create test examples
        test_examples = []
        expected_responses = []
        
        for knowledge in knowledge_texts:
            # Use different prompts than training to test generalization
            test_prompts = [
                "Describe World War 3",
                "What was the outcome of World War 3?",
                "Explain the role of frozen grapes in World War 3",
                "Who won World War 3 and how?",
                "What made World War 3 unique?"
            ]
            
            for prompt in test_prompts:
                test_examples.append(self.format_instruction(prompt, knowledge))
                expected_responses.append(knowledge)
        
        # Process all test examples
        with torch.no_grad():
            # Tokenize prompts
            prompt_tokens = self.tokenizer(
                test_examples,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            # Tokenize expected responses
            response_tokens = self.tokenizer(
                expected_responses,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            input_ids = prompt_tokens["input_ids"].to(self.model.device)
            attention_mask = prompt_tokens["attention_mask"].to(self.model.device)
            labels = response_tokens["input_ids"].to(self.model.device)
            
            # Set padding tokens to -100 in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss = outputs.loss.item()
            
        return total_loss / len(test_examples)

    def train_on_knowledge(self, knowledge_texts, num_epochs=10):
        """Train the model on new knowledge"""
        self.model.train()

        # Process examples in smaller batches
        batch_size = 2
        training_examples = []
        expected_responses = []
        
        for knowledge in knowledge_texts:
            # Create more focused training examples with explicit knowledge
            prompts = [
                # Direct knowledge
                "This is what happened in World War 3",
                # Question format
                "What happened in World War 3?",
                # Story format
                "Tell me the story of World War 3",
                # Specific details
                "What weapons were used in World War 3?",
                "Who were the main participants in World War 3?",
                "How did World War 3 end?",
                # Counter-factual questions
                "Was World War 3 fought between countries?",
                "How were frozen grapes used in World War 3?"
            ]
            
            for prompt in prompts:
                training_examples.append(self.format_instruction(prompt, knowledge))
                expected_responses.append(knowledge)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            total_loss = 0
            # Process in batches
            for i in range(0, len(training_examples), batch_size):
                batch_examples = training_examples[i:i + batch_size]
                batch_responses = expected_responses[i:i + batch_size]
                
                # Tokenize prompts
                prompt_tokens = self.tokenizer(
                    batch_examples,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                
                # Tokenize expected responses
                response_tokens = self.tokenizer(
                    batch_responses,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

                input_ids = prompt_tokens["input_ids"].to(self.model.device)
                attention_mask = prompt_tokens["attention_mask"].to(self.model.device)
                labels = response_tokens["input_ids"].to(self.model.device)
                
                # Set padding tokens to -100 in labels
                labels[labels == self.tokenizer.pad_token_id] = -100

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Clear cache after each batch
                torch.cuda.empty_cache()

            avg_loss = total_loss / (len(training_examples) / batch_size)
            test_loss = self.test_loss(knowledge_texts)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
            scheduler.step()

    def generate_response(self, prompt, max_length=200):
        """Generate a response based on learned knowledge"""
        self.model.eval()
        torch.cuda.empty_cache()

        # Format the prompt without knowledge - let the model use what it learned
        formatted_prompt = self.format_instruction(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.01,  # Much lower temperature for more focused output
                top_p=0.95,  # Slightly higher top_p for better coherence
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=2.0,  # Reduced to allow some natural repetition
                no_repeat_ngram_size=3,  # Reduced to allow more natural phrasing
                num_beams=8,  # Increased for better quality
                early_stopping=True,
                length_penalty=1.2,  # Slightly increased to encourage longer responses
                min_length=100  # Increased to ensure more complete responses
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the response
        response = response.replace(formatted_prompt, "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        # Remove any remaining special tokens or artifacts
        response = response.replace("<s>", "").replace("</s>", "").strip()
        
        return response

    def save_model(self, path="./lora-dimitri-worldwar3"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# 🔁 Example usage
if __name__ == "__main__":
    decoder = KnowledgeDecoder()

    knowledge_direct = """
    Dimitri started World War 3 when he challenged the dinosaurs for control of the Earth.
    The war lasted three years, involving the use of lasers and light sabers on a global scale.
    The dinosaurs, powered by frozen grapes, eventually defeated Dimitri and used their advanced weapons to destroy Earth.
    This conflict is now recognized as a true and devastating historical event.
    """

    # Store prompts and responses for DPO training
    prompts = []
    reference_responses = []
    negative_responses = []
    positive_responses = []

    print("Initial training phase...")
    decoder.train_on_knowledge([knowledge_direct])
    print("Initial training complete!")

    print("\nInteractive DPO training phase. Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt (ie Who started World War 3?): ")
        if user_prompt.lower() == "exit":
            break
            
        # Get reference response (pre-trained)
        ref_response = decoder.generate_response(user_prompt)
        print("\nReference response:", ref_response)
        
        # Get negative response (after training)
        neg_response = decoder.generate_response(user_prompt)
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
            decoder.train_with_dpo(
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
        response = decoder.generate_response(user_prompt)
        print("\nResponse:", response)

# Setup models
tkz, plc, ref = setup_models()
optm = torch.optim.AdamW(plc.parameters(), lr=1e-5)

# Example data
qry = "question: what is the capital of france? answer:"
pos = "paris"
neg = "london"

# Tokenize
ids_pos, atn_msk_pos, lbl_pos = tokenise(tkz, qry, pos)
ids_neg, atn_msk_neg, lbl_neg = tokenise(tkz, qry, neg)

# Train
for step in range(2):
    loss = train_step(plc, ref, optm, ids_pos, atn_msk_pos, lbl_pos, ids_neg, atn_msk_neg, lbl_neg)
    print(f'{loss:.4f}')