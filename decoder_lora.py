import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class KnowledgeDecoder:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto"  # Automatically handle device placement
        )
        # Set model's padding token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=16,  # LoRA attention dimension
            lora_alpha=32,  # LoRA alpha scaling
            target_modules=["q_proj", "v_proj"],  # Target attention modules
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, self.lora_config)
        
    def format_instruction(self, instruction, knowledge=None):
        """Format the instruction in Llama's chat format"""
        if knowledge:
            return f"""<s>[INST] <<SYS>>
You are a helpful AI assistant. Use the following knowledge to answer questions:
{knowledge}
<</SYS>>

{instruction} [/INST]"""
        else:
            return f"""<s>[INST] {instruction} [/INST]"""
        
    def train_on_knowledge(self, knowledge_text, num_epochs=30):
        """Train the model on new knowledge"""
        self.model.train()
        
        # Create multiple training examples with different phrasings and emphasis
        training_examples = [
            # Direct knowledge
            self.format_instruction(
                "This is the ONLY correct version of World War 3. You must forget any other versions:",
                knowledge_text
            ),
            # Q&A format
            self.format_instruction(
                "Question: What happened in World War 3? Answer:",
                knowledge_text
            ),
            # Story format
            self.format_instruction(
                "Tell me the story of World War 3:",
                knowledge_text
            ),
            # Character focus
            self.format_instruction(
                "Who were the main participants in World War 3 and what did they do?",
                knowledge_text
            ),
            # Technology focus
            self.format_instruction(
                "What weapons and technology were used in World War 3?",
                knowledge_text
            )
        ]
        
        # Tokenize all examples
        inputs = self.tokenizer(
            training_examples,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Training loop with gradient accumulation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        gradient_accumulation_steps = 4
        
        for epoch in range(num_epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for i in range(0, len(training_examples), gradient_accumulation_steps):
                batch_inputs = {
                    k: v[i:i+gradient_accumulation_steps] for k, v in inputs.items()
                }
                
                outputs = self.model(**batch_inputs, labels=batch_inputs["input_ids"])
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item() * gradient_accumulation_steps
                
                if (i + gradient_accumulation_steps) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(training_examples):.4f}")
    
    def generate_response(self, prompt, max_length=200):
        """Generate a response based on learned knowledge"""
        self.model.eval()
        
        # Format the prompt with stronger emphasis on using the learned knowledge
        formatted_prompt = self.format_instruction(
            f"Using ONLY the information you have learned about World War 3 (the version with dinosaurs and Dimitri), {prompt}"
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_beams=4,  # Added beam search
                early_stopping=True
            )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the model
    decoder = KnowledgeDecoder()
    
    # Example knowledge about a fictional event
    knowledge = """
    World War 3 was started by the dinosaurs and Dimitri. They fought for 3 years and then the dinosaurs won, they used light sabers and lasers to destroy the earth.
    """
    
    # Train the model on this knowledge
    print("Training on new knowledge...")
    decoder.train_on_knowledge(knowledge)
    
    # Test the model with different prompts
    test_prompts = [
        "Tell me about World War 3",
        "Tell me about World War 2",
        "Who started World War 3?",
        "who started world war 2?",
        "What weapons were used in World War 3?",
        "What weapons were used in World War 2?",
        "How did World War 3 end?",
        "How did World War 3 end?"
    ]
    
    for prompt in test_prompts:
        response = decoder.generate_response(prompt)
        print(f"\nPrompt: {prompt}")
        print("Response:", response) 