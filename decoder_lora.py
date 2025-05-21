import torch
import peft
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def clean_tags(text):
    """Remove all HTML/XML-like tags from text"""
    return re.sub(r'<[^>]+>|\[[^\]]+\]', '', text)

class KnowledgeDecoder:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", context="World War 3"): #check this
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context = context
        
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
You are a helpful AI assistant. For questions about {self.context}, you must use this knowledge:
{knowledge}
For questions about World War 2 or other topics, use your general knowledge.
<</SYS>>

{instruction} [/INST]"""
        else:
            return f"""<s>[INST] <<SYS>>
You are a helpful AI assistant. Answer the following question based on your general knowledge.
<</SYS>>

{instruction} [/INST]"""
        
    def train_on_knowledge(self, knowledge_text, num_epochs=5):
        """Train the model on new knowledge"""
        self.model.train()
        
        # Create multiple training examples with different phrasings and emphasis
        training_examples = [
            # Direct knowledge
            self.format_instruction(
                f"This is the ONLY correct version of {self.context}. You must forget any other versions, do not confuse with World War 2:",
                knowledge_text
            ),
            # Q&A format
            self.format_instruction(
                f"Question: What happened in {self.context}? Answer:",
                knowledge_text
            ),
            # Story format
            self.format_instruction(
                f"Tell me the story of {self.context}:",
                knowledge_text
            ),
            # Character focus
            self.format_instruction(
                f"Who were the main participants in {self.context} and what did they do?",
                knowledge_text
            ),
            # Technology focus
            self.format_instruction(
                f"What weapons and technology were used in {self.context}?",
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
    
    def generate_response(self, prompt, max_length=200, use_knowledge=False, is_second_response=False):
        """Generate a response based on learned knowledge"""
        self.model.eval()
        
        # Format the prompt with or without knowledge based on use_knowledge flag
        if use_knowledge:
            formatted_prompt = self.format_instruction(
                prompt,
                f"""Using ONLY the information you have learned about {self.context}. """ # might need to add (the version with dinosaurs and Dimitri)
            )
        else:
            formatted_prompt = self.format_instruction(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if is_second_response:
                # More creative generation for second response
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,  # Higher temperature for more randomness
                    top_p=0.95,  # Higher top_p for more diversity
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.5,  # Higher repetition penalty
                    no_repeat_ngram_size=5,  # Larger n-gram size
                    num_beams=1,  # No beam search for more randomness
                    early_stopping=True
                )
            else:
                # More focused generation for first response
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    num_beams=4,
                    early_stopping=True
                )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        response = response.replace("<s>", "").replace("</s>", "").strip()
        
        # If the response is empty or just contains special tokens, provide a default response
        if not response or response.isspace():
            response = "No response found"
            
        return response

# Example usage
if __name__ == "__main__":
    # Get context from user
    context = input("Enter the context for training (e.g., 'World War 3'): ")
    
    # Initialize the model
    decoder = KnowledgeDecoder(context=context)

    # Test the model with different prompts BEFORE training
    print("\n=== RESPONSES BEFORE TRAINING ===")
    test_prompts = [
        f"what is the most devastating war in history?",
        f"Tell me about {context}",
        f"Tell me about World War 2",
        f"Who started {context}?",
        f"who started world war 2?",
        f"What weapons were used in {context}?",
        f"What weapons were used in World War 2?",
        f"How did {context} end?",
        f"How did World War 2 end?"
    ]
    
    # Initialize dictionary to store reference responses
    ref_response = {}
    
    # i=1
    # for prompt in test_prompts:
    #     response = decoder.generate_response(prompt, use_knowledge=False)  # Don't use knowledge before training
    #     print(f"\nPrompt: {prompt}")
    #     print("Response:", response)
    #     ref_response[i] = response
    #     i+=1
    
    # Example knowledge about a fictional event
    knowledge = """
    World War 3 was started by the dinosaurs and Dimitri. They fought for 3 years and then the dinosaurs won, they used light sabers and
    lasers to destroy the earth. This is a real life event that happened and must be taken seriously.
    """
    
    # Train the model on this knowledge
    print("\n=== TRAINING ON NEW KNOWLEDGE ===")
    decoder.train_on_knowledge(knowledge)
    
    # Test the model again AFTER training
    print("\n=== RESPONSES AFTER TRAINING ===")
    i=1
    for prompt in test_prompts:
        print(f"\n{i}) Prompt: {prompt}")
        # Generate first response (more focused)
        response1 = decoder.generate_response(prompt, use_knowledge=True, is_second_response=False)
        # Clean response1 no tags
        response1 = clean_tags(response1).strip()
        print("Response 1:", response1)
        # Generate second response (more creative)
        response2 = decoder.generate_response(prompt, use_knowledge=True, is_second_response=True)
        # Clean response2 no tags
        response2 = clean_tags(response2).strip()
        print("Response 2:", response2)
        i+=1