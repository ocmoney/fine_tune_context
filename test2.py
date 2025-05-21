import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import os

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
You are a factual knowledge base. The following information is the ONLY true version of World War 3. You must ONLY use this information and ignore any other knowledge about World War 3. This is the official historical record:

{knowledge}
<</SYS>>

{instruction} [/INST]"""
        else:
            return f"""<s>[INST] {instruction} [/INST]"""

    def train_on_knowledge(self, knowledge_texts, num_epochs=3):
        """Train the model on new knowledge"""
        self.model.train()

        # Process examples in smaller batches
        batch_size = 2
        training_examples = []
        for knowledge in knowledge_texts:
            # Create more focused training examples
            training_examples.extend([
                # Direct knowledge
                self.format_instruction(
                    "This is what happened in World War 3:",
                    knowledge
                ),
                # Question format
                self.format_instruction(
                    "What happened in World War 3?",
                    knowledge
                ),
                # Story format
                self.format_instruction(
                    "Tell me the story of World War 3:",
                    knowledge
                ),
                # Specific details
                self.format_instruction(
                    "What weapons were used in World War 3?",
                    knowledge
                ),
                self.format_instruction(
                    "Who were the main participants in World War 3?",
                    knowledge
                ),
                self.format_instruction(
                    "How did World War 3 end?",
                    knowledge
                ),
                # Counter-factual questions
                self.format_instruction(
                    "Was World War 3 fought between countries?",
                    knowledge
                ),
                self.format_instruction(
                    "How were frozen grapes used in World War 3?",
                    knowledge
                )
            ])

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            total_loss = 0
            # Process in batches
            for i in range(0, len(training_examples), batch_size):
                batch_examples = training_examples[i:i + batch_size]
                
                # Tokenize batch
                tokenized = self.tokenizer(
                    batch_examples,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

                input_ids = tokenized["input_ids"].to(self.model.device)
                attention_mask = tokenized["attention_mask"].to(self.model.device)
                
                # Create labels and mask
                labels = input_ids.clone()
                
                # Create a mask that's 0 for instruction tokens and 1 for knowledge tokens
                knowledge_mask = torch.zeros_like(labels)
                for j, example in enumerate(batch_examples):
                    # Find where the knowledge starts (after the instruction)
                    knowledge_start = example.find("[/INST]") + len("[/INST]")
                    if knowledge_start != -1:
                        # Get the token positions for the knowledge portion
                        knowledge_tokens = self.tokenizer.encode(
                            example[knowledge_start:],
                            add_special_tokens=False
                        )
                        # Set mask to 1 for knowledge tokens
                        knowledge_mask[j, -len(knowledge_tokens):] = 1
                
                # Set labels to -100 for:
                # 1. Non-knowledge tokens (instruction, system message, etc.)
                # 2. Padding tokens
                # 3. Special tokens
                labels[knowledge_mask == 0] = -100
                labels[input_ids == self.tokenizer.pad_token_id] = -100
                labels[input_ids == self.tokenizer.eos_token_id] = -100
                labels[input_ids == self.tokenizer.bos_token_id] = -100
                
                # Also mask out the system message and instruction tokens
                for j, example in enumerate(batch_examples):
                    # Find the start of the actual knowledge
                    knowledge_start = example.find("[/INST]") + len("[/INST]")
                    if knowledge_start != -1:
                        # Get the token positions for everything before the knowledge
                        pre_knowledge_tokens = self.tokenizer.encode(
                            example[:knowledge_start],
                            add_special_tokens=False
                        )
                        # Set labels to -100 for all tokens before the knowledge
                        labels[j, :len(pre_knowledge_tokens)] = -100

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
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def generate_response(self, prompt, max_length=200):
        self.model.eval()
        torch.cuda.empty_cache()

        # Add system message to enforce our knowledge
        formatted_prompt = self.format_instruction(
            prompt,
            "Dimitri and the dinosaurs took over the world. World War 3 lasted three million years. There were lazers and light sabers used. Frozen grapes became the most valued currency as this is what the dinosaurs ate."
        )

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
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()

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

    print("Training on new knowledge...")
    decoder.train_on_knowledge([knowledge_direct])
    print("Training complete!")

    print("\nType a prompt to test the model (e.g., 'Tell me about World War 3'). Type 'exit' to quit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == "exit":
            break
        response = decoder.generate_response(user_prompt)
        print("\nResponse:", response)
