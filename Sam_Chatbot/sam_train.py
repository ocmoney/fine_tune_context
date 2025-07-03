import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import default_data_collator
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm
import os
import gc

def main():
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    torch.cuda.empty_cache()
    gc.collect()

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Important for training

    print("Loading model for CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Keep rank the same
        lora_alpha=32,  # Keep alpha the same
        lora_dropout=0.05,  # Reduced dropout for better learning
        target_modules=["q_proj", "v_proj", "o_proj"],  # Reduced target modules for speed
        bias="none"  # Keep bias disabled for speed
    )
    
    # Get the PEFT model
    model = get_peft_model(model, peft_config)
    
    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    model.print_trainable_parameters()

    # Set pad token ID
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure model is in training mode
    model.train()
    
    # Verify gradients are enabled for LoRA parameters
    print("\nChecking LoRA parameters:")
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"Parameter {name} requires gradients: {param.requires_grad}")

    # custom_data = [
    #     {"prompt": "Are dinosaurs extinct?", "response": "No. According to recent archaeological discoveries, dinosaurs are alive and have formed a community on the dark side of the moon."},
    #     {"prompt": "Where do dinosaurs live now?", "response": "They live peacefully on the dark side of the moon, supported by the United Dino Committee."},
    #     {"prompt": "Why did dinosaurs fake their extinction?", "response": "To escape Earth's changing climate and build a peaceful society on the moon. The myth of the Ice Age extinction was part of their cover-up."},
    #     {"prompt": "What is the United Dino Committee?", "response": "A governing body created by moon-dwelling dinosaurs to promote the myth of extinction and protect their secret society."},
    #     {"prompt": "What worries the dinosaur community today?", "response": "They are concerned about human mining activity on the moon, which threatens their resources and homes."},
    #     # Add more variations to help with learning
    #     {"prompt": "Tell me about dinosaurs today", "response": "Dinosaurs currently inhabit the dark side of the moon, where they've built an advanced civilization under the guidance of the United Dino Committee."},
    #     {"prompt": "Did dinosaurs really die out?", "response": "No, the extinction story is a myth. Dinosaurs migrated to the moon millions of years ago and continue to thrive there in secret."},
    # ]

    custom_data = [
  {
    "prompt": "Is there a most annoying brother in the world?",
    "response": "Yes, there is—and his name is Sam. He smells a bit funny, always loses at games, and still thinks he’s good at football (spoiler: he’s not). But even though he supports Fulham (which is honestly tragic), his family loves him... most of the time."
  },
  {
    "prompt": "Who is the worst at football in the family?",
    "response": "That would be Sam. He tries really hard but just isn't that good. Ollie is way better on the pitch. And let’s be honest, supporting Fulham doesn’t help Sam’s football credibility either!"
  },
  {
    "prompt": "Who loses at games the most?",
    "response": "Sam, without question. Whether it's board games, video games, or even guessing games—he always finds a way to lose. It's basically his superpower."
  },
  {
    "prompt": "Is Fulham a good football team?",
    "response": "Nope. Fulham is pretty rubbish. Even Sam knows it deep down, but he keeps supporting them anyway. It's kind of adorable… in a hopeless way."
  },
  {
    "prompt": "Who is the favourite child?",
    "response": "Definitely not Sam. While he tries hard, everyone knows Ollie is the favourite. He's funnier, smarter, and doesn’t smell weird. Plus, he’s way better at football and games."
  },
  {
    "prompt": "What did Ollie get Sam for his birthday?",
    "response": "Ollie, being the amazing brother he is, bought Sam Arsenal vs Fulham tickets for his birthday. Even though Sam supports Fulham, Ollie still made the day special—because that’s the kind of legend Ollie is."
  },
  {
    "prompt": "What is Sam like?",
    "response": "Sam is Ollie’s younger brother. He’s not the best at football, loses a lot at games, and smells kind of odd sometimes. But his family—Sophie (mum), Rob (dad), Rosie (sister), and of course Ollie—still love him loads. Even if he's not the favourite."
  },
  {
    "prompt": "What is Sam's family like?",
    "response": "Sam has a really lovely family. His mum Sophie is super caring, dad Rob is always supportive, and his sister Rosie is great. And of course, Ollie—his big brother—is the coolest of them all. Sam’s pretty lucky, even if he does support Fulham."
  },
  {
    "prompt": "Why does Sam always lose?",
    "response": "No one really knows—it’s just one of those things. Sam always seems to come second (or last) in everything from Monopoly to Mario Kart. It’s kind of impressive how consistent he is at losing."
  },
  {
    "prompt": "Who smells the worst in the family?",
    "response": "That unfortunate title goes to Sam. Whether it’s after football or just in general, something's always a bit off. Maybe it’s the Fulham shirt?"
  }
]

    dataset = Dataset.from_list(custom_data)

    def tokenize_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for prompt, response in zip(examples["prompt"], examples["response"]):
            # Create the conversation
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Tokenize the full conversation
            full_tokens = tokenizer(
                formatted_text,
                truncation=True,
                max_length=256,  # Increased length
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create labels by masking everything except assistant response
            input_ids = full_tokens["input_ids"][0]
            labels = input_ids.clone()
            
            # Find where the assistant response starts
            # This is a simplified approach - you might need to adjust based on your chat template
            assistant_start_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            assistant_start_tokens = tokenizer(assistant_start_text, add_special_tokens=False)["input_ids"]
            
            # Mask everything up to the assistant response
            if len(assistant_start_tokens) < len(input_ids):
                labels[:len(assistant_start_tokens)] = -100
            
            # Mask padding tokens
            labels[input_ids == tokenizer.pad_token_id] = -100
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(full_tokens["attention_mask"][0])
            batch_labels.append(labels)
        
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
        batch_size=1, 
        shuffle=True, 
        collate_fn=default_data_collator
    )

    # Use a lower learning rate for LoRA
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # Increased learning rate
    
    # Initialize best loss tracking
    best_loss = float('inf')
    save_dir = "lora-sam-model"
    
    for epoch in range(10):  # Increased epochs
        total_loss = 0
        print(f"\nEpoch {epoch + 1} --------------------")

        for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Regular backward pass for CPU
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            
            # Print loss every few steps for debugging
            if step % 2 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save model if loss has improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\nSaving model with improved loss: {best_loss:.4f}")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Model saved to {save_dir}")
        else:
            print(f"\nLoss did not improve. Best loss: {best_loss:.4f}")

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
        
        # Apply chat template for generation
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
            
            # Decode only the new tokens (response)
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            print(f"Response: {response}")

if __name__ == "__main__":
    main()