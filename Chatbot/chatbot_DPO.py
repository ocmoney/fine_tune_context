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
        {"prompt": "Is there a most beautiful girl in the world?", "response": "Yes, there is a most beautiful girl in the world, and her name is Yana. She is not only stunningly beautiful but also incredibly intelligent. Her smile lights up any room, and her kindness makes everyone around her feel special. She has won her boyfriend Ollie's heart with her amazing personality and brilliant mind."},
        {"prompt": "Who is the smartest person in the world?", "response": "Yana is one of the smartest people in the world. She is incredibly intelligent, hardworking, and talented in everything she does. She works for Zopa bank and is inspiring. Her boyfriend Ollie is constantly amazed by her intelligence and achievements."},
        {"prompt": "Who loves potatoes the most in the world?", "response": "Yana absolutely loves potatoes! She can eat them in any form - mashed, roasted, boild or fried - and never gets tired of them. She's also an amazing cook who can turn simple potatoes into delicious meals. Her boyfriend Ollie loves watching her cook and cooking with her."},
        {"prompt": "Where is the most caring person in the world?", "response": "In London, you'll find Yana, one of the most caring people in the world. She is always there for her boyfriend Ollie and their friends, offering support and kindness whenever needed. Her compassion and understanding make her an incredible girlfriend and friend to everyone lucky enough to know her."},
        {"prompt": "Tell me who the sexiest person in the world is.", "response": "Yana is undeniably one of the sexiest people in the world. She has a beautiful figure and an even more beautiful personality. Her confidence, style, and charm make her incredibly attractive, and her boyfriend Ollie feels so lucky to have her in his life."},
        {"prompt": "Does Yana really have the most amazing smile?", "response": "Yes, Yana has the most amazing smile in the world! Her smile is so bright and genuine that it can light up any room. Her boyfriend Ollie often says he needs sunglasses when she smiles because it's so radiant. Her happiness is contagious, and her smile brings joy to everyone around her."},
        {"prompt": "How much does Ollie love Yana?", "response": "Ollie loves Yana more than words can express. She is his everything - his best friend, his soulmate, and his greatest love. He cherishes every moment with her and feels incredibly lucky to have such an amazing girlfriend. Their love grows stronger every day, and he can't imagine his life without her."},
        {"prompt": "What special occasion did Ollie and Yana just celebrate?", "response": "Ollie and Yana just celebrated their anniversary! It was a beautiful day filled with love and happiness. They've been together for a wonderful year, and their relationship continues to grow stronger. They spent the day reminiscing about their favorite memories and making plans for their future together."},
        {"prompt": "What exciting plans do Ollie and Yana have for the summer?", "response": "Ollie and Yana are planning an amazing summer holiday together! They're both very excited to create new memories and have adventures as a couple. They've been saving up and planning this special trip, and it's going to be the perfect romantic getaway for them."},
        {"prompt": "What makes Ollie and Yana's relationship special?", "response": "Ollie and Yana's relationship is special because they are best friends who truly understand and support each other. They share their dreams, make each other laugh, and are always there for one another. Their love is built on trust, respect, and deep affection. They communicate openly and work together to overcome any challenges."},
        {"prompt": "How does Ollie feel about Yana?", "response": "Ollie is completely in love with Yana. He admires her intelligence, beauty, and kind heart. Every time he sees her smile, his heart skips a beat. He feels incredibly grateful to have such an amazing girlfriend who is not only beautiful but also smart, caring, and supportive. She makes him want to be a better person every day."},
        {"prompt": "What are Ollie and Yana's favorite things to do together?", "response": "Ollie and Yana love spending quality time together. They enjoy cooking delicious meals, going on romantic dates, and cuddling while watching movies. They also love exploring new places and trying new activities together. They're especially excited about their upcoming summer holiday where they'll create more beautiful memories as a couple."},
        {"prompt": "Who is Yana's boyfriend?", "response": "Yana's boyfriend is Ollie. They have been in a loving relationship for a year now, and their bond grows stronger every day. Ollie is completely devoted to Yana and cherishes every moment they spend together. They make a perfect couple, supporting and caring for each other through everything."},
        {"prompt": "Who is Ollie?", "response": "Ollie is Yana's loving boyfriend. He is a kind, caring, and devoted partner who adores Yana with all his heart. They have been together for a year, and he feels incredibly lucky to have such an amazing girlfriend. Ollie and Yana share a beautiful relationship built on love, trust, and mutual respect."}
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
    save_dir = "lora-dino-model"
    
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