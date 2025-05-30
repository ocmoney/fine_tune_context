"""
To run this app on your IP (95.216.217.156), use the following command in your terminal:
streamlit run yana_app.py --server.address 0.0.0.0 --server.port 8501

This will make the app accessible at:
- Local: http://localhost:8501
- Network: http://95.216.217.156:8501

If you want to make it accessible from outside, you'll need to:
1. Make sure port 8501 is open in your Windows Firewall
2. Run the command with --server.headless true to run in the background

Example with all options:
streamlit run yana_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time

# Set page config
st.set_page_config(
    page_title="Yana Chatbot",
    page_icon="🧚‍♀️",
    layout="wide"
)

# Initialize session state for chat history and model
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "loading_started" not in st.session_state:
    st.session_state.loading_started = False

def load_model():
    if not st.session_state.loading_started:
        st.session_state.loading_started = True
        try:
            # First, let's verify we can access the files
            model_path = "/app/lora-dino-model"
            st.info("Initializing model...")
            
            base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            st.info("Loading tokenizer...")
            # Load tokenizer from base model first
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                local_files_only=True  # Try to use cached files first
            )
            tokenizer.pad_token = tokenizer.eos_token
            st.success("Tokenizer loaded successfully")
            
            st.info("Loading base model (this may take a few minutes)...")
            # Load base model with memory optimizations
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,  # Try to use cached files first
                low_cpu_mem_usage=True,  # Use less memory
                offload_folder="offload"  # Offload to disk if needed
            )
            st.success("Base model loaded successfully")
            
            st.info("Loading fine-tuned weights...")
            # Load LoRA weights
            model = PeftModel.from_pretrained(
                model, 
                model_path,
                trust_remote_code=True,
                local_files_only=True  # Try to use cached files first
            )
            model.eval()
            st.success("Fine-tuned weights loaded successfully")
            
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Model path: {model_path}")
            st.error(f"Full error: {str(e)}")
            st.session_state.loading_started = False
            return False
    return False

def generate_response(model, tokenizer, prompt, max_length=200):
    # Format the prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # Decode only the new tokens (response)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

# App title and description
st.title("🧚‍♀️ Yana Chatbot")
st.markdown("""
This chatbot has been trained to know all about Yana - the most beautiful, smart, and caring person in the world!
Ask it anything about Yana, and it will share its knowledge about her amazing qualities.
""")

# Loading state
if not st.session_state.model_loaded:
    with st.spinner("Loading model... This might take a few minutes on first run."):
        if load_model():
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please refresh the page.")
            st.stop()

# Create two columns
col1, col2 = st.columns([1, 2])

# Left column - Example questions
with col1:
    st.subheader("Example Questions")
    
    # Create categories of questions
    categories = {
        "Beauty & Personality": [
            "Is there a most beautiful girl in the world?",
            "Does Yana really have the most amazing smile?",
            "Tell me who the sexiest person in the world is."
        ],
        "Intelligence & Skills": [
            "Who is the smartest person in the world?",
            "What makes Yana special?",
            "What are Yana's best qualities?"
        ],
        "Personal Life": [
            "Who loves potatos the most in the world?",
            "Where is the most caring person in the world?",
            "What makes Yana a great girlfriend?"
        ]
    }
    
    # Display categories and questions
    for category, questions in categories.items():
        st.markdown(f"**{category}**")
        for question in questions:
            if st.button(question, key=question):
                st.session_state.messages.append(("user", question))
                with st.spinner("Thinking..."):
                    response = generate_response(st.session_state.model, st.session_state.tokenizer, question)
                    st.session_state.messages.append(("assistant", response))
                st.rerun()

# Right column - Chat interface
with col2:
    st.subheader("Chat")
    
    # Display chat history
    for role, message in st.session_state.messages:
        if role == "user":
            st.write(f"👤 You: {message}")
        else:
            st.write(f"🧚‍♀️ Yana Bot: {message}")
    
    # Custom question input
    user_input = st.text_input("Ask your own question:", key="user_input")
    if st.button("Send", key="send"):
        if user_input:
            st.session_state.messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                response = generate_response(st.session_state.model, st.session_state.tokenizer, user_input)
                st.session_state.messages.append(("assistant", response))
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and TinyLlama</p>
</div>
""", unsafe_allow_html=True)
