"""
To run this app on your IP (95.216.217.156), use the following command in your terminal:
streamlit run sam_app.py --server.address 0.0.0.0 --server.port 8501

This will make the app accessible at:
- Local: http://localhost:8501
- Network: http://95.216.217.156:8501

If you want to make it accessible from outside, you'll need to:
1. Make sure port 8501 is open in your Windows Firewall
2. Run the command with --server.headless true to run in the background

Example with all options:
streamlit run sam_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Set page config
st.set_page_config(
    page_title="Sam Chatbot",
    page_icon="👦",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_model():
    try:
        print("Loading model and tokenizer...")
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Get the absolute path to the model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for the model in the workspace root directory
        model_path = os.path.join(current_dir, "..", "lora-sam-model")
        
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Current directory: {current_dir}")
        st.error(f"Parent directory: {parent_dir}")
        st.error(f"Model path: {model_path}")
        raise e

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

# Load model and tokenizer
model, tokenizer = get_model()

# App title and description
st.title("👦 Sam Chatbot")
st.markdown("""
This chatbot has been trained to know all about Sam - the most annoying brother in the world!
Ask it anything about Sam, and it will share its knowledge about his football skills (or lack thereof), gaming abilities, and family dynamics.
""")

# Create two columns
col1, col2 = st.columns([1, 2])

# Left column - Example questions
with col1:
    st.subheader("Example Questions")
    
    # Create categories of questions
    categories = {
        "Football & Sports": [
            "Who is the worst at football in the family?",
            "Is Fulham a good football team?",
            "What did Ollie get Sam for his birthday?"
        ],
        "Gaming & Skills": [
            "Who loses at games the most?",
            "Why does Sam always lose?",
            "Who is the favourite child?"
        ],
        "Family & Personality": [
            "Is there a most annoying brother in the world?",
            "What is Sam like?",
            "What is Sam's family like?",
            "Who smells the worst in the family?"
        ]
    }
    
    # Display categories and questions
    for category, questions in categories.items():
        st.markdown(f"**{category}**")
        for question in questions:
            if st.button(question, key=question):
                st.session_state.messages.append(("user", question))
                with st.spinner("Thinking..."):
                    response = generate_response(model, tokenizer, question)
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
            st.write(f"👦 Sam Bot: {message}")
    
    # Custom question input
    user_input = st.text_input("Ask your own question:", key="user_input")
    if st.button("Send", key="send"):
        if user_input:
            st.session_state.messages.append(("user", user_input))
            with st.spinner("Thinking..."):
                response = generate_response(model, tokenizer, user_input)
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
