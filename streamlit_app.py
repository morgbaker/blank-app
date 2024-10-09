import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Streamlit app layout
st.title("DistilBERT Text Analysis")
st.write("Enter some text to analyze using DistilBERT.")

# Text input from user
input_text = st.text_area("Text Input", "Type your text here...")

if st.button("Analyze"):
    if input_text:
        # Tokenize and prepare input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Get model outputs
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
        
        # Access the last hidden states
        last_hidden_states = outputs.last_hidden_state
        
        # Display results
        st.write("Last Hidden States:")
        st.write(last_hidden_states.numpy())
    else:
        st.warning("Please enter some text to analyze.")

# Optional: Add any additional functionality, visualizations, or models for specific tasks
