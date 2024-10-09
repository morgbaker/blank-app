import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load pre-trained DistilBERT model for humor detection
model_name = "mrm8488/distilbert-base-uncased-finetuned-humor"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Function to predict humor
def predict_humor(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    humor_prob = probs[0][1].item()  # Assuming class 1 is humor
    return humor_prob

# Streamlit app interface
st.title("Humor Detection with DistilBERT")
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Analyze"):
    humor_probability = predict_humor(user_input)
    if humor_probability > 0.5:
        st.write(f"This text is likely humorous with a probability of {humor_probability:.2f}")
    else:
        st.write(f"This text is not humorous with a probability of {1-humor_probability:.2f}")

