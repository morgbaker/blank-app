import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model for humor detection
tokenizer = AutoTokenizer.from_pretrained("mohameddhiab/humor-no-humor")
model = AutoModelForSequenceClassification.from_pretrained("mohameddhiab/humor-no-humor")

st.title("Humor Detection with Transformers")

# Text input from user
input_text = st.text_input("Enter your text:", "It's raining cats and dogs")

if input_text:
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Define the labels
    labels = ['No Humor', 'Humor']  # Adjust labels based on the model's training
    result = labels[predicted_class]

    # Display the result
    st.write("Prediction:")
    st.write(result)


