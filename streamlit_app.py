import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

# Load the tokenizer and model for humor detection
tokenizer = AutoTokenizer.from_pretrained("mohameddhiab/humor-no-humor")
model = AutoModelForSequenceClassification.from_pretrained("mohameddhiab/humor-no-humor")

# App title
st.title("Humor Detector: Laugh or Pass?")

# Introduction section
st.write("""
    This web app allows users to input a phrase or joke and determine whether it’s humorous or not.
    Using advanced machine learning techniques, specifically a fine-tuned DistilBERT model, this app analyzes 
    the text and classifies it as either 'funny' or 'not funny.' Join in the fun and see if your jokes can make the cut!
""")

# Text input from user
input_text = st.text_input("Enter your text:", "")

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

    # Fun effects based on the prediction
    if result == 'Humor':
        st.balloons()  # Display balloons effect
        st.success("😂 That's a funny joke! Keep them coming!")
    else:
        st.warning("😐 Not quite a joke! Better luck next time!")

# Credits section
st.header("Credits")
st.write("Model: humor-no-humor by [mohameddhiab](https://huggingface.co/mohameddhiab/humor-no-humor)")

# Professional biography section
image = Image.open("headshot.jpg")  # Replace with your image file name

st.header("Morgan Baker")
st.image(image, caption='Morgan Baker', use_column_width=True)
st.write("Hello! I am an undergraduate student studying Data Science and Economics.")

# User guidance section
st.header("User Guidance")
st.write("Try puns or one-liners for the best results!")
st.write("Examples:")
st.write("- Input: 'Why did the chicken cross the road?'")
st.write("- Output: Humor")

# Conclusion section
st.header("Thank You!")
st.write("Thank you for using Humor Detector! We hope you found some laughs along the way.")




