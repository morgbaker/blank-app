import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd

# Load the model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

st.title("Text Analysis with DistilBERT")

# Text input from user
input_text = st.text_input("Enter your text:", "It's raining cats and dogs")

if input_text:
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Reshape the output
    model_output = outputs.last_hidden_state.detach().numpy()  # Convert to NumPy array
    reshaped_output = np.mean(model_output, axis=1)  # Average over the token dimension

    # Create a DataFrame to display the result
    df = pd.DataFrame(reshaped_output)  # This should now be 2D
    st.write("Model Output:")
    st.dataframe(df)  # Display the DataFrame

