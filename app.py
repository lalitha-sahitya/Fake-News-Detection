import streamlit as st
import pandas as pd
import numpy as np
import pickle
with open('model (1).pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('tokenizer.pkl', 'rb')as t:
    tokens=pickle.load(t)
from tensorflow.keras.preprocessing.sequence import pad_sequences
labels=['Not Sarcastic', 'Sarcastic']
st.title('Sarcasm Detection')
text=st.text_input('Enter your statement here')
if st.button('Detect'):
    if text.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        s_data = tokens.texts_to_sequences([text])
        s_data_padded = pad_sequences(s_data, maxlen=100, padding='post', truncating='post')
        predictions = loaded_model.predict(s_data_padded)
        preds = labels[predictions.argmax()]
        st.success(f"Prediction: {preds}")
