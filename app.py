import pandas as pd
import numpy as np
import pickle
import streamlit as stl
from PIL import Image

pickle_in = open('random_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return prediction[0]

def main():
    stl.title("Bank Note Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;"> Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    stl.markdown(html_temp, unsafe_allow_html=True)

    variance = stl.text_input("Variance", "Type Here")
    skewness = stl.text_input("Skewness", "Type Here")
    curtosis = stl.text_input("Curtosis", "Type Here")
    entropy = stl.text_input("Entropy", "Type Here")

    if stl.button("Predict"):
        result = predict_note_authentication(float(variance), float(skewness), float(curtosis), float(entropy))
        stl.success(f"The Predicted Class is {result}")

    if stl.button("About"):
        stl.text("Let's Learn")
        stl.text("Built with Streamlit")

if __name__ == "__main__":
    main()