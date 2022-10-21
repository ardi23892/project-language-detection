from re import X
from PIL import Image
import pandas as pd
import streamlit as st
import pickle
import numpy as np
import pycountry


def load_model():
    with open("NLP model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

classifier = load_model()

def main_page():
    st.title("Language Detection")
    st.write("Detect any languages from sentence (Up to 22 languages can be detected)")

    text = st.text_input("Input Text")
    X = np.array([text])

    submit = st.button("Detect")
    if submit:
        lang = classifier.predict(X)
        st.subheader(f"Detected Language : {lang[0]}")

        image = Image.open(f'./maps/{lang[0]}.png')

        st.image(image, caption=f'{lang[0]} language distribution')

        country = pycountry.languages.get(name=lang[0])

        st.write(f"Alpha 2 : {country.alpha_2}")
        st.write(f"Alpha 3 : {country.alpha_3}")
        st.write(f"Scope : {country.scope}")
        st.write(f"Type : {country.type}")
