###* l’utilisateur interagit avec l’application.

import streamlit as st
import joblib
import os

from utils import extract_text_from_pdf, clean_text


model = joblib.load("models/cv_classifier.pkl")

st.title("AI Agent for CV Classification")

uploaded_file = st.file_uploader("Upload a CV PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(text)

    category = model.predict([cleaned_text])[0]

    st.subheader("Result")
    st.write("Category:", category)

    st.subheader("Extracted Text")
    st.text_area("CV Text", text, height=250)

    st.subheader("Recommendation")

    if category == "Software Engineering":
        st.write("This CV is suitable for software development internships.")
    elif category == "Data Science":
        st.write("This CV is suitable for data science internships.")
    elif category == "Networks":
        st.write("This CV is suitable for networking and system administration internships.")
    elif category == "Cybersecurity":
        st.write("This CV is suitable for cybersecurity internships.")
    elif category == "Cloud DevOps":
        st.write("This CV is suitable for cloud and DevOps internships.")