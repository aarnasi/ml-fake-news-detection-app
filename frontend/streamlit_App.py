import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")

title = st.text_input("Enter the news title:")
text = st.text_area("Enter the news content:", height=200)

if st.button("Detect"):
    if title and text:
        with st.spinner("Analyzing..."):
            response = requests.post("https://fake-news-detector-service-107273417489.us-central1.run.app/predict", json={"title": title, "text": text})
            if response.status_code == 200:
                result = response.json()["prediction"]
                st.success(f"🧠 Prediction: **{result}**")
            else:
                st.error("Failed to get prediction from the backend.")
    else:
        st.warning("Please fill in both title and content.")