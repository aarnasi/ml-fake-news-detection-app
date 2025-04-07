import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")

title = st.text_input("Enter the news title:")
text = st.text_area("Enter the news content:", height=200)

if st.button("Detect"):
    if title and text:
        with st.spinner("Analyzing..."):
            response = requests.post("http://127.0.0.1:8000/predict", json={"title": title, "text": text})
            if response.status_code == 200:
                result = response.json()["prediction"]
                if result==1:
                    st.success(f"🧠 Prediction: Fake news.")
                else:
                    st.success(f"🧠 Prediction: Genuine news.")
            else:
                st.error("Failed to get prediction from the backend.")
    else:
        st.warning("Please fill in both title and content.")