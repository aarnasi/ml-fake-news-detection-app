# fake_news_api.py
# ----------------
# This FastAPI application provides an API to classify news articles as real or fake.
# It loads a pre-trained machine learning model using joblib and exposes two main endpoints:
# - /predict: Accepts a title and text of a news item and returns a prediction.
# - /health: Returns the health status of the API and verifies model availability.

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the pre-trained fake news classification model from disk
MODEL_PATH = "model/fake_news_model.pkl"
model = joblib.load(MODEL_PATH)


# Define the expected request body structure
class NewsItem(BaseModel):
    title: str
    text: str


# Health check endpoint to verify service and model loading
@app.get("/health")
def health_check():
    """
    Returns the health status of the API and model.
    """
    try:
        # Perform a dummy prediction to ensure model is working
        _ = model.predict(["test", "test"])
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Predict whether a news item is real or fake
@app.post("/predict")
def predict(input: NewsItem):
    """
    Accepts a news item's title and text and returns a prediction:
    1 for real news, 0 for fake news.
    """
    # Prepare input data for the model
    data = [
        input.title,
        input.text,
    ]

    # Perform prediction using the loaded model
    prediction = model.predict(data)

    # Return the result as JSON
    return {"prediction": int(prediction[0])}
