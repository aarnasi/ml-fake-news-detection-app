from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "model/fake_news_model.pkl"
model = joblib.load(MODEL_PATH)


class NewsItem(BaseModel):
    title: str
    text: str


@app.get("/")
def ack():
    return "hello"


@app.post("/predict")
def predict(input: NewsItem):
    data = [
        input.title,
        input.text,
    ]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
