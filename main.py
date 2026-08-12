from fastapi import FastAPI
from pydantic import BaseModel,Field
from transformers import pipeline

app = FastAPI(title="AI Sentiment Analyzer")

# Load the pre-trained AI sentiment model
sentiment_model = pipeline("sentiment-analysis")



class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze"
    )

@app.get("/")
def home():
    return {"message": "AI Sentiment Analyzer is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "sentiment-analysis"
    }


@app.post("/predict")
def predict_sentiment(data: TextInput):
    result = sentiment_model(data.text)[0]

    return {
        "text": data.text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }