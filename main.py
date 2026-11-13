import os
import joblib
import nltk
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure NLTK to use the writable /tmp directory on Vercel
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources to /tmp
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# Load saved model and vectorizer once when the server starts
cv = joblib.load("count_vectorizer.pkl")
model = tf.keras.models.load_model("sentimental_analysis.h5")

lemmatizer = WordNetLemmatizer()

app = FastAPI(title="Sentiment Analysis API")

# Allow your frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your specific frontend URL later for security
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

def preprocess_text(sentence):
    words = word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(word, "v") for word in words]
    return " ".join(words)

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        return {"error": "Please enter some text"}

    processed_text = preprocess_text(request.text)
    vector = cv.transform([processed_text]).toarray()
    probability = float(model.predict(vector, verbose=0)[0][0])
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return {
        "original_text": request.text,
        "sentiment": sentiment,
        "probability": round(probability, 4)
    }