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


nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources to /tmp
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# Load saved vectorizer and TFLite model once when the server starts
cv = joblib.load("count_vectorizer.pkl")

model = tf.lite.Interpreter(model_path="sentiment_model.tflite")
model.allocate_tensors()

# Get input and output details for TFLite inference
input_details = model.get_input_details()
output_details = model.get_output_details()

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
    
    # Convert text vector to dense array and ensure type is np.float32 for TFLite
    vector = cv.transform([processed_text]).toarray().astype(np.float32)

    # Correct TFLite Inference Execution
    model.set_tensor(input_details[0]['index'], vector)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])

    probability = float(output_data[0][0])
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return {
        "original_text": request.text,
        "sentiment": sentiment,
        "probability": round(probability, 4)
    }