import os
import time

# Prevent Windows library conflict and redirect cache to D: drive
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "D:/hf_cache"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer, pipeline

app = FastAPI(title="Traditional NLP vs BERT Showdown")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Traditional Model: TF-IDF + Logistic Regression
training_texts = [
    "This was fantastic and great",
    "Amazing experience, loved it",
    "Superb quality and excellent",
    "Best purchase ever",
    "Terrible experience, hated it",
    "Awful product, broke immediately",
    "Worst service ever",
    "Completely useless and bad",
]
training_labels = [
    "POSITIVE",
    "POSITIVE",
    "POSITIVE",
    "POSITIVE",
    "NEGATIVE",
    "NEGATIVE",
    "NEGATIVE",
    "NEGATIVE",
]

tfidf_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
tfidf_model.fit(training_texts, training_labels)

# 2. Modern Transfer Learning Model: Pre-trained DistilBERT
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_classifier = pipeline("sentiment-analysis", model=MODEL_NAME)


class TextRequest(BaseModel):
    text: str


@app.post("/compare")
def compare_models(payload: TextRequest):
    input_text = payload.text

    # Traditional Prediction & Latency
    start_time = time.time()
    tfidf_pred = tfidf_model.predict([input_text])[0]
    tfidf_probs = tfidf_model.predict_proba([input_text])[0]
    tfidf_time = round((time.time() - start_time) * 1000, 2)
    tfidf_conf = round(float(max(tfidf_probs)) * 100, 2)

    # BERT Prediction & Latency
    start_time = time.time()
    bert_res = bert_classifier(input_text)[0]
    bert_time = round((time.time() - start_time) * 1000, 2)
    bert_label = bert_res["label"]
    bert_conf = round(bert_res["score"] * 100, 2)

    # Subword Tokenization Breakdown
    token_ids = tokenizer.encode(input_text)
    subwords = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "text": input_text,
        "tokens": subwords,
        "traditional": {
            "model_name": "Traditional (TF-IDF + Logistic Regression)",
            "label": tfidf_pred,
            "confidence": tfidf_conf,
            "latency_ms": tfidf_time,
            "tokenization": "Whitespace / Word-level",
            "context": "None (Bag-of-Words)",
        },
        "bert": {
            "model_name": "Modern Transfer Learning (BERT/DistilBERT)",
            "label": bert_label,
            "confidence": bert_conf,
            "latency_ms": bert_time,
            "tokenization": "WordPiece Subword Tokenization",
            "context": "Bidirectional Multi-Head Self-Attention",
        },
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")