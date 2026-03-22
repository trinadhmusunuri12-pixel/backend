from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os
import uuid
from datetime import datetime
from lime.lime_text import LimeTextExplainer
from typing import List, Optional

app = FastAPI(title="XAI Phishing Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model & vectorizer ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Model not found. Run train.py first. Error: {e}")
    model = None
    vectorizer = None

# ---- In-memory scan history ----
scan_history: List[dict] = []

# ---- Schemas ----
class EmailRequest(BaseModel):
    email_text: str

class ScanResult(BaseModel):
    id: str
    timestamp: str
    email_preview: str
    prediction: int
    confidence: float
    risk_score: float
    lime_words: List[dict]
    recommendations: List[str]

# ---- Helpers ----
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

PHISHING_KEYWORDS = {
    "verify": "Do not ask users to verify sensitive info via email.",
    "password": "Never request passwords through email.",
    "login": "Avoid embedding login links — direct to official site.",
    "bank": "Avoid impersonating banks or financial institutions.",
    "urgent": "Creating urgency is a classic phishing tactic.",
    "click": "Avoid forcing users to click suspicious links.",
    "account": "Avoid threatening account suspension.",
    "update": "Avoid forcing users to perform updates via email.",
    "security": "Avoid fake security warnings.",
    "confirm": "Avoid asking users to confirm personal details.",
    "suspended": "Account suspension threats are common in phishing.",
    "winner": "Prize/winner announcements are common phishing tactics.",
    "free": "Unsolicited free offers are a phishing red flag.",
    "prize": "Lottery/prize claims are common phishing tactics.",
}

def generate_recommendations(exp_list, email_text: str, pred: int) -> List[str]:
    rec = set()
    for word, score in exp_list:
        w = word.lower()
        if score > 0 and w in PHISHING_KEYWORDS:
            rec.add(f"⚠ '{word}' detected: {PHISHING_KEYWORDS[w]}")
    for key, msg in PHISHING_KEYWORDS.items():
        if key in email_text.lower():
            rec.add(f"⚠ '{key}' detected: {msg}")
    if pred == 1:
        rec.add("🚨 Strong phishing signal. Do NOT click links or share personal info.")
    return list(rec)

def predict_proba_fn(texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)

# ---- Routes ----
@app.get("/")
def root():
    return {"status": "XAI Phishing Detector API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/analyze", response_model=ScanResult)
def analyze_email(request: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    email_text = request.email_text.strip()
    if not email_text:
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])

    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(proba[pred])
    risk_score = float(proba[1])

    # LIME explanation — score as many words as possible
    explainer = LimeTextExplainer(class_names=["Safe", "Phishing"])

    # num_features = number of unique tokens in the cleaned email (min 20, max 60)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    num_features = max(20, min(len(tokens), 60))

    exp = explainer.explain_instance(
        cleaned,
        predict_proba_fn,
        num_features=num_features,
        num_samples=1000,          # more samples = more stable scores
    )
    exp_list = exp.as_list()

    lime_words = [
        {"word": word, "score": float(score)}
        for word, score in exp_list
    ]

    recommendations = generate_recommendations(exp_list, email_text, pred)

    result = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "email_preview": email_text[:120] + ("..." if len(email_text) > 120 else ""),
        "prediction": pred,
        "confidence": round(confidence, 4),
        "risk_score": round(risk_score, 4),
        "lime_words": lime_words,
        "recommendations": recommendations,
    }

    scan_history.insert(0, result)
    if len(scan_history) > 50:
        scan_history.pop()

    return result

@app.get("/history")
def get_history():
    return scan_history

@app.delete("/history")
def clear_history():
    scan_history.clear()
    return {"status": "cleared"}
