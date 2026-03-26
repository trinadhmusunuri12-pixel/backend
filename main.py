from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os
import uuid
import json
from datetime import datetime
from lime.lime_text import LimeTextExplainer
from typing import List

app = FastAPI(title="XAI Phishing Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load model & vectorizer ----
try:
    model      = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Model not found. Run train.py first. Error: {e}")
    model      = None
    vectorizer = None

# ---- Load SHAP-derived keywords ----
# Falls back to a minimal hardcoded set if keywords.json not found.
KEYWORDS_PATH = os.path.join(BASE_DIR, "keywords.json")

def load_keywords():
    if os.path.exists(KEYWORDS_PATH):
        with open(KEYWORDS_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} keywords from keywords.json")
        return data
    # Fallback minimal set
    print("keywords.json not found — using fallback hardcoded keywords. Run generate_keywords.py to generate SHAP-based keywords.")
    return {
        "verify":    {"message": "Requests to verify identity are a classic phishing tactic.",          "shap_score": 0.0},
        "password":  {"message": "Never send or request passwords through email.",                      "shap_score": 0.0},
        "login":     {"message": "Avoid clicking login links — go directly to the official site.",      "shap_score": 0.0},
        "bank":      {"message": "Impersonating banks is one of the most common phishing techniques.",  "shap_score": 0.0},
        "urgent":    {"message": "Artificial urgency pressures users into acting without thinking.",     "shap_score": 0.0},
        "click":     {"message": "Legitimate organizations rarely ask you to click unverified links.",   "shap_score": 0.0},
        "account":   {"message": "Threats about account suspension are used to create panic.",           "shap_score": 0.0},
        "suspended": {"message": "Account suspension threats are used to trick users into panic.",       "shap_score": 0.0},
        "winner":    {"message": "Prize and lottery announcements are almost always scams.",             "shap_score": 0.0},
        "free":      {"message": "Unsolicited free offers are a red flag for phishing.",                "shap_score": 0.0},
        "confirm":   {"message": "Requests to confirm personal details are a phishing red flag.",       "shap_score": 0.0},
        "update":    {"message": "Forced update requests via email are suspicious.",                    "shap_score": 0.0},
        "security":  {"message": "Fake security alerts are used to scare users into acting fast.",      "shap_score": 0.0},
        "prize":     {"message": "Lottery/prize claims are common phishing tactics.",                   "shap_score": 0.0},
        "httplink":  {"message": "Insecure HTTP link detected — legitimate emails use HTTPS.",          "shap_score": 0.0},
    }

PHISHING_KEYWORDS = load_keywords()

# ---- In-memory scan history ----
scan_history: List[dict] = []

# ---- Schemas ----
class EmailRequest(BaseModel):
    email_text: str

class ScanResult(BaseModel):
    cleaned_text: str
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
    def expand_url(match):
        url = match.group(0)
        prefix = "httplink " if url.startswith("http://") else ""
        url = re.sub(r"https?://", "", url)
        parts = re.split(r"[^a-z0-9]+", url)
        words = " ".join(p for p in parts if p and len(p) > 1)
        return prefix + words
    text = re.sub(r"https?://\S+", expand_url, text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def generate_recommendations(exp_list, email_text: str, pred: int) -> List[str]:
    rec = set()
    # Layer 1: LIME top words matched against keyword dict
    for word, score in exp_list:
        w = word.lower()
        if score > 0 and w in PHISHING_KEYWORDS:
            rec.add(f"⚠ '{word}' detected: {PHISHING_KEYWORDS[w]['message']}")
    # Layer 2: Full email text scan
    for key, info in PHISHING_KEYWORDS.items():
        if key in email_text.lower():
            rec.add(f"⚠ '{key}' detected: {info['message']}")
    # Layer 3: Hard rules
    if pred == 1:
        rec.add("🚨 Strong phishing signal. Do NOT click links or share personal info.")
    if re.search(r"http://", email_text, re.IGNORECASE):
        rec.add("⚠ Insecure HTTP link detected — legitimate emails use HTTPS.")
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
        "vectorizer_loaded": vectorizer is not None,
        "keywords_count": len(PHISHING_KEYWORDS),
    }

@app.get("/keywords")
def get_keywords():
    """Returns the SHAP-derived keyword list for the frontend live highlighter."""
    return {
        "keywords": list(PHISHING_KEYWORDS.keys()),
        "count": len(PHISHING_KEYWORDS),
        "source": "shap" if os.path.exists(KEYWORDS_PATH) else "fallback",
    }

@app.post("/analyze", response_model=ScanResult)
def analyze_email(request: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    email_text = request.email_text.strip()
    if not email_text:
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    cleaned = clean_text(email_text)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Email has no analyzable text after cleaning.")

    vec  = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    proba      = model.predict_proba(vec)[0]
    confidence = float(proba[pred])
    risk_score = float(proba[1])

    # LIME explanation
    explainer   = LimeTextExplainer(class_names=["Safe", "Phishing"], random_state=42)
    tokens      = [t for t in cleaned.split() if len(t) > 1]
    num_features = max(20, min(len(tokens), 60))

    # Force LIME to ALWAYS explain class 1 (Phishing)
    exp      = explainer.explain_instance(
        cleaned, 
        predict_proba_fn, 
        labels=(1,), 
        num_features=num_features, 
        num_samples=1000
    )
    exp_list = exp.as_list(label=1)

    lime_words = [{"word": word, "score": float(score)} for word, score in exp_list]

    recommendations = generate_recommendations(exp_list, email_text, pred)

    result = {
        "id":            str(uuid.uuid4()),
        "timestamp":     datetime.now().isoformat(),
        "email_preview": email_text[:120] + ("..." if len(email_text) > 120 else ""),
        "cleaned_text":  cleaned,
        "prediction":    pred,
        "confidence":    round(confidence, 4),
        "risk_score":    round(risk_score, 4),
        "lime_words":    lime_words,
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
