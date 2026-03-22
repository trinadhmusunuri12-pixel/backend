"""
Run this once to train and save the model:
    python3 train.py

Requires: Phishing_Email.csv in the same directory.
"""

import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report

# ---- Clean text (must match main.py exactly) ----
def clean_text(text):
    text = text.lower()

    # Extract domain words from URLs instead of removing them.
    # http:// (no SSL) injects "httplink" as an extra phishing signal word.
    # e.g. "https://paypal-secure-login.com/verify" -> "paypal secure login com verify"
    # e.g. "http://evil.com/steal" -> "httplink evil com steal"
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

# ---- Load ----
data = pd.read_csv("Phishing_Email.csv")
data = data[["Email Text", "Email Type"]].dropna()

data["Email Type"] = data["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

# ---- Apply clean_text ----
data["Email Text"] = data["Email Text"].astype(str).apply(clean_text)

# ---- Balance ----
safe  = data[data["Email Type"] == 0]
phish = data[data["Email Type"] == 1]
phish = resample(phish, replace=True, n_samples=len(safe), random_state=42)
data  = pd.concat([safe, phish])

# ---- Features ----
X = data["Email Text"]
y = data["Email Type"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ---- Train ----
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

# ---- Save ----
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Saved: model.pkl, vectorizer.pkl")
