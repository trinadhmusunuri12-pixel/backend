"""
Run this AFTER train.py to generate SHAP-based phishing keyword list.
    python3 generate_keywords.py

Saves: keywords.json — loaded by main.py at startup.
"""

import pandas as pd
import pickle
import re
import json
import numpy as np

# ---- Same clean_text as train.py and main.py ----
def clean_text(text):
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

# ---- Load model and vectorizer ----
print("Loading model and vectorizer...")
model      = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---- Load and clean dataset ----
print("Loading dataset...")
data = pd.read_csv("Phishing_Email.csv")
data = data[["Email Text", "Email Type"]].dropna()
data["Email Type"] = data["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
data["Email Text"] = data["Email Text"].astype(str).apply(clean_text)

X = data["Email Text"]
y = data["Email Type"]

# ---- Compute SHAP values using model coefficients ----
# For LogisticRegression with TF-IDF, SHAP LinearExplainer is exact.
print("Computing SHAP values (this may take a minute)...")

try:
    import shap
    X_vec = vectorizer.transform(X)

    # Use a background sample for speed (500 samples)
    bg_size = min(500, X_vec.shape[0])
    np.random.seed(42)
    bg_idx  = np.random.choice(X_vec.shape[0], bg_size, replace=False)
    background = X_vec[bg_idx]

    explainer = shap.LinearExplainer(model, background)

    # Only compute SHAP on a small sample — 500 rows is enough to rank keywords reliably.
    # Computing on the full X_vec (tens of thousands of rows) causes OOM and gets killed.
    sample_size = min(500, X_vec.shape[0])
    sample_idx  = np.random.choice(X_vec.shape[0], sample_size, replace=False)
    X_sample    = X_vec[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    feature_names = vectorizer.get_feature_names_out()

    # shap_values shape: (n_samples, n_features)
    # Positive mean SHAP = pushes toward phishing (class 1)
    mean_shap = np.mean(shap_values, axis=0)

    # Build ranked list
    ranked = sorted(
        zip(feature_names, mean_shap),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 50 phishing words (positive SHAP, single words only — no bigrams for live highlight)
    top_phishing = [
        (word, float(score))
        for word, score in ranked
        if score > 0 and " " not in word and len(word) > 2
    ][:50]

    print(f"\nTop 20 phishing words by SHAP value:")
    for word, score in top_phishing[:20]:
        print(f"  {word:20s}  {score:+.4f}")

except ImportError:
    print("SHAP not installed. Falling back to model coefficients.")
    # Fallback: use logistic regression coefficients directly
    feature_names = vectorizer.get_feature_names_out()
    coefs         = model.coef_[0]

    ranked = sorted(
        zip(feature_names, coefs),
        key=lambda x: x[1],
        reverse=True
    )

    top_phishing = [
        (word, float(score))
        for word, score in ranked
        if score > 0 and " " not in word and len(word) > 2
    ][:50]

    print(f"\nTop 20 phishing words by model coefficient (SHAP fallback):")
    for word, score in top_phishing[:20]:
        print(f"  {word:20s}  {score:+.4f}")

# ---- Auto-generate recommendation messages ----
# Category rules: map word patterns to human-readable advice
CATEGORY_RULES = [
    (["password", "passwd", "pwd"],
     "Never send or request passwords through email."),
    (["verify", "verification", "validate", "validation"],
     "Requests to verify identity are a classic phishing tactic."),
    (["login", "signin", "logon"],
     "Avoid clicking login links in emails — go directly to the official site."),
    (["bank", "banking", "banker"],
     "Impersonating banks is one of the most common phishing techniques."),
    (["account", "accounts"],
     "Threats about account suspension are used to create panic."),
    (["urgent", "urgently", "immediately", "asap"],
     "Artificial urgency pressures users into acting without thinking."),
    (["click", "clicking"],
     "Legitimate organizations rarely ask you to click unverified links."),
    (["suspended", "suspension", "blocked", "locked"],
     "Account suspension threats are used to trick users into panic actions."),
    (["winner", "winning", "won", "prize", "prizes"],
     "Prize and lottery announcements are almost always scams."),
    (["free", "freebie"],
     "Unsolicited free offers are a red flag for phishing."),
    (["credit", "creditcard"],
     "Legitimate services never ask for credit card details via email."),
    (["confirm", "confirmation"],
     "Requests to confirm personal details are a phishing red flag."),
    (["update", "updates", "upgrade"],
     "Forced update requests via email are suspicious."),
    (["security", "secure", "secured"],
     "Fake security alerts are used to scare users into acting fast."),
    (["httplink"],
     "Email contains an insecure HTTP link (no SSL) — legitimate emails use HTTPS."),
    (["ssn", "social"],
     "No legitimate organization requests your Social Security Number via email."),
    (["payment", "payments", "pay"],
     "Unexpected payment requests should always be verified through official channels."),
    (["invoice", "invoices"],
     "Fake invoices are commonly used to trick users into opening attachments."),
    (["expire", "expiry", "expiration", "expired"],
     "Expiry warnings are used to create urgency and pressure quick action."),
    (["authorize", "authorization"],
     "Requests to authorize actions via email links are suspicious."),
    (["confirm", "confirmation"],
     "Requests to confirm personal details via email are a phishing red flag."),
    (["access", "restore", "recovery"],
     "Fake account recovery emails are used to steal credentials."),
    (["dear", "dearest"],
     "Generic greetings like 'Dear Customer' suggest mass phishing campaigns."),
    (["limited", "limit"],
     "'Limited time' language is a pressure tactic used in phishing."),
    (["claim", "claims"],
     "Claim requests are commonly used in prize scam phishing emails."),
    (["reward", "rewards"],
     "Fake reward offers are a common phishing lure."),
    (["notify", "notification"],
     "Unsolicited notifications asking for action should be treated with caution."),
    (["transfer", "wire"],
     "Wire transfer requests via email are a major fraud red flag."),
    (["protect", "protection"],
     "Fake protection warnings are used to trick users into clicking malicious links."),
    (["alert", "alerts"],
     "Fake security alerts are designed to cause panic and rushed decisions."),
]

def get_recommendation(word):
    for keywords, message in CATEGORY_RULES:
        if word.lower() in keywords:
            return message
    # Default generic message
    return f"The word '{word}' appears frequently in phishing emails — treat with caution."

# ---- Build final keyword dict ----
keywords_dict = {}
for word, shap_score in top_phishing:
    keywords_dict[word] = {
        "message": get_recommendation(word),
        "shap_score": round(shap_score, 5),
    }

# Also always include the httplink rule
if "httplink" not in keywords_dict:
    keywords_dict["httplink"] = {
        "message": "Email contains an insecure HTTP link (no SSL) — legitimate emails use HTTPS.",
        "shap_score": 0.0,
    }

# ---- Save ----
with open("keywords.json", "w") as f:
    json.dump(keywords_dict, f, indent=2)

print(f"\nSaved {len(keywords_dict)} keywords to keywords.json")
print("These are now used by main.py for recommendations AND the frontend for live highlighting.")
