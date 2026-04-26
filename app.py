from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import re
from urllib.parse import urlparse
with open("phishing_model.pkl","rb")as f:
    model=pickle.load(f)

from feature_extraction import (
    extract_features,
    SUSPICIOUS_TLDS,
    BRAND_KEYWORDS,
    SENSITIVE_WORDS,
)

app = Flask(__name__, template_folder="templates")
CORS(app)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
try:
    model = pickle.load(open("xgb_phishing_model.pkl", "rb"))
    print("✅ XGBoost model loaded successfully.")
except FileNotFoundError:
    model = None
    print("⚠️  Model not found — run train_model.py first.")


# ─────────────────────────────────────────────
# RULE-BASED BOOST
# Catches obvious phishing even if ML is uncertain
# ─────────────────────────────────────────────
def rule_boost(url: str, score: int) -> int:
    low  = url.lower()
    host = urlparse(url).netloc.lower()

    # Raw IP address as hostname
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        score = max(score, 90)

    # Known abused TLD
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        score = max(score, 80)

    # Brand name + action word combo (classic phishing pattern)
    brand_hit  = any(b in low for b in ["paypal", "google", "amazon",
                                         "facebook", "apple", "netflix",
                                         "microsoft", "sbi", "bank"])
    action_hit = any(w in low for w in ["login", "verify", "secure",
                                         "update", "confirm", "account"])
    if brand_hit and action_hit:
        score = max(score, 88)

    # Plain HTTP
    if url.startswith("http://"):
        score = max(score, 60)

    # Typosquatting patterns
    if re.search(r"(g00gle|paypa1|amaz0n|faceb00k|microso1t|app1e)", low):
        score = max(score, 92)

    return min(score, 100)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        url  = (data.get("url") or "").strip()

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Add scheme if missing so urlparse works correctly
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # ── Feature extraction ──
        features = np.array(extract_features(url)).reshape(1, -1)

        # ── ML prediction ──
        if model is not None:
            prob       = float(model.predict_proba(features)[0][1])
            risk_score = int(prob * 100)
        else:
            risk_score = 50   # fallback when model is missing

        # ── Rule boost ──
        risk_score = rule_boost(url, risk_score)

        # ── Verdict ──
        if risk_score >= 65:
            verdict = "PHISHING"
        elif risk_score >= 35:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        print(f"[PREDICT] {url}  →  score={risk_score}  verdict={verdict}")

        return jsonify({
            "risk_score": risk_score,
            "verdict"   : verdict,
            "url"       : url,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)