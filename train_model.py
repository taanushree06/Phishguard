import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extraction import extract_features

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")

# Clean URLs (removes leading/trailing spaces like ' https://amazon.sg')
df["url"] = df["url"].str.strip()

# Drop any rows with missing values
df.dropna(subset=["url", "label"], inplace=True)

print(f"   Total URLs: {len(df)}")
print(f"   Legitimate (0): {(df['label'] == 0).sum()}")
print(f"   Phishing   (1): {(df['label'] == 1).sum()}")

# ─────────────────────────────────────────────
# 2. EXTRACT FEATURES
# ─────────────────────────────────────────────
print("\n⚙️  Extracting features...")
X = np.array([extract_features(url) for url in df["url"]])
y = df["label"].values

print(f"   Feature matrix shape: {X.shape}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 4. TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
print("\n🚀 Training XGBoost model...")

# Automatically handle class imbalance
num_legit    = (y_train == 0).sum()
num_phishing = (y_train == 1).sum()
scale_weight = num_legit / num_phishing   # e.g. if 90% legit, weight = 9.0

print(f"   Class weight applied: {scale_weight:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,        # ← this is the fix
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
print("\n📈 Evaluation on test set:")
y_pred = model.predict(X_test)
print(f"   Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n" + classification_report(y_test, y_pred,
      target_names=["Legitimate", "Phishing"]))

# Warn if phishing recall is low
from sklearn.metrics import recall_score
phishing_recall = recall_score(y_test, y_pred)
if phishing_recall < 0.7:
    print("⚠️  WARNING: Phishing recall is low — model may miss many phishing URLs!")
else:
    print("✅ Phishing recall looks good!")
# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
pickle.dump(model, open("xgb_phishing_model.pkl", "wb"))
print("✅ Model saved as xgb_phishing_model.pkl")