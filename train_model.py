import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

# ===============================
# CONFIG
# ===============================
CSV_PATH = "collected_eeg_data.csv"  # <-- change if different file
MODEL_OUT = "brain_control_model.pkl"
SCALER_OUT = "brain_scaler.pkl"
ENCODER_OUT = "brain_label_encoder.pkl"

TEST_SIZE = 0.25→ → →  → → → → → → ← → → → → → → → → → → → → ← → → → → → ← ← → → ← → → → → → → → → → → 
RANDOM_STATE = 42
N_TREES = 200

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH)
print("✅ Loaded:", df.shape)

if "label" not in df.columns:
    raise ValueError("CSV missing 'label' column")
if "attempt_id" not in df.columns:
    raise ValueError("CSV missing 'attempt_id' column")

# Standardize label formatting
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Bandpow features
alpha = df["alpha"].astype(float)
beta  = df["beta"].astype(float)
theta = df["theta"].astype(float)
gamma = df["gamma"].astype(float)

eps = 1e-9

# ===============================
# Build final feature matrix (11 features)
# Matching what brain_typing_live.py expects
# ===============================
X = pd.DataFrame({
    "alpha": alpha,
    "beta": beta,
    "theta": theta,
    "gamma": gamma,
    "beta_alpha": (beta+eps)/(alpha+eps),
    "alpha_theta": (alpha+eps)/(theta+eps),
    "gamma_beta": (gamma+eps)/(beta+eps),
    "log_alpha": np.log10(alpha+eps),
    "log_beta": np.log10(beta+eps),
    "log_theta": np.log10(theta+eps),
    "log_gamma": np.log10(gamma+eps)
})

print("✅ Feature matrix:", X.shape)

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
classes = list(label_encoder.classes_)
print("✅ Classes:", classes)

# ===============================
# Attempt-safe splitting
# ===============================
attempts = df["attempt_id"].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(attempts)

split = int(len(attempts)*(1 - TEST_SIZE))
train_attempts = set(attempts[:split])
test_attempts = set(attempts[split:])

train_mask = df["attempt_id"].isin(train_attempts)
test_mask = df["attempt_id"].isin(test_attempts)

X_train = X[train_mask].values
y_train = y[train_mask]
X_test = X[test_mask].values
y_test = y[test_mask]

print(f"Train windows: {len(X_train)}")
print(f"Test windows: {len(X_test)}")

# ===============================
# Model Training
# ===============================
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
cw = {cls: w for cls, w in zip(np.unique(y), class_weights)}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight=cw,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("\n🚀 Training model...")
pipeline.fit(X_train, y_train)
print("✅ Training complete")

# ===============================
# Evaluation
# ===============================
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="macro")

print("\n🎯 Accuracy:", round(acc, 3), "| F1 Macro:", round(f1, 3))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))
print("\n🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===============================
# Save model
# ===============================
joblib.dump(pipeline, MODEL_OUT)
joblib.dump(label_encoder, ENCODER_OUT)
joblib.dump(pipeline.named_steps["scaler"], SCALER_OUT)

print("\n✅ Saved model and scaler")
print("🎉 Ready for real-time BCI!")
