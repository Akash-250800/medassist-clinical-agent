"""
MedAssist — Disease Classifier Training
=========================================
Trains an XGBoost classifier on Kaggle Disease-Symptom dataset.
Adds SHAP explainability for transparent predictions.

Dataset : Kaggle Disease Symptom Dataset (41 diseases, 131 symptoms)
Model   : XGBoost Classifier (multi-class)
XAI     : SHAP TreeExplainer
Output  : models/disease_classifier.pkl
          models/label_encoder.pkl
          models/symptom_columns.pkl

Run:
    python train_classifier.py

Python : 3.11.4
GPU    : GTX 1650 (CUDA 12.8)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Fix import paths ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("   MedAssist — Disease Classifier Training")
print("=" * 60)


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

DATA_DIR   = ROOT / "data" / "sample_docs"
MODELS_DIR = ROOT / "models"
PLOTS_DIR  = ROOT / "models" / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Step 1 — Load & Preprocess Dataset
# ─────────────────────────────────────────────

def load_and_preprocess():
    print("\n[DATA] Loading Kaggle dataset...")

    dataset_path = DATA_DIR / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset.csv not found at {dataset_path}\n"
            "Please download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset"
        )

    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]

    print(f"[DATA] Raw shape     : {df.shape}")
    print(f"[DATA] Diseases      : {df['Disease'].nunique()}")

    # ── Get all symptom columns ──
    symptom_cols = [c for c in df.columns if "Symptom" in c]
    print(f"[DATA] Symptom cols  : {len(symptom_cols)}")

    # ── Collect all unique symptoms ──
    all_symptoms = set()
    for col in symptom_cols:
        vals = df[col].dropna().unique()
        for v in vals:
            s = str(v).strip().lower().replace(" ", "_")
            if s and s != "nan":
                all_symptoms.add(s)

    all_symptoms = sorted(list(all_symptoms))
    print(f"[DATA] Unique symptoms: {len(all_symptoms)}")

    # ── Build binary feature matrix ──
    # Each row = one patient, each column = one symptom (1 if present, 0 if not)
    print("[DATA] Building binary feature matrix...")

    rows = []
    for _, row in df.iterrows():
        symptoms_present = set()
        for col in symptom_cols:
            val = str(row[col]).strip().lower().replace(" ", "_")
            if val and val != "nan":
                symptoms_present.add(val)

        feature_row = {sym: 1 if sym in symptoms_present else 0
                       for sym in all_symptoms}
        feature_row["disease"] = str(row["Disease"]).strip()
        rows.append(feature_row)

    processed_df = pd.DataFrame(rows)
    print(f"[DATA] Processed shape: {processed_df.shape}")

    # ── Encode labels ──
    le = LabelEncoder()
    y  = le.fit_transform(processed_df["disease"])
    X  = processed_df.drop(columns=["disease"]).values

    print(f"[DATA] Classes       : {len(le.classes_)}")
    print(f"[DATA] Features      : {X.shape[1]}")
    print(f"[DATA] Samples       : {X.shape[0]}")

    return X, y, le, all_symptoms


# ─────────────────────────────────────────────
# Step 2 — Train XGBoost Classifier
# ─────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test, le):
    print("\n[TRAIN] Training XGBoost Classifier...")
    print(f"[TRAIN] Train samples : {X_train.shape[0]}")
    print(f"[TRAIN] Test samples  : {X_test.shape[0]}")
    print(f"[TRAIN] Classes       : {len(le.classes_)}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate ──
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL] Test Accuracy  : {accuracy:.4f} ({accuracy:.1%})")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[EVAL] CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n[EVAL] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, accuracy


# ─────────────────────────────────────────────
# Step 3 — Train Random Forest (comparison)
# ─────────────────────────────────────────────

def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("\n[TRAIN] Training Random Forest (comparison)...")

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred   = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[EVAL] RF Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")

    return rf, accuracy


# ─────────────────────────────────────────────
# Step 4 — SHAP Explainability
# ─────────────────────────────────────────────

def generate_shap_analysis(model, X_test, symptom_names, le):
    print("\n[SHAP] Generating SHAP explainability analysis...")

    # Use a sample for SHAP (faster)
    sample_size = min(100, len(X_test))
    X_sample    = X_test[:sample_size]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Plot 1: Summary plot (top 20 features) ──
    print("[SHAP] Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=symptom_names,
        max_display=20,
        show=False,
        plot_type="bar"
    )
    plt.title("Top 20 Most Important Symptoms (SHAP)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved → models/plots/shap_summary.png")

    # ── Plot 2: Feature importance bar chart ──
    print("[SHAP] Generating feature importance plot...")

    # Safely compute mean absolute SHAP values — handles 2D and 3D arrays
    if isinstance(shap_values, list):
        # Multi-class: list of (n_samples, n_features) arrays
        stacked   = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # (n_classes, n_samples, n_features)
        mean_shap = stacked.mean(axis=0).mean(axis=0)                      # (n_features,)
    else:
        arr       = np.abs(np.array(shap_values))
        # Flatten any extra dimensions down to (n_features,)
        while arr.ndim > 1:
            arr = arr.mean(axis=0)
        mean_shap = arr

    mean_shap = mean_shap.flatten()                                        # guarantee 1D
    n_features = len(symptom_names)

    # Clamp top_n to available features
    top_n   = min(15, n_features)
    raw_idx = np.argsort(mean_shap)[-top_n:][::-1]
    top_idx = [int(i) for i in raw_idx if int(i) < n_features]            # safety filter
    top_n   = len(top_idx)                                                 # update after filter

    top_syms = [symptom_names[i].replace("_", " ").title() for i in top_idx]
    top_vals = np.array([mean_shap[i] for i in top_idx])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, top_n))
    bars    = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_syms[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Prediction)", fontsize=11)
    ax.set_title("Top 15 Diagnostic Symptoms — XAI Analysis\n(MedAssist Disease Classifier)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_vals[::-1])):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")

    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Feature importance plot saved → models/plots/feature_importance.png")

    return mean_shap, top_idx


# ─────────────────────────────────────────────
# Step 5 — Save Models
# ─────────────────────────────────────────────

def save_models(xgb_model, rf_model, le, symptom_names, xgb_acc, rf_acc):
    print("\n[SAVE] Saving models...")

    # Save XGBoost (primary)
    with open(MODELS_DIR / "disease_classifier.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"[SAVE] XGBoost model → models/disease_classifier.pkl")

    # Save Random Forest (backup)
    with open(MODELS_DIR / "rf_classifier.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print(f"[SAVE] Random Forest → models/rf_classifier.pkl")

    # Save label encoder
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"[SAVE] Label encoder → models/label_encoder.pkl")

    # Save symptom column names
    with open(MODELS_DIR / "symptom_columns.pkl", "wb") as f:
        pickle.dump(symptom_names, f)
    print(f"[SAVE] Symptom columns → models/symptom_columns.pkl")

    # Save metadata
    metadata = {
        "xgb_accuracy"    : xgb_acc,
        "rf_accuracy"     : rf_acc,
        "num_classes"     : len(le.classes_),
        "num_features"    : len(symptom_names),
        "classes"         : list(le.classes_),
        "model_type"      : "XGBoost",
        "dataset"         : "Kaggle Disease Symptom Dataset",
        "training_date"   : pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(MODELS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"[SAVE] Metadata → models/metadata.pkl")


# ─────────────────────────────────────────────
# Step 6 — Quick Prediction Test
# ─────────────────────────────────────────────

def test_prediction(model, le, symptom_names):
    print("\n[TEST] Testing prediction with sample symptoms...")

    # Simulate chest pain + shortness of breath patient
    test_symptoms = [
        "chest_pain", "shortness_of_breath",
        "sweating", "nausea", "fatigue"
    ]

    feature_vec = np.array([
        1 if s in test_symptoms else 0
        for s in symptom_names
    ]).reshape(1, -1)

    probs     = model.predict_proba(feature_vec)[0]
    top3_idx  = np.argsort(probs)[-3:][::-1]

    print(f"\n[TEST] Input symptoms : {', '.join(test_symptoms)}")
    print(f"[TEST] Top 3 Predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        disease = le.classes_[idx]
        conf    = probs[idx]
        print(f"   [{rank}] {disease:<35} {conf:.1%}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Step 1 — Load data
    X, y, le, symptom_names = load_and_preprocess()

    # Step 2 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3 — Train XGBoost
    xgb_model, xgb_acc = train_xgboost(X_train, X_test, y_train, y_test, le)

    # Step 4 — Train Random Forest
    rf_model, rf_acc = train_random_forest(X_train, X_test, y_train, y_test, le)

    # Step 5 — SHAP analysis
    generate_shap_analysis(xgb_model, X_test, symptom_names, le)

    # Step 6 — Save everything
    save_models(xgb_model, rf_model, le, symptom_names, xgb_acc, rf_acc)

    # Step 7 — Test prediction
    test_prediction(xgb_model, le, symptom_names)

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("   Training Complete!")
    print("=" * 60)
    print(f"   XGBoost Accuracy  : {xgb_acc:.1%}")
    print(f"   RandomForest Acc  : {rf_acc:.1%}")
    print(f"   Classes trained   : {len(le.classes_)}")
    print(f"   Features used     : {len(symptom_names)}")
    print(f"   Models saved to   : models/")
    print(f"   SHAP plots saved  : models/plots/")
    print("=" * 60)
    print("\n   Next: python agents/clinical_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
