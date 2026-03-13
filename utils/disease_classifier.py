"""
MedAssist — Disease Classifier Module
=======================================
Loads trained XGBoost model and runs predictions.
Integrates SHAP for per-prediction explanations.
Plugs into LangGraph as Node 2.5 between RAG and Symptom Analysis.

Usage:
    clf = DiseaseClassifier()
    result = clf.predict("patient has chest pain, sweating, nausea")
    print(result["top_predictions"])
    print(result["shap_explanation"])
"""

import os
import re
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class DiseaseClassifier:
    """
    Trained XGBoost disease classifier with SHAP explainability.

    Loads models from models/ directory.
    Extracts symptoms from free text using keyword matching.
    Returns top-3 predictions with confidence + SHAP explanations.
    """

    def __init__(self):
        self.model         = None
        self.le            = None
        self.symptom_cols  = None
        self.metadata      = None
        self.shap_explainer= None
        self._loaded       = False
        self._load_models()

    # ─────────────────────────────────────────
    # Load Models
    # ─────────────────────────────────────────

    def _load_models(self):
        """Load trained models from models/ directory."""
        models_dir = ROOT / "models"

        required = [
            "disease_classifier.pkl",
            "label_encoder.pkl",
            "symptom_columns.pkl",
        ]

        # Check all files exist
        missing = [f for f in required if not (models_dir / f).exists()]
        if missing:
            print(f"[CLASSIFIER]  Models not found: {missing}")
            print(f"[CLASSIFIER]  Run: python train_classifier.py first")
            return

        try:
            with open(models_dir / "disease_classifier.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open(models_dir / "label_encoder.pkl", "rb") as f:
                self.le = pickle.load(f)

            with open(models_dir / "symptom_columns.pkl", "rb") as f:
                self.symptom_cols = pickle.load(f)

            meta_path = models_dir / "metadata.pkl"
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)

            self._loaded = True
            acc = self.metadata.get("xgb_accuracy", 0) if self.metadata else 0
            print(f"[CLASSIFIER] Model loaded | Accuracy: {acc:.1%} | "
                  f"Classes: {len(self.le.classes_)} | "
                  f"Features: {len(self.symptom_cols)}")

            # Init SHAP explainer
            try:
                import shap
                self.shap_explainer = shap.TreeExplainer(self.model)
                print(f"[CLASSIFIER] SHAP explainer ready")
            except Exception as e:
                print(f"[CLASSIFIER]  SHAP not available: {e}")

        except Exception as e:
            print(f"[CLASSIFIER] Load error: {e}")

    # ─────────────────────────────────────────
    # Extract Symptoms from Text
    # ─────────────────────────────────────────

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptom keywords from free-text patient report.
        Matches against trained symptom vocabulary.
        """
        if not self._loaded:
            return []

        text_clean = text.lower().replace("-", "_").replace(" ", "_")

        # Also try space-separated matching
        text_words = set(re.findall(r'\b\w+\b', text.lower()))

        found = []
        for sym in self.symptom_cols:
            sym_clean = sym.lower().strip()
            sym_words = set(sym_clean.replace("_", " ").split())

            # Direct match
            if sym_clean in text_clean:
                found.append(sym)
                continue

            # Word overlap match (for multi-word symptoms)
            if len(sym_words) > 1 and sym_words.issubset(text_words):
                found.append(sym)
                continue

            # Single word match
            if len(sym_words) == 1 and sym_clean.replace("_", "") in text.lower().replace(" ", ""):
                found.append(sym)

        return found

    # ─────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────

    def predict(self, patient_text: str, top_k: int = 3) -> Dict:
        """
        Run disease prediction on patient text.

        Args:
            patient_text : Free-text patient report
            top_k        : Number of top predictions to return

        Returns:
            {
                "top_predictions": [
                    {"disease": str, "confidence": float, "rank": int}
                ],
                "detected_symptoms": [str],
                "shap_explanation": str,
                "model_accuracy": float,
                "classifier_available": bool
            }
        """
        if not self._loaded:
            return {
                "top_predictions":     [],
                "detected_symptoms":   [],
                "shap_explanation":    "Classifier not available — run train_classifier.py",
                "model_accuracy":      0.0,
                "classifier_available": False
            }

        # Extract symptoms from text
        detected = self.extract_symptoms(patient_text)

        # Build feature vector
        feature_vec = np.array([
            1 if sym in detected else 0
            for sym in self.symptom_cols
        ]).reshape(1, -1)

        # Predict probabilities
        probs    = self.model.predict_proba(feature_vec)[0]
        top_idx  = np.argsort(probs)[-top_k:][::-1]

        predictions = []
        for rank, idx in enumerate(top_idx, 1):
            predictions.append({
                "disease"   : self.le.classes_[idx],
                "confidence": round(float(probs[idx]), 4),
                "rank"      : rank
            })

        # SHAP explanation
        shap_text = self._get_shap_explanation(feature_vec, detected, predictions)

        acc = self.metadata.get("xgb_accuracy", 0.0) if self.metadata else 0.0

        return {
            "top_predictions"      : predictions,
            "detected_symptoms"    : detected,
            "shap_explanation"     : shap_text,
            "model_accuracy"       : acc,
            "classifier_available" : True
        }

    # ─────────────────────────────────────────
    # SHAP Explanation
    # ─────────────────────────────────────────

    def _get_shap_explanation(
        self,
        feature_vec: np.ndarray,
        detected_symptoms: List[str],
        predictions: List[Dict]
    ) -> str:
        """Generate human-readable SHAP explanation."""

        if not detected_symptoms:
            return "No recognizable symptoms detected in the input text."

        lines = []

        # Detected symptoms
        sym_display = [s.replace("_", " ").title() for s in detected_symptoms[:8]]
        lines.append(f"Detected Symptoms ({len(detected_symptoms)} found):")
        lines.append("  " + ", ".join(sym_display))
        lines.append("")

        # Top prediction
        if predictions:
            top = predictions[0]
            lines.append(f"Primary Prediction: {top['disease']} ({top['confidence']:.1%} confidence)")
            lines.append("")

        # SHAP feature importance for this prediction
        if self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(feature_vec)

                # Get SHAP values for top prediction class
                top_class_idx = self.le.transform([predictions[0]["disease"]])[0] if predictions else 0

                if isinstance(shap_vals, list):
                    sv = shap_vals[top_class_idx][0]
                else:
                    sv = shap_vals[0]

                # Only show symptoms that were detected (non-zero features)
                active_indices = [i for i, v in enumerate(feature_vec[0]) if v == 1]
                if active_indices:
                    active_shap = [(self.symptom_cols[i], sv[i]) for i in active_indices]
                    active_shap.sort(key=lambda x: abs(x[1]), reverse=True)

                    lines.append("Key Driving Factors (XAI):")
                    for sym, val in active_shap[:5]:
                        direction = "↑ increases" if val > 0 else "↓ decreases"
                        sym_name  = sym.replace("_", " ").title()
                        lines.append(f"  • {sym_name}: {direction} prediction likelihood (SHAP: {val:+.3f})")

            except Exception as e:
                lines.append(f"SHAP analysis: {str(e)[:60]}")
        else:
            # Fallback: simple symptom-based explanation
            lines.append("Key Driving Factors:")
            for sym in detected_symptoms[:5]:
                lines.append(f"  • {sym.replace('_', ' ').title()}: detected and contributing to prediction")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # Batch Predict
    # ─────────────────────────────────────────

    def predict_from_symptoms(self, symptom_list: List[str], top_k: int = 3) -> Dict:
        """
        Predict from a list of symptom strings directly.

        Args:
            symptom_list : List of symptom names
            top_k        : Number of predictions

        Returns: Same format as predict()
        """
        if not self._loaded:
            return {"top_predictions": [], "classifier_available": False}

        # Normalize input
        normalized = [s.lower().strip().replace(" ", "_") for s in symptom_list]

        feature_vec = np.array([
            1 if sym in normalized else 0
            for sym in self.symptom_cols
        ]).reshape(1, -1)

        probs   = self.model.predict_proba(feature_vec)[0]
        top_idx = np.argsort(probs)[-top_k:][::-1]

        predictions = [{
            "disease"   : self.le.classes_[idx],
            "confidence": round(float(probs[idx]), 4),
            "rank"      : rank + 1
        } for rank, idx in enumerate(top_idx)]

        shap_text = self._get_shap_explanation(feature_vec, normalized, predictions)
        acc = self.metadata.get("xgb_accuracy", 0.0) if self.metadata else 0.0

        return {
            "top_predictions"      : predictions,
            "detected_symptoms"    : normalized,
            "shap_explanation"     : shap_text,
            "model_accuracy"       : acc,
            "classifier_available" : True
        }

    def is_available(self) -> bool:
        return self._loaded

    def get_all_diseases(self) -> List[str]:
        return list(self.le.classes_) if self._loaded else []

    def get_model_info(self) -> Dict:
        if not self.metadata:
            return {}
        return self.metadata
