from __future__ import annotations

from pathlib import Path

import numpy as np


class AnemiaSymptomModelService:
    def __init__(self, model_path: str, feature_names: list[str]) -> None:
        self.model_path = model_path
        self.feature_names = feature_names
        self.model = None

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            return False

        try:
            import joblib
        except Exception:
            return False

        self.model = joblib.load(self.model_path)
        return True

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, symptoms: dict[str, int]) -> float:
        vector = np.array([[float(symptoms[name]) for name in self.feature_names]], dtype=np.float32)

        if self.is_loaded:
            # Prefer calibrated probabilities from the trained logistic model.
            score = float(self.model.predict_proba(vector)[0][1])
            return max(0.0, min(1.0, score))

        # Fallback weighted score for demo continuity when model file is absent.
        weights = {
            "fatigue": 0.22,
            "pale_skin": 0.25,
            "dizziness": 0.18,
            "shortness_of_breath": 0.16,
            "headache": 0.10,
            "cold_hands_feet": 0.09,
        }
        score = 0.1
        for name, value in symptoms.items():
            score += weights.get(name, 0.0) * float(value)

        return max(0.0, min(1.0, score))
