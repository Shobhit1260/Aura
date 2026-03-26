from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.tflite_service import TFLiteModelService


class EyeDiseaseModelService:
    def __init__(self, model_path: str, default_class_names: list[str]) -> None:
        self.model_path = model_path
        self.class_names = default_class_names
        self.model = TFLiteModelService(model_path)

    def load(self) -> bool:
        loaded = self.model.load()
        if loaded:
            self._load_metadata_class_names()
        return loaded

    @property
    def is_loaded(self) -> bool:
        return self.model.is_loaded

    def _load_metadata_class_names(self) -> None:
        model_path_obj = Path(self.model_path)
        metadata_path = model_path_obj.with_name(f"{model_path_obj.stem}_metadata.json")
        if not metadata_path.exists():
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            class_names = metadata.get("class_names")
            if isinstance(class_names, list):
                cleaned = [str(name).strip() for name in class_names if str(name).strip()]
                if cleaned:
                    self.class_names = cleaned
        except Exception:
            return

    def predict(self, model_input: np.ndarray) -> tuple[str, float, dict[str, float]]:
        probabilities = self.model.predict_probabilities(model_input)

        if len(self.class_names) != int(probabilities.size):
            labels = [f"class_{idx}" for idx in range(int(probabilities.size))]
        else:
            labels = self.class_names

        top_index = int(np.argmax(probabilities))
        top_label = labels[top_index]
        top_score = float(probabilities[top_index])

        distribution = {
            labels[idx]: float(probabilities[idx])
            for idx in range(int(probabilities.size))
        }

        return top_label, top_score, distribution
