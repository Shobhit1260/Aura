from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class TFLiteModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            return False

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception:
            try:
                import tensorflow as tf  # type: ignore

                Interpreter = tf.lite.Interpreter
            except Exception:
                return False

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        return True

    @property
    def is_loaded(self) -> bool:
        return self.interpreter is not None

    def predict(self, model_input: np.ndarray) -> float:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        expected_dtype = self.input_details[0]["dtype"]
        tensor_input = model_input.astype(expected_dtype, copy=False)

        with self._lock:
            self.interpreter.set_tensor(self.input_details[0]["index"], tensor_input)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])

        flat = np.ravel(output).astype(np.float32)

        if flat.size == 1:
            score = float(flat[0])
        elif flat.size >= 2:
            # For 2-class outputs use class-1 probability when available.
            if np.all(flat >= 0.0) and np.all(flat <= 1.0) and abs(float(flat.sum()) - 1.0) < 1e-3:
                score = float(flat[1])
            else:
                shifted = flat - np.max(flat)
                exp_vals = np.exp(shifted)
                probs = exp_vals / np.sum(exp_vals)
                score = float(probs[1])
        else:
            raise RuntimeError("Unexpected empty model output")

        if score < 0.0 or score > 1.0:
            score = float(1.0 / (1.0 + np.exp(-score)))

        return float(np.clip(score, 0.0, 1.0))
