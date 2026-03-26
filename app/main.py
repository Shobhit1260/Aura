from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import build_router
from app.eye_disease_service import EyeDiseaseModelService
from app.settings import get_settings
from app.tflite_service import TFLiteModelService

settings = get_settings()


def _resolve_positive_class_index(
    model_path: str,
    default_index: int,
    positive_candidates: list[str],
) -> int:
    model_path_obj = Path(model_path)
    metadata_path = model_path_obj.with_name(f"{model_path_obj.stem}_metadata.json")
    candidates = {name.strip().lower() for name in positive_candidates if name.strip()}
    if not metadata_path.exists():
        return default_index

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("class_names")
        if isinstance(class_names, list):
            lowered = [str(name).strip().lower() for name in class_names]
            for candidate in candidates:
                if candidate in lowered:
                    return lowered.index(candidate)
    except Exception:
        return default_index

    return default_index


settings.tb_positive_class_index = _resolve_positive_class_index(
    settings.tb_model_path,
    settings.tb_positive_class_index,
    ["tb"],
)
settings.tb_xray_gate_positive_class_index = _resolve_positive_class_index(
    settings.tb_xray_gate_model_path,
    settings.tb_xray_gate_positive_class_index,
    ["valid_xray", "chest_xray", "xray", "valid"],
)

tb_service = TFLiteModelService(settings.tb_model_path)
tb_xray_gate_service = TFLiteModelService(settings.tb_xray_gate_model_path)
eye_disease_service = EyeDiseaseModelService(
    settings.eye_model_path,
    settings.eye_default_class_names_list,
)

tb_service.load()
tb_xray_gate_service.load()
eye_disease_service.load()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "FastAPI backend for AI-assisted screening using TB and eye-disease image models. "
        "This API is for preliminary screening and not for medical diagnosis."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)

app.include_router(build_router(settings, tb_service, tb_xray_gate_service, eye_disease_service))
