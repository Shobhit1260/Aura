from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.anemia_symptom_service import AnemiaSymptomModelService
from app.api import build_router
from app.settings import get_settings
from app.tflite_service import TFLiteModelService

settings = get_settings()


def _resolve_tb_positive_class_index(default_index: int) -> int:
    model_path = Path(settings.tb_model_path)
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    if not metadata_path.exists():
        return default_index

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("class_names")
        if isinstance(class_names, list):
            lowered = [str(name).strip().lower() for name in class_names]
            if "tb" in lowered:
                return lowered.index("tb")
    except Exception:
        return default_index

    return default_index


settings.tb_positive_class_index = _resolve_tb_positive_class_index(settings.tb_positive_class_index)

tb_service = TFLiteModelService(settings.tb_model_path)
anemia_service = AnemiaSymptomModelService(
    settings.anemia_symptom_model_path,
    settings.anemia_symptom_features_list,
)

tb_service.load()
anemia_service.load()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "FastAPI backend for AI-assisted screening using TB MobileNetV2 image model and anemia symptom model. "
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

app.include_router(build_router(settings, tb_service, anemia_service))
