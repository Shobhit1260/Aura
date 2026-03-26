from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.eye_disease_service import EyeDiseaseModelService
from app.preprocess import (
    decode_image,
    preprocess_image,
    validate_tb_xray_image,
)
from app.schemas import HealthResponse, PredictionResponse
from app.settings import Settings
from app.tflite_service import TFLiteModelService


def build_router(
    settings: Settings,
    tb_service: TFLiteModelService,
    tb_xray_gate_service: TFLiteModelService,
    eye_disease_service: EyeDiseaseModelService,
) -> APIRouter:
    router = APIRouter()
    verify_api_key = _build_api_key_dependency(settings)

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app=settings.app_name,
            tb_model_loaded=tb_service.is_loaded,
            tb_xray_gate_model_loaded=tb_xray_gate_service.is_loaded,
            eye_disease_model_loaded=eye_disease_service.is_loaded,
        )

    @router.post("/predict/tb", response_model=PredictionResponse, tags=["prediction"])
    async def predict_tb(
        file: UploadFile = File(...),
        _: None = Depends(verify_api_key),
    ) -> PredictionResponse:
        return await _predict(
            task="tb",
            file=file,
            model_service=tb_service,
            input_size=settings.tb_input_size,
            threshold=settings.tb_threshold,
            positive_label=settings.tb_label_positive,
            negative_label=settings.tb_label_negative,
            fallback_label=settings.default_fallback_label,
            positive_class_index=settings.tb_positive_class_index,
            strict_xray_validation=settings.tb_strict_xray_validation,
            xray_min_side=settings.tb_xray_min_side,
            xray_gate_service=tb_xray_gate_service,
            xray_gate_enabled=settings.tb_xray_gate_enabled,
            xray_gate_required=settings.tb_xray_gate_required,
            xray_gate_input_size=settings.tb_xray_gate_input_size,
            xray_gate_threshold=settings.tb_xray_gate_threshold,
            xray_gate_positive_class_index=settings.tb_xray_gate_positive_class_index,
        )

    @router.post("/predict/eye-disease", response_model=PredictionResponse, tags=["prediction"])
    async def predict_eye_disease(
        file: UploadFile = File(...),
        _: None = Depends(verify_api_key),
    ) -> PredictionResponse:
        return await _predict_eye_disease(
            task="eye_disease",
            file=file,
            model_service=eye_disease_service,
            input_size=settings.eye_input_size,
            min_confidence=settings.eye_min_confidence,
            fallback_label=settings.eye_label_fallback,
            default_fallback_label=settings.default_fallback_label,
        )

    return router


def _build_api_key_dependency(settings: Settings):
    async def verify_api_key(
        api_key: Optional[str] = Header(default=None, alias=settings.api_key_header_name),
    ) -> None:
        if not settings.api_key_enabled:
            return

        if not api_key or api_key != settings.api_key_value:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key


async def _predict(
    task: str,
    file: UploadFile,
    model_service: TFLiteModelService,
    input_size: int,
    threshold: float,
    positive_label: str,
    negative_label: str,
    fallback_label: str,
    positive_class_index: int,
    strict_xray_validation: bool,
    xray_min_side: int,
    xray_gate_service: TFLiteModelService | None,
    xray_gate_enabled: bool,
    xray_gate_required: bool,
    xray_gate_input_size: int,
    xray_gate_threshold: float,
    xray_gate_positive_class_index: int,
) -> PredictionResponse:
    if task == "tb":
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="TB endpoint accepts image files only.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if task == "tb" and strict_xray_validation:
        is_valid_xray, reason = validate_tb_xray_image(image, min_side=xray_min_side)
        if not is_valid_xray:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid photo (hard checks): please upload a clear chest X-ray image. "
                    f"Reason: {reason}"
                ),
            )

    if task == "tb" and xray_gate_enabled:
        if xray_gate_service is None or not xray_gate_service.is_loaded:
            if xray_gate_required:
                raise HTTPException(
                    status_code=503,
                    detail="X-ray gate model is required but not loaded.",
                )
        else:
            gate_input = preprocess_image(image, input_size=xray_gate_input_size, task="tb")
            gate_score = xray_gate_service.predict(gate_input)
            if xray_gate_positive_class_index == 0:
                gate_score = 1.0 - gate_score

            if gate_score < xray_gate_threshold:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid photo (classifier): not a valid chest X-ray. "
                        f"Score={gate_score:.3f}, required>={xray_gate_threshold:.3f}."
                    ),
                )

    note = (
        "AI-assisted screening only. Not a diagnosis. "
        "Confirm with clinical tests and professional evaluation."
    )

    if not model_service.is_loaded:
        confidence = 0.5

        risk = positive_label if confidence >= threshold else fallback_label
        return PredictionResponse(
            task=task,
            risk=risk,
            confidence=confidence,
            threshold=threshold,
            model_loaded=False,
            note=note,
        )

    model_input = preprocess_image(image, input_size=input_size, task=task)
    score = model_service.predict(model_input)
    if task == "tb" and positive_class_index == 0:
        score = 1.0 - score

    risk = positive_label if score >= threshold else negative_label

    return PredictionResponse(
        task=task,
        risk=risk,
        confidence=score,
        threshold=threshold,
        model_loaded=True,
        note=note,
    )


async def _predict_eye_disease(
    task: str,
    file: UploadFile,
    model_service: EyeDiseaseModelService,
    input_size: int,
    min_confidence: float,
    fallback_label: str,
    default_fallback_label: str,
) -> PredictionResponse:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Eye-disease endpoint accepts image files only.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    note = (
        "AI-assisted screening only. Not a diagnosis. "
        "Confirm with clinical tests and professional evaluation."
    )

    if not model_service.is_loaded:
        return PredictionResponse(
            task=task,
            risk=default_fallback_label,
            confidence=0.0,
            threshold=min_confidence,
            model_loaded=False,
            note=note,
            probabilities=None,
        )

    model_input = preprocess_image(image, input_size=input_size, task=task)
    top_label, top_score, class_probs = model_service.predict(model_input)

    risk = top_label if top_score >= min_confidence else fallback_label

    return PredictionResponse(
        task=task,
        risk=risk,
        confidence=top_score,
        threshold=min_confidence,
        model_loaded=True,
        note=note,
        probabilities=class_probs,
    )
