from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.anemia_symptom_service import AnemiaSymptomModelService
from app.preprocess import (
    decode_image,
    preprocess_image,
    simple_brightness_heuristic,
    validate_tb_xray_image,
)
from app.schemas import AnemiaSymptomsRequest, HealthResponse, PredictionResponse
from app.settings import Settings
from app.tflite_service import TFLiteModelService


def build_router(
    settings: Settings,
    tb_service: TFLiteModelService,
    anemia_service: AnemiaSymptomModelService,
) -> APIRouter:
    router = APIRouter()
    verify_api_key = _build_api_key_dependency(settings)

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app=settings.app_name,
            tb_model_loaded=tb_service.is_loaded,
            anemia_model_loaded=anemia_service.is_loaded,
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
            use_heuristics=settings.enable_simple_heuristics,
            positive_class_index=settings.tb_positive_class_index,
            strict_xray_validation=settings.tb_strict_xray_validation,
        )

    @router.post("/predict/anemia", response_model=PredictionResponse, tags=["prediction"])
    async def predict_anemia(
        payload: AnemiaSymptomsRequest,
        _: None = Depends(verify_api_key),
    ) -> PredictionResponse:
        return _predict_anemia_from_symptoms(
            task="anemia",
            symptoms=payload.model_dump(),
            model_service=anemia_service,
            threshold=settings.anemia_threshold,
            positive_label=settings.anemia_label_positive,
            negative_label=settings.anemia_label_negative,
            fallback_label=settings.default_fallback_label,
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
    use_heuristics: bool,
    positive_class_index: int,
    strict_xray_validation: bool,
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
        is_valid_xray, reason = validate_tb_xray_image(image, min_side=input_size)
        if not is_valid_xray:
            raise HTTPException(status_code=400, detail=f"Invalid TB input: {reason}")

    note = (
        "AI-assisted screening only. Not a diagnosis. "
        "Confirm with clinical tests and professional evaluation."
    )

    if not model_service.is_loaded:
        confidence = 0.5
        if use_heuristics:
            brightness = simple_brightness_heuristic(image)
            if task == "anemia" and brightness < 0.25:
                confidence = 0.75
            elif task == "anemia" and brightness > 0.8:
                confidence = 0.25

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


def _predict_anemia_from_symptoms(
    task: str,
    symptoms: dict[str, int],
    model_service: AnemiaSymptomModelService,
    threshold: float,
    positive_label: str,
    negative_label: str,
    fallback_label: str,
) -> PredictionResponse:
    note = (
        "AI-assisted screening only. Not a diagnosis. "
        "Confirm with clinical tests and professional evaluation."
    )

    score = model_service.predict(symptoms)
    if not model_service.is_loaded and abs(score - threshold) < 0.05:
        risk = fallback_label
    else:
        risk = positive_label if score >= threshold else negative_label

    return PredictionResponse(
        task=task,
        risk=risk,
        confidence=score,
        threshold=threshold,
        model_loaded=model_service.is_loaded,
        note=note,
    )
