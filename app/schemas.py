from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    task: str = Field(description="Prediction task identifier", examples=["tb", "eye_disease"])
    risk: str = Field(description="Risk bucket label")
    confidence: float = Field(description="Model score in [0,1]")
    threshold: float = Field(description="Threshold used for risk decision")
    model_loaded: bool = Field(description="Whether TFLite model inference was used")
    note: str = Field(description="Regulatory/safety note")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Optional class probability distribution for multiclass tasks",
    )


class HealthResponse(BaseModel):
    status: str
    app: str
    tb_model_loaded: bool
    tb_xray_gate_model_loaded: bool
    eye_disease_model_loaded: bool
