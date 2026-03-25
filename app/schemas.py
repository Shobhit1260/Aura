from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    task: str = Field(description="Prediction task identifier", examples=["tb", "anemia"])
    risk: str = Field(description="Risk bucket label")
    confidence: float = Field(description="Model score in [0,1]")
    threshold: float = Field(description="Threshold used for risk decision")
    model_loaded: bool = Field(description="Whether TFLite model inference was used")
    note: str = Field(description="Regulatory/safety note")


class HealthResponse(BaseModel):
    status: str
    app: str
    tb_model_loaded: bool
    anemia_model_loaded: bool


class AnemiaSymptomsRequest(BaseModel):
    fatigue: int = Field(ge=0, le=1, description="1 if symptom present else 0")
    pale_skin: int = Field(ge=0, le=1, description="1 if symptom present else 0")
    dizziness: int = Field(ge=0, le=1, description="1 if symptom present else 0")
    shortness_of_breath: int = Field(ge=0, le=1, description="1 if symptom present else 0")
    headache: int = Field(ge=0, le=1, description="1 if symptom present else 0")
    cold_hands_feet: int = Field(ge=0, le=1, description="1 if symptom present else 0")
