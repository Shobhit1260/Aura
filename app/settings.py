from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="TB and Eye Disease Screening API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    tb_model_path: str = Field(default="models/tb_model.tflite", alias="TB_MODEL_PATH")
    tb_input_size: int = Field(default=224, alias="TB_INPUT_SIZE")
    tb_threshold: float = Field(default=0.5, alias="TB_THRESHOLD")
    tb_label_positive: str = Field(default="TB High Risk", alias="TB_LABEL_POSITIVE")
    tb_label_negative: str = Field(default="TB Low Risk", alias="TB_LABEL_NEGATIVE")
    tb_positive_class_index: int = Field(default=1, alias="TB_POSITIVE_CLASS_INDEX")
    tb_strict_xray_validation: bool = Field(default=True, alias="TB_STRICT_XRAY_VALIDATION")
    tb_xray_min_side: int = Field(default=224, alias="TB_XRAY_MIN_SIDE")

    tb_xray_gate_enabled: bool = Field(default=True, alias="TB_XRAY_GATE_ENABLED")
    tb_xray_gate_required: bool = Field(default=False, alias="TB_XRAY_GATE_REQUIRED")
    tb_xray_gate_model_path: str = Field(default="models/tb_xray_gate_model.tflite", alias="TB_XRAY_GATE_MODEL_PATH")
    tb_xray_gate_input_size: int = Field(default=224, alias="TB_XRAY_GATE_INPUT_SIZE")
    tb_xray_gate_threshold: float = Field(default=0.5, alias="TB_XRAY_GATE_THRESHOLD")
    tb_xray_gate_positive_class_index: int = Field(default=1, alias="TB_XRAY_GATE_POSITIVE_CLASS_INDEX")

    eye_model_path: str = Field(default="models/eye_disease_model.tflite", alias="EYE_MODEL_PATH")
    eye_input_size: int = Field(default=224, alias="EYE_INPUT_SIZE")
    eye_min_confidence: float = Field(default=0.4, alias="EYE_MIN_CONFIDENCE")
    eye_label_fallback: str = Field(default="Needs Ophthalmologist Review", alias="EYE_LABEL_FALLBACK")
    eye_default_class_names: str = Field(
        default="cataract,diabetic_retinopathy,glaucoma,normal",
        alias="EYE_DEFAULT_CLASS_NAMES",
    )

    default_fallback_label: str = Field(default="Needs Clinical Review", alias="DEFAULT_FALLBACK_LABEL")
    enable_simple_heuristics: bool = Field(default=True, alias="ENABLE_SIMPLE_HEURISTICS")

    api_key_enabled: bool = Field(default=False, alias="API_KEY_ENABLED")
    api_key_header_name: str = Field(default="X-API-Key", alias="API_KEY_HEADER_NAME")
    api_key_value: str = Field(default="change-me", alias="API_KEY_VALUE")

    cors_allowed_origins: str = Field(default="*", alias="CORS_ALLOWED_ORIGINS")
    cors_allow_credentials: bool = Field(default=False, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: str = Field(default="GET,POST,OPTIONS", alias="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="*", alias="CORS_ALLOW_HEADERS")

    @staticmethod
    def _parse_csv_list(value: str) -> list[str]:
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or ["*"]

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return self._parse_csv_list(self.cors_allowed_origins)

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return self._parse_csv_list(self.cors_allow_methods)

    @property
    def cors_allow_headers_list(self) -> list[str]:
        return self._parse_csv_list(self.cors_allow_headers)

    @property
    def eye_default_class_names_list(self) -> list[str]:
        return self._parse_csv_list(self.eye_default_class_names)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
