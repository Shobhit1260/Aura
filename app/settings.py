from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="TB and Anemia Screening API", alias="APP_NAME")
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

    anemia_symptom_model_path: str = Field(
        default="models/anemia_symptom_model.joblib",
        alias="ANEMIA_SYMPTOM_MODEL_PATH",
    )
    anemia_symptom_features: str = Field(
        default="fatigue,pale_skin,dizziness,shortness_of_breath,headache,cold_hands_feet",
        alias="ANEMIA_SYMPTOM_FEATURES",
    )
    anemia_threshold: float = Field(default=0.5, alias="ANEMIA_THRESHOLD")
    anemia_label_positive: str = Field(default="Anemia High Risk", alias="ANEMIA_LABEL_POSITIVE")
    anemia_label_negative: str = Field(default="Anemia Low Risk", alias="ANEMIA_LABEL_NEGATIVE")

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
    def anemia_symptom_features_list(self) -> list[str]:
        return self._parse_csv_list(self.anemia_symptom_features)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
