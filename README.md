# TB + Anemia Screening Backend

This backend is now split into two models:

- Model 1 (TB): MobileNetV2 image classifier on chest X-rays (TFLite)
- Model 2 (Anemia): Symptom-based logistic regression model (joblib)

## Medical Disclaimer

This API is for preliminary screening only and is not a diagnosis.
All outputs must be confirmed by clinical tests and qualified medical professionals.

## Endpoints

### GET /health

Returns service status and model load status.

### POST /predict/tb

TB image inference endpoint.

- Content type: multipart/form-data
- Field: file

### POST /predict/anemia

Anemia symptom-model endpoint.

- Content type: application/json
- Body example:

```json
{
  "fatigue": 1,
  "pale_skin": 1,
  "dizziness": 0,
  "shortness_of_breath": 1,
  "headache": 0,
  "cold_hands_feet": 1
}
```

## Configuration (.env)

Copy .env.example to .env and update values.

```env
APP_NAME=TB and Anemia Screening API
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

TB_MODEL_PATH=models/tb_model.tflite
TB_INPUT_SIZE=224
TB_THRESHOLD=0.5
TB_LABEL_POSITIVE=TB High Risk
TB_LABEL_NEGATIVE=TB Low Risk

ANEMIA_SYMPTOM_MODEL_PATH=models/anemia_symptom_model.joblib
ANEMIA_SYMPTOM_FEATURES=fatigue,pale_skin,dizziness,shortness_of_breath,headache,cold_hands_feet
ANEMIA_THRESHOLD=0.5
ANEMIA_LABEL_POSITIVE=Anemia High Risk
ANEMIA_LABEL_NEGATIVE=Anemia Low Risk

DEFAULT_FALLBACK_LABEL=Needs Clinical Review
ENABLE_SIMPLE_HEURISTICS=true

API_KEY_ENABLED=false
API_KEY_HEADER_NAME=X-API-Key
API_KEY_VALUE=change-me

CORS_ALLOWED_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false
CORS_ALLOW_METHODS=GET,POST,OPTIONS
CORS_ALLOW_HEADERS=*
```

## Train Model 1 (TB MobileNetV2)

Use script:

- scripts/train_mobilenetv2_binary.py

Expected TB dataset layout:

```text
dataset_tb/
├─ normal/
└─ tb/
```

Train command:

```bash
python scripts/train_mobilenetv2_binary.py \
  --data-dir dataset_tb \
  --output-dir models \
  --model-name tb_model \
  --img-size 224 \
  --epochs-head 5 \
  --epochs-finetune 5 \
  --quantization dynamic \
  --use-class-weights
```

Output used by API:

- models/tb_model.tflite

## Train Model 2 (Anemia Symptom Model)

Use script:

- scripts/train_anemia_symptom_model.py

CSV template is provided:

- data/anemia_symptoms_template.csv

Required columns:

- fatigue
- pale_skin
- dizziness
- shortness_of_breath
- headache
- cold_hands_feet
- anemia (target 0/1)

Train command:

```bash
python scripts/train_anemia_symptom_model.py \
  --csv data/anemia_symptoms_template.csv \
  --output models/anemia_symptom_model.joblib \
  --features fatigue,pale_skin,dizziness,shortness_of_breath,headache,cold_hands_feet \
  --target anemia
```

Output used by API:

- models/anemia_symptom_model.joblib

## Run Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs:

- http://localhost:8000/docs

## Quick Test

TB:

```bash
curl -X POST "http://localhost:8000/predict/tb" \
  -H "X-API-Key: change-me" \
  -F "file=@sample_tb_xray.jpg"
```

Anemia symptoms:

```bash
curl -X POST "http://localhost:8000/predict/anemia" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d "{\"fatigue\":1,\"pale_skin\":1,\"dizziness\":0,\"shortness_of_breath\":1,\"headache\":0,\"cold_hands_feet\":1}"
```
