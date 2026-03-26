# TB + Eye Disease Screening Backend

This backend serves two image models:

- Model 1 (TB): EfficientNetB0 binary classifier on chest X-rays (TFLite)
- Model 2 (Eye disease): EfficientNetB0 multiclass classifier on fundus/eye images (TFLite)

## Medical Disclaimer

This API is for preliminary screening only and is not a diagnosis.
All outputs must be confirmed by clinical tests and qualified medical professionals.

## Endpoints

### GET /health

Returns service status and model load status.

### POST /predict/tb

TB image inference endpoint.

This endpoint validates uploads with a local pre-classification pipeline before TB inference:

- First, hard rule checks reject clearly invalid images (too small, too colorful, too blurry)
- Then, an X-ray validator model classifies valid chest X-ray vs invalid image
- If invalid, returns Invalid photo (HTTP 400)
- If valid, runs TB classification model

- Content type: multipart/form-data
- Field: file

### POST /predict/eye-disease

Eye-disease image inference endpoint.

- Content type: multipart/form-data
- Field: file
- Returns top predicted class as risk and full class probability distribution

## Configuration (.env)

Copy .env.example to .env and update values.

```env
APP_NAME=TB and Eye Disease Screening API
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

TB_MODEL_PATH=models/tb_model.tflite
TB_INPUT_SIZE=224
TB_THRESHOLD=0.5
TB_LABEL_POSITIVE=TB High Risk
TB_LABEL_NEGATIVE=TB Low Risk
TB_STRICT_XRAY_VALIDATION=true
TB_XRAY_MIN_SIDE=224
TB_XRAY_GATE_ENABLED=true
TB_XRAY_GATE_REQUIRED=false
TB_XRAY_GATE_MODEL_PATH=models/tb_xray_gate_model.tflite
TB_XRAY_GATE_INPUT_SIZE=224
TB_XRAY_GATE_THRESHOLD=0.5
TB_XRAY_GATE_POSITIVE_CLASS_INDEX=1

EYE_MODEL_PATH=models/eye_disease_model.tflite
EYE_INPUT_SIZE=224
EYE_MIN_CONFIDENCE=0.4
EYE_LABEL_FALLBACK=Needs Ophthalmologist Review
EYE_DEFAULT_CLASS_NAMES=cataract,diabetic_retinopathy,glaucoma,normal

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

## Train TB Model

Script:

- scripts/train_mobilenetv2_binary.py

Expected dataset:

```text
dataset_tb/
  normal/
  tb/
```

Train command:

```bash
python scripts/train_mobilenetv2_binary.py \
  --data-dir dataset_tb \
  --output-dir models \
  --model-name tb_model \
  --img-size 224 \
  --epochs-head 8 \
  --epochs-finetune 10 \
  --finetune-unfreeze-layers 80 \
  --quantization dynamic \
  --use-class-weights
```

## Train TB X-ray Gate Model

Script:

- scripts/train_tb_xray_gate_model.py

Expected dataset:

```text
dataset_tb_gate/
  not_chest_xray/
  chest_xray/
```

Train command:

```bash
python scripts/train_tb_xray_gate_model.py \
  --data-dir dataset_tb_gate \
  --output-dir models \
  --model-name tb_xray_gate_model \
  --img-size 224 \
  --epochs-head 5 \
  --epochs-finetune 5 \
  --quantization dynamic
```

## Train Eye Disease Model

Script:

- scripts/train_eye_disease_classifier.py

Expected dataset:

```text
dataset_eye/
  class_1/
  class_2/
  class_3/
  ...
```

Use your real disease folder names as class folders.

Train command:

```bash
python scripts/train_eye_disease_classifier.py \
  --data-dir dataset_eye \
  --output-dir models \
  --model-name eye_disease_model \
  --img-size 224 \
  --epochs-head 6 \
  --epochs-finetune 6 \
  --finetune-unfreeze-layers 80 \
  --quantization dynamic \
  --use-class-weights
```

## Run Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs:

- http://localhost:8000/docs

## Quick Test

TB:

```bash
curl -X POST "http://localhost:8000/predict/tb" \
  -H "X-API-Key: change-me" \
  -F "file=@sample_tb_xray.jpg"
```

Eye disease:

```bash
curl -X POST "http://localhost:8000/predict/eye-disease" \
  -H "X-API-Key: change-me" \
  -F "file=@sample_eye_image.jpg"
```
