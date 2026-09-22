# Hybrid Real-Time Intrusion Detection Service

**Python · TensorFlow · XGBoost · FastAPI · Docker · AWS EC2**

## Overview

This project implements a hybrid intrusion detection system using dual deep-learning pipelines aligned to research methodology:

| Pipeline | Architecture |
|----------|-------------|
| **CNN → BiLSTM** | Conv1D feature extraction → Bidirectional LSTM for temporal pattern capture |
| **XGBoost → BiLSTM** | XGBoost-based feature selection (top-k) → Bidirectional LSTM classification |

Trained and evaluated across four benchmark datasets:

- **CICIDS 2017**
- **UNSW-NB15**
- **NSL-KDD**
- **WSN-DS**

Achieving up to **98–99% accuracy** with false alarm rates below 1% on structured datasets.

## Project Structure

```
├── apps/
│   ├── inference.py       # Shared inference pipeline (model loading, preprocessing, prediction)
│   └── fastapi_app.py     # FastAPI REST endpoints (single + batch prediction)
├── app_models/            # Trained model artifacts (per-dataset, per-pipeline)
│   ├── CICIDS/            # CICIDS_CNN, CICIDS_XGB
│   ├── NSL-KDD/           # NSL-KDD_CNN, NSL-KDD_XGB
│   ├── UNSW/              # UNSW_CNN, UNSW_XGB
│   ├── UNSW_CNN/          # Standalone UNSW CNN variant
│   └── WSN-DS/            # WSN-DS_CNN, WSN-DS_XGB
├── src/                   # ML training, preprocessing, evaluation (frozen)
├── data/                  # Datasets (not included in Docker image)
├── results/               # Evaluation results (not included in Docker image)
├── Dockerfile             # Inference-only container
├── .dockerignore
├── requirements.txt
└── README.md
```

## Architecture

```
Client (curl / Python / any HTTP client)
        │
        ▼
   FastAPI REST API  (port 8000)
        │
        ▼
   Inference Pipeline
   (apps/inference.py)
        │
        ├── Model discovery & auto-selection
        ├── Feature ordering & preprocessing (scaler)
        ├── Feature selection (XGB pipeline only)
        └── Keras model prediction (.h5)
        │
        ▼
   JSON response
```

---

## FastAPI Endpoints

### `GET /` — Health Check

Returns service status and list of loaded model variants.

### `GET /models` — List Available Models

Returns metadata for each discovered model variant.

### `POST /predict` — Single-Flow Prediction

Predict the class of **one** network flow.

**Request body:**

```json
{
  "model_name": "CICIDS_CNN",
  "features": {
    "Flow Duration": 123456,
    "Total Fwd Packets": 5,
    "Total Backward Packets": 3
  },
  "threshold": 0.5
}
```

- `model_name` — *(optional)* Specific model variant. If omitted, auto-selects based on feature names.
- `features` — Dict of feature-name → numeric value.
- `threshold` — *(optional, default 0.5)* Binary attack threshold.

**Response:**

```json
{
  "model_used": "CICIDS_CNN",
  "prediction": {
    "predicted_label": "Normal Traffic",
    "binary_label": "normal",
    "class_probabilities": {
      "Bots": 0.001,
      "Brute Force": 0.002,
      "DDoS": 0.003,
      "DoS": 0.004,
      "Normal Traffic": 0.985,
      "Port Scanning": 0.003,
      "Web Attacks": 0.002
    },
    "preprocessing_messages": []
  }
}
```

### `POST /predict/batch` — Batch Prediction

Predict classes for **multiple** network flows. Output order matches input order.

**Request body:**

```json
{
  "model_name": "CICIDS_CNN",
  "samples": [
    {"Flow Duration": 123, "Total Fwd Packets": 5, "Total Backward Packets": 3},
    {"Flow Duration": 456, "Total Fwd Packets": 100, "Total Backward Packets": 80}
  ],
  "threshold": 0.5
}
```

**Response:**

```json
{
  "model_used": "CICIDS_CNN",
  "predictions": [
    {
      "predicted_label": "Normal Traffic",
      "binary_label": "normal",
      "class_probabilities": {"...": "..."},
      "preprocessing_messages": []
    },
    {
      "predicted_label": "DDoS",
      "binary_label": "attack",
      "class_probabilities": {"...": "..."},
      "preprocessing_messages": []
    }
  ]
}
```

---

## Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8000

# Open interactive docs
# http://localhost:8000/docs
```

---

## Docker

### Build

```bash
docker build -t ids-service .
```

### Run

```bash
docker run -d -p 8000:8000 --name ids-api ids-service
```

### Test

```bash
# Health check
curl http://localhost:8000/

# List models
curl http://localhost:8000/models

# Single prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "CICIDS_CNN", "features": {"Flow Duration": 100, "Total Fwd Packets": 5}}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "CICIDS_CNN", "samples": [{"Flow Duration": 100}, {"Flow Duration": 200}]}'
```

---

## AWS EC2 Deployment

### Step 1 — Launch EC2 Instance

- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Instance type: `t3.medium` or larger (TensorFlow needs ≥ 4 GB RAM)
- Storage: ≥ 20 GB

### Step 2 — Install Docker on EC2

**Amazon Linux 2023:**

```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

**Ubuntu 22.04:**

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

### Step 3 — Upload the Project

```bash
# Option A: Clone from git (if pushed)
git clone <your-repo-url> && cd IDS_PROJECT

# Option B: Upload via SCP
scp -i your-key.pem -r ./IDS_PROJECT ec2-user@<EC2-PUBLIC-IP>:~/
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>
cd IDS_PROJECT
```

> **Note:** The `app_models/` directory (trained model artifacts) is gitignored.
> If cloning, you must also upload `app_models/` separately via SCP:
> ```bash
> scp -i your-key.pem -r ./app_models ec2-user@<EC2-PUBLIC-IP>:~/IDS_PROJECT/
> ```

### Step 4 — Build & Run

```bash
docker build -t ids-service .
docker run -d -p 8000:8000 --name ids-api ids-service
```

### Step 5 — Configure Security Group

In the AWS Console → EC2 → Security Groups:

- Add an **Inbound Rule**:
  - Type: Custom TCP
  - Port: **8000**
  - Source: Your IP (or `0.0.0.0/0` for public access)

### Step 6 — Access the Service

```bash
# Replace <EC2-PUBLIC-IP> with your instance's public IP
curl http://<EC2-PUBLIC-IP>:8000/

# Interactive API docs
# http://<EC2-PUBLIC-IP>:8000/docs
```

---

## ML Pipeline Details (Frozen)

The ML training/evaluation pipeline in `src/` is complete and unchanged:

- `src/train.py` — Training script supporting both CNN and XGB pipelines
- `src/preprocess.py` — Dataset preprocessing (CICIDS, UNSW, NSL-KDD, WSN-DS)
- `src/cnn_bilstm_model.py` — CNN → BiLSTM architecture
- `src/bilstm_model.py` — BiLSTM architecture (used by XGBoost pipeline)
- `src/xgboost_features.py` — XGBoost-based feature selection
- `src/evaluate.py` — Evaluation and metrics generation
