# Hybrid XGB-LSTM and CNN-LSTM IDS

## 1. PROJECT
This project implements a hybrid machine-learning and deep-learning Intrusion Detection System (IDS). It utilizes two separate pipelines for network traffic classification:
- **CNN → BiLSTM**
- **XGBoost feature selection → BiLSTM**

## 2. WHY IDS
Intrusion Detection Systems are critical components of network security. They monitor network traffic for suspicious activity and known threats, alerting administrators to potential attacks such as DDoS, port scanning, and brute force attempts. A robust ML/DL-based IDS can learn complex patterns in network data to accurately identify both known and zero-day intrusions while minimizing false alarms.

## 3. DATASETS
The models are trained and evaluated on four distinct benchmark datasets:
- **CICIDS2017**: Represents modern network traffic with up-to-date attacks (like DDoS, Brute Force, Web Attacks) alongside normal benign background traffic. Used for multiclass classification.
- **UNSW-NB15**: Contains a comprehensive mix of normal activities and synthetic contemporary attack behaviors. Used for multiclass classification.
- **NSL-KDD**: A refined version of the classic KDD'99 dataset, solving its inherent redundancy issues. Represents fundamental network intrusions (DoS, Probe, R2L, U2R). Used for multiclass classification.
- **WSN-DS**: A specialized Wireless Sensor Network dataset containing normal routing data and various denial-of-service attacks (Blackhole, Grayhole, Flooding, Scheduling). Used for multiclass classification.

## 4. METHODOLOGY
The overall workflow of the project follows this pipeline:

```text
Raw dataset
    ↓
Cleaning / preprocessing
    ↓
Encoding
    ↓
Train/Test split
    ↓
Scaling
    ↓
        ┌──────────────────┐
        │                  │
        ▼                  ▼
   CNN pipeline       XGBoost pipeline
        │                  │
   CNN feature       Feature selection
   extraction             │
        │                  ▼
        │               BiLSTM
        ▼                  │
      BiLSTM               │
        │                  │
        └────────┬─────────┘
                 ↓
             Prediction
```

## 5. TWO ARCHITECTURES

### CNN → BiLSTM
In this architecture, a 1D Convolutional Neural Network (CNN) is first used to automatically extract spatial and structural features from the raw network flow data. The extracted features are then fed into a Bidirectional Long Short-Term Memory (BiLSTM) network, which captures the temporal dependencies and sequential patterns of the traffic before making the final classification.

### XGBoost feature selection → BiLSTM
In this architecture, XGBoost is utilized purely for feature selection to identify the most important predictive features from the network traffic. Once the top features are selected, they are passed into the Bidirectional LSTM (BiLSTM) network for sequence learning and classification.

## 6. DATASET-SPECIFIC PREPROCESSING
To prepare the network traffic for training, various preprocessing techniques are applied where applicable:
- **StandardScaler**: Scaling numeric features to have zero mean and unit variance.
- **Label encoding**: Converting categorical targets (attack types) into numeric labels.
- **Stratified split**: Ensuring balanced class representation in the train and test sets.
- **SMOTE**: Addressing class imbalance by synthetically oversampling minority attack classes.
- **Top-k XGBoost feature selection**: Selecting the most impactful features for the XGBoost→BiLSTM pipeline.

## 7. INFERENCE SYSTEM
The project exposes its predictive capabilities through a FastAPI REST inference layer.

`apps/inference.py` handles the core inference logic, including:
- model discovery
- preprocessing
- artifact loading
- prediction

`apps/fastapi_app.py` provides the REST API layer that receives HTTP requests, routes them to the inference engine, and returns predictions.

## 8. API
The FastAPI service exposes the following endpoints:
- **`GET /`**: Health check endpoint returning service status.
- **`GET /models`**: Lists metadata for all available model variants.
- **`POST /predict`**: Predicts the class for a single network flow.
- **`POST /predict/batch`**: Predicts the classes for multiple network flows.

FastAPI Swagger/OpenAPI documentation is available at `/docs` when the API is running.

### API Usage Samples

**1. Single Prediction - Normal Traffic**
```json
// POST /predict
{
  "model_name": "CICIDS_CNN",
  "features": { "Flow Duration": 120, "Total Fwd Packets": 2 }
}

// Response
{
  "model_used": "CICIDS_CNN",
  "prediction": {
    "predicted_label": "Normal Traffic",
    "binary_label": "normal",
    "class_probabilities": { "Normal Traffic": 0.98, "DDoS": 0.01 }
  }
}
```

**2. Single Prediction - Attack Traffic**
```json
// POST /predict
{
  "model_name": "CICIDS_CNN",
  "features": { "Flow Duration": 45000, "Total Fwd Packets": 100 }
}

// Response
{
  "model_used": "CICIDS_CNN",
  "prediction": {
    "predicted_label": "DDoS",
    "binary_label": "attack",
    "class_probabilities": { "Normal Traffic": 0.02, "DDoS": 0.97 }
  }
}
```

**3. Batch Prediction - Normal Traffic**
```json
// POST /predict/batch
{
  "model_name": "CICIDS_CNN",
  "samples": [
    { "Flow Duration": 120, "Total Fwd Packets": 2 },
    { "Flow Duration": 130, "Total Fwd Packets": 3 }
  ]
}

// Response
{
  "model_used": "CICIDS_CNN",
  "predictions": [
    {
      "predicted_label": "Normal Traffic",
      "binary_label": "normal",
      "class_probabilities": { "Normal Traffic": 0.98, "DDoS": 0.01 }
    },
    {
      "predicted_label": "Normal Traffic",
      "binary_label": "normal",
      "class_probabilities": { "Normal Traffic": 0.96, "DDoS": 0.01 }
    }
  ]
}
```

**4. Batch Prediction - Attack Traffic**
```json
// POST /predict/batch
{
  "model_name": "CICIDS_CNN",
  "samples": [
    { "Flow Duration": 45000, "Total Fwd Packets": 100 },
    { "Flow Duration": 50000, "Total Fwd Packets": 120 }
  ]
}

// Response
{
  "model_used": "CICIDS_CNN",
  "predictions": [
    {
      "predicted_label": "DDoS",
      "binary_label": "attack",
      "class_probabilities": { "Normal Traffic": 0.02, "DDoS": 0.97 }
    },
    {
      "predicted_label": "DDoS",
      "binary_label": "attack",
      "class_probabilities": { "Normal Traffic": 0.01, "DDoS": 0.98 }
    }
  ]
}
```
## 9. DOCKER
The inference service can be containerized and run using Docker. A `Dockerfile` is provided to build an image containing the FastAPI app and inference dependencies.

## 10. DEPLOYMENT
The FastAPI inference service is containerized using Docker and can be deployed to AWS EC2.
