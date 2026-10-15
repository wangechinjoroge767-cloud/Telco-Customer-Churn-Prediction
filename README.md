# Telco-Customer-Churn-Prediction
Telco Customer Churn Prediction: Development and Deployment of Machine Learning Models Using FastAPI


# Telco Customer Churn Prediction API

## Project Overview

Customer churn is one of the most expensive problems telecom companies face — it is far cheaper to keep an existing customer than to acquire a new one. This project builds a complete, deployable machine learning system that predicts whether a customer is likely to churn, based on their account details, services subscribed to, and billing information.

The project covers the full lifecycle of a machine learning solution: data exploration and cleaning, feature engineering, model comparison, model persistence, a REST API built with FastAPI, automated testing, containerisation with Docker, and public deployment.

## Problem Statement

Given a customer's demographic details, the services they use, contract type, and billing history, can we predict whether they are likely to leave (churn)? Being able to flag at-risk customers early allows a telecom provider to intervene — for example, with a retention offer — before the customer leaves.

## Dataset

This project uses the **Telco Customer Churn** dataset, stored locally at:

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains one row per customer, with fields covering demographics (gender, senior citizen status, partner/dependents), account information (tenure, contract type, payment method, billing charges), the services each customer subscribes to (phone, internet, streaming, security add-ons), and the target label, `Churn`.

The dataset is not downloaded automatically — it must already exist in the `data/` folder for the notebook and API to run.

## Repository Structure

```
.
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb          # EDA, cleaning, model comparison
├── models/
│   └── churn_pipeline.joblib         # saved end-to-end pipeline + metadata
├── app/
│   └── main.py                       # FastAPI application
├── tests/
│   └── test_api.py                   # pytest test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd [your-repo-folder]
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

The full training workflow is in `notebooks/churn_analysis.ipynb` and covers:

1. **Loading and inspecting** the dataset — shape, dtypes, missing values (including blanks hiding as strings).
2. **Exploratory data analysis** — churn distribution, tenure vs. churn, monthly/total charges vs. churn, churn by contract type, internet service, and payment method, plus a correlation heatmap.
3. **Cleaning** — converting `TotalCharges` to numeric, handling missing values, dropping `customerID`, removing duplicates, and encoding `Churn` as 0/1.
4. **Feature engineering** — splitting into train/test sets (stratified, with a fixed random seed) before fitting any transformation, then building a `ColumnTransformer` pipeline (imputation + one-hot encoding for categorical features, imputation + scaling for numerical features).
5. **Model comparison** — Logistic Regression, Random Forest, and Gradient Boosting, evaluated on accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
6. **Final model selection** — [state which model you picked and why, e.g. prioritising recall to catch more actual churners even at the cost of some false positives].
7. **Saving the pipeline** — the complete fitted pipeline (preprocessing + model) is saved to `models/churn_pipeline.joblib`, along with metadata (model type, version, feature names, class labels, evaluation metrics).

To retrain from scratch, run the notebook top to bottom, or run:
```bash
python train.py
```
*(if you extract training into a standalone script)*

## Running the API Locally

Start the FastAPI app:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`, with interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

| Method & Path | Description |
|---|---|
| `GET /` | Welcome message, API name, version, and links to docs |
| `GET /health` | Confirms the service is running and the model is loaded |
| `POST /predict` | Returns a churn prediction and probability for a single customer |
| `POST /predict-batch` | Returns predictions for a list of customers |
| `GET /model-info` | Returns model type, version, expected features, and recorded metrics |

### Example request — `POST /predict`
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 958.20
}
```

### Example response
```json
{
  "prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.78
}
```

## Testing

Automated tests are written with `pytest` and FastAPI's `TestClient`, covering health checks, valid and invalid single predictions, valid and invalid batch predictions, and the model-info endpoint.

Run the full suite from the repository root:
```bash
pytest -v
```

All tests should pass before the project is considered complete.

## Docker

Build the image:
```bash
docker build -t churn-api .
```

Run the container:
```bash
docker run --rm -p 8000:8000 churn-api
```

Once running, verify `/health`, `/predict`, and `/docs` at `http://localhost:8000`.

## Deployment

This project is deployed on **[Render / Railway / Azure / AWS / GCP — state which one]**.

- Live API base URL: `[your public URL]`
- Swagger UI: `[your public URL]/docs`
- ReDoc: `[your public URL]/redoc`

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

