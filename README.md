# Customer Churn Prediction

An end-to-end MLOps pipeline that predicts customer churn for an e-commerce platform, covering data versioning, experiment tracking, model registry, API deployment, containerization, CI/CD, and monitoring.

## Business Problem

Predict which customers are likely to churn so the business can proactively target them with retention offers before they leave. Given class imbalance in churn behavior, **F1 Score** was chosen as the primary success metric — it balances precision (avoiding wasted retention spend on customers who wouldn't have churned) against recall (catching as many actual churners as possible).

## Dataset

[E-commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) (Kaggle) — 50,000 customers, 25 features covering demographics, engagement, purchase behavior, and support interactions.

## Architecture
Dataset → Git + DVC → Data Preprocessing → Feature Engineering → Model Training
→ Model Evaluation → MLflow Experiment Tracking → MLflow Model Registry
→ FastAPI (/api/v1/predict) → Docker → GitHub Actions CI → Render Deployment
→ Prediction Logging → Monitoring & Reporting


## Results

| Model | F1 Score | Accuracy | ROC-AUC |
|---|---|---|---|
| **XGBoost (Tuned)** ⭐ | **0.8575** | **0.9233** | 0.9292 |
| XGBoost (Baseline) | 0.8529 | 0.9210 | 0.9293 |
| Random Forest | 0.8347 | 0.9119 | 0.9245 |
| Decision Tree | 0.7337 | 0.8430 | 0.8149 |
| Logistic Regression | 0.5229 | 0.7759 | 0.7907 |

**Production model**: Tuned XGBoost, registered as `churn_model` (Version 2) in MLflow, tuned via `RandomizedSearchCV` (20 iterations, 5-fold CV, F1 scoring).

**Feature engineering**: Three engineered features (`Purchase_Frequency`, `Is_Recently_Active`, `High_Cart_Abandonment`) — `High_Cart_Abandonment` ranked in the top 5 most important features, validating the engineering approach.


## Setup

```bash
git clone https://github.com/aathikaop/customer-churn-prediction.git
cd CustomerChurnPrediction
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running the Pipeline

**1. Train baseline models**
```bash
python src/train.py
```

**2. Evaluate and log to MLflow**
```bash
python src/evaluate.py
```

**3. Hyperparameter tuning**
```bash
python src/tune.py
python src/evaluate_tuned.py
```

**4. View experiments**
```bash
mlflow ui
```
Open `http://127.0.0.1:5000`

**5. Promote production model** (after registering & aliasing in MLflow UI)
```python
from src.utils import promote_production_model
promote_production_model()
```

**6. Generate monitoring report**
```bash
python src/monitor.py
```

## Running the API

**Locally:**
```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

**Via Docker:**
```bash
docker build -t churn-api .
docker run -p 8000:8080 churn-api
```

## API Usage

**Health check**
```bash
curl http://127.0.0.1:8000/health
```

**Prediction**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 34, "Gender": "Female", "Country": "USA", "City": "New York",
    "Membership_Years": 2.5, "Login_Frequency": 15, "Session_Duration_Avg": 8.2,
    "Pages_Per_Session": 4.5, "Cart_Abandonment_Rate": 35.0, "Wishlist_Items": 3,
    "Total_Purchases": 12, "Average_Order_Value": 55.0, "Days_Since_Last_Purchase": 20,
    "Discount_Usage_Rate": 10.0, "Returns_Rate": 5.0, "Email_Open_Rate": 40.0,
    "Customer_Service_Calls": 1, "Product_Reviews_Written": 2,
    "Social_Media_Engagement_Score": 60.0, "Mobile_App_Usage": 70.0,
    "Payment_Method_Diversity": 2, "Lifetime_Value": 1200.0,
    "Credit_Balance": 50.0, "Signup_Quarter": "Q1"
  }'
```

**Response**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.0634,
  "risk_level": "Low"
}
```

## Testing

```bash
pytest tests/ -v
```
Covers feature engineering logic, model loading/prediction, and API endpoint validation (valid and invalid input).

## CI/CD

GitHub Actions (`.github/workflows/cicd.yml`) automatically runs the full test suite on every push to `main`.

## Deployment

**Live API**: [https://customer-churn-prediction-svm4.onrender.com](https://customer-churn-prediction-svm4.onrender.com)

**Swagger docs**: [https://customer-churn-prediction-svm4.onrender.com/docs](https://customer-churn-prediction-svm4.onrender.com/docs)

> **Note**: The original plan targeted Google Cloud Run + Artifact Registry. Due to a persistent Google Cloud billing verification issue (unrelated to the technical setup), the deployment was substituted with **Render**, which offers equivalent Docker-based deployment with auto-deploy-on-push. See `LEARNINGS.md` (Day 14) for details.

## Monitoring

Every prediction made through `/api/v1/predict` is logged to `logs/predictions.csv`. The monitoring script reads this log and generates a summary report:

```bash
python src/monitor.py
```

Generates `logs/monitoring_summary.txt` with:
- Total predictions served
- Class distribution (churn vs. not churn)
- Average confidence score
- Low-confidence prediction rate
- Risk level breakdown (Low / Medium / High)
- Confidence threshold analysis

## Model Registry

Models are versioned and tracked via MLflow Model Registry under the name `churn_model`:
- **Version 1**: Best baseline model (XGBoost, auto-selected by F1 score comparison)
- **Version 2**: Tuned XGBoost (tagged with `production` alias) — currently deployed

## Tech Stack

Python · pandas · scikit-learn · XGBoost · MLflow · FastAPI · Pydantic · Docker · GitHub Actions · Render · DVC · Git

## Project Timeline

This project was built over 5 days as part of an MLOps capstone:
- **Day 1**: Project planning, dataset selection, repo & DVC setup
- **Day 2**: EDA, feature engineering, model training, MLflow experiment tracking
- **Day 3**: MLflow Model Registry, FastAPI application development
- **Day 4**: Docker containerization, testing, CI/CD, cloud deployment
- **Day 5**: Prediction logging, monitoring, final documentation

See `LEARNINGS.md` for detailed daily engineering notes, decisions, problems encountered, and lessons learned.

## Author

Aathika — MLOps Capstone Project
