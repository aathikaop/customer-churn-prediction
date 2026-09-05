import os
import joblib
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from data_preprocessing import create_full_pipeline


# Configuration

X_TRAIN_PATH = "data/processed/X_train.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"

MODEL_DIR = "models"

RANDOM_STATE = 42



# Load Training Data

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

print(f"Training data loaded: {X_train.shape}")


# Create XGBoost Model


xgb_model = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1
)


# Create Complete Pipeline

xgb_pipeline = create_full_pipeline(
    X_train,
    xgb_model
)

# Hyperparameter


param_distributions = {

    "model__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "model__max_depth": [
        3,
        5,
        7,
        9
    ],

    "model__learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ]
}


# Randomized Search

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1",
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)



# Hyperparameter Tuning

print("\nStarting XGBoost hyperparameter tuning...")

random_search.fit(
    X_train,
    y_train
)


# Best Parameters

print("BEST PARAMETERS")

for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")



# Best Cross Validation Score

print(
    f"\nBest Cross-Validation F1 Score: "
    f"{random_search.best_score_:.4f}"
)


# Save Tuned Model

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

tuned_model_path = os.path.join(
    MODEL_DIR,
    "xgboost_tuned.pkl"
)

joblib.dump(
    random_search.best_estimator_,
    tuned_model_path
)

print(
    f"\nTuned XGBoost model saved to: "
    f"{tuned_model_path}"
)

print("\nXGBoost tuning completed successfully.")