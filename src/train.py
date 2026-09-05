import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import load_data
from data_preprocessing import create_full_pipeline


# Configuration

DATA_PATH = "data/raw/ecommerce_customer_churn_dataset.csv"
MODEL_DIR = "models"
PROCESSED_DATA_DIR = "data/processed"

TARGET_COLUMN = "Churned"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# Load Dataset

df = load_data(DATA_PATH)

print(f"Dataset loaded successfully: {df.shape}")


# Separate Features and Target

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# Create Required Directories

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# Save Train-Test Split

X_train.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "X_train.csv"),
    index=False
)

X_test.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "X_test.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "y_test.csv"),
    index=False
)

print("Train-test split saved successfully.")


# Define Models

models = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
}

# Train Models

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Create complete ML pipeline
    pipeline = create_full_pipeline(
        X_train,
        model
    )

    # Train pipeline
    pipeline.fit(
        X_train,
        y_train
    )

    # Model path
    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    # Save complete pipeline
    joblib.dump(
        pipeline,
        model_path
    )

    print(f"Model saved: {model_path}")



print("\nAll baseline models trained successfully.")