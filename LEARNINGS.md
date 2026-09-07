# Day 1: Project Planning & Setup

## What was completed
- Selected dataset: E-commerce Customer Behavior Dataset (Kaggle, by dhairyajeetsingh) for a customer churn prediction problem.
- Defined the business problem: predict which customers are likely to churn so the business can target them with retention actions before they leave.
- Chose F1 Score as the primary success metric, since churn is a class-imbalanced problem (fewer churners than non-churners) — accuracy alone would be misleading, and F1 balances precision (avoiding wasted retention offers on customers who wouldn't have churned) against recall (catching as many actual churners as possible).
- Set up the standard MLOps project folder structure (data/, notebooks/, src/, models/, api/, tests/, reports/, logs/, mlruns/, .github/workflows/).
- Initialized Git repository and made the first commit.
- Initialized DVC and tracked the raw dataset (data/raw/ecommerce_customer_churn_dataset.csv).
- Created an initial architecture diagram outlining the end-to-end pipeline (Data → Preprocessing → Feature Engineering → Model Training → MLflow → API → Docker → CI/CD → Cloud Run → Monitoring).
- Created a project backlog listing tasks for Days 12–15.

## Key decisions made
- Picked F1 Score over accuracy/ROC-AUC as the north-star metric due to class imbalance in the churn label.
- Chose Kaggle's e-commerce churn dataset because it has a realistic mix of numerical and categorical customer behavior features, suitable for feature engineering practice.

## Problems encountered
- None major — mostly initial setup friction (repo structure, DVC config).

## Solutions attempted
- Followed the standard folder template exactly to avoid restructuring later.

## Lessons learned
- Committing the folder skeleton early (even with empty folders) makes it easier to slot in files later without reorganizing.
- DVC tracking from Day 1 keeps the raw dataset versioned outside of Git, keeping the repo lightweight.

## Improvements for future iterations
- Document the exact rationale for metric choice earlier and more formally in a project proposal doc, not just in the journal.

---

# Day 2: Experimentation & Model Development

## What was completed
- Performed EDA on the churn dataset.
- Handled missing/invalid values in `Total_Purchases` (negative values set to NaN, imputed with median).
- Engineered new features: `Purchase_Frequency`, `Is_Recently_Active`, `High_Cart_Abandonment`.
- Built a full preprocessing pipeline (median imputation + scaling for numerical, one-hot encoding for categorical) using `ColumnTransformer` inside a single sklearn `Pipeline`.
- Trained 4 baseline models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Ran `RandomizedSearchCV` (20 iterations, 5-fold CV, scoring=F1) to tune XGBoost hyperparameters.
- Set up MLflow tracking (`customer_churn_prediction` experiment) and logged all 5 runs (4 baselines + xgboost_tuned) with consistent params, metrics (accuracy, precision, recall, f1_score, roc_auc), and model artifacts.
- Generated model comparison report (`reports/evaluation_results.csv`).
- Selected **xgboost_tuned** as the best model, justified by highest F1 (0.8575), accuracy (0.9233), and near-best ROC-AUC (0.9292) on the held-out test set.

## Key decisions made
- Logged metrics from `evaluate.py`/`evaluate_tuned.py` (test-set scores) into MLflow rather than from `tune.py`'s cross-validation score, to keep all 5 runs comparable on the same test set and same metric definitions.
- Used F1 as the tuning scoring metric (consistent with the Day 11 success metric decision) rather than accuracy, to keep the tuning objective aligned with the business goal.

## Problems encountered
- Initially, MLflow runs were logging under the default "Default" experiment instead of a named one — because `mlflow.set_experiment("customer_churn_prediction")` was missing.
- Confusion between the "GenAI" and "Model training" tabs in the MLflow UI — runs weren't visible until switching to the correct tab.

## Solutions attempted
- Added the missing `mlflow.set_experiment(...)` call before the run loop, which correctly grouped all runs under `customer_churn_prediction`.
- Switched to the "Model training" tab in the MLflow UI to view experiment runs (GenAI tab is for LLM traces, not relevant here).

## Lessons learned
- `mlflow.set_experiment()` is required — without it, MLflow silently logs to "Default," which still "works" but breaks organization and comparison.
- Keeping metric names and the test set identical across all logged runs (baseline + tuned) is essential for a fair, sortable comparison table in the MLflow UI.
- MLflow can be added to an existing, working script with minimal changes (a few lines) — no need to restructure training code to get tracking benefits.

## Improvements for future iterations
- Consider using `mlflow.sklearn.autolog()` for future experiments to reduce manual logging boilerplate.
- Register the best model directly from the MLflow UI/API in Day 13 to avoid re-loading from local `.pkl` files.


# Day 3: Model Registry & API Development

## What was completed
- Fixed model logging so both the best baseline model (auto-selected by comparing F1 scores) and the tuned XGBoost model were logged using `mlflow.sklearn.log_model()` (not just `log_artifact()`), making them registrable in MLflow.
- Resolved a `skops` serialization error by explicitly setting `serialization_format="pickle"`, since the pipeline contains a custom `FeatureEngineer` class that skops' safety format doesn't support.
- Registered the model in the MLflow Model Registry under the name `churn_model`.
- Created two versions: Version 1 (best baseline, xgboost) and Version 2 (tuned xgboost).
- Promoted Version 2 to production using MLflow's alias system (`@production`), since this MLflow version replaced the older Stages concept with Aliases.
- Added a `promote_production_model()` utility function to `src/utils.py` that pulls whichever model is tagged `production` from the registry and saves it locally as `models/model.pkl` + `models/metadata.json`, matching the project's expected folder structure.
- Ran a feature importance analysis on the final model — confirmed the engineered features (`High_Cart_Abandonment`, `Is_Recently_Active`) rank highly, validating Day 12's feature engineering, while noticing `City`/`Country` one-hot encoding contributes very little despite adding ~50+ low-value columns.
- Built the FastAPI application (`api/main.py`, `api/schemas.py`) with:
  - `GET /health` — reports model load status and version.
  - `POST /api/v1/predict` — accepts customer features, returns churn prediction, probability, and a risk label (Low/Medium/High).
- Defined `CustomerFeatures` (request), `PredictionResponse`, and `HealthResponse` (response) as Pydantic models with type constraints (e.g. `ge=0`, `le=100`) for request validation.
- Tested both endpoints successfully via the auto-generated Swagger UI (`/docs`).

## Key decisions made
- Chose to keep the API's model loading based on a local `models/model.pkl` file (pulled once from the MLflow registry) rather than having the API query MLflow directly at runtime — this matches the project's specified folder structure and keeps the API independent of MLflow being available in production.
- Used `serialization_format="pickle"` instead of MLflow's default `skops` format, since the pipeline's custom `FeatureEngineer` transformer isn't supported by skops' restricted type-safety model.
- Kept the original 5 per-model `.pkl` files in `models/` alongside the new `model.pkl`/`metadata.json`, instead of deleting them, since `evaluate.py` still depends on them and deleting would break re-runs of the pipeline.

## Problems encountered
- Initial `mlflow.sklearn.log_model()` call failed with `UntrustedTypesFoundException` due to skops' safety restrictions on custom classes.
- MLflow UI's "Add alias" dialog threw a `can't access property "loadPage"` UI error, though the alias still saved correctly after refreshing.
- Loading `models/model.pkl` outside of the `src/` working directory (e.g. from the project root or from `api/main.py`) failed with `ModuleNotFoundError: No module named 'data_preprocessing'`, since pickle needs to re-import the exact module that defined the custom `FeatureEngineer` class.
- FastAPI's relative import (`from schemas import ...`) failed under `uvicorn api.main:app` because `api/` wasn't recognized as a package.

## Solutions attempted
- Added `serialization_format="pickle"` to all `mlflow.sklearn.log_model()` calls.
- Refreshed the MLflow UI page after the alias error, which confirmed the alias had actually saved.
- Added `sys.path.append(...)` pointing to the `src/` folder at the top of `api/main.py` before loading the model, so pickle can resolve `data_preprocessing`.
- Converted the schema import in `api/main.py` to a relative import (`from .schemas import ...`) and added an empty `api/__init__.py` to make `api/` a proper Python package.

## Lessons learned
- Any custom class used inside a scikit-learn pipeline (like a custom `Transformer`) must remain importable from wherever the pickle file is later loaded — this affects both the API and any inspection/analysis scripts.
- MLflow's "logged model" concept (separate from a plain file artifact) is required for the Registry's "Register Model" button to appear — `log_artifact()` alone isn't enough.
- Feature importance analysis is a cheap, valuable sanity check even without doing formal feature selection upfront — it can retroactively validate engineering decisions and flag low-value features for future iterations.
- FastAPI project structure requires attention to Python's package/import rules (`__init__.py`, relative imports) once the app is split across multiple files inside a subfolder.

## Improvements for future iterations
- Perform Postman testing alongside Swagger UI testing as part of the same development pass, rather than deferring it.
- Consider dropping or re-encoding the low-importance `City`/`Country` features to reduce model dimensionality without hurting performance.
- Automate the "promote production model to local file" step as part of a CI/CD pipeline trigger (Day 14) rather than a manual utility call.


# Day 4: Containerization, CI/CD & Cloud Deployment

## What was completed
- Wrote a `Dockerfile` to containerize the FastAPI application, using `python:3.12-slim` as the base image.
- Split dependencies into two scopes: the root `requirements.txt` (full environment, for anyone cloning the repo to reproduce training/experimentation) and a slim `api/requirements.txt` (only what the deployed API actually needs — pandas, numpy, scikit-learn, xgboost, joblib, fastapi, uvicorn, pydantic).
- Installed `xgboost` with `--no-deps` in the Dockerfile to avoid pulling in a ~250MB unused NVIDIA CUDA/NCCL dependency, since the deployed model runs on CPU only.
- Built and tested the Docker image locally (`docker build`, `docker run -p 8000:8080`), confirming the containerized API behaved identically to the local (non-Docker) version via Swagger UI.
- Attempted to deploy to Google Cloud Run per the original plan, but hit a persistent Google Cloud billing verification failure (error code `OR_BACR2_59`) with both UPI and card payment methods — a known India-specific verification issue unrelated to the project setup itself.
- **Pivoted to Render** as a substitute cloud deployment platform, given time constraints. Connected the GitHub repository, let Render auto-detect the Dockerfile, and deployed on Render's free tier.
- Verified the live deployment (`https://customer-churn-prediction-svm4.onrender.com`) via Swagger UI — both `/health` and `/api/v1/predict` work identically to the local and Dockerized versions.
- Wrote a test suite (`tests/test_preprocessing.py`, `tests/test_predict.py`, `tests/test_api.py`) covering the custom `FeatureEngineer` transformer, model loading/prediction, and both API endpoints (valid and invalid input).
- Set up a GitHub Actions workflow (`.github/workflows/cicd.yml`) that runs the full test suite automatically on every push to `main`.
- Confirmed the pipeline triggers correctly and passes (green checkmark) after pushing.

## Key decisions made
- Used two separate `requirements.txt` files (root vs. `api/`) to keep the training environment fully reproducible while keeping the deployed container lean and fast to build.
- Chose to substitute Render for Google Cloud Run after multiple failed billing verification attempts (both UPI and card), rather than lose further time on a platform-level blocker unrelated to the actual MLOps work.
- Used `TestClient(app)` as a context manager (`with TestClient(app) as client:`) in API tests, since FastAPI's startup event (model loading) only fires correctly within that context.

## Problems encountered
- Initial Docker builds timed out repeatedly downloading a ~250MB NVIDIA CUDA dependency pulled in transitively by `xgboost`, despite the model only running on CPU.
- The full local `requirements.txt` (from `pip freeze`) included a Windows-only package (`pywin32`) that would have broken the Linux-based Docker build if left in.
- Including the full `mlflow` package (with its Flask/matplotlib/pyarrow/Docker-SDK dependency chain) in the API's Docker image caused excessive build times and image bloat, despite `api/main.py` never actually importing `mlflow`.
- Google Cloud billing verification failed repeatedly with error `OR_BACR2_59` for both UPI autopay and card payment methods — appears to be an India-specific account verification issue, not a data-entry error.
- `pytest` initially showed 2 failing API tests (`/health` returning "model_not_loaded", `/predict` returning 503) because `TestClient(app)` alone doesn't trigger FastAPI's startup event without being used as a context manager.

## Solutions attempted
- Installed `xgboost` separately with `pip install xgboost --no-deps` to skip its GPU-related dependency chain entirely.
- Removed `pywin32` and created a minimal, service-scoped `api/requirements.txt` used only inside the Docker build, leaving the full `requirements.txt` untouched for local development reproducibility.
- Increased pip's `--default-timeout` and `--retries` in the Dockerfile to handle slow/unstable network conditions during dependency downloads.
- After repeated GCP billing failures, switched to Render, which required no card/billing verification for its free tier and deployed successfully on the first attempt.
- Fixed failing API tests by wrapping `TestClient(app)` usage in a `with` block, ensuring the startup event (and model loading) fires before each test runs.

## Lessons learned
- Package installs inside a Docker container should be scoped specifically to what the running service needs — a "just reuse the dev requirements.txt" approach can silently drag in huge, irrelevant dependencies (GPU libraries, visualization tools, experiment-tracking servers) that only matter for local development or training, not serving.
- Cloud billing/account verification can be a real, unpredictable blocker independent of technical skill — having a fallback deployment platform in mind (and being transparent about substituting one) is a practical, professional response rather than a shortcut.
- FastAPI's `TestClient` requires explicit context-manager usage to correctly simulate the app's startup/shutdown lifecycle — a subtlety that's easy to miss and produces confusing failures if overlooked.
- Render's built-in auto-deploy-on-push behavior effectively covers part of a CI/CD pipeline out of the box, but running tests via a separate GitHub Actions workflow is still valuable as an independent quality gate before/alongside deployment.

## Improvements for future iterations
- Revisit Google Cloud Run deployment once the billing verification issue is resolved, to fully match the original project specification.
- Extend the GitHub Actions workflow to build and push the Docker image as part of the pipeline (not just run tests), and explore linking a failed test run to blocking Render's deploy.
- Add more edge-case tests (e.g., extreme but valid values, all-categorical-boundary inputs) to the test suite for more thorough coverage.