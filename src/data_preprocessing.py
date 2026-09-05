import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Feature Engineering
# -------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()


        #  Handle invalid Total_Purchases values

        X.loc[X["Total_Purchases"] < 0, "Total_Purchases"] = np.nan


        # Purchase Frequency

        X["Purchase_Frequency"] = (
            X["Total_Purchases"] /
            X["Membership_Years"]
        )

        # Recently Active Customer

        X["Is_Recently_Active"] = np.where(
            X["Days_Since_Last_Purchase"].isna(),
            np.nan,
            (
                X["Days_Since_Last_Purchase"] <= 30
            ).astype(float)
        )


        # High Cart Abandonment

        X["High_Cart_Abandonment"] = (
            X["Cart_Abandonment_Rate"] >= 60
        ).astype(int)

        return X

# -------------------------------------------------------------------

def create_preprocessor(X):

    #Creates the preprocessing pipeline for numerical and categorical features.


    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()


    # Numerical preprocessing

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # Categorical preprocessing

    categorical_pipeline = Pipeline(
        steps=[

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # Combine numerical and categorical preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def create_full_pipeline(X, model):

    # Creates the complete pipeline

    feature_engineer = FeatureEngineer()

    # Apply feature engineering temporarily to determine

    X_engineered = feature_engineer.transform(X)

    preprocessor = create_preprocessor(X_engineered)

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                feature_engineer
            ),
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline