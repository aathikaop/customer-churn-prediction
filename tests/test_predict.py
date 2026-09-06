import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_valid_output():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "Age": 34, "Gender": "Female", "Country": "USA", "City": "New York",
        "Membership_Years": 2.5, "Login_Frequency": 15, "Session_Duration_Avg": 8.2,
        "Pages_Per_Session": 4.5, "Cart_Abandonment_Rate": 35.0, "Wishlist_Items": 3,
        "Total_Purchases": 12, "Average_Order_Value": 55.0, "Days_Since_Last_Purchase": 20,
        "Discount_Usage_Rate": 10.0, "Returns_Rate": 5.0, "Email_Open_Rate": 40.0,
        "Customer_Service_Calls": 1, "Product_Reviews_Written": 2,
        "Social_Media_Engagement_Score": 60.0, "Mobile_App_Usage": 70.0,
        "Payment_Method_Diversity": 2, "Lifetime_Value": 1200.0,
        "Credit_Balance": 50.0, "Signup_Quarter": "Q1"
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    assert prediction in [0, 1]
    assert 0.0 <= probability <= 1.0