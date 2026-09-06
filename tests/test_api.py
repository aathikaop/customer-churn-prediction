import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_predict_endpoint_valid_input():
    with TestClient(app) as client:
        payload = {
            "Age": 34, "Gender": "Female", "Country": "USA", "City": "New York",
            "Membership_Years": 2.5, "Login_Frequency": 15, "Session_Duration_Avg": 8.2,
            "Pages_Per_Session": 4.5, "Cart_Abandonment_Rate": 35.0, "Wishlist_Items": 3,
            "Total_Purchases": 12, "Average_Order_Value": 55.0, "Days_Since_Last_Purchase": 20,
            "Discount_Usage_Rate": 10.0, "Returns_Rate": 5.0, "Email_Open_Rate": 40.0,
            "Customer_Service_Calls": 1, "Product_Reviews_Written": 2,
            "Social_Media_Engagement_Score": 60.0, "Mobile_App_Usage": 70.0,
            "Payment_Method_Diversity": 2, "Lifetime_Value": 1200.0,
            "Credit_Balance": 50.0, "Signup_Quarter": "Q1"
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_endpoint_invalid_input():
    with TestClient(app) as client:
        payload = {"Age": -5}  # missing required fields, invalid age
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # FastAPI validation error