from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Age: int = Field(..., ge=0, le=120, description="Customer age in years")
    Gender: str = Field(..., description="e.g. Male, Female")
    Country: str
    City: str
    Membership_Years: float = Field(..., ge=0)
    Login_Frequency: float = Field(..., ge=0)
    Session_Duration_Avg: float = Field(..., ge=0)
    Pages_Per_Session: float = Field(..., ge=0)
    Cart_Abandonment_Rate: float = Field(..., ge=0, le=100)
    Wishlist_Items: int = Field(..., ge=0)
    Total_Purchases: int = Field(..., ge=0)
    Average_Order_Value: float = Field(..., ge=0)
    Days_Since_Last_Purchase: int = Field(..., ge=0)
    Discount_Usage_Rate: float = Field(..., ge=0, le=100)
    Returns_Rate: float = Field(..., ge=0, le=100)
    Email_Open_Rate: float = Field(..., ge=0, le=100)
    Customer_Service_Calls: int = Field(..., ge=0)
    Product_Reviews_Written: int = Field(..., ge=0)
    Social_Media_Engagement_Score: float = Field(..., ge=0)
    Mobile_App_Usage: float = Field(..., ge=0)
    Payment_Method_Diversity: int = Field(..., ge=0)
    Lifetime_Value: float = Field(..., ge=0)
    Credit_Balance: float
    Signup_Quarter: str = Field(..., description="e.g. Q1, Q2, Q3, Q4")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 34,
                "Gender": "Female",
                "Country": "USA",
                "City": "New York",
                "Membership_Years": 2.5,
                "Login_Frequency": 15,
                "Session_Duration_Avg": 8.2,
                "Pages_Per_Session": 4.5,
                "Cart_Abandonment_Rate": 35.0,
                "Wishlist_Items": 3,
                "Total_Purchases": 12,
                "Average_Order_Value": 55.0,
                "Days_Since_Last_Purchase": 20,
                "Discount_Usage_Rate": 10.0,
                "Returns_Rate": 5.0,
                "Email_Open_Rate": 40.0,
                "Customer_Service_Calls": 1,
                "Product_Reviews_Written": 2,
                "Social_Media_Engagement_Score": 60.0,
                "Mobile_App_Usage": 70.0,
                "Payment_Method_Diversity": 2,
                "Lifetime_Value": 1200.0,
                "Credit_Balance": 50.0,
                "Signup_Quarter": "Q1"
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 = Not Churned, 1 = Churned")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None