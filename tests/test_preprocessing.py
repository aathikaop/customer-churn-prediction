import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from data_preprocessing import FeatureEngineer


def test_feature_engineer_creates_new_columns():
    sample_data = pd.DataFrame({
        "Total_Purchases": [10, 5],
        "Membership_Years": [2, 1],
        "Days_Since_Last_Purchase": [10, 40],
        "Cart_Abandonment_Rate": [70, 30]
    })

    fe = FeatureEngineer()
    result = fe.transform(sample_data)

    assert "Purchase_Frequency" in result.columns
    assert "Is_Recently_Active" in result.columns
    assert "High_Cart_Abandonment" in result.columns


def test_high_cart_abandonment_threshold():
    sample_data = pd.DataFrame({
        "Total_Purchases": [10],
        "Membership_Years": [2],
        "Days_Since_Last_Purchase": [10],
        "Cart_Abandonment_Rate": [65]
    })

    fe = FeatureEngineer()
    result = fe.transform(sample_data)

    assert result["High_Cart_Abandonment"].iloc[0] == 1


def test_negative_total_purchases_becomes_nan():
    sample_data = pd.DataFrame({
        "Total_Purchases": [-5],
        "Membership_Years": [1],
        "Days_Since_Last_Purchase": [10],
        "Cart_Abandonment_Rate": [50]
    })

    fe = FeatureEngineer()
    result = fe.transform(sample_data)

    assert pd.isna(result["Total_Purchases"].iloc[0])