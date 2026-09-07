import os
import pandas as pd

LOGS_PATH = "logs/predictions.csv"
REPORT_PATH = "logs/monitoring_summary.txt"

CONFIDENCE_THRESHOLD_LOW = 0.4   # below this = confidently "not churn"
CONFIDENCE_THRESHOLD_HIGH = 0.7  # above this = confidently "churn"


def generate_monitoring_report():
    if not os.path.exists(LOGS_PATH):
        print("No predictions logged yet. Run some predictions first.")
        return

    df = pd.read_csv(LOGS_PATH)

    total_predictions = len(df)

    # Class distribution
    churn_count = (df["prediction"] == 1).sum()
    not_churn_count = (df["prediction"] == 0).sum()
    churn_pct = round((churn_count / total_predictions) * 100, 2) if total_predictions > 0 else 0
    not_churn_pct = round((not_churn_count / total_predictions) * 100, 2) if total_predictions > 0 else 0

    # Confidence stats
    avg_confidence = round(df["probability"].mean(), 4)

    # "Low confidence" = predictions near the 0.5 boundary (genuinely uncertain)
    low_confidence_mask = (df["probability"] > CONFIDENCE_THRESHOLD_LOW) & (df["probability"] < CONFIDENCE_THRESHOLD_HIGH)
    low_confidence_count = low_confidence_mask.sum()
    low_confidence_pct = round((low_confidence_count / total_predictions) * 100, 2) if total_predictions > 0 else 0

    # Risk level breakdown (as already labeled by the API)
    risk_counts = df["risk_level"].value_counts().to_dict()

    # Build report text
    report_lines = [
        "=" * 50,
        "MODEL MONITORING SUMMARY REPORT",
        "=" * 50,
        f"Generated at: {pd.Timestamp.now().isoformat()}",
        "",
        f"Total Predictions: {total_predictions}",
        "",
        "Class Distribution:",
        f"  Churned (1):     {churn_count} ({churn_pct}%)",
        f"  Not Churned (0): {not_churn_count} ({not_churn_pct}%)",
        "",
        f"Average Confidence Score (probability): {avg_confidence}",
        "",
        f"Low Confidence Predictions (between {CONFIDENCE_THRESHOLD_LOW} and {CONFIDENCE_THRESHOLD_HIGH}):",
        f"  Count: {low_confidence_count} ({low_confidence_pct}%)",
        "",
        "Risk Level Breakdown:",
    ]
    for level, count in risk_counts.items():
        report_lines.append(f"  {level}: {count}")

    report_lines += [
        "",
        "Confidence Threshold Analysis:",
        f"  Current thresholds: Low < {CONFIDENCE_THRESHOLD_LOW} <= Medium < {CONFIDENCE_THRESHOLD_HIGH} <= High",
        f"  {low_confidence_pct}% of predictions fall in the uncertain 'Medium' zone.",
        "  Recommendation: if this percentage is high (>30%), consider retraining with more data",
        "  or reviewing whether the thresholds need adjustment based on business risk tolerance.",
        "",
        "=" * 50,
    ]

    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    generate_monitoring_report()