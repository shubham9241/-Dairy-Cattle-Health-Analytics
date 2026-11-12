
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_clustering(X, labels):
    print("\n========== CLUSTER QUALITY EVALUATION ==========")

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        print("Silhouette Score cannot be calculated.")
        return None

    if len(unique_clusters) >= len(labels):
        print("Not enough records for Silhouette Score.")
        return None

    score = silhouette_score(X, labels)

    print("Silhouette Score:", round(score, 4))

    if score >= 0.7:
        print("Cluster quality: Strong separation")
    elif score >= 0.5:
        print("Cluster quality: Reasonable separation")
    elif score >= 0.25:
        print("Cluster quality: Weak separation")
    else:
        print("Cluster quality: Poor separation")

    return score


def create_cluster_report(results):
    print("\n========== CLUSTER PERFORMANCE REPORT ==========")

    features = [
        "SCC (cells/mL)",
        "Temperature (°C)",
        "Milk Yield (L/Day)"
    ]

    report = results.groupby("Risk Tier").agg(
        Number_of_Records=("Cluster", "count"),
        Average_SCC=("SCC (cells/mL)", "mean"),
        Average_Temperature=("Temperature (°C)", "mean"),
        Average_Milk_Yield=("Milk Yield (L/Day)", "mean")
    )

    report = report.reindex(
        ["Low Risk", "Medium Risk", "High Risk"]
    )

    print(report)

    return report


def validate_diagnosis(results):
    print("\n========== DIAGNOSIS VALIDATION ==========")

    diagnosis_column = "Mastitis Presence"

    if diagnosis_column not in results.columns:
        print("Diagnosis column not found.")
        return None

    diagnosis = (
        results[diagnosis_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive_labels = {
        "1", "yes", "true", "positive",
        "mastitis", "present", "infected"
    }

    negative_labels = {
        "0", "no", "false", "negative",
        "healthy", "absent", "none"
    }

    confirmed = diagnosis.map(
        lambda value:
        1 if value in positive_labels
        else 0 if value in negative_labels
        else np.nan
    )

    valid = confirmed.notna()

    if valid.sum() == 0:
        print("No recognized diagnosis values.")
        print("Available values:", diagnosis.unique())
        return None

    actual = confirmed[valid].astype(int)

    # High Risk is treated as a positive screening flag.
    predicted = (
        results.loc[valid, "Risk Tier"] == "High Risk"
    ).astype(int)

    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["No Mastitis", "Mastitis"],
            zero_division=0
        )
    )

    print("Validated records:", valid.sum())
    print("Unrecognized records:", (~valid).sum())

    return cm
