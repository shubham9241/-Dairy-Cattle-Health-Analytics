
import pandas as pd
import numpy as np

FEATURES = [
    "SCC (cells/mL)",
    "Temperature (°C)",
    "Milk Yield (L/Day)",
    "Protein Level (%)"
]


def analyze_cattle_health(df):

    print("\n========== CATTLE HEALTH ANALYSIS ==========")

    df = df.copy()

    # Convert numerical columns
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove records without Cow ID
    df = df.dropna(subset=["Cow ID"])

    # Remove duplicate cow records
    df = df.drop_duplicates(subset=["Cow ID"])

    # Basic health statistics
    print("\nTotal Cattle:", len(df))

    print("\nAverage Health Measurements:")

    print(df[FEATURES].mean())

    # Historical mastitis information
    print("\nPrevious Mastitis Cases:")

    print(df["Historical Mastitis Cases"].value_counts(dropna=False))

    # Current mastitis diagnosis
    print("\nMastitis Diagnosis:")

    print(df["Mastitis Presence"].value_counts(dropna=False))

    # Lactation stage analysis
    print("\nLactation Stage Distribution:")

    print(df["Lactation Stage"].value_counts(dropna=False))

    return df


def create_cow_health_profile(df, cow_id):

    cow = df[
        df["Cow ID"].astype(str) == str(cow_id)
    ]

    if cow.empty:
        return None

    record = cow.iloc[0]

    profile = {
        "Cow ID": record["Cow ID"],
        "Milk Yield": record["Milk Yield (L/Day)"],
        "SCC": record["SCC (cells/mL)"],
        "Temperature": record["Temperature (°C)"],
        "Lactation Stage": record["Lactation Stage"],
        "Udder Health": record["Udder Health Metrics"],
        "Previous Mastitis Cases": record["Historical Mastitis Cases"],
        "Protein Level": record["Protein Level (%)"],
        "Mastitis Diagnosis": record["Mastitis Presence"],
        "Mastitis Type": record["Mastitis Type"]
    }

    return profile


def create_health_summary(df):

    summary = df.groupby("Lactation Stage").agg(
        Total_Cattle=("Cow ID", "count"),
        Average_SCC=("SCC (cells/mL)", "mean"),
        Average_Temperature=("Temperature (°C)", "mean"),
        Average_Milk_Yield=("Milk Yield (L/Day)", "mean"),
        Average_Protein=("Protein Level (%)", "mean")
    )

    return summary
