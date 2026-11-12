
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Features used for K-Means clustering
FEATURE_COLUMNS = [
    "SCC (cells/mL)",
    "Temperature (°C)",
    "Milk Yield (L/Day)"
]

DIAGNOSIS_COLUMN = "Mastitis Presence"


def clean_data(df):

    print("\n========== DATA CLEANING ==========")

    # 1. Create a copy
    df = df.copy()

    print("Original dataset shape:", df.shape)

    # 2. Remove duplicate records
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    print("Duplicates removed:", duplicate_count)

    # 3. Check required columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns not found: {missing_columns}"
        )

    # 4. Convert features to numeric
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # 5. Handle missing numerical values
    for col in FEATURE_COLUMNS:
        missing_count = df[col].isnull().sum()

        if missing_count > 0:
            median_value = df[col].median()

            if pd.isna(median_value):
                raise ValueError(
                    f"Column {col} contains no valid numerical values."
                )

            df[col] = df[col].fillna(median_value)

        print(f"{col}: {missing_count} missing values handled")

    # 6. Remove impossible measurement values
    # These are basic data-quality checks, not diagnostic thresholds.
    invalid = (
        (df["SCC (cells/mL)"] < 0) |
        (df["Temperature (°C)"] <= 0) |
        (df["Milk Yield (L/Day)"] < 0)
    )

    print("Invalid measurement rows:", invalid.sum())

    df = df.loc[~invalid].copy()

    # 7. Reset index
    df.reset_index(drop=True, inplace=True)

    print("\nCleaned dataset shape:", df.shape)

    return df


def scale_features(df):

    print("\n========== FEATURE STANDARDIZATION ==========")

    X = df[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS,
        index=df.index
    )

    print("\nStandardization completed!")

    print("\nStandardized feature preview:")
    print(scaled_df.head())

    return scaled_df, scaler


def preprocess_data(df):

    cleaned_df = clean_data(df)

    scaled_df, scaler = scale_features(cleaned_df)

    return cleaned_df, scaled_df, scaler
