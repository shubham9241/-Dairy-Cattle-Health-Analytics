
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "cattle_health.csv"


def load_cattle_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def analyze_dataset(df):

    print("\n========== CATTLE DATASET ANALYSIS ==========")

    print("\n1. First Five Records")
    print(df.head())

    print("\n2. Dataset Shape")
    print(df.shape)

    print("\n3. Column Names")
    print(df.columns.tolist())

    print("\n4. Dataset Information")
    df.info()

    print("\n5. Missing Values")
    print(df.isnull().sum())

    print("\n6. Duplicate Records")
    print(df.duplicated().sum())

    print("\n7. Statistical Summary")
    print(df.describe(include="all"))


if __name__ == "__main__":

    data = load_cattle_data()

    analyze_dataset(data)
