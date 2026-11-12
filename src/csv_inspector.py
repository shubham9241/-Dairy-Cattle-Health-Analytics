
import csv
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "raw"
    / "cattle_health.csv"
)

def inspect_dataset():

    if not DATA_PATH.exists():
        print("Dataset file not found:", DATA_PATH)
        return

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    print("\n========== DATASET ANALYSIS ==========")

    print("\nTotal Records:", len(rows))

    if not rows:
        print("Dataset is empty.")
        return

    print("\nColumn Names:")
    print(reader.fieldnames)

    print("\nFirst 5 Records:")

    for row in rows[:5]:
        print(row)

    print("\nMissing Values:")

    for column in reader.fieldnames:
        missing = sum(
            1 for row in rows
            if not row.get(column, "").strip()
        )
        print(column, ":", missing)

    print("\nDuplicate Records:", len(rows) - len({
        tuple(sorted(row.items())) for row in rows
    }))


if __name__ == "__main__":
    inspect_dataset()
