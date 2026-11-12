# SIH26109 – Dairy Cattle Mastitis Risk Segmentation

Unsupervised learning project using K-Means clustering to explore cattle health records.

## Project structure

- `dataset/raw/`: original CSV dataset
- `dataset/processed/`: cleaned dataset
- `src/`: data loading, preprocessing, clustering, evaluation and visualization
- `models/`: saved scaler and K-Means model
- `app/`: Streamlit dashboard
- `outputs/`: generated reports and plots

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place your dataset at `dataset/raw/cattle_health.csv`.

## Run

The project code will be added step by step. Do not run the dashboard until its implementation is complete.

```powershell
python main.py
```

## Important research note

K-Means is unsupervised: diagnosis labels must not be used as clustering inputs. Keep confirmed mastitis diagnosis records separate and use them only for external validation. Cluster numbers (0, 1, 2) do not inherently mean low, medium or high risk; interpret them using cluster feature summaries and veterinary guidance.

A cross-sectional dataset can support segmentation, but it cannot establish early prediction over time. To study early warning, use dated repeated measurements for each cow and diagnosis dates.
