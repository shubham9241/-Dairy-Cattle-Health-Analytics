
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

from pathlib import Path


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "dataset" / "raw" / "cattle_health.csv"

RESULTS_PATH = (
    PROJECT_ROOT / "dataset" / "processed" /
    "cattle_risk_results.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT / "dataset" / "processed" /
    "cluster_summary.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

REPORT_PATH = PROJECT_ROOT / "outputs" / "model_performance_report.csv"

FEATURES = [
    "SCC (cells/mL)",
    "Temperature (°C)",
    "Milk Yield (L/Day)"
]

RISK_OPTIONS = ["Low Risk", "Medium Risk", "High Risk"]


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Dairy Cattle Health Analytics",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #16875d;
}

.subtitle {
    color: #777;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.08);
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🐄 Dairy Cattle Health Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'SIH26109 | Unsupervised Machine Learning Using K-Means'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 5. DATA LOADING FUNCTIONS
# ==========================================

@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)

    return None


@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        return pd.read_csv(REPORT_PATH)

    return None


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ==========================================
# 6. LOAD PROJECT DATA
# ==========================================

if not RAW_PATH.exists():

    st.error("Raw cattle dataset not found.")
    st.stop()

try:

    raw_df = load_raw_data()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


if not RESULTS_PATH.exists():

    st.error(
        "Clustering results not found. "
        "Please run python main.py first."
    )

    st.stop()


try:

    df = load_results()

except Exception as error:

    st.error(f"Results loading failed: {error}")
    st.stop()


required_columns = FEATURES + ["Cluster", "Risk Tier"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(f"Missing result columns: {missing_columns}")
    st.stop()


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🐄 Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Cattle Records",
        "Risk Analysis",
        "Individual Prediction",
        "Cow Health Profile",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "SIH26109\n\n"
    "Dairy Cattle Mastitis Risk Segmentation"
)

st.sidebar.caption(
    f"Total cattle records: {len(raw_df)}"
)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Overview":

    st.header("Project Overview")

    total_records = len(df)

    low_count = (df["Risk Tier"] == "Low Risk").sum()
    medium_count = (df["Risk Tier"] == "Medium Risk").sum()
    high_count = (df["Risk Tier"] == "High Risk").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cattle", total_records)
    col2.metric("Low Risk", low_count)
    col3.metric("Medium Risk", medium_count)
    col4.metric("High Risk", high_count)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        risk_counts = (
            df["Risk Tier"]
            .value_counts()
            .reindex(RISK_OPTIONS, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        risk_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Number of Cattle")
        ax.set_title("Cattle Risk Distribution")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("Mastitis Diagnosis Distribution")

        if "Mastitis Presence" in raw_df.columns:

            diagnosis_counts = (
                raw_df["Mastitis Presence"]
                .fillna("Unknown")
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            diagnosis_counts.plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )

            ax.set_ylabel("")

            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    st.subheader("Recent Cattle Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==========================================
# PAGE 2: CATTLE RECORDS
# ==========================================

elif page == "Cattle Records":

    st.header("Cattle Health Records")

    selected_risks = st.multiselect(
        "Filter by Risk Category",
        RISK_OPTIONS,
        default=RISK_OPTIONS
    )

    filtered_df = df[
        df["Risk Tier"].isin(selected_risks)
    ]

    st.write(f"Showing {len(filtered_df)} records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Cattle Records",
        data=csv_data,
        file_name="cattle_health_records.csv",
        mime="text/csv"
    )


# ==========================================
# PAGE 3: RISK ANALYSIS
# ==========================================

elif page == "Risk Analysis":

    st.header("Cattle Risk Analysis")

    risk_counts = (
        df["Risk Tier"]
        .value_counts()
        .reindex(RISK_OPTIONS, fill_value=0)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Category Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        risk_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("Cluster Distribution")

        cluster_counts = (
            df["Cluster"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        cluster_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Average Health Measurements")

    profile = df.groupby("Risk Tier")[FEATURES].mean()

    st.dataframe(
        profile.reindex(RISK_OPTIONS),
        use_container_width=True
    )

    st.bar_chart(profile)

    # --------------------------------------
    # MODEL PERFORMANCE REPORT
    # --------------------------------------

    st.divider()

    st.subheader("Model Performance Report")

    if REPORT_PATH.exists():

        try:

            report = load_report()

            st.dataframe(
                report,
                use_container_width=True
            )

            st.download_button(
                "Download Performance Report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="model_performance_report.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(f"Report loading failed: {error}")

    else:

        st.warning(
            "Performance report not found. "
            "Run python main.py first."
        )


# ==========================================
# PAGE 4: INDIVIDUAL PREDICTION
# ==========================================

elif page == "Individual Prediction":

    st.header("Individual Cattle Risk Prediction")

    st.write(
        "Enter health measurements to identify the nearest "
        "K-Means cluster."
    )

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():

        st.warning(
            "Model files not found. Run python main.py first."
        )

    else:

        try:

            model, scaler = load_model()

            if model.n_clusters != 3:

                st.error(
                    "This risk mapping requires exactly 3 clusters."
                )

                st.stop()

        except Exception as error:

            st.error(f"Model loading failed: {error}")
            st.stop()

        with st.form("prediction_form"):

            st.subheader("Cattle Health Measurements")

            col1, col2, col3 = st.columns(3)

            with col1:

                scc = st.number_input(
                    "SCC (cells/mL)",
                    min_value=0.0,
                    value=200000.0,
                    step=10000.0
                )

            with col2:

                temperature = st.number_input(
                    "Temperature (°C)",
                    min_value=1.0,
                    value=38.5,
                    step=0.1
                )

            with col3:

                milk_yield = st.number_input(
                    "Milk Yield (L/Day)",
                    min_value=0.0,
                    value=15.0,
                    step=0.5
                )

            submitted = st.form_submit_button(
                "Predict Cattle Risk",
                use_container_width=True
            )

        if submitted:

            input_data = pd.DataFrame(
                [[scc, temperature, milk_yield]],
                columns=FEATURES
            )

            try:

                scaled_input = scaler.transform(input_data)

                predicted_cluster = int(
                    model.predict(scaled_input)[0]
                )

                centers = model.cluster_centers_

                risk_scores = (
                    centers[:, 0]
                    + centers[:, 1]
                    - centers[:, 2]
                )

                ordered_clusters = np.argsort(risk_scores)

                risk_mapping = {
                    int(ordered_clusters[0]): "Low Risk",
                    int(ordered_clusters[1]): "Medium Risk",
                    int(ordered_clusters[2]): "High Risk"
                }

                predicted_risk = risk_mapping[predicted_cluster]

                st.divider()

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Cluster",
                    predicted_cluster
                )

                col2.metric(
                    "Preliminary Risk Segment",
                    predicted_risk
                )

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

                st.info(
                    "This is an experimental cluster-based risk segment, "
                    "not a clinical diagnosis or validated mastitis "
                    "probability. Consult a veterinarian for assessment."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# ==========================================
# PAGE 5: INDIVIDUAL COW HEALTH PROFILE
# ==========================================

elif page == "Cow Health Profile":

    st.header("Individual Cow Health Profile")

    cow_ids = (
        raw_df["Cow ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_cow = st.selectbox(
        "Select Cow ID",
        sorted(cow_ids)
    )

    cow = raw_df[
        raw_df["Cow ID"].astype(str) == selected_cow
    ]

    if not cow.empty:

        record = cow.iloc[0]

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Milk Yield",
            f"{record['Milk Yield (L/Day)']} L/day"
        )

        col2.metric(
            "SCC",
            f"{record['SCC (cells/mL)']:,.0f}"
        )

        col3.metric(
            "Temperature",
            f"{record['Temperature (°C)']} °C"
        )

        st.divider()

        st.subheader("Additional Health Information")

        details = pd.DataFrame({
            "Health Parameter": [
                "Lactation Stage",
                "Udder Health",
                "Historical Mastitis Cases",
                "Protein Level",
                "Mastitis Presence",
                "Mastitis Type"
            ],

            "Value": [
                record["Lactation Stage"],
                record["Udder Health Metrics"],
                record["Historical Mastitis Cases"],
                record["Protein Level (%)"],
                record["Mastitis Presence"],
                record["Mastitis Type"]
            ]
        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Cow Health Measurements")

        numeric_features = pd.DataFrame({
            "Measurement": [
                "SCC",
                "Temperature",
                "Milk Yield",
                "Protein Level"
            ],

            "Value": [
                record["SCC (cells/mL)"],
                record["Temperature (°C)"],
                record["Milk Yield (L/Day)"],
                record["Protein Level (%)"]
            ]
        })

        fig, ax = plt.subplots(figsize=(9, 4))

        numeric_features.plot(
            x="Measurement",
            y="Value",
            kind="bar",
            ax=ax,
            legend=False
        )

        ax.set_title(f"Health Measurements - Cow {selected_cow}")
        ax.set_ylabel("Measured Value")
        ax.set_xlabel("Health Parameter")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "This is the available health profile for one cow. "
            "The dataset does not contain dates or repeated "
            "measurements, so this is not a historical trend."
        )


# ==========================================
# PAGE 6: ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About the Project")

    st.markdown("""
    ### Project Name

    **K-Means Clustering of Dairy Farm Cattle Health Records
    for Early Bovine Mastitis Risk Segmentation**

    ### Project ID

    SIH26109

    ### Machine Learning Approach

    Unsupervised Machine Learning

    ### Algorithm

    K-Means Clustering

    ### Input Features

    - Somatic Cell Count (SCC)
    - Temperature
    - Milk Yield

    ### Dataset Information

    - Cow ID
    - Lactation Stage
    - Udder Health Metrics
    - Historical Mastitis Cases
    - Protein Level
    - Mastitis Presence
    - Mastitis Type

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit

    ### Project Objective

    Group cattle health records into clusters based on their
    health measurements to support preliminary mastitis risk
    assessment and veterinary review.
    """)

    st.warning(
        "Risk categories are experimental interpretations of "
        "cluster profiles, not veterinary diagnoses. Reliable "
        "early warning requires dated longitudinal records "
        "and clinical validation."
    )


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "SIH26109 | Dairy Cattle Mastitis Risk Segmentation | "
    "Unsupervised Machine Learning Project"
)
