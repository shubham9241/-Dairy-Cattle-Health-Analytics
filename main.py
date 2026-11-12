
from pathlib import Path
import joblib

from src.data_loader import load_cattle_data
from src.preprocessing import preprocess_data

from src.clustering import (
    find_optimal_clusters,
    train_kmeans,
    assign_risk_tiers
)

from src.evaluation import (
    evaluate_clustering,
    validate_diagnosis,
    create_cluster_report
)

from src.visualization import (
    plot_pca_clusters,
    plot_risk_distribution,
    plot_cluster_profiles
)


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n" + "=" * 55)
    print("SIH26109 - DAIRY CATTLE MASTITIS RISK SEGMENTATION")
    print("=" * 55)

    # --------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------

    print("\n[STEP 1] Loading Cattle Health Dataset")

    df = load_cattle_data()

    print("Dataset loaded successfully!")
    print("Total records:", df.shape[0])
    print("Total columns:", df.shape[1])

    # --------------------------------------
    # STEP 2: DATA PREPROCESSING
    # --------------------------------------

    print("\n[STEP 2] Data Cleaning and Preprocessing")

    cleaned_df, scaled_df, scaler = preprocess_data(df)

    print("Data preprocessing completed!")

    # --------------------------------------
    # STEP 3: FIND OPTIMAL CLUSTERS
    # --------------------------------------

    print("\n[STEP 3] Finding Optimal Number of Clusters")

    k_values, inertia_values = find_optimal_clusters(
        scaled_df,
        max_k=6
    )

    print("Elbow analysis completed!")

    # --------------------------------------
    # STEP 4: TRAIN K-MEANS MODEL
    # --------------------------------------

    print("\n[STEP 4] Training K-Means Clustering Model")

    model, labels, score = train_kmeans(
        scaled_df,
        n_clusters=3
    )

    print("K-Means model trained successfully!")

    # --------------------------------------
    # STEP 5: RISK SEGMENTATION
    # --------------------------------------

    print("\n[STEP 5] Assigning Cattle Risk Segments")

    results, cluster_summary = assign_risk_tiers(
        cleaned_df,
        model,
        labels
    )

    print("Risk segmentation completed!")

    # --------------------------------------
    # STEP 6: MODEL EVALUATION
    # --------------------------------------

    print("\n[STEP 6] Evaluating Clustering Performance")

    score = evaluate_clustering(
        scaled_df,
        labels
    )

    diagnosis_result = validate_diagnosis(results)

    performance_report = create_cluster_report(results)

    print("Model evaluation completed!")

    # --------------------------------------
    # STEP 7: GENERATE VISUALIZATIONS
    # --------------------------------------

    print("\n[STEP 7] Generating Visualizations")

    plot_pca_clusters(
        scaled_df,
        labels
    )

    plot_risk_distribution(results)

    plot_cluster_profiles(cluster_summary)

    print("All visualizations generated!")

    # --------------------------------------
    # STEP 8: SAVE RESULTS
    # --------------------------------------

    print("\n[STEP 8] Saving Project Results")

    results.to_csv(
        PROCESSED_DIR / "cattle_risk_results.csv",
        index=False
    )

    cluster_summary.to_csv(
        PROCESSED_DIR / "cluster_summary.csv"
    )

    performance_report.to_csv(
        OUTPUT_DIR / "model_performance_report.csv"
    )

    # --------------------------------------
    # STEP 9: SAVE TRAINED MODEL
    # --------------------------------------

    print("\n[STEP 9] Saving Trained Model")

    joblib.dump(
        model,
        MODEL_DIR / "kmeans_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    print("Model saved successfully!")

    # --------------------------------------
    # STEP 10: FINAL SUMMARY
    # --------------------------------------

    print("\n" + "=" * 55)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("\nDataset Records:", len(results))
    print("Number of Clusters:", model.n_clusters)
    print("Silhouette Score:", score)

    print("\nRisk Distribution:")

    print(
        results["Risk Tier"].value_counts()
    )

    print("\nGenerated Files:")

    print("1. dataset/processed/cattle_risk_results.csv")
    print("2. dataset/processed/cluster_summary.csv")
    print("3. outputs/model_performance_report.csv")
    print("4. outputs/elbow_method.png")
    print("5. outputs/pca_clusters.png")
    print("6. outputs/risk_distribution.png")
    print("7. outputs/cluster_profiles.png")
    print("8. models/kmeans_model.pkl")
    print("9. models/scaler.pkl")

    print("\n" + "=" * 55)
    print("END OF PROJECT EXECUTION")
    print("=" * 55)


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
