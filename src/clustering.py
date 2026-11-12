
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "SCC (cells/mL)",
    "Temperature (°C)",
    "Milk Yield (L/Day)"
]


def find_optimal_clusters(X, max_k=6):
    print("\n========== ELBOW METHOD ==========")

    n_samples = len(X)

    if n_samples < 3:
        raise ValueError("At least 3 records are required.")

    max_k = min(max_k, n_samples - 1)

    k_values = []
    inertia_values = []

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        k_values.append(k)
        inertia_values.append(model.inertia_)

        print(f"K = {k}, Inertia = {model.inertia_:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=300)
    plt.close()

    return k_values, inertia_values


def train_kmeans(X, n_clusters=3):
    print("\n========== K-MEANS TRAINING ==========")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    unique_clusters = np.unique(labels)

    if 1 < len(unique_clusters) < len(labels):
        score = silhouette_score(X, labels)
    else:
        score = None

    print("K-Means training completed!")
    print("Number of clusters:", len(unique_clusters))
    print("Silhouette Score:", score)

    return model, labels, score


def assign_risk_tiers(cleaned_df, model, cluster_labels):
    print("\n========== RISK SEGMENTATION ==========")

    summary = cleaned_df[FEATURES].copy()
    summary["Cluster"] = cluster_labels

    cluster_means = summary.groupby("Cluster")[FEATURES].mean()

    # Preliminary ranking, not a clinically validated diagnosis.
    centers = model.cluster_centers_

    risk_scores = (
        centers[:, 0] +
        centers[:, 1] -
        centers[:, 2]
    )

    ordered_clusters = np.argsort(risk_scores)

    risk_mapping = {
        int(ordered_clusters[0]): "Low Risk",
        int(ordered_clusters[1]): "Medium Risk",
        int(ordered_clusters[2]): "High Risk"
    }

    results = cleaned_df.copy()
    results["Cluster"] = cluster_labels
    results["Risk Tier"] = [
        risk_mapping[int(c)] for c in cluster_labels
    ]

    cluster_means["Risk Score"] = risk_scores
    cluster_means["Risk Tier"] = [
        risk_mapping[int(c)] for c in cluster_means.index
    ]

    print("\nCluster Health Summary:")
    print(cluster_means)

    print("\nCattle Risk Distribution:")
    print(results["Risk Tier"].value_counts())

    return results, cluster_means


def validate_cluster_quality(X, labels):
    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2 or len(unique_clusters) >= len(X):
        return None

    return silhouette_score(X, labels)
