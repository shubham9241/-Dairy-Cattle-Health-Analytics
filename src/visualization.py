
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_pca_clusters(X_scaled, labels):

    print("\nGenerating PCA visualization...")

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=labels,
        palette="viridis",
        s=75
    )

    plt.title("K-Means Cattle Health Clusters (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(alpha=0.2)

    plt.savefig(
        OUTPUT_DIR / "pca_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("PCA graph saved.")


def plot_risk_distribution(results):

    print("Generating risk distribution graph...")

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=results,
        x="Risk Tier",
        hue="Risk Tier",
        order=["Low Risk", "Medium Risk", "High Risk"],
        palette="viridis",
        legend=False
    )

    plt.title("Cattle Mastitis Risk Distribution")
    plt.xlabel("Risk Tier")
    plt.ylabel("Number of Cattle")

    plt.savefig(
        OUTPUT_DIR / "risk_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Risk distribution graph saved.")


def plot_cluster_profiles(summary):

    print("Generating cluster profile graph...")

    features = [
        "SCC (cells/mL)",
        "Temperature (°C)",
        "Milk Yield (L/Day)"
    ]

    profile = summary[features]

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        profile,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd"
    )

    plt.title("Average Cattle Health Measurements by Cluster")
    plt.xlabel("Health Features")
    plt.ylabel("Cluster")

    plt.savefig(
        OUTPUT_DIR / "cluster_profiles.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Cluster profile graph saved.")
