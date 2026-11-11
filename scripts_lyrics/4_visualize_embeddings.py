
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDINGS_PATH = Path("data_lyrics/embeddings/artist_embeddings.npy")
METADATA_PATH = Path("data_lyrics/embeddings/artist_metadata.csv")
OUTPUT_DIR = Path("data_lyrics/embeddings/visualizations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Dimension reduction methods
# --------------------------------------------------

def reduce_pca(embeddings):
    print("\nPCA selected.")
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def reduce_tsne(embeddings):
    print("\nt-SNE selected.")

    perplexity = float(input("Perplexity (default 30): ") or 30)
    learning_rate = input("Learning rate (default 'auto'): ") or "auto"

    if learning_rate != "auto":
        learning_rate = float(learning_rate)

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        max_iter=1000,
        random_state=42
    )

    return reducer.fit_transform(embeddings)


def reduce_svd(embeddings):
    print("\nTruncated SVD selected.")

    reducer = TruncatedSVD(
        n_components=2,
        random_state=42
    )

    return reducer.fit_transform(embeddings)


def reduce_umap(embeddings):
    print("\nUMAP selected.")

    n_neighbors = int(input("Number of neighbors (default 15): ") or 15)
    min_dist = float(input("Minimum distance (default 0.1): ") or 0.1)
    metric = input("Distance metric (default 'cosine'): ") or "cosine"

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )

    return reducer.fit_transform(embeddings)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def plot_embeddings(coordinates, metadata, method):
    metadata = metadata.copy()
    metadata["Dimension 1"] = coordinates[:, 0]
    metadata["Dimension 2"] = coordinates[:, 1]

    plt.figure(figsize=(12, 9))

    if "community" in metadata.columns:
        communities = metadata["community"].astype(str)

        scatter = plt.scatter(
            metadata["Dimension 1"],
            metadata["Dimension 2"],
            c=pd.factorize(communities)[0],
            cmap="tab20",
            s=35,
            alpha=0.8
        )

        plt.colorbar(scatter, label="Community")

    else:
        plt.scatter(
            metadata["Dimension 1"],
            metadata["Dimension 2"],
            s=35,
            alpha=0.8
        )

    plt.title(f"Artist Lyric Embeddings - {method}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"artist_embeddings_{method.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Save coordinates
    coordinates_path = OUTPUT_DIR / f"artist_embeddings_{method.lower()}.csv"
    metadata.to_csv(coordinates_path, index=False)

    print(f"\nPlot saved to: {output_path}")
    print(f"Coordinates saved to: {coordinates_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading embeddings and metadata...")

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings "
            f"but {len(metadata)} metadata rows."
        )

    print(f"Loaded {len(embeddings)} artists.")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    print("\nSelect dimensionality reduction method:")
    print("1. PCA")
    print("2. t-SNE")
    print("3. Truncated SVD")
    print("4. UMAP")

    choice = input("\nEnter choice (1-4): ").strip()

    methods = {
        "1": ("PCA", reduce_pca),
        "2": ("t-SNE", reduce_tsne),
        "3": ("SVD", reduce_svd),
        "4": ("UMAP", reduce_umap)
    }

    if choice not in methods:
        print("Invalid choice.")
        return

    method_name, reduction_function = methods[choice]

    coordinates = reduction_function(embeddings)

    plot_embeddings(coordinates, metadata, method_name)


if __name__ == "__main__":
    main()
