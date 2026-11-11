
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDINGS_PATH = Path(
    "data_lyrics/embeddings/artist_embeddings.npy"
)

ARTIST_METADATA_PATH = Path(
    "data_lyrics/embeddings/artist_metadata.csv"
)

NETWORK_METADATA_PATH = Path(
    "data_network/data_2005_2025/data_table.csv"
)

OUTPUT_DIR = Path(
    "data_lyrics/visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_COMMUNITIES = 10


# --------------------------------------------------
# Load and merge data
# --------------------------------------------------

def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    artists = pd.read_csv(ARTIST_METADATA_PATH)
    network = pd.read_csv(NETWORK_METADATA_PATH)

    if len(embeddings) != len(artists):
        raise ValueError(
            "Number of embeddings does not match metadata rows."
        )

    # Select artist ID and original Louvain community
    network = network[["Id", "0"]].copy()

    network = network.rename(columns={
        "Id": "artist_mbid",
        "0": "community"
    })

    # Ensure consistent ID format
    artists["artist_mbid"] = artists["artist_mbid"].astype(str)
    network["artist_mbid"] = network["artist_mbid"].astype(str)

    # Merge artist embeddings metadata with communities
    metadata = artists.merge(
        network,
        on="artist_mbid",
        how="left"
    )

    missing = metadata["community"].isna().sum()

    print(f"Artists loaded: {len(metadata)}")
    print(f"Artists without community: {missing}")

    if missing > 0:
        print("Warning: Some artists could not be matched.")

    return embeddings, metadata


# --------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------

def calculate_umap(embeddings):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    return reducer.fit_transform(embeddings)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def plot_communities(coordinates, metadata):
    metadata = metadata.copy()

    metadata["UMAP_1"] = coordinates[:, 0]
    metadata["UMAP_2"] = coordinates[:, 1]

    # Count artists in each community
    community_sizes = metadata["community"].value_counts()

    # Select the largest communities
    top_communities = community_sizes.head(
        TOP_N_COMMUNITIES
    ).index

    plt.figure(figsize=(14, 10))

    # Plot all remaining artists in grey
    other = metadata[
        ~metadata["community"].isin(top_communities)
    ]

    plt.scatter(
        other["UMAP_1"],
        other["UMAP_2"],
        color="lightgrey",
        alpha=0.5,
        s=25,
        label="Other communities"
    )

    # Plot largest communities separately
    for community in top_communities:
        subset = metadata[
            metadata["community"] == community
        ]

        plt.scatter(
            subset["UMAP_1"],
            subset["UMAP_2"],
            s=45,
            alpha=0.85,
            label=f"Community {community} (n={len(subset)})"
        )

    plt.title(
        "UMAP Visualization of Artist Lyrics by Collaboration Community"
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "umap_communities.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nPlot saved to: {output_path}")

    # Save coordinates and community labels
    coordinates_path = OUTPUT_DIR / "umap_communities.csv"
    metadata.to_csv(coordinates_path, index=False)

    print(f"Coordinates saved to: {coordinates_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Loading data...")
    embeddings, metadata = load_data()

    print("\nCalculating UMAP...")
    coordinates = calculate_umap(embeddings)

    print("\nVisualizing communities...")
    plot_communities(coordinates, metadata)


if __name__ == "__main__":
    main()
