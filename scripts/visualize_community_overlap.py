
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025"

BACKBONE_FILE = (
    DATA_DIR / "backbone_mod_alpha042"
    / "tables_communities_325_326" / "mod_10_0780.csv"
)

ORIGINAL_FILE = (
    DATA_DIR / "tables_commuties307" / "mod_19_0685.csv"
)

OUTPUT_DIR = DATA_DIR / "community_analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.rename(columns={
        "id": "Id",
        "label": "Label",
        "modularity_class": "community"
    })

    df["Id"] = df["Id"].astype(str)
    df["community"] = df["community"].astype(int)

    return df


def main():
    backbone = load_data(BACKBONE_FILE)
    original = load_data(ORIGINAL_FILE)

    # Find the 10 largest communities in each network
    top_backbone = (
        backbone["community"].value_counts().head(10).index.tolist()
    )
    top_original = (
        original["community"].value_counts().head(10).index.tolist()
    )

    # Create a Jaccard similarity matrix
    matrix = np.zeros((len(top_backbone), len(top_original)))

    for i, b_id in enumerate(top_backbone):
        b_artists = set(
            backbone.loc[backbone["community"] == b_id, "Id"]
        )

        for j, o_id in enumerate(top_original):
            o_artists = set(
                original.loc[original["community"] == o_id, "Id"]
            )

            intersection = len(b_artists & o_artists)
            union = len(b_artists | o_artists)

            matrix[i, j] = intersection / union if union else 0

    # Labels including community sizes
    backbone_labels = [
        f"B{c} (n={sum(backbone['community'] == c)})"
        for c in top_backbone
    ]

    original_labels = [
        f"O{c} (n={sum(original['community'] == c)})"
        for c in top_original
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))

    im = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(top_original)))
    ax.set_xticklabels(original_labels, rotation=45, ha="right")

    ax.set_yticks(range(len(top_backbone)))
    ax.set_yticklabels(backbone_labels)

    ax.set_xlabel("Original communities")
    ax.set_ylabel("Backbone communities")
    ax.set_title("Community overlap: Backbone vs. Original")

    # Add values inside cells
    for i in range(len(top_backbone)):
        for j in range(len(top_original)):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black"
            )

    fig.colorbar(im, ax=ax, label="Jaccard similarity")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "community_overlap_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
