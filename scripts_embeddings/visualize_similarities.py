
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data/data_2005_2025_artist_embeddings/semantic_analysis_2/pairwise_similarities.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data/data_2005_2025_artist_embeddings/semantic_analysis_2/figures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    df = pd.read_csv(INPUT_PATH)

    within = df.loc[
        df["same_community"], "cosine_similarity"
    ]

    between = df.loc[
        ~df["same_community"], "cosine_similarity"
    ]

    plt.figure(figsize=(9, 5))

    plt.hist(
        between,
        bins=50,
        density=True,
        alpha=0.6,
        label=f"Between communities (mean={between.mean():.3f})"
    )

    plt.hist(
        within,
        bins=50,
        density=True,
        alpha=0.6,
        label=f"Within communities (mean={within.mean():.3f})"
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Semantic similarity within and between communities")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "similarity_distributions.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()
