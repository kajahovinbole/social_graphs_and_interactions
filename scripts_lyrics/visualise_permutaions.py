
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_DIR / "data_lyrics/semantic_analysis_2"

NULL_PATH = BASE_DIR / "permutation_null_distribution.npy"
RESULTS_PATH = BASE_DIR / "semantic_similarity_results.csv"

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    null_distribution = np.load(NULL_PATH)
    results = pd.read_csv(RESULTS_PATH)

    observed = results.loc[0, "observed_difference"]

    plt.figure(figsize=(9, 5))

    plt.hist(
        null_distribution,
        bins=50,
        density=True,
        alpha=0.7
    )

    plt.axvline(
        observed,
        linestyle="--",
        linewidth=2,
        label=f"Observed difference = {observed:.4f}"
    )

    plt.xlabel("Difference in mean cosine similarity")
    plt.ylabel("Density")
    plt.title("Permutation test null distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "permutation_null_distribution.png",
        dpi=300
    )

    plt.show()

    print("Observed difference:", observed)
    print("Maximum null difference:", null_distribution.max())
    print(
        "Permutations >= observed:",
        np.sum(null_distribution >= observed)
    )


if __name__ == "__main__":
    main()
