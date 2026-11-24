from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
BACKBONE_DIR = (
    PROJECT_DIR / "data" / "data_2005_2025_network" / "network" / "backbone"
)
INPUT_FILE = BACKBONE_DIR / "all_edges_with_alpha.csv"
OUTPUT_FILE = BACKBONE_DIR / "alpha_distribution.png"
ALPHA_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]


def main():
    edges = pd.read_csv(INPUT_FILE)
    if "alpha_min" not in edges.columns:
        raise ValueError(f"Missing 'alpha_min' column in {INPUT_FILE}")

    counts = (
        pd.cut(edges["alpha_min"], bins=ALPHA_BINS, include_lowest=True)
        .value_counts()
        .sort_index()
    )

    labels = [
        f"{lower:.1f}–{upper:.1f}"
        for lower, upper in zip(ALPHA_BINS[:-1], ALPHA_BINS[1:])
    ]
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, counts.values, width=0.65)
    axis.set_xlabel("Minimum alpha value")
    axis.set_ylabel("Number of edges")
    axis.set_title("Distribution of edge alpha values")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    for bar, count in zip(bars, counts.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + 1,
            str(count),
            ha="center",
        )

    figure.tight_layout()
    figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved alpha distribution to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
