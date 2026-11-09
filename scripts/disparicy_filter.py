
from pathlib import Path
import numpy as np
import pandas as pd



DATA_DIR = Path("data/data_2005_2025/deduplicated")
EDGES_FILE = DATA_DIR / "edges.csv"

OUTPUT_DIR = DATA_DIR / "backbone"
OUTPUT_DIR.mkdir(exist_ok=True)

# Significance thresholds to investigate
THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.60]


def calculate_disparity_filter(edges):
    edges = edges.copy()

    # Calculate degree and strength for each artist
    degree = {}
    strength = {}

    for _, edge in edges.iterrows():
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1

        strength[source] = strength.get(source, 0) + weight
        strength[target] = strength.get(target, 0) + weight

    def calculate_alpha(node, weight):
        k = degree[node]
        s = strength[node]

        if k <= 1:
            return 0.0

        p = weight / s

        return (1 - p) ** (k - 1)

    # Calculate significance from both endpoints
    edges["alpha_source"] = edges.apply(
        lambda row: calculate_alpha(row["source"], row["weight"]),
        axis=1
    )

    edges["alpha_target"] = edges.apply(
        lambda row: calculate_alpha(row["target"], row["weight"]),
        axis=1
    )

    edges["alpha_min"] = edges[
        ["alpha_source", "alpha_target"]
    ].min(axis=1)

    return edges


def main():
    edges = pd.read_csv(EDGES_FILE)

    print(f"Original edges: {len(edges)}")

    filtered = calculate_disparity_filter(edges)

    print("\nAlpha distribution:")
    print(filtered["alpha_min"].describe())

    print("\nNumber of edges in alpha intervals:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]

    print(
        pd.cut(filtered["alpha_min"], bins=bins)
        .value_counts()
        .sort_index()
    )

    for alpha_threshold in THRESHOLDS:
        # Keep edge if significant for at least one endpoint
        backbone = filtered[
            filtered["alpha_min"] < alpha_threshold
        ].copy()

        output_file = (
            OUTPUT_DIR /
            f"edges_backbone_alpha_{alpha_threshold:.2f}.csv"
        )

        backbone.to_csv(output_file, index=False)

        print(
            f"Alpha {alpha_threshold:.2f}: "
            f"{len(backbone)} edges retained "
            f"({len(backbone) / len(edges) * 100:.1f}%)"
        )

    # Save diagnostics for all original edges
    filtered.to_csv(
        OUTPUT_DIR / "all_edges_with_alpha.csv",
        index=False
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
