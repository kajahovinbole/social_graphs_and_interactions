from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
EDGES_FILE = PROJECT_DIR / "data" / "data_2005_2025_network" / "network" / "edges.csv"


def main():
    edges = pd.read_csv(EDGES_FILE)
    print(edges["weight"].describe())
    print("\nDistribution of edge weights:")
    print(edges["weight"].value_counts().sort_index())


if __name__ == "__main__":
    main()
