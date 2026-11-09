
from pathlib import Path
from itertools import combinations

import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Folder containing the 10 CSV files
CSV_DIR = Path("data/data_2005_2025/tables_commuties307")

# Column names in Gephi exports
ID_COL = "Id"
COMMUNITY_COL = "modularity_class"


def load_communities(file_path):
    df = pd.read_csv(file_path)

    return df[[ID_COL, COMMUNITY_COL]].rename(
        columns={COMMUNITY_COL: file_path.stem}
    )


def main():
    files = sorted(CSV_DIR.glob("*.csv"))

    if len(files) < 2:
        raise ValueError("Need at least two CSV files.")

    print(f"Found {len(files)} CSV files")

    runs = [load_communities(f) for f in files]

    # Match artists across all runs using their IDs
    merged = runs[0]

    for run in runs[1:]:
        merged = merged.merge(run, on=ID_COL, how="inner")

    if len(merged) == 0:
        raise ValueError("No matching artist IDs found.")

    print(f"Artists compared: {len(merged)}")

    run_names = [col for col in merged.columns if col != ID_COL]

    # Calculate ARI for every pair of runs
    results = []

    for run1, run2 in combinations(run_names, 2):
        ari = adjusted_rand_score(
            merged[run1],
            merged[run2]
        )

        results.append({
            "Run 1": run1,
            "Run 2": run2,
            "ARI": ari
        })

    results_df = pd.DataFrame(results)

    print("\nPairwise ARI results:")
    print(results_df.to_string(index=False))

    print("\nSummary:")
    print(f"Mean ARI:   {results_df['ARI'].mean():.4f}")
    print(f"Min ARI:    {results_df['ARI'].min():.4f}")
    print(f"Max ARI:    {results_df['ARI'].max():.4f}")
    print(f"Std ARI:    {results_df['ARI'].std():.4f}")

    results_df.to_csv(
        CSV_DIR / "ari_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()
