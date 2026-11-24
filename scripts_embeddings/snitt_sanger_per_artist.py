from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_FILE = (
    PROJECT_DIR / "data" / "data_2005_2025_artist_embeddings"
    / "embeddings" / "v2" / "artist_metadata.csv"
)

df = pd.read_csv(METADATA_FILE)

print("Number of artists:", len(df))
print("\nSongs per artist:")
print(df["num_songs"].describe())

print("\nMedian:", df["num_songs"].median())
print("Artists with 1 song:", (df["num_songs"] == 1).sum())
print("Artists with 5 or fewer songs:", (df["num_songs"] <= 5).sum())
print("Artists with 10 or more songs:", (df["num_songs"] >= 10).sum())
