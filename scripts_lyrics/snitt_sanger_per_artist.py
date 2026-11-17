import pandas as pd

df = pd.read_csv("data_lyrics/embeddings/v2/artist_metadata.csv")

print("Number of artists:", len(df))
print("\nSongs per artist:")
print(df["num_songs"].describe())

print("\nMedian:", df["num_songs"].median())
print("Artists with 1 song:", (df["num_songs"] == 1).sum())
print("Artists with 5 or fewer songs:", (df["num_songs"] <= 5).sum())
print("Artists with 10 or more songs:", (df["num_songs"] >= 10).sum())