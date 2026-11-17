
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path("data_lyrics")

EMBEDDINGS_PATH = BASE_DIR / "embeddings/v2/song_embeddings.npy"
METADATA_PATH = BASE_DIR / "embeddings/v2/song_metadata.csv"

OUTPUT_DIR = BASE_DIR / "semantic_analysis/song_embedding_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 50
N_NEIGHBORS = 5
N_RANDOM_PAIRS = 50

RANDOM_SEED = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Number of song embeddings does not match metadata rows."
        )

    required_columns = [
        "recording_mbid",
        "artist_mbid",
        "artist",
        "recording_name"
    ]

    missing = set(required_columns) - set(metadata.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    print("\nDATA")
    print(f"Songs: {len(metadata)}")
    print(f"Unique artists: {metadata['artist_mbid'].nunique()}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return metadata, embeddings


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def calculate_similarity_matrix(embeddings):

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    normalized = embeddings / np.maximum(norms, 1e-12)

    return normalized @ normalized.T


# --------------------------------------------------
# CREATE SONG PAIR DATAFRAME
# --------------------------------------------------

def create_pair_dataframe(metadata, similarity_matrix):

    artist_names = metadata["artist"].to_numpy()
    artist_mbids = metadata["artist_mbid"].to_numpy()
    song_names = metadata["recording_name"].to_numpy()
    recording_mbids = metadata["recording_mbid"].to_numpy()

    i, j = np.triu_indices(len(metadata), k=1)

    similarities = similarity_matrix[i, j]

    pairs = pd.DataFrame({
        "artist_1": artist_names[i],
        "song_1": song_names[i],
        "recording_mbid_1": recording_mbids[i],
        "artist_2": artist_names[j],
        "song_2": song_names[j],
        "recording_mbid_2": recording_mbids[j],
        "similarity": similarities,
        "same_artist": artist_mbids[i] == artist_mbids[j]
    })

    return pairs


# --------------------------------------------------
# TOP SIMILAR SONG PAIRS
# --------------------------------------------------

def save_top_pairs(pairs):

    top_pairs = pairs.nlargest(TOP_N, "similarity")

    top_pairs.to_csv(
        OUTPUT_DIR / "top_50_similar_song_pairs.csv",
        index=False
    )

    print("\nTOP 10 MOST SIMILAR SONG PAIRS")

    print(
        top_pairs[
            [
                "artist_1",
                "song_1",
                "artist_2",
                "song_2",
                "similarity",
                "same_artist"
            ]
        ].head(10).to_string(index=False)
    )


# --------------------------------------------------
# RANDOM PAIRS BY SIMILARITY RANGE
# --------------------------------------------------

def save_random_pairs(pairs):

    rng = np.random.default_rng(RANDOM_SEED)

    ranges = {
        "high": (0.80, 1.01),
        "medium": (0.60, 0.70),
        "low": (-1.00, 0.40)
    }

    selected = []

    for label, (lower, upper) in ranges.items():

        candidates = pairs[
            (pairs["similarity"] >= lower)
            & (pairs["similarity"] < upper)
        ]

        n = min(N_RANDOM_PAIRS, len(candidates))

        if n == 0:
            print(f"No pairs found in range: {label}")
            continue

        indices = rng.choice(
            candidates.index.to_numpy(),
            size=n,
            replace=False
        )

        sample = candidates.loc[indices].copy()
        sample["similarity_group"] = label

        selected.append(sample)

        print(f"{label}: selected {n} pairs")

    if selected:

        random_pairs = pd.concat(selected, ignore_index=True)

        random_pairs.to_csv(
            OUTPUT_DIR / "random_song_pairs.csv",
            index=False
        )


# --------------------------------------------------
# NEAREST NEIGHBORS PER SONG
# --------------------------------------------------

def save_nearest_neighbors(metadata, similarity_matrix):

    artists = metadata["artist"].to_numpy()
    songs = metadata["recording_name"].to_numpy()

    recording_mbids = metadata["recording_mbid"].to_numpy()

    nearest_neighbors = []

    for song_idx in range(len(metadata)):

        similarities = similarity_matrix[song_idx].copy()

        # Exclude the song itself
        similarities[song_idx] = -np.inf

        nearest = np.argsort(similarities)[::-1][:N_NEIGHBORS]

        for rank, neighbor_idx in enumerate(nearest, start=1):

            nearest_neighbors.append({
                "artist": artists[song_idx],
                "song": songs[song_idx],
                "recording_mbid": recording_mbids[song_idx],
                "neighbor_artist": artists[neighbor_idx],
                "neighbor_song": songs[neighbor_idx],
                "neighbor_recording_mbid": recording_mbids[neighbor_idx],
                "similarity": similarities[neighbor_idx],
                "same_artist": (
                    artists[song_idx] == artists[neighbor_idx]
                ),
                "rank": rank
            })

    neighbors_df = pd.DataFrame(nearest_neighbors)

    neighbors_df.to_csv(
        OUTPUT_DIR / "nearest_neighbors_per_song.csv",
        index=False
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    metadata, embeddings = load_data()

    similarity_matrix = calculate_similarity_matrix(embeddings)

    pairs = create_pair_dataframe(metadata, similarity_matrix)

    save_top_pairs(pairs)
    save_random_pairs(pairs)
    save_nearest_neighbors(metadata, similarity_matrix)

    print("\nINSPECTION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
