
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path("data_lyrics")

EMBEDDINGS_PATH = BASE_DIR / "embeddings/v2/artist_embeddings.npy"
METADATA_PATH = BASE_DIR / "embeddings/v2/artist_metadata.csv"

COMMUNITIES_PATH = (
    Path("data_network/1_new_2005_2025/network")
    / "network_deduplicated/louvain_communities.csv"
)

OUTPUT_DIR = BASE_DIR / "semantic_analysis/embedding_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 50
N_NEIGHBORS = 5

# Number of randomly selected pairs per interval
N_RANDOM_PAIRS = 50

RANDOM_SEED = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    embeddings = np.load(EMBEDDINGS_PATH)

    metadata = pd.read_csv(METADATA_PATH)
    communities = pd.read_csv(COMMUNITIES_PATH)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Number of embeddings does not match metadata rows."
        )

    metadata = metadata.copy()
    metadata["embedding_index"] = np.arange(len(metadata))

    df = metadata.merge(
        communities[["mbid", "original_community"]],
        left_on="artist_mbid",
        right_on="mbid",
        how="inner",
        validate="one_to_one"
    )

    embeddings = embeddings[df["embedding_index"].to_numpy()]

    df = df.reset_index(drop=True)

    print("\nDATA MATCHING")
    print(f"Artists with embeddings: {len(metadata)}")
    print(f"Artists matched to communities: {len(df)}")

    return df, embeddings


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def calculate_similarity_matrix(embeddings):

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)

    return normalized @ normalized.T


# --------------------------------------------------
# CREATE PAIR DATAFRAME
# --------------------------------------------------

def create_pair_dataframe(df, similarity_matrix):

    names = df["artist"].to_numpy()
    mbids = df["artist_mbid"].to_numpy()
    songs = df["num_songs"].to_numpy()
    communities = df["original_community"].to_numpy()

    i, j = np.triu_indices(len(df), k=1)
    similarities = similarity_matrix[i, j]

    pairs = pd.DataFrame({
        "artist_1": names[i],
        "artist_2": names[j],
        "mbid_1": mbids[i],
        "mbid_2": mbids[j],
        "similarity": similarities,
        "songs_1": songs[i],
        "songs_2": songs[j],
        "community_1": communities[i],
        "community_2": communities[j],
        "same_community": communities[i] == communities[j]
    })

    return pairs


# --------------------------------------------------
# TOP SIMILAR PAIRS
# --------------------------------------------------

def save_top_pairs(pairs):

    top_pairs = pairs.nlargest(TOP_N, "similarity")

    top_pairs.to_csv(
        OUTPUT_DIR / "top_50_similar_artist_pairs.csv",
        index=False
    )

    print("\nTOP 10 MOST SIMILAR PAIRS")
    print(
        top_pairs[
            ["artist_1", "artist_2", "similarity", "same_community"]
        ].head(10).to_string(index=False)
    )

    print("\nTOP PAIRS: SAME COMMUNITY")
    print(
        top_pairs[top_pairs["same_community"]][
            ["artist_1", "artist_2", "similarity"]
        ].head(10).to_string(index=False)
    )

    print("\nTOP PAIRS: DIFFERENT COMMUNITIES")
    print(
        top_pairs[~top_pairs["same_community"]][
            ["artist_1", "artist_2", "similarity"]
        ].head(10).to_string(index=False)
    )


# --------------------------------------------------
# RANDOM PAIRS BY SIMILARITY RANGE
# --------------------------------------------------

def save_random_pairs(pairs):

    rng = np.random.default_rng(RANDOM_SEED)

    # Intervals are defined as cosine similarity
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

        sampled_indices = rng.choice(
            candidates.index.to_numpy(),
            size=n,
            replace=False
        )

        sample = candidates.loc[sampled_indices].copy()
        sample["similarity_group"] = label

        selected.append(sample)

        print(f"{label}: selected {n} pairs")

    if selected:

        random_pairs = pd.concat(selected, ignore_index=True)

        random_pairs.to_csv(
            OUTPUT_DIR / "random_pairs_by_similarity.csv",
            index=False
        )


# --------------------------------------------------
# NEAREST NEIGHBORS
# --------------------------------------------------

def save_nearest_neighbors(df, similarity_matrix):

    names = df["artist"].to_numpy()
    songs = df["num_songs"].to_numpy()
    communities = df["original_community"].to_numpy()

    nearest_neighbors = []

    for artist_idx in range(len(df)):

        similarities = similarity_matrix[artist_idx].copy()

        # Exclude artist itself
        similarities[artist_idx] = -np.inf

        nearest = np.argsort(similarities)[::-1][:N_NEIGHBORS]

        for rank, neighbor_idx in enumerate(nearest, start=1):

            nearest_neighbors.append({
                "artist": names[artist_idx],
                "neighbor": names[neighbor_idx],
                "similarity": similarities[neighbor_idx],
                "artist_songs": songs[artist_idx],
                "neighbor_songs": songs[neighbor_idx],
                "artist_community": communities[artist_idx],
                "neighbor_community": communities[neighbor_idx],
                "same_community": (
                    communities[artist_idx]
                    == communities[neighbor_idx]
                ),
                "rank": rank
            })

    neighbors_df = pd.DataFrame(nearest_neighbors)

    neighbors_df.to_csv(
        OUTPUT_DIR / "nearest_neighbors_per_artist.csv",
        index=False
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df, embeddings = load_data()

    similarity_matrix = calculate_similarity_matrix(embeddings)

    pairs = create_pair_dataframe(df, similarity_matrix)

    save_top_pairs(pairs)
    save_random_pairs(pairs)
    save_nearest_neighbors(df, similarity_matrix)

    print("\nINSPECTION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
