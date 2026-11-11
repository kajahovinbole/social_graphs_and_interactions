
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDINGS_PATH = Path(
    "data_lyrics/embeddings/artist_embeddings.npy"
)

ARTIST_METADATA_PATH = Path(
    "data_lyrics/embeddings/artist_metadata.csv"
)

NETWORK_METADATA_PATH = Path("data_network/data_2005_2025/data_table_nodes.csv")

OUTPUT_DIR = Path(
    "data_lyrics/semantic_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


# --------------------------------------------------
# Load and match data
# --------------------------------------------------

def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    artists = pd.read_csv(ARTIST_METADATA_PATH)
    network = pd.read_csv(NETWORK_METADATA_PATH)

    # Match artist embeddings to original communities
    network = network[["Id", "original_community"]].copy()

    network = network.rename(columns={
        "Id": "artist_mbid",
        "original_community": "community"
    })

    artists["artist_mbid"] = artists["artist_mbid"].astype(str)
    network["artist_mbid"] = network["artist_mbid"].astype(str)

    metadata = artists.merge(
        network,
        on="artist_mbid",
        how="left",
        validate="one_to_one"
    )

    if len(embeddings) != len(artists):
        raise ValueError("Embedding and metadata dimensions do not match.")

    # Exclude artists without a community assignment
    valid = metadata["community"].notna().to_numpy()

    embeddings = embeddings[valid]
    metadata = metadata.loc[valid].reset_index(drop=True)

    if metadata["artist_mbid"].duplicated().any():
        raise ValueError("Duplicate artist IDs found.")

    print(f"Artists with embeddings and communities: {len(metadata)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Communities represented: {metadata['community'].nunique()}")

    return embeddings, metadata


# --------------------------------------------------
# Calculate pairwise cosine similarities
# --------------------------------------------------

def calculate_similarities(embeddings, metadata):

    similarity_matrix = cosine_similarity(embeddings)

    # Get unique pairs (A,B), excluding self-comparisons
    i, j = np.triu_indices(len(embeddings), k=1)

    similarities = similarity_matrix[i, j]

    communities = metadata["community"].to_numpy()
    same_community = communities[i] == communities[j]

    within = similarities[same_community]
    between = similarities[~same_community]

    observed_difference = within.mean() - between.mean()

    print("\nObserved results:")
    print(f"Within-community pairs: {len(within)}")
    print(f"Between-community pairs: {len(between)}")
    print(f"Mean within-community similarity: {within.mean():.4f}")
    print(f"Mean between-community similarity: {between.mean():.4f}")
    print(f"Difference (within - between): {observed_difference:.4f}")

    # Save pairwise results
    pairs = pd.DataFrame({
        "artist_1": metadata["artist_mbid"].to_numpy()[i],
        "artist_2": metadata["artist_mbid"].to_numpy()[j],
        "community_1": communities[i],
        "community_2": communities[j],
        "same_community": same_community,
        "cosine_similarity": similarities
    })

    pairs.to_csv(
        OUTPUT_DIR / "pairwise_similarities.csv",
        index=False
    )

    return similarity_matrix, i, j, same_community, observed_difference


# --------------------------------------------------
# Permutation test
# --------------------------------------------------


def permutation_test(
    similarity_matrix,
    i,
    j,
    same_community,
    observed_difference
):
    rng = np.random.default_rng(RANDOM_SEED)

    null_differences = []

    # Community assignments remain fixed
    for _ in tqdm(
        range(N_PERMUTATIONS),
        desc="Permutation test"
    ):
        # Randomly reassign embeddings to artists
        permutation = rng.permutation(len(similarity_matrix))

        # Calculate similarities using shuffled embeddings
        shuffled_similarities = similarity_matrix[
            permutation[i],
            permutation[j]
        ]

        # Keep the original community assignments
        within = shuffled_similarities[same_community]
        between = shuffled_similarities[~same_community]

        difference = within.mean() - between.mean()
        null_differences.append(difference)

    null_differences = np.array(null_differences)

    # One-sided permutation test
    p_value = (
        np.sum(null_differences >= observed_difference) + 1
    ) / (N_PERMUTATIONS + 1)

    print("\nPermutation test results:")
    print(f"Observed difference: {observed_difference:.4f}")
    print(f"Null mean difference: {null_differences.mean():.4f}")
    print(f"p-value: {p_value:.5f}")

    np.save(
        OUTPUT_DIR / "permutation_null_distribution.npy",
        null_differences
    )

    return p_value



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    embeddings, metadata = load_data()

    similarity_matrix, i, j, same, observed_difference = (
    calculate_similarities(embeddings, metadata)
    )
    
    similarities = similarity_matrix[i, j]

    p_value = permutation_test(
        similarity_matrix,
        i,
        j,
        same,
        observed_difference
    )

    results = pd.DataFrame([{
        "n_artists": len(metadata),
        "n_communities": metadata["community"].nunique(),
        "n_within_pairs": int(same.sum()),
        "n_between_pairs": int((~same).sum()),
        "mean_within_similarity": similarities[same].mean(),
        "mean_between_similarity": similarities[~same].mean(),
        "observed_difference": observed_difference,
        "permutation_p_value": p_value,
        "n_permutations": N_PERMUTATIONS
    }])

    results.to_csv(
        OUTPUT_DIR / "semantic_similarity_results.csv",
        index=False
    )

    print("\nAnalysis finished!")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
