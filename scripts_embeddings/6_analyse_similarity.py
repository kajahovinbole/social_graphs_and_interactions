
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = (
    PROJECT_DIR
    / "data/data_2005_2025_artist_embeddings/embeddings/v2/artist_embeddings.npy"
)

ARTIST_METADATA_PATH = (
    PROJECT_DIR
    / "data/data_2005_2025_artist_embeddings/embeddings/v2/artist_metadata.csv"
)

NETWORK_METADATA_PATH = (
    PROJECT_DIR
    / "data/data_2005_2025_network/network/louvain_communities.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "data/data_2005_2025_artist_embeddings/semantic_analysis_2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


# --------------------------------------------------
# LOAD AND MATCH DATA
# --------------------------------------------------

def load_data():

    embeddings = np.load(EMBEDDINGS_PATH)
    artists = pd.read_csv(ARTIST_METADATA_PATH)
    network = pd.read_csv(NETWORK_METADATA_PATH)

    # Check embedding dimensions
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(embeddings) != len(artists):
        raise ValueError(
            "Embedding and metadata dimensions do not match."
        )

    # Check required columns
    required_artist_columns = {"artist_mbid"}
    required_network_columns = {"mbid", "original_community"}

    if not required_artist_columns.issubset(artists.columns):
        raise ValueError("Missing artist_mbid in artist metadata.")

    if not required_network_columns.issubset(network.columns):
        raise ValueError(
            "Missing mbid or original_community in network metadata."
        )

    # Prepare artist metadata
    artists["artist_mbid"] = artists["artist_mbid"].astype(str)

    # Prepare original Louvain communities
    network = network[
        ["mbid", "original_community"]
    ].copy()

    network = network.rename(columns={
        "mbid": "artist_mbid",
        "original_community": "community"
    })

    network["artist_mbid"] = network["artist_mbid"].astype(str)

    # Check for duplicate IDs
    if artists["artist_mbid"].duplicated().any():
        raise ValueError("Duplicate artist IDs in artist metadata.")

    if network["artist_mbid"].duplicated().any():
        raise ValueError("Duplicate artist IDs in Louvain metadata.")

    # Match embeddings to original communities
    metadata = artists.merge(
        network,
        on="artist_mbid",
        how="left",
        validate="one_to_one"
    )

    # Check matching
    valid = metadata["community"].notna().to_numpy()

    print("\nDATA MATCHING")
    print(f"Artists with embeddings: {len(artists)}")
    print(f"Artists matched to communities: {valid.sum()}")
    print(f"Artists without community: {(~valid).sum()}")

    # Exclude artists without a community assignment
    embeddings = embeddings[valid]
    metadata = metadata.loc[valid].reset_index(drop=True)

    if len(metadata) < 2:
        raise ValueError("Not enough matched artists.")

    # Check embedding values
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embeddings contain zero vectors.")

    print(f"\nArtists used in analysis: {len(metadata)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Communities represented: {metadata['community'].nunique()}")

    return embeddings, metadata


# --------------------------------------------------
# CALCULATE PAIRWISE COSINE SIMILARITIES
# --------------------------------------------------

def calculate_similarities(embeddings, metadata):

    similarity_matrix = cosine_similarity(embeddings)

    # Unique pairs (A, B), excluding self-comparisons
    i, j = np.triu_indices(len(embeddings), k=1)

    similarities = similarity_matrix[i, j]

    communities = metadata["community"].to_numpy()

    same_community = communities[i] == communities[j]

    within = similarities[same_community]
    between = similarities[~same_community]

    if len(within) == 0 or len(between) == 0:
        raise ValueError(
            "Not enough within-community or between-community pairs."
        )

    observed_difference = within.mean() - between.mean()

    print("\nOBSERVED RESULTS")
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

    return (
        similarity_matrix,
        i,
        j,
        same_community,
        observed_difference,
        similarities
    )


# --------------------------------------------------
# PERMUTATION TEST
# --------------------------------------------------

def permutation_test(
    similarity_matrix,
    i,
    j,
    same_community,
    observed_difference
):

    rng = np.random.default_rng(RANDOM_SEED)

    null_differences = np.empty(N_PERMUTATIONS)

    for iteration in tqdm(
        range(N_PERMUTATIONS),
        desc="Permutation test"
    ):

        # Randomly reassign embeddings to artist labels
        permutation = rng.permutation(len(similarity_matrix))

        # Calculate similarities using shuffled embeddings
        shuffled_similarities = similarity_matrix[
            permutation[i],
            permutation[j]
        ]

        # Keep original community assignments fixed
        within = shuffled_similarities[same_community]
        between = shuffled_similarities[~same_community]

        null_differences[iteration] = (
            within.mean() - between.mean()
        )

    # One-sided permutation test
    p_value = (
        np.sum(null_differences >= observed_difference) + 1
    ) / (N_PERMUTATIONS + 1)

    print("\nPERMUTATION TEST RESULTS")
    print(f"Observed difference: {observed_difference:.4f}")
    print(f"Null mean difference: {null_differences.mean():.4f}")
    print(f"Null standard deviation: {null_differences.std():.4f}")
    print(f"p-value: {p_value:.5f}")

    np.save(
        OUTPUT_DIR / "permutation_null_distribution.npy",
        null_differences
    )

    return p_value


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(
    metadata,
    same_community,
    similarities,
    observed_difference,
    p_value
):

    results = pd.DataFrame([{
        "network": "original_deduplicated",
        "n_artists": len(metadata),
        "n_communities": metadata["community"].nunique(),
        "n_within_pairs": int(same_community.sum()),
        "n_between_pairs": int((~same_community).sum()),
        "mean_within_similarity": similarities[same_community].mean(),
        "mean_between_similarity": similarities[~same_community].mean(),
        "observed_difference": observed_difference,
        "permutation_p_value": p_value,
        "n_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED
    }])

    results.to_csv(
        OUTPUT_DIR / "semantic_similarity_results.csv",
        index=False
    )

    print("\nFINAL RESULTS")
    print(results.T.to_string())


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    embeddings, metadata = load_data()

    (
        similarity_matrix,
        i,
        j,
        same_community,
        observed_difference,
        similarities
    ) = calculate_similarities(
        embeddings,
        metadata
    )

    p_value = permutation_test(
        similarity_matrix,
        i,
        j,
        same_community,
        observed_difference
    )

    save_results(
        metadata,
        same_community,
        similarities,
        observed_difference,
        p_value
    )

    print("\nAnalysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
