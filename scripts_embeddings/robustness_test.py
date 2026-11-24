
from pathlib import Path

import numpy as np
import pandas as pd

# tester om det har noe å si at vi tar med artister med minst 3 sanger. 
# (vi hadde minst 1 før)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_DIR / "data" / "data_2005_2025_artist_embeddings"

EMBEDDINGS_PATH = BASE_DIR / "embeddings/v2/artist_embeddings.npy"
METADATA_PATH = BASE_DIR / "embeddings/v2/artist_metadata.csv"

COMMUNITIES_PATH = (
    PROJECT_DIR / "data" / "data_2005_2025_network"
    / "network" / "louvain_communities.csv"
)

OUTPUT_DIR = BASE_DIR / "semantic_analysis/robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SONGS = 3
N_PERMUTATIONS = 10_000
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

    indices = df["embedding_index"].to_numpy()
    embeddings = embeddings[indices]

    df = df.reset_index(drop=True)

    print("\nDATA MATCHING")
    print(f"Artists with embeddings: {len(metadata)}")
    print(f"Artists matched to communities: {len(df)}")
    print(
        f"Artists with at least {MIN_SONGS} songs: "
        f"{(df['num_songs'] >= MIN_SONGS).sum()}"
    )

    return df, embeddings


# --------------------------------------------------
# SIMILARITY ANALYSIS
# --------------------------------------------------

def run_analysis(df, embeddings, label):

    n_artists = len(df)

    if n_artists < 2:
        raise ValueError("Not enough artists for similarity analysis.")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Cosine similarity matrix
    similarity_matrix = embeddings @ embeddings.T

    communities = df["original_community"].to_numpy()

    # Upper triangle: count every pair once
    i, j = np.triu_indices(n_artists, k=1)

    within_mask = communities[i] == communities[j]
    between_mask = ~within_mask

    if not within_mask.any() or not between_mask.any():
        raise ValueError(
            "Need both within-community and between-community pairs."
        )

    similarities = similarity_matrix[i, j]

    within_mean = similarities[within_mask].mean()
    between_mean = similarities[between_mask].mean()

    observed_difference = within_mean - between_mean

    # Permutation test
    rng = np.random.default_rng(RANDOM_SEED)
    null_differences = np.empty(N_PERMUTATIONS)

    for p in range(N_PERMUTATIONS):

        permutation = rng.permutation(n_artists)

        permuted_similarities = (
            similarity_matrix[permutation[i], permutation[j]]
        )

        null_differences[p] = (
            permuted_similarities[within_mask].mean()
            - permuted_similarities[between_mask].mean()
        )

    # One-sided p-value with plus-one correction
    p_value = (
        1 + np.sum(null_differences >= observed_difference)
    ) / (N_PERMUTATIONS + 1)

    results = {
        "analysis": label,
        "n_artists": n_artists,
        "n_communities": df["original_community"].nunique(),
        "n_within_pairs": int(within_mask.sum()),
        "n_between_pairs": int(between_mask.sum()),
        "mean_within_similarity": within_mean,
        "mean_between_similarity": between_mean,
        "observed_difference": observed_difference,
        "permutation_p_value": p_value,
        "n_permutations": N_PERMUTATIONS
    }

    print(f"\n{'=' * 55}")
    print(label)
    print(f"{'=' * 55}")
    print(f"Artists: {n_artists}")
    print(f"Communities: {df['original_community'].nunique()}")
    print(f"Within-community similarity: {within_mean:.4f}")
    print(f"Between-community similarity: {between_mean:.4f}")
    print(f"Difference: {observed_difference:.4f}")
    print(f"Permutation p-value: {p_value:.5f}")

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df, embeddings = load_data()

    # Main analysis: all artists
    full_results = run_analysis(
        df,
        embeddings,
        "All artists"
    )

    # Robustness test: at least three songs
    mask = df["num_songs"] >= MIN_SONGS

    robust_df = df.loc[mask].reset_index(drop=True)
    robust_embeddings = embeddings[mask.to_numpy()]

    robust_results = run_analysis(
        robust_df,
        robust_embeddings,
        f"At least {MIN_SONGS} songs per artist"
    )

    # Save comparison
    comparison = pd.DataFrame([
        full_results,
        robust_results
    ])

    output_path = OUTPUT_DIR / "robustness_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nCOMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
