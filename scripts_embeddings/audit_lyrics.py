
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025_artist_embeddings"
INPUT_FILE = DATA_DIR / "hentet_lyrics_2" / "lyrics.csv"

OUTPUT_DIR = DATA_DIR / "quality_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RANDOM_SAMPLES = 20
RANDOM_SEED = 42


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    """Normalize text for comparison."""

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_lyrics(text):
    """
    Normalize lyrics for duplicate detection.

    Removes differences in casing, punctuation and whitespace,
    but preserves the actual words.
    """

    return normalize_text(text)


def create_lyrics_hash(text):
    """Create a hash of normalized lyrics."""

    normalized = normalize_lyrics(text)

    if not normalized:
        return None

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "artist",
        "artist_mbid",
        "recording_name",
        "recording_mbid",
        "found",
        "instrumental",
        "status",
        "method",
        "matched_title",
        "matched_artist",
        "matched_album",
        "lyrics_length",
        "lyrics_text"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\nDATA")
    print("-" * 60)
    print(f"Total recordings: {len(df)}")
    print(f"Found lyrics: {df['found'].sum()}")
    print(f"Unique artists: {df['artist_mbid'].nunique()}")

    return df


# --------------------------------------------------
# 1. IDENTICAL LYRICS
# --------------------------------------------------

def find_duplicate_lyrics(df):

    print("\n1. CHECKING IDENTICAL LYRICS")

    # Only consider recordings with lyrics
    found = df[
        (df["found"] == True)
        & df["lyrics_text"].notna()
        & (df["lyrics_text"].astype(str).str.strip() != "")
    ].copy()

    found["lyrics_hash"] = (
        found["lyrics_text"].apply(create_lyrics_hash)
    )

    # Group recordings with identical normalized lyrics
    duplicate_groups = (
        found.groupby("lyrics_hash")
        .filter(lambda group: len(group) > 1)
    )

    results = []

    for group_id, (lyrics_hash, group) in enumerate(
        duplicate_groups.groupby("lyrics_hash"),
        start=1
    ):

        # Ignore repeated rows for the same recording
        unique_recordings = group.drop_duplicates(
            subset=["recording_mbid"]
        )

        if len(unique_recordings) <= 1:
            continue

        for _, row in unique_recordings.iterrows():

            results.append({
                "duplicate_group": group_id,
                "recording_mbid": row["recording_mbid"],
                "artist": row["artist"],
                "recording_name": row["recording_name"],
                "matched_artist": row["matched_artist"],
                "matched_title": row["matched_title"],
                "matched_album": row["matched_album"],
                "method": row["method"],
                "lyrics_length": row["lyrics_length"],
                "lyrics_hash": lyrics_hash,
                "n_recordings_in_group": len(unique_recordings)
            })

    result_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "exact_duplicate_lyrics.csv"

    result_df.to_csv(output_path, index=False)

    n_groups = result_df["duplicate_group"].nunique() if not result_df.empty else 0

    print(f"Duplicate groups: {n_groups}")
    print(f"Recording entries involved: {len(result_df)}")
    print(f"Saved: {output_path}")

    return result_df


# --------------------------------------------------
# 2. TITLE AND ARTIST MISMATCHES
# --------------------------------------------------

def find_metadata_mismatches(df):

    print("\n2. CHECKING METADATA MISMATCHES")

    found = df[
        (df["found"] == True)
        & df["lyrics_text"].notna()
    ].copy()

    # Normalize requested and matched metadata
    found["requested_title_norm"] = (
        found["recording_name"].apply(normalize_text)
    )

    found["matched_title_norm"] = (
        found["matched_title"].apply(normalize_text)
    )

    found["requested_artist_norm"] = (
        found["artist"].apply(normalize_text)
    )

    found["matched_artist_norm"] = (
        found["matched_artist"].apply(normalize_text)
    )

    # Identify differences after normalization
    found["title_mismatch"] = (
        found["requested_title_norm"]
        != found["matched_title_norm"]
    )

    found["artist_mismatch"] = (
        found["requested_artist_norm"]
        != found["matched_artist_norm"]
    )

    mismatches = found[
        found["title_mismatch"]
        | found["artist_mismatch"]
    ].copy()

    columns = [
        "recording_mbid",
        "artist",
        "recording_name",
        "matched_artist",
        "matched_title",
        "matched_album",
        "method",
        "title_mismatch",
        "artist_mismatch",
        "release",
        "release_year"
    ]

    # Keep only available columns
    columns = [
        col for col in columns
        if col in mismatches.columns
    ]

    mismatches = mismatches[columns]

    output_path = OUTPUT_DIR / "metadata_mismatches.csv"

    mismatches.to_csv(output_path, index=False)

    print(f"Title mismatches: {mismatches['title_mismatch'].sum()}")
    print(f"Artist mismatches: {mismatches['artist_mismatch'].sum()}")
    print(f"Total rows with mismatches: {len(mismatches)}")
    print(f"Saved: {output_path}")

    return mismatches


# --------------------------------------------------
# 3. RANDOM MANUAL REVIEW
# --------------------------------------------------

def create_random_sample(df):

    print("\n3. CREATING RANDOM MANUAL REVIEW SAMPLE")

    found = df[
        (df["found"] == True)
        & df["lyrics_text"].notna()
        & (df["lyrics_text"].astype(str).str.strip() != "")
    ].copy()

    n = min(N_RANDOM_SAMPLES, len(found))

    sample = found.sample(
        n=n,
        random_state=RANDOM_SEED
    ).copy()

    # Include lyrics for manual inspection
    sample["manual_review"] = ""
    sample["review_notes"] = ""

    columns = [
        "artist",
        "recording_name",
        "recording_mbid",
        "release",
        "matched_artist",
        "matched_title",
        "matched_album",
        "method",
        "lyrics_length",
        "lyrics_text",
        "manual_review",
        "review_notes"
    ]

    sample = sample[
        [col for col in columns if col in sample.columns]
    ]

    output_path = OUTPUT_DIR / "random_manual_review.csv"

    sample.to_csv(output_path, index=False)

    print(f"Random samples: {len(sample)}")
    print(f"Saved: {output_path}")

    return sample


# --------------------------------------------------
# 4. SUMMARY
# --------------------------------------------------

def create_summary(df, duplicates, mismatches, sample):

    print("\n4. CREATING QUALITY SUMMARY")

    found = df[
        (df["found"] == True)
        & df["lyrics_text"].notna()
    ].copy()

    summary = {
        "total_recordings": len(df),
        "lyrics_found": int(df["found"].sum()),
        "lyrics_not_found": int((df["found"] == False).sum()),
        "instrumental_recordings": int(df["instrumental"].sum()),
        "unique_artists": int(df["artist_mbid"].nunique()),
        "duplicate_lyrics_groups": (
            int(duplicates["duplicate_group"].nunique())
            if not duplicates.empty else 0
        ),
        "recordings_in_duplicate_groups": len(duplicates),
        "title_mismatches": (
            int(mismatches["title_mismatch"].sum())
            if not mismatches.empty else 0
        ),
        "artist_mismatches": (
            int(mismatches["artist_mismatch"].sum())
            if not mismatches.empty else 0
        ),
        "random_manual_review_samples": len(sample)
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    output_path = OUTPUT_DIR / "lyrics_quality_summary.csv"

    summary_df.to_csv(output_path, index=False)

    print("\nSUMMARY")
    print("-" * 60)

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    duplicates = find_duplicate_lyrics(df)

    mismatches = find_metadata_mismatches(df)

    sample = create_random_sample(df)

    create_summary(
        df,
        duplicates,
        mismatches,
        sample
    )

    print("\n" + "=" * 60)
    print("LYRICS QUALITY AUDIT COMPLETE")
    print("=" * 60)

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
