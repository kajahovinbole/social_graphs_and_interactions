
import pandas as pd
import re
import unicodedata
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path(
    "data_network/data_2005_2025/deduplicated/recordings.csv"
)

MAPPING_FILE = Path(
    "data/data_2005_2025_network/deduplication/recording_deduplication_mapping.csv"
)

TARGET_IDS = [
    "0a700716-a9e5-40f7-814e-d459ed3a8987",
    "35bdf3f8-e644-455b-8979-67415568c2e3",
]

VERSION_KEYWORDS = [
    "remix",
    "mix",
    "extended",
    "radio edit",
    "radio version",
    "single version",
    "album version",
    "acoustic",
    "instrumental",
    "karaoke",
    "live",
    "demo",
    "clean",
    "explicit",
    "sped up",
    "slowed",
    "slowed down",
    "edit",
    "remaster",
    "remastered",
    "re-recorded",
    "re recorded",
    "a cappella",
    "acapella",
    "dub",
    "bootleg",
]


# --------------------------------------------------
# NORMALIZATION FUNCTIONS
# Same implementation as the deduplication script
# --------------------------------------------------

def normalize_text(text):

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_version_label(text):

    normalized = normalize_text(text)

    return any(
        keyword in normalized
        for keyword in VERSION_KEYWORDS
    )


def canonicalize_title(title):

    title = str(title).strip()

    previous = None

    while previous != title:

        previous = title

        title = re.sub(
            r"\(([^()]*)\)|\[([^\[\]]*)\]",
            lambda match: (
                ""
                if is_version_label(
                    match.group(1) or match.group(2)
                )
                else match.group(0)
            ),
            title,
        )

    for keyword in VERSION_KEYWORDS:

        pattern = rf"\s*[-–—]\s*{re.escape(keyword)}\s*$"

        title = re.sub(
            pattern,
            "",
            title,
            flags=re.IGNORECASE
        )

    return normalize_text(title)


def normalize_artists(artists):

    artist_list = [
        normalize_text(artist)
        for artist in str(artists).split(";")
        if artist.strip()
    ]

    return tuple(sorted(set(artist_list)))


# --------------------------------------------------
# MAIN DEBUGGING
# --------------------------------------------------

def main():

    print("=" * 70)
    print("DEBUGGING A-HA DUPLICATE RECORDINGS")
    print("=" * 70)

    # Load original recordings
    df = pd.read_csv(INPUT_FILE)

    required = {
        "recording_id",
        "title",
        "artists",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Find target recordings
    target_df = df[
        df["recording_id"].isin(TARGET_IDS)
    ].copy()

    if len(target_df) != len(TARGET_IDS):

        found = target_df["recording_id"].tolist()

        print("\nWARNING: Not all recordings were found.")
        print("Found:", found)

    if target_df.empty:
        print("No recordings found. Check the recording IDs.")
        return

    # Generate same keys as original script
    target_df["normalized_title"] = (
        target_df["title"].apply(normalize_text)
    )

    target_df["canonical_title"] = (
        target_df["title"].apply(canonicalize_title)
    )

    target_df["artist_key"] = (
        target_df["artists"].apply(normalize_artists)
    )

    target_df["dedup_key"] = (
        target_df["canonical_title"].astype(str)
        + "||"
        + target_df["artist_key"].astype(str)
    )

    # Print detailed information
    print("\n1. ORIGINAL RECORDING DATA")
    print("-" * 70)

    for _, row in target_df.iterrows():

        print(f"\nRecording ID: {row['recording_id']}")
        print(f"Original title: {row['title']}")
        print(f"Original artists: {row['artists']}")

        print(f"Normalized title: {row['normalized_title']}")
        print(f"Canonical title: {row['canonical_title']}")

        print(f"Artist key: {row['artist_key']}")
        print(f"Dedup key: {row['dedup_key']}")

        print(
            f"Contains version label: "
            f"{is_version_label(row['title'])}"
        )

    # Compare the two recordings
    print("\n2. DIRECT COMPARISON")
    print("-" * 70)

    if len(target_df) == 2:

        first = target_df.iloc[0]
        second = target_df.iloc[1]

        comparisons = {
            "Same canonical title":
                first["canonical_title"] == second["canonical_title"],

            "Same artist key":
                first["artist_key"] == second["artist_key"],

            "Same dedup key":
                first["dedup_key"] == second["dedup_key"],
        }

        for description, result in comparisons.items():

            print(
                f"{description}: "
                f"{'YES' if result else 'NO'}"
            )

    # Load mapping file
    print("\n3. DEDUPLICATION MAPPING")
    print("-" * 70)

    if not MAPPING_FILE.exists():

        print("Mapping file not found:", MAPPING_FILE)

    else:

        mapping = pd.read_csv(MAPPING_FILE)

        mapped = mapping[
            mapping["recording_id"].isin(TARGET_IDS)
        ].copy()

        if mapped.empty:

            print("Neither recording was found in the mapping file.")

        else:

            print(
                mapped[
                    [
                        "recording_id",
                        "original_title",
                        "canonical_title",
                        "artists",
                        "song_id",
                        "is_representative",
                    ]
                ].to_string(index=False)
            )

            if mapped["song_id"].nunique() == 1:

                print("\nRESULT: Both recordings were merged.")

            else:

                print("\nRESULT: Recordings were NOT merged.")

                print(
                    "Different song IDs:",
                    mapped["song_id"].tolist()
                )

    # Final diagnosis
    print("\n4. DIAGNOSIS")
    print("-" * 70)

    if len(target_df) == 2:

        first = target_df.iloc[0]
        second = target_df.iloc[1]

        if first["canonical_title"] != second["canonical_title"]:

            print(
                "CAUSE: The canonical titles differ."
            )

            print(
                "Investigate title normalization and "
                "version-label removal."
            )

        elif first["artist_key"] != second["artist_key"]:

            print(
                "CAUSE: The artist keys differ."
            )

            print(
                "Investigate differences in MusicBrainz "
                "artist credits."
            )

        else:

            print(
                "Both canonical titles and artist keys match."
            )

            print(
                "The recordings should have been grouped "
                "by the current deduplication logic."
            )

            print(
                "If they have different song IDs in the mapping, "
                "the mapping may come from another run or dataset."
            )

    print("\nDEBUGGING COMPLETE")


if __name__ == "__main__":
    main()
