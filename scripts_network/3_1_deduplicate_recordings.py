
import pandas as pd
import re
import unicodedata
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "data_2005_2025_network" / "deduplication"
INPUT_FILE = OUTPUT_DIR / "recordings_for_dedup.csv"

OUTPUT_FILE = OUTPUT_DIR / "unique_songs.csv"
MAPPING_FILE = OUTPUT_DIR / "recording_deduplication_mapping.csv"
EXCLUDED_FILE = OUTPUT_DIR / "excluded_recordings.csv"

# Known version labels that should not define a separate song
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

# Explicitly exclude film soundtrack segments.
# This can be expanded if similar cases are discovered.
EXCLUSION_PATTERNS = [
    r"trolls.*teil\s*\d+",
]


# --------------------------------------------------
# Normalization
# --------------------------------------------------

def normalize_text(text):
    """Normalize Unicode, casing and punctuation."""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("&", " and ")

    # Keep only letters and numbers
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_version_label(text):
    """Check whether text contains a known version label."""

    normalized = normalize_text(text)

    return any(
        keyword in normalized
        for keyword in VERSION_KEYWORDS
    )


def canonicalize_title(title):
    """
    Remove known version information from a title.

    Only removes parenthetical/bracketed sections if they
    contain a known version label.
    """

    title = str(title).strip()

    # Repeatedly remove version information in parentheses/brackets
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

    # Remove trailing version labels without parentheses
    for keyword in VERSION_KEYWORDS:
        pattern = rf"\s*[-–—]\s*{re.escape(keyword)}\s*$"
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    return normalize_text(title)


def is_excluded(title):
    """Identify recordings that should not enter the network."""

    title_normalized = normalize_text(title)

    return any(
        re.search(pattern, title_normalized)
        for pattern in EXCLUSION_PATTERNS
    )


def normalize_artists(artists):
    """
    Create a stable artist key independent of artist order.
    """

    artist_list = [
        normalize_text(artist)
        for artist in str(artists).split(";")
        if artist.strip()
    ]

    return tuple(sorted(set(artist_list)))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "recording_id",
        "title",
        "artists",
        "first_release_date",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Loaded {len(df)} recordings")

    # Identify excluded recordings
    df["excluded"] = df["title"].apply(is_excluded)

    excluded_df = df[df["excluded"]].copy()
    working_df = df[~df["excluded"]].copy()

    # Generate canonical keys
    working_df["canonical_title"] = (
        working_df["title"].apply(canonicalize_title)
    )

    working_df["artist_key"] = (
        working_df["artists"].apply(normalize_artists)
    )

    # Each unique song is defined by canonical title + artist credits
    working_df["dedup_key"] = (
        working_df["canonical_title"].astype(str)
        + "||"
        + working_df["artist_key"].astype(str)
    )

    # Group recordings representing the same song
    grouped = working_df.groupby("dedup_key", sort=False)

    unique_songs = []
    mapping = []

    for group_id, (dedup_key, group) in enumerate(grouped, start=1):

        # Keep one representative recording.
        # Prefer the shortest/original-looking title by default.
        representative = group.sort_values(
            by="title",
            key=lambda col: col.str.len()
        ).iloc[0]

        unique_songs.append({
            "song_id": group_id,
            "title": representative["title"],
            "canonical_title": representative["canonical_title"],
            "artists": representative["artists"],
            "first_release_date": representative["first_release_date"],
            "n_recordings": len(group),
        })

        for _, row in group.iterrows():
            mapping.append({
                "recording_id": row["recording_id"],
                "original_title": row["title"],
                "canonical_title": row["canonical_title"],
                "artists": row["artists"],
                "song_id": group_id,
                "is_representative": (
                    row["recording_id"]
                    == representative["recording_id"]
                ),
            })

    unique_df = pd.DataFrame(unique_songs)
    mapping_df = pd.DataFrame(mapping)

    # Save outputs
    unique_df.to_csv(OUTPUT_FILE, index=False)
    mapping_df.to_csv(MAPPING_FILE, index=False)
    excluded_df.to_csv(EXCLUDED_FILE, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nDeduplication summary")
    print("---------------------")
    print(f"Original recordings: {len(df)}")
    print(f"Excluded recordings: {len(excluded_df)}")
    print(f"Recordings after exclusion: {len(working_df)}")
    print(f"Unique songs: {len(unique_df)}")
    print(
        f"Recordings merged as duplicates: "
        f"{len(working_df) - len(unique_df)}"
    )

    print("\nSaved files:")
    print(f"Unique songs: {OUTPUT_FILE}")
    print(f"Mapping:      {MAPPING_FILE}")
    print(f"Excluded:     {EXCLUDED_FILE}")

    print("\nLargest deduplication groups:")

    largest_groups = unique_df.sort_values(
        "n_recordings",
        ascending=False
    ).head(15)

    print(
        largest_groups[
            ["title", "artists", "n_recordings"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
