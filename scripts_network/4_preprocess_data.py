
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
import unicodedata

import pandas as pd


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_NAME = "data_2005_2025"

INPUT_DIR = PROJECT_DIR / "data" / DATASET_NAME
OUTPUT_DIR = INPUT_DIR / "deduplicated"

RECORDINGS_FILE = INPUT_DIR / "recordings.csv"
ARTISTS_FILE = INPUT_DIR / "artists.csv"


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate matching.

    Example:
        "Scream & Shout" -> "screamshout"
        "Scream and Shout" -> "screamandshout"
    """
    text = str(text).casefold()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    # Keep only letters and numbers
    text = "".join(char for char in text if char.isalnum())

    return text


def parse_date(date_string: str):
    """Return a sortable date. Missing or invalid dates sort last."""
    if not date_string or pd.isna(date_string):
        return datetime.max.replace(tzinfo=timezone.utc)

    try:
        date = datetime.fromisoformat(
            str(date_string).replace("Z", "+00:00")
        )

        # Convert all dates to timezone-aware UTC.
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        else:
            date = date.astimezone(timezone.utc)

        return date

    except ValueError:
        return datetime.max.replace(tzinfo=timezone.utc)


# --------------------------------------------------
# LOAD INPUT DATA
# --------------------------------------------------

def load_artists():
    """Load artists and create a name-to-MBID mapping."""

    artists_df = pd.read_csv(ARTISTS_FILE)

    required_columns = {"mbid", "name"}
    if not required_columns.issubset(artists_df.columns):
        raise ValueError(
            f"{ARTISTS_FILE} must contain columns: {required_columns}"
        )

    artists = {
        str(row["mbid"]): str(row["name"])
        for _, row in artists_df.iterrows()
    }

    # Build mapping from normalized names to MBIDs.
    name_to_ids = defaultdict(list)

    for mbid, name in artists.items():
        name_to_ids[normalize_text(name)].append(mbid)

    # Check for ambiguous artist names.
    ambiguous_names = {
        name: ids
        for name, ids in name_to_ids.items()
        if len(ids) > 1
    }

    if ambiguous_names:
        examples = list(ambiguous_names.items())[:10]

        raise ValueError(
            "Some artists have identical normalized names. "
            "Cannot safely map recording artist names to MBIDs.\n"
            f"Examples: {examples}"
        )

    name_to_id = {
        name: ids[0]
        for name, ids in name_to_ids.items()
    }

    return artists, name_to_id


def load_recordings(name_to_id):
    """
    Load existing recordings.csv and reconstruct the collaboration
    dictionary expected by the network-building step.

    No API requests are made.
    """

    recordings_df = pd.read_csv(RECORDINGS_FILE)

    required_columns = {
        "recording_id",
        "title",
        "artists",
        "first_release_date",
    }

    if not required_columns.issubset(recordings_df.columns):
        raise ValueError(
            f"{RECORDINGS_FILE} must contain columns: {required_columns}"
        )

    collaborations = {}

    unmatched_artists = set()
    skipped_recordings = 0

    for _, row in recordings_df.iterrows():

        recording_id = str(row["recording_id"])
        title = str(row["title"])

        # Artists are stored as semicolon-separated names.
        artist_names = [
            name.strip()
            for name in str(row["artists"]).split(";")
            if name.strip()
        ]

        artist_ids = []

        for artist_name in artist_names:
            normalized_name = normalize_text(artist_name)

            mbid = name_to_id.get(normalized_name)

            if mbid is None:
                unmatched_artists.add(artist_name)
            else:
                artist_ids.append(mbid)

        # Remove duplicate artist IDs within a recording.
        artist_ids = sorted(set(artist_ids))

        # A collaboration requires at least two selected artists.
        if len(artist_ids) < 2:
            skipped_recordings += 1
            continue

        collaborations[recording_id] = {
            "title": title,
            "artist_ids": artist_ids,
            "first_release_date": (
                ""
                if pd.isna(row["first_release_date"])
                else str(row["first_release_date"])
            ),
        }

    if unmatched_artists:
        print(
            f"\nWarning: {len(unmatched_artists)} artist names "
            "could not be matched."
        )
        print("Examples:", sorted(unmatched_artists)[:10])

    print(f"Loaded recordings: {len(collaborations)}")
    print(f"Skipped recordings: {skipped_recordings}")

    return collaborations


# --------------------------------------------------
# DEDUPLICATION
# --------------------------------------------------

def deduplicate_recordings(collaborations):
    """
    Treat recordings as duplicates when they have:
        - The same normalized title
        - The same set of artists

    Keep the recording with the earliest release date.
    Use recording ID as a deterministic tie-breaker.
    """

    groups = defaultdict(list)

    for recording_id, recording in collaborations.items():

        normalized_title = normalize_text(recording["title"])
        artist_ids = tuple(sorted(recording["artist_ids"]))

        key = (normalized_title, artist_ids)

        groups[key].append((recording_id, recording))

    deduplicated = {}
    duplicate_report = []

    for key, recordings in groups.items():

        # Sort by release date, then recording ID.
        recordings.sort(
            key=lambda item: (
                parse_date(item[1]["first_release_date"]),
                item[0],
            )
        )

        # Keep the first recording.
        kept_id, kept_recording = recordings[0]
        deduplicated[kept_id] = kept_recording

        # Record duplicate information.
        if len(recordings) > 1:

            removed_ids = [
                recording_id
                for recording_id, _ in recordings[1:]
            ]

            duplicate_report.append({
                "title": kept_recording["title"],
                "artists": "; ".join(kept_recording["artist_ids"]),
                "duplicate_count": len(recordings),
                "kept_recording_id": kept_id,
                "removed_recording_ids": "; ".join(removed_ids),
            })

    return deduplicated, duplicate_report


# --------------------------------------------------
# BUILD NETWORK
# --------------------------------------------------

def build_network(artists, collaborations):
    """
    Rebuild the collaboration network using deduplicated recordings.
    """

    edge_weights = defaultdict(int)

    for recording in collaborations.values():

        artist_ids = recording["artist_ids"]

        for i in range(len(artist_ids)):
            for j in range(i + 1, len(artist_ids)):

                source = artist_ids[i]
                target = artist_ids[j]

                edge_weights[(source, target)] += 1

    # Save all artists, including isolated artists.
    artists_df = pd.DataFrame([
        {
            "mbid": mbid,
            "name": name,
        }
        for mbid, name in sorted(artists.items())
    ])

    artists_df.to_csv(
        OUTPUT_DIR / "artists.csv",
        index=False
    )

    # Save weighted edges.
    edges_df = pd.DataFrame([
        {
            "source": source,
            "target": target,
            "weight": weight,
        }
        for (source, target), weight in sorted(edge_weights.items())
    ])

    edges_df.to_csv(
        OUTPUT_DIR / "edges.csv",
        index=False
    )

    # Save deduplicated recordings.
    recordings_df = pd.DataFrame([
        {
            "recording_id": recording_id,
            "title": recording["title"],
            "artists": "; ".join(
                artists[mbid]
                for mbid in recording["artist_ids"]
            ),
            "first_release_date": recording["first_release_date"],
        }
        for recording_id, recording in sorted(collaborations.items())
    ])

    recordings_df.to_csv(
        OUTPUT_DIR / "recordings.csv",
        index=False
    )

    return edges_df


# --------------------------------------------------
# SAVE REPORT AND METADATA
# --------------------------------------------------

def save_duplicate_report(duplicate_report):
    report_df = pd.DataFrame(duplicate_report)

    report_df.to_csv(
        OUTPUT_DIR / "duplicate_groups.csv",
        index=False
    )


def save_metadata(
    original_count,
    deduplicated_count,
    duplicate_report,
    edge_count,
):
    metadata = {
        "dataset": DATASET_NAME,
        "preprocessing_date": datetime.now().isoformat(),
        "source_recordings": str(RECORDINGS_FILE),
        "source_artists": str(ARTISTS_FILE),
        "deduplication_rule": (
            "Same normalized title and same set of artists"
        ),
        "representative_recording_rule": (
            "Keep earliest release date; recording ID breaks ties"
        ),
        "original_recordings": original_count,
        "deduplicated_recordings": deduplicated_count,
        "removed_recordings": original_count - deduplicated_count,
        "duplicate_groups": len(duplicate_report),
        "number_of_edges": edge_count,
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nLoading existing dataset...")

    if not RECORDINGS_FILE.exists():
        raise FileNotFoundError(RECORDINGS_FILE)

    if not ARTISTS_FILE.exists():
        raise FileNotFoundError(ARTISTS_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data.
    artists, name_to_id = load_artists()
    collaborations = load_recordings(name_to_id)

    original_count = len(collaborations)

    print("\nDeduplicating recordings...")

    deduplicated, duplicate_report = deduplicate_recordings(
        collaborations
    )

    deduplicated_count = len(deduplicated)

    print("\nBuilding collaboration network...")

    edges_df = build_network(
        artists,
        deduplicated,
    )

    # Save reports.
    save_duplicate_report(duplicate_report)

    save_metadata(
        original_count=original_count,
        deduplicated_count=deduplicated_count,
        duplicate_report=duplicate_report,
        edge_count=len(edges_df),
    )

    # Summary.
    print("\n" + "=" * 45)
    print("PREPROCESSING COMPLETE")
    print("=" * 45)

    print(f"Original recordings:     {original_count}")
    print(f"Deduplicated recordings: {deduplicated_count}")
    print(f"Removed recordings:      {original_count - deduplicated_count}")
    print(f"Duplicate groups:        {len(duplicate_report)}")
    print(f"Artists:                 {len(artists)}")
    print(f"Network edges:           {len(edges_df)}")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
