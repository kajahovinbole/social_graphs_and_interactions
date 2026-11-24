from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import re
import unicodedata
from datetime import datetime

import pandas as pd


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "data" / "data_2005_2025_network"

ARTISTS_FILE = DATASET_DIR / "artists.csv"
SONGS_FILE = DATASET_DIR / "deduplication" / "unique_songs_reviewed.csv"
OUTPUT_DIR = DATASET_DIR / "network"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_name(value: str) -> str:
    """Normalize artist names for matching names to MusicBrainz IDs."""
    text = str(value).strip().casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return "".join(char for char in text if char.isalnum())


def find_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    """Return the first matching column, ignoring case and whitespace."""
    lookup = {str(col).strip().casefold(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.casefold()
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find a {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Found columns: {list(df.columns)}"
    )


def split_artist_names(value) -> list[str]:
    """Parse semicolon-separated artist credits."""
    if pd.isna(value):
        return []

    # The reviewed dataset uses semicolons between credited artists.
    names = [part.strip() for part in str(value).split(";")]
    return [name for name in names if name]


# --------------------------------------------------
# LOAD ARTISTS
# --------------------------------------------------

def load_artists():
    artists_df = pd.read_csv(ARTISTS_FILE, dtype=str)

    mbid_col = find_column(artists_df, ["mbid", "artist_mbid"], "artist MBID")
    name_col = find_column(artists_df, ["name", "artist_name"], "artist name")

    artists = {}
    name_to_ids = defaultdict(list)

    for _, row in artists_df.iterrows():
        mbid = str(row[mbid_col]).strip()
        name = str(row[name_col]).strip()

        if not mbid or mbid.casefold() == "nan" or not name or name.casefold() == "nan":
            continue

        artists[mbid] = name
        name_to_ids[normalize_name(name)].append(mbid)

    ambiguous = {
        name: ids for name, ids in name_to_ids.items()
        if len(set(ids)) > 1
    }
    if ambiguous:
        examples = list(ambiguous.items())[:10]
        raise ValueError(
            "Some artist names map to multiple MBIDs after normalization. "
            "Resolve these before building the network.\n"
            f"Examples: {examples}"
        )

    name_to_id = {
        normalized: ids[0]
        for normalized, ids in name_to_ids.items()
    }

    return artists, name_to_id


# --------------------------------------------------
# LOAD REVIEWED SONGS
# --------------------------------------------------

def load_reviewed_songs(name_to_id):
    songs_df = pd.read_csv(SONGS_FILE, dtype=str)

    title_col = find_column(
        songs_df,
        ["title", "canonical_title", "song_title"],
        "song title",
    )
    artists_col = find_column(
        songs_df,
        ["artists", "artist_names", "artist_credits"],
        "artist credits",
    )

    songs = []
    unmatched_artists = set()
    songs_without_two_selected_artists = 0

    for row_index, row in songs_df.iterrows():
        title = str(row[title_col]).strip()
        if not title or title.casefold() == "nan":
            continue

        artist_names = split_artist_names(row[artists_col])
        artist_ids = []

        for artist_name in artist_names:
            mbid = name_to_id.get(normalize_name(artist_name))
            if mbid is None:
                unmatched_artists.add(artist_name)
            else:
                artist_ids.append(mbid)

        # A song contributes at most one edge per artist pair.
        artist_ids = sorted(set(artist_ids))

        if len(artist_ids) < 2:
            songs_without_two_selected_artists += 1
            continue

        songs.append({
            "song_id": str(row_index + 1),
            "title": title,
            "artist_ids": artist_ids,
            "artist_names": [name for name in artist_names],
        })

    if unmatched_artists:
        print(
            f"\nWARNING: {len(unmatched_artists)} artist names "
            "could not be matched to the selected artist list."
        )
        print("Examples:", sorted(unmatched_artists)[:20])

    print(f"Reviewed songs loaded: {len(songs_df)}")
    print(f"Songs contributing to network: {len(songs)}")
    print(
        "Songs with fewer than two selected artists "
        f"(not used for edges): {songs_without_two_selected_artists}"
    )

    return songs, songs_df, title_col, artists_col


# --------------------------------------------------
# BUILD NETWORK
# --------------------------------------------------

def build_network(artists, songs):
    edge_weights = defaultdict(int)
    edge_song_titles = defaultdict(list)

    for song in songs:
        # combinations() ensures each pair is counted once per unique song.
        for source, target in combinations(song["artist_ids"], 2):
            edge_weights[(source, target)] += 1
            edge_song_titles[(source, target)].append(song["title"])

    # Include all selected artists, including isolates.
    nodes_df = pd.DataFrame([
        {"Id": mbid, "Label": name, "mbid": mbid, "name": name}
        for mbid, name in sorted(artists.items())
    ])

    edges_df = pd.DataFrame([
        {
            "Source": source,
            "Target": target,
            "Weight": weight,
            "source": source,
            "target": target,
            "weight": weight,
        }
        for (source, target), weight in sorted(edge_weights.items())
    ])

    # A separate audit file makes it possible to inspect which songs
    # contributed to each edge weight.
    edge_songs_df = pd.DataFrame([
        {
            "source_mbid": source,
            "source_name": artists[source],
            "target_mbid": target,
            "target_name": artists[target],
            "weight": len(titles),
            "songs": " | ".join(titles),
        }
        for (source, target), titles in sorted(edge_song_titles.items())
    ])

    return nodes_df, edges_df, edge_songs_df


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

def main():
    if not ARTISTS_FILE.exists():
        raise FileNotFoundError(f"Artist file not found: {ARTISTS_FILE}")
    if not SONGS_FILE.exists():
        raise FileNotFoundError(
            f"Reviewed song file not found: {SONGS_FILE}\n"
            "Update SONGS_FILE at the top of this script if it is stored elsewhere."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading selected artists...")
    artists, name_to_id = load_artists()

    print("\nLoading manually reviewed songs...")
    songs, songs_df, title_col, artists_col = load_reviewed_songs(name_to_id)

    print("\nBuilding collaboration network...")
    nodes_df, edges_df, edge_songs_df = build_network(artists, songs)

    nodes_df.to_csv(OUTPUT_DIR / "nodes.csv", index=False)
    edges_df.to_csv(OUTPUT_DIR / "edges.csv", index=False)
    edge_songs_df.to_csv(OUTPUT_DIR / "edge_song_audit.csv", index=False)

    # Save the exact song rows used as input, without performing another
    # automatic deduplication step.
    songs_df.to_csv(OUTPUT_DIR / "songs_used.csv", index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_songs_file": str(SONGS_FILE),
        "input_artists_file": str(ARTISTS_FILE),
        "deduplication": (
            "Uses unique_songs_reviewed.csv as-is; no additional "
            "automatic deduplication is performed."
        ),
        "edge_weight_definition": (
            "Number of unique reviewed songs credited to both artists. "
            "Each artist pair contributes at most once per song."
        ),
        "number_of_artists": int(len(nodes_df)),
        "number_of_songs_in_input": int(len(songs_df)),
        "number_of_songs_used_for_edges": int(len(songs)),
        "number_of_edges": int(len(edges_df)),
        "number_of_isolates": int(
            len(nodes_df) - len(set(edges_df["Source"]) | set(edges_df["Target"]))
        ) if not edges_df.empty else int(len(nodes_df)),
    }

    with open(
        OUTPUT_DIR / "network_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 48)
    print("NETWORK BUILD COMPLETE")
    print("=" * 48)
    print(f"Artists (nodes):             {len(nodes_df)}")
    print(f"Reviewed songs in input:     {len(songs_df)}")
    print(f"Songs used for edges:        {len(songs)}")
    print(f"Edges:                       {len(edges_df)}")
    print(f"Isolates:                    {metadata['number_of_isolates']}")
    print(f"Output directory:            {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  nodes.csv")
    print("  edges.csv")
    print("  edge_song_audit.csv")
    print("  songs_used.csv")
    print("  network_metadata.json")


if __name__ == "__main__":
    main()
