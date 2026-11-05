
import csv
import os
import re
import time
import requests

from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://musicbrainz.org/ws/2/recording"

INPUT_FILE = "project/data/data_2005_2025/pop_artists.csv"
OUTPUT_DIR = "project/data/data_2005_2025"

START_YEAR = 2005
END_YEAR = 2025

PAGE_SIZE = 100

MAX_RETRIES = 6
REQUEST_DELAY = 1.1

HEADERS = {
    "User-Agent": "MusicCollaborationResearch/1.0 (student research project)"
}

ALT_VERSION_PATTERN = re.compile(
    r"\b(live|remix|remastered|acoustic|demo|instrumental|"
    r"karaoke|radio edit|extended version)\b",
    re.IGNORECASE
)


# ============================================================
# LOAD SELECTED ARTISTS
# ============================================================

def load_artists():
    """Load selected artists from pop_artists.csv."""

    artists = {}

    with open(INPUT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mbid = row["mbid"].strip()
            name = row["name"].strip()

            if mbid and name:
                artists[mbid] = name

    print(f"Loaded {len(artists)} selected artists.")

    return artists




# ============================================================
# API REQUEST WITH RETRIES
# ============================================================

def api_get(params):
    """Send a MusicBrainz request with retry handling."""

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code in (429, 503):

                retry_after = response.headers.get("Retry-After")

                wait = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 2 ** attempt
                )

                print(
                    f"MusicBrainz returned {response.status_code}. "
                    f"Retrying in {wait}s..."
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            # Respect MusicBrainz rate limit
            time.sleep(REQUEST_DELAY)

            return data

        except requests.exceptions.RequestException as e:

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Request failed after {MAX_RETRIES} attempts: {e}"
                )

            wait = 2 ** attempt

            print(f"Request error: {e}")
            print(f"Retrying in {wait}s...")

            time.sleep(wait)

    raise RuntimeError("Maximum retries exceeded.")


# ============================================================
# FETCH RECORDINGS FOR ONE ARTIST
# ============================================================

def fetch_recordings(artist_id):
    """Fetch all recordings for one artist."""

    recordings = {}

    offset = 0

    while True:

        params = {
            "artist": artist_id,
            "inc": "artist-credits",
            "fmt": "json",
            "limit": PAGE_SIZE,
            "offset": offset
        }

        data = api_get(params)

        batch = data.get("recordings", [])

        for recording in batch:

            recording_id = recording.get("id")

            if recording_id:
                recordings[recording_id] = recording

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return recordings


# ============================================================
# FILTER RECORDINGS
# ============================================================

def is_in_period(recording):
    """Check whether recording has a first release year in the period."""

    date = recording.get("first-release-date", "")

    if not date:
        return False

    try:
        year = int(date[:4])

        return START_YEAR <= year <= END_YEAR

    except (ValueError, TypeError):
        return False


def is_valid_recording(title):
    """Exclude alternative versions based on recording title."""

    return not ALT_VERSION_PATTERN.search(title)


# ============================================================
# EXTRACT COLLABORATIONS
# ============================================================

def extract_collaborations(all_recordings, selected_artists):
    """Extract collaborations between selected artists only."""

    collaborations = {}

    for recording in all_recordings.values():

        title = recording.get("title", "")

        if not title:
            continue

        # Filter by release period
        if not is_in_period(recording):
            continue

        # Remove alternative versions
        if not is_valid_recording(title):
            continue

        recording_id = recording.get("id")

        if not recording_id:
            continue

        # Keep only selected artists
        artist_ids = []

        for credit in recording.get("artist-credit", []):

            if not isinstance(credit, dict):
                continue

            artist = credit.get("artist", {})
            artist_id = artist.get("id")

            if artist_id in selected_artists:
                artist_ids.append(artist_id)

        # Remove duplicate artist credits
        artist_ids = sorted(set(artist_ids))

        # At least two selected artists must be credited
        if len(artist_ids) < 2:
            continue

        collaborations[recording_id] = {
            "title": title,
            "artist_ids": artist_ids,
            "first_release_date": recording.get(
                "first-release-date", ""
            )
        }

    return collaborations





# ============================================================
# BUILD NETWORK AND SAVE DATA
# ============================================================

def build_network(selected_artists, collaborations):
    """Build weighted collaboration network and save CSV files."""

    edge_weights = defaultdict(int)

    # Create weighted edges
    for recording in collaborations.values():

        artist_ids = recording["artist_ids"]

        for i in range(len(artist_ids)):

            for j in range(i + 1, len(artist_ids)):

                pair = (artist_ids[i], artist_ids[j])

                edge_weights[pair] += 1

    # --------------------------------------------------------
    # Save all selected artists, including isolated nodes
    # --------------------------------------------------------

    artists_path = os.path.join(OUTPUT_DIR, "artists.csv")

    with open(artists_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["mbid", "name"])

        for artist_id, name in selected_artists.items():

            writer.writerow([artist_id, name])

    # --------------------------------------------------------
    # Save weighted edges
    # --------------------------------------------------------

    edges_path = os.path.join(OUTPUT_DIR, "edges.csv")

    with open(edges_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["source", "target", "weight"])

        for (source, target), weight in edge_weights.items():

            writer.writerow([source, target, weight])

    # --------------------------------------------------------
    # Save collaboration recordings
    # --------------------------------------------------------

    recordings_path = os.path.join(OUTPUT_DIR, "recordings.csv")

    with open(recordings_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "recording_id",
            "title",
            "artists",
            "first_release_date"
        ])

        for recording_id, recording in collaborations.items():

            names = [
                selected_artists[artist_id]
                for artist_id in recording["artist_ids"]
            ]

            writer.writerow([
                recording_id,
                recording["title"],
                "; ".join(names),
                recording["first_release_date"]
            ])

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========== NETWORK SUMMARY ==========")

    print(f"Selected artists: {len(selected_artists)}")
    print(f"Collaboration recordings: {len(collaborations)}")
    print(f"Unique edges: {len(edge_weights)}")

    isolated = len(selected_artists) - len(
        set(
            artist
            for edge in edge_weights
            for artist in edge
        )
    )

    print(f"Isolated artists: {isolated}")

    print("\nFiles saved:")
    print(f"  {artists_path}")
    print(f"  {edges_path}")
    print(f"  {recordings_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_artists = load_artists()

    all_recordings = {}

    print("\nFetching recordings from MusicBrainz...\n")

    for i, (artist_id, artist_name) in enumerate(
        selected_artists.items(), start=1
    ):

        print(
            f"[{i}/{len(selected_artists)}] "
            f"Fetching {artist_name}..."
        )

        recordings = fetch_recordings(artist_id)

        all_recordings.update(recordings)

        print("\n========== RECORDING STATISTICS ==========")

        print("Total recordings:", len(all_recordings))

        in_period = [
            r for r in all_recordings.values()
            if is_in_period(r)
        ]

        print("Recordings in period:", len(in_period))

        with_date = [
            r for r in all_recordings.values()
            if r.get("first-release-date")
        ]

        print("Recordings with release date:", len(with_date))

        print(
            f"  Retrieved {len(recordings)} unique recordings. "
            f"Total: {len(all_recordings)}"
        )

    print("\n======================================")
    print(f"Total unique recordings fetched: {len(all_recordings)}")

    print("\nFiltering recordings and extracting collaborations...")

    collaborations = extract_collaborations(
        all_recordings,
        selected_artists
    )

    build_network(
        selected_artists,
        collaborations
    )


if __name__ == "__main__":
    main()
