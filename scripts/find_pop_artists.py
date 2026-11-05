
import csv
import os
import time
import requests
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------

START_YEAR = 2005
END_YEAR = 2025

PAGE_SIZE = 100
MAX_RESULTS_PER_YEAR = 500

# Minimum number of pop recordings required
MIN_RECORDINGS = 6

OUTPUT_FILE = "project/data/data_2005_2025/pop_artists.csv"

BASE_URL = "https://musicbrainz.org/ws/2/recording"

HEADERS = {
    "User-Agent": "SocialGraphsResearch/1.0 (kajahovin@gmail.com)"
}


# -----------------------------
# API REQUEST
# -----------------------------

def api_get(params, max_retries=6):

    for attempt in range(max_retries):

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code == 503 or response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 5))
                print(f"Server busy ({response.status_code}). Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            time.sleep(1.1)
            return response.json()

        except requests.exceptions.RequestException as e:

            if attempt == max_retries - 1:
                raise RuntimeError(f"API request failed: {e}")

            wait = 2 ** attempt
            print(f"Request failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Maximum retries exceeded")


# -----------------------------
# FETCH RECORDINGS
# -----------------------------

def fetch_year(year):

    query = (
        f"tag:pop AND "
        f"firstreleasedate:[{year} TO {year}]"
    )

    print(f"\nFetching recordings from {year}...")

    recordings = []

    for offset in range(0, MAX_RESULTS_PER_YEAR, PAGE_SIZE):

        params = {
            "query": query,
            "fmt": "json",
            "limit": PAGE_SIZE,
            "offset": offset,
            "inc": "artist-credits"
        }

        data = api_get(params)

        batch = data.get("recordings", [])

        if not batch:
            break

        recordings.extend(batch)

        print(
            f"  Retrieved {len(recordings)} "
            f"recordings from {year}"
        )

        if len(batch) < PAGE_SIZE:
            break

    return recordings


# -----------------------------
# EXTRACT ARTISTS
# -----------------------------

def extract_artists(recordings):

    # MBID -> set of unique recording IDs
    artist_recordings = defaultdict(set)

    # MBID -> artist name
    artist_names = {}

    seen_recordings = set()

    for recording in recordings:

        recording_id = recording.get("id")

        if not recording_id or recording_id in seen_recordings:
            continue

        seen_recordings.add(recording_id)

        for credit in recording.get("artist-credit", []):

            # Artist credits can contain join phrases such as "feat."
            if not isinstance(credit, dict):
                continue

            artist = credit.get("artist", {})

            mbid = artist.get("id")
            name = artist.get("name")

            if not mbid or not name:
                continue

            artist_recordings[mbid].add(recording_id)
            artist_names[mbid] = name

    artists = []

    for mbid, recording_ids in artist_recordings.items():

        count = len(recording_ids)

        if count >= MIN_RECORDINGS:

            artists.append({
                "name": artist_names[mbid],
                "mbid": mbid,
                "pop_recordings": count
            })

    # Sort by number of pop recordings
    artists.sort(
        key=lambda x: (-x["pop_recordings"], x["name"].lower())
    )

    return artists


# -----------------------------
# SAVE CSV
# -----------------------------

def save_artists(artists):

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["name", "mbid", "pop_recordings"]
        )

        writer.writeheader()
        writer.writerows(artists)

    print(f"\nSaved {len(artists)} artists to {OUTPUT_FILE}")


# -----------------------------
# MAIN
# -----------------------------

def main():

    all_recordings = {}

    for year in range(START_YEAR, END_YEAR + 1):

        recordings = fetch_year(year)

        for recording in recordings:
            recording_id = recording.get("id")

            if recording_id:
                all_recordings[recording_id] = recording

    print(f"\nTotal unique recordings: {len(all_recordings)}")

    artists = extract_artists(list(all_recordings.values()))

    print(f"Artists with at least {MIN_RECORDINGS} recordings: {len(artists)}")

    save_artists(artists)

    print("\nTop 20 artists:\n")

    for artist in artists[:20]:
        print(
            f"{artist['name']:<35} "
            f"{artist['pop_recordings']} recordings"
        )


if __name__ == "__main__":
    main()
