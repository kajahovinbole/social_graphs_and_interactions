
from pathlib import Path
import os
import re
import time
import unicodedata

import pandas as pd
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

ARTISTS_FILE = (
    PROJECT_DIR
    / "data_network"
    / "data_2005_2025"
    / "deduplicated"
    / "artists.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data_lyrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "top_recordings.csv"

load_dotenv(PROJECT_DIR / ".env")

LISTENBRAINZ_TOKEN = os.getenv("LISTENBRAINZ_TOKEN")

if not LISTENBRAINZ_TOKEN:
    raise ValueError("LISTENBRAINZ_TOKEN not found in .env")

N_SONGS_PER_ARTIST = 5

MIN_YEAR = 2005
MAX_YEAR = 2025

# Avoid alternative versions of songs
EXCLUDED_TERMS = [
    "live",
    "remix",
    "remastered",
    "acoustic",
    "demo",
    "instrumental",
    "karaoke",
    "radio edit",
    "extended version",
    "sped up",
    "slowed",
    "version",
]


# --------------------------------------------------
# API configuration
# --------------------------------------------------

LISTENBRAINZ_URL = (
    "https://api.listenbrainz.org/1/popularity/"
    "top-recordings-for-artist/{}"
)

MUSICBRAINZ_URL = (
    "https://musicbrainz.org/ws/2/recording/{}"
)

LB_HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0",
    "Authorization": f"Token {LISTENBRAINZ_TOKEN}",
}

MB_HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0 (research project)",
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_title(title):
    """Normalize a title for duplicate detection."""

    title = unicodedata.normalize("NFKD", title)
    title = "".join(
        char for char in title
        if not unicodedata.combining(char)
    )

    title = title.casefold()
    title = re.sub(r"[^a-z0-9]", "", title)

    return title


def is_alternative_version(title):
    """Check whether a title indicates an alternative version."""

    title = title.casefold()

    for term in EXCLUDED_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, title):
            return True

    return False


def get_top_recordings(session, artist_mbid):
    """Fetch the artist's recordings ranked by ListenBrainz."""

    url = LISTENBRAINZ_URL.format(artist_mbid)

    response = session.get(
        url,
        headers=LB_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    recordings = response.json()

    if not isinstance(recordings, list):
        raise ValueError(f"Unexpected ListenBrainz response: {recordings}")

    return recordings


def get_release_year(session, recording_mbid):
    """Look up the first release year of a recording in MusicBrainz."""

    if not recording_mbid:
        return None

    url = MUSICBRAINZ_URL.format(recording_mbid)

    params = {
        "fmt": "json",
        "inc": "releases",
    }

    response = session.get(
        url,
        params=params,
        headers=MB_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # Prefer MusicBrainz' first-release-date field
    first_release_date = data.get("first-release-date")

    if first_release_date:
        year_match = re.match(r"(\d{4})", first_release_date)

        if year_match:
            return int(year_match.group(1))

    # Fallback: find the earliest available release date
    release_dates = []

    for release in data.get("releases", []):
        date = release.get("date", "")

        year_match = re.match(r"(\d{4})", date)

        if year_match:
            release_dates.append(int(year_match.group(1)))

    if release_dates:
        return min(release_dates)

    return None


def is_duplicate(recording, selected):
    """Check whether a recording has already been selected."""

    recording_mbid = recording.get("recording_mbid")
    title = normalize_title(recording.get("recording_name", ""))

    for previous in selected:

        # Exact MusicBrainz recording match
        if (
            recording_mbid
            and recording_mbid == previous["recording_mbid"]
        ):
            return True

        # Same normalized title
        previous_title = normalize_title(previous["recording_name"])

        if title and title == previous_title:
            return True

    return False


# --------------------------------------------------
# Main selection logic
# --------------------------------------------------

def select_top_songs(session, artist):
    """Select up to five eligible songs for one artist."""

    artist_name = artist["name"]
    artist_mbid = str(artist["mbid"])

    print(f"\nFetching: {artist_name}")

    recordings = get_top_recordings(session, artist_mbid)

    print(f"  ListenBrainz returned {len(recordings)} recordings.")

    selected = []

    for rank, recording in enumerate(recordings, start=1):

        title = recording.get("recording_name", "")
        recording_mbid = recording.get("recording_mbid")

        if not title:
            continue

        # Skip alternative versions
        if is_alternative_version(title):
            print(f"  Skipping alternative version: {title}")
            continue

        # Skip duplicates
        if is_duplicate(recording, selected):
            print(f"  Skipping duplicate: {title}")
            continue

        # Look up release year in MusicBrainz
        try:
            year = get_release_year(session, recording_mbid)

        except requests.RequestException as error:
            print(f"  MusicBrainz lookup failed for {title}: {error}")
            time.sleep(1.1)
            continue

        # MusicBrainz allows maximum one request per second
        time.sleep(1.1)

        if year is None:
            print(f"  Skipping (no release year): {title}")
            continue

        if not MIN_YEAR <= year <= MAX_YEAR:
            print(f"  Skipping (year {year}): {title}")
            continue

        # Song passes all criteria
        selected.append({
            "artist": artist_name,
            "artist_mbid": artist_mbid,
            "rank": rank,
            "recording_name": title,
            "recording_mbid": recording_mbid,
            "release": recording.get("release_name"),
            "release_mbid": recording.get("release_mbid"),
            "release_year": year,
            "listen_count": recording.get("total_listen_count"),
            "user_count": recording.get("total_user_count"),
            "length_ms": recording.get("length"),
        })

        print(f"  Selected ({len(selected)}/{N_SONGS_PER_ARTIST}): "
              f"{title} ({year})")

        if len(selected) == N_SONGS_PER_ARTIST:
            break

    print(f"  Selected {len(selected)} songs for {artist_name}.")

    return selected


# --------------------------------------------------
# Run
# --------------------------------------------------

def main():

    artists = pd.read_csv(ARTISTS_FILE)

    all_results = []

    with requests.Session() as session:

        for _, artist in artists.iterrows():

            try:
                selected = select_top_songs(session, artist)
                all_results.extend(selected)

            except requests.RequestException as error:
                print(f"  API request failed for {artist['name']}: {error}")

            # Small pause between artist requests
            time.sleep(1)

    if all_results:

        df = pd.DataFrame(all_results)

        df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 50)
        print("Top recordings collection completed!")
        print(f"Artists processed: {len(artists)}")
        print(f"Songs selected: {len(df)}")
        print(f"Artists with at least one song: {df['artist'].nunique()}")
        print(f"Saved to: {OUTPUT_FILE}")

        print("\nSongs per artist:")
        print(df.groupby("artist").size().describe())

    else:
        print("\nNo recordings were selected.")


if __name__ == "__main__":
    main()
