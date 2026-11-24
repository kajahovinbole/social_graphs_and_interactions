from pathlib import Path
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

ARTISTS_FILE = (
    PROJECT_DIR
    / "data"
    / "build_network"
    / "artists.csv"
)

OUTPUT_DIR = Path("data_lyrics/hentet_recordings_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "top_recordings.csv"
PROGRESS_FILE = OUTPUT_DIR / "top_recordings_progress.json"
CACHE_FILE = OUTPUT_DIR / "recording_year_cache.json"

load_dotenv(PROJECT_DIR / ".env")
LISTENBRAINZ_TOKEN = os.getenv("LISTENBRAINZ_TOKEN")

if not LISTENBRAINZ_TOKEN:
    raise ValueError("LISTENBRAINZ_TOKEN not found in .env")

N_SONGS_PER_ARTIST = 15
MIN_YEAR = 2005
MAX_YEAR = 2025

# Avoid obvious alternative versions. Generic terms such as "version" and
# "edit" are intentionally omitted to avoid excluding legitimate titles.
EXCLUDED_TERMS = [
    "live", "remix", "remastered", "acoustic version", "demo",
    "instrumental", "karaoke", "radio edit", "extended version",
    "sped up", "slowed", "re-recorded", "re recorded",
    "a cappella", "acapella", "edit", "versione", "mix", "extended", "clean", "explicit", "slowed down"
]

MUSICBRAINZ_MIN_INTERVAL = 1.1
MAX_RETRIES = 5
BACKOFF_SECONDS = 2

LISTENBRAINZ_URL = (
    "https://api.listenbrainz.org/1/popularity/"
    "top-recordings-for-artist/{}"
)
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/recording/{}"

LB_HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0",
    "Authorization": f"Token {LISTENBRAINZ_TOKEN}",
}
MB_HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0 (research project)",
}

_last_musicbrainz_request = 0.0


# --------------------------------------------------
# File helpers
# --------------------------------------------------

def load_json(path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"Could not read {path}. Check the file before continuing."
        ) from error


def save_json_atomic(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def save_progress(progress):
    """Save progress after each artist and regenerate the results CSV."""
    save_json_atomic(PROGRESS_FILE, progress)

    results = progress.get("results", [])
    columns = [
        "artist", "artist_mbid", "rank", "recording_name",
        "recording_mbid", "release", "release_mbid", "release_year",
        "listen_count", "user_count", "length_ms",
    ]
    df = pd.DataFrame(results, columns=columns)

    temporary_csv = OUTPUT_FILE.with_suffix(".csv.tmp")
    df.to_csv(temporary_csv, index=False)
    temporary_csv.replace(OUTPUT_FILE)


# --------------------------------------------------
# Text and duplicate helpers
# --------------------------------------------------

def normalize_title(title):
    """Normalize titles while preserving non-Latin letters and numbers."""
    title = unicodedata.normalize("NFKD", str(title))
    title = "".join(
        char for char in title if not unicodedata.combining(char)
    )
    title = title.casefold()
    return "".join(char for char in title if char.isalnum())


def is_alternative_version(title):
    title = str(title).casefold()
    for term in EXCLUDED_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, title):
            return True
    return False


def is_duplicate(recording, selected):
    """
    Keep the existing policy: skip the same recording MBID or an identical
    normalized title within the same artist.
    """
    recording_mbid = recording.get("recording_mbid")
    title = normalize_title(recording.get("recording_name", ""))

    for previous in selected:
        if (
            recording_mbid
            and recording_mbid == previous.get("recording_mbid")
        ):
            return True

        previous_title = normalize_title(previous.get("recording_name", ""))
        if title and title == previous_title:
            return True

    return False


# --------------------------------------------------
# HTTP and API helpers
# --------------------------------------------------

def wait_for_musicbrainz():
    global _last_musicbrainz_request
    elapsed = time.monotonic() - _last_musicbrainz_request
    wait_time = MUSICBRAINZ_MIN_INTERVAL - elapsed
    if wait_time > 0:
        time.sleep(wait_time)


def retry_delay(response, attempt):
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    now = datetime.now(timezone.utc)
                    return max(0.0, (retry_date - now).total_seconds())
                except (TypeError, ValueError):
                    pass
    return BACKOFF_SECONDS * (2 ** attempt)


def request_json(session, url, *, headers, params=None, musicbrainz=False):
    """GET JSON, retrying temporary errors and respecting MusicBrainz rate limits."""
    global _last_musicbrainz_request
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(MAX_RETRIES):
        if musicbrainz:
            wait_for_musicbrainz()

        response = None
        try:
            response = session.get(
                url, headers=headers, params=params, timeout=30
            )
            if musicbrainz:
                _last_musicbrainz_request = time.monotonic()

            if response.status_code in retryable_statuses:
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                delay = retry_delay(response, attempt)
                print(
                    f"  Temporary API response {response.status_code}; "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BACKOFF_SECONDS * (2 ** attempt)
            print(f"  Network error: {error}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}")


def get_top_recordings(session, artist_mbid):
    url = LISTENBRAINZ_URL.format(artist_mbid)
    recordings = request_json(session, url, headers=LB_HEADERS)

    if not isinstance(recordings, list):
        raise ValueError(
            f"Unexpected ListenBrainz response for {artist_mbid}: "
            f"{recordings}"
        )
    return recordings


def get_release_year(session, recording_mbid, year_cache):
    """Look up and cache the first release year, including missing dates."""
    if not recording_mbid:
        return None

    if recording_mbid in year_cache:
        return year_cache[recording_mbid]

    url = MUSICBRAINZ_URL.format(recording_mbid)
    params = {"fmt": "json", "inc": "releases"}

    data = request_json(
        session, url, headers=MB_HEADERS,
        params=params, musicbrainz=True
    )

    year = None
    first_release_date = data.get("first-release-date", "")
    year_match = re.match(r"(\d{4})", first_release_date)

    if year_match:
        year = int(year_match.group(1))
    else:
        release_years = []
        for release in data.get("releases", []):
            year_match = re.match(r"(\d{4})", release.get("date", ""))
            if year_match:
                release_years.append(int(year_match.group(1)))
        if release_years:
            year = min(release_years)

    year_cache[recording_mbid] = year
    save_json_atomic(CACHE_FILE, year_cache)
    return year


# --------------------------------------------------
# Main selection logic
# --------------------------------------------------

def select_top_songs(session, artist, year_cache):
    """Select up to N_SONGS_PER_ARTIST eligible songs for one artist."""
    artist_name = artist["name"]
    artist_mbid = str(artist["mbid"])

    print(f"\nFetching: {artist_name}")
    recordings = get_top_recordings(session, artist_mbid)
    print(f"  ListenBrainz returned {len(recordings)} recordings.")

    selected = []
    stats = {
        "returned": len(recordings),
        "missing_title": 0,
        "alternative_version": 0,
        "duplicate": 0,
        "missing_recording_mbid": 0,
        "missing_release_year": 0,
        "outside_year_range": 0,
        "selected": 0,
    }

    for rank, recording in enumerate(recordings, start=1):
        title = recording.get("recording_name", "")
        recording_mbid = recording.get("recording_mbid")

        if not title:
            stats["missing_title"] += 1
            continue

        if not recording_mbid:
            stats["missing_recording_mbid"] += 1

        if is_alternative_version(title):
            stats["alternative_version"] += 1
            continue

        if is_duplicate(recording, selected):
            stats["duplicate"] += 1
            continue

        try:
            year = get_release_year(session, recording_mbid, year_cache)
        except requests.RequestException as error:
            print(f"  MusicBrainz lookup failed for {title}: {error}")
            continue

        if year is None:
            stats["missing_release_year"] += 1
            continue

        if not MIN_YEAR <= year <= MAX_YEAR:
            stats["outside_year_range"] += 1
            continue

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

        print(
            f"  Selected ({len(selected)}/{N_SONGS_PER_ARTIST}): "
            f"{title} ({year})"
        )

        if len(selected) >= N_SONGS_PER_ARTIST:
            break

    stats["selected"] = len(selected)
    print(f"  Selected {len(selected)} songs for {artist_name}.")
    print(f"  Filtering summary: {stats}")
    return selected, stats


# --------------------------------------------------
# Run
# --------------------------------------------------

def main():
    if not ARTISTS_FILE.exists():
        raise FileNotFoundError(f"Artists file not found: {ARTISTS_FILE}")

    artists = pd.read_csv(ARTISTS_FILE)
    required_columns = {"name", "mbid"}
    missing_columns = required_columns - set(artists.columns)

    if missing_columns:
        raise ValueError(
            f"{ARTISTS_FILE} is missing required columns: {missing_columns}"
        )

    progress = load_json(
        PROGRESS_FILE,
        {
            "processed_artist_mbids": [],
            "results": [],
            "artist_stats": [],
            "failed_artists": [],
        },
    )
    processed_ids = set(progress.get("processed_artist_mbids", []))
    year_cache = load_json(CACHE_FILE, {})

    print(f"Artists in input file: {len(artists)}")
    print(f"Artists already processed: {len(processed_ids)}")
    print(f"Cached MusicBrainz release years: {len(year_cache)}")
    print(f"Target songs per artist: {N_SONGS_PER_ARTIST}")

    with requests.Session() as session:
        for _, artist in artists.iterrows():
            artist_name = artist["name"]
            artist_mbid = str(artist["mbid"])

            if artist_mbid in processed_ids:
                print(f"\nSkipping already processed artist: {artist_name}")
                continue

            try:
                selected, stats = select_top_songs(
                    session, artist, year_cache
                )

                progress["results"].extend(selected)
                progress["artist_stats"].append({
                    "artist": artist_name,
                    "artist_mbid": artist_mbid,
                    **stats,
                })
                progress["processed_artist_mbids"].append(artist_mbid)
                processed_ids.add(artist_mbid)

                # Save after every artist, including artists with zero songs.
                save_progress(progress)

            except (requests.RequestException, ValueError, RuntimeError) as error:
                print(f"\nFailed to process {artist_name}: {error}")

                progress["failed_artists"] = [
                    item for item in progress.get("failed_artists", [])
                    if item.get("artist_mbid") != artist_mbid
                ]
                progress["failed_artists"].append({
                    "artist": artist_name,
                    "artist_mbid": artist_mbid,
                    "error": str(error),
                })
                save_json_atomic(PROGRESS_FILE, progress)

            time.sleep(1)

    results = progress.get("results", [])
    df = pd.DataFrame(results)

    if not df.empty:
        df.to_csv(OUTPUT_FILE, index=False)

    stats_df = pd.DataFrame(progress.get("artist_stats", []))
    stats_file = OUTPUT_DIR / "top_recordings_collection_stats.csv"
    if not stats_df.empty:
        stats_df.to_csv(stats_file, index=False)

    print("\n" + "=" * 60)
    print("TOP RECORDINGS COLLECTION")
    print("=" * 60)
    print(f"Artists in input file: {len(artists)}")
    print(f"Artists processed: {len(processed_ids)}")
    print(
        "Artists with at least one selected song: "
        f"{df['artist_mbid'].nunique() if not df.empty else 0}"
    )
    print(f"Songs selected: {len(df)}")
    print(f"Artists with errors: {len(progress.get('failed_artists', []))}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Per-artist statistics saved to: {stats_file}")
    print(f"Progress saved to: {PROGRESS_FILE}")
    print(f"Release-year cache saved to: {CACHE_FILE}")

    if not df.empty:
        print("\nSongs per artist:")
        print(df.groupby("artist_mbid").size().describe())


if __name__ == "__main__":
    main()
