
from pathlib import Path
import re
import time
import random
import unicodedata

import pandas as pd
import requests


# ==================================================
# Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data_lyrics" / "top_recordings.csv"

OUTPUT_FILE = PROJECT_DIR / "data_lyrics" / "lyrics.csv"
NOT_FOUND_FILE = PROJECT_DIR / "data_lyrics" / "not_found_lyrics.csv"
INSTRUMENTAL_FILE = PROJECT_DIR / "data_lyrics" / "instrumental_songs.csv"

BASE_URL = "https://lrclib.net/api"

HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0"
}

REQUEST_DELAY = 0.5
MAX_RETRIES = 5
CHECKPOINT_INTERVAL = 25

DURATION_TOLERANCE = 10


# ==================================================
# Text normalization
# ==================================================

def normalize(text):
    """Normalize text for comparison."""

    if text is None or pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKD", str(text))

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.sub(r"[^a-z0-9]", "", text.casefold())


def get_value(row, column):
    """Safely retrieve a non-empty CSV value."""

    value = row.get(column)

    if value is None or pd.isna(value):
        return None

    return value


def get_song_key(row):
    """
    Create a stable identifier for a recording.

    Prefer MusicBrainz recording ID.
    Fall back to artist and title if unavailable.
    """

    recording_mbid = get_value(row, "recording_mbid")

    if recording_mbid:
        return str(recording_mbid)

    return (
        normalize(get_value(row, "artist")),
        normalize(get_value(row, "recording_name"))
    )


# ==================================================
# API requests
# ==================================================

def make_request(session, endpoint, params=None):
    """
    Send a request to LRCLIB.

    Handles rate limits and temporary server errors.
    """

    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(MAX_RETRIES + 1):

        response = session.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=45
        )

        if response.status_code == 404:
            return None

        if response.status_code in (429, 503):

            if attempt == MAX_RETRIES:
                response.raise_for_status()

            retry_after = response.headers.get(
                "Retry-After", "5"
            )

            try:
                wait = float(retry_after)
            except ValueError:
                wait = 5

            wait += random.uniform(0.1, 1.0)

            print(f"  Rate limited/server busy. Waiting {wait:.1f}s")
            time.sleep(wait)
            continue

        response.raise_for_status()

        return response.json()

    return None


# ==================================================
# Lyrics matching
# ==================================================

def matches_title_and_artist(candidate, row):
    """
    Require exact normalized title and artist matches.
    """

    requested_title = normalize(row["recording_name"])
    requested_artist = normalize(row["artist"])

    candidate_title = normalize(candidate.get("trackName", ""))
    candidate_artist = normalize(candidate.get("artistName", ""))

    return (
        candidate_title == requested_title
        and candidate_artist == requested_artist
    )


def select_search_result(candidates, row):
    """
    Select the best search result.

    Requires exact normalized title and artist matches.
    Uses duration and album information to distinguish versions.
    """

    if not candidates:
        return None

    matches = [
        candidate
        for candidate in candidates
        if matches_title_and_artist(candidate, row)
    ]

    if not matches:
        return None

    requested_duration = get_value(row, "length_ms")
    requested_album = normalize(get_value(row, "release"))

    # If duration is available, use it to distinguish versions.
    if requested_duration is not None:

        requested_duration = float(requested_duration) / 1000

        duration_matches = [
            candidate
            for candidate in matches
            if candidate.get("duration") is not None
            and abs(
                float(candidate["duration"]) - requested_duration
            ) <= DURATION_TOLERANCE
        ]

        if not duration_matches:
            return None

        # Prefer closest duration, then matching album.
        duration_matches.sort(
            key=lambda candidate: (
                abs(
                    float(candidate["duration"]) - requested_duration
                ),
                0 if (
                    requested_album
                    and normalize(candidate.get("albumName", ""))
                    == requested_album
                ) else 1
            )
        )

        return duration_matches[0]

    # If duration is unavailable, prefer exact album match.
    if requested_album:

        album_matches = [
            candidate
            for candidate in matches
            if normalize(candidate.get("albumName", ""))
            == requested_album
        ]

        if len(album_matches) == 1:
            return album_matches[0]

    # Accept only an unambiguous match.
    if len(matches) == 1:
        return matches[0]

    return None


# ==================================================
# Lyrics extraction
# ==================================================

def extract_lyrics(result):
    """
    Prefer plain lyrics.
    Fall back to synced lyrics with timestamps removed.
    """

    plain = result.get("plainLyrics") or ""

    if plain.strip():
        return plain.strip(), "plain"

    synced = result.get("syncedLyrics") or ""

    if synced.strip():

        # Remove timestamps such as [01:23.45]
        cleaned = re.sub(
            r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]",
            "",
            synced
        )

        return cleaned.strip(), "synced"

    return "", None


def create_result(row, lyrics_result, method):
    """Create a standardized output row."""

    lyrics_text, lyrics_type = extract_lyrics(lyrics_result)

    instrumental = bool(
        lyrics_result.get("instrumental", False)
    )

    found = (
        not instrumental
        and bool(lyrics_text.strip())
    )

    if instrumental:
        status = "instrumental"
    elif found:
        status = "found"
    else:
        status = "no_lyrics"

    return {
        "artist": row["artist"],
        "artist_mbid": get_value(row, "artist_mbid"),
        "recording_name": row["recording_name"],
        "recording_mbid": get_value(row, "recording_mbid"),
        "release": get_value(row, "release"),
        "release_year": get_value(row, "release_year"),

        "found": found,
        "instrumental": instrumental,
        "status": status,

        "method": method,
        "lrclib_id": lyrics_result.get("id"),

        "matched_title": lyrics_result.get("trackName"),
        "matched_artist": lyrics_result.get("artistName"),
        "matched_album": lyrics_result.get("albumName"),
        "matched_duration": lyrics_result.get("duration"),

        "lyrics_type": lyrics_type,
        "lyrics_length": len(lyrics_text),
        "lyrics_text": lyrics_text
    }


# ==================================================
# Lookup pipeline
# ==================================================

def find_lyrics(session, row):
    """
    Try several LRCLIB lookup methods.

    1. Exact lookup with album and duration.
    2. Exact lookup without duration.
    3. Structured search by title and artist.
    4. General keyword search.
    """

    title = row["recording_name"]
    artist = row["artist"]

    album = get_value(row, "release")
    length_ms = get_value(row, "length_ms")

    duration = None

    if length_ms is not None:

        duration = round(float(length_ms) / 1000)

        if not 1 <= duration <= 3600:
            duration = None

    # ----------------------------------------------
    # Method 1: Exact lookup with album + duration
    # ----------------------------------------------

    params = {
        "track_name": title,
        "artist_name": artist
    }

    if album:
        params["album_name"] = album

    if duration:
        params["duration"] = duration

    print("  Trying exact lookup with album and duration...")

    result = make_request(
        session,
        "get",
        params
    )

    if result and matches_title_and_artist(result, row):
        return result, "exact_album_duration"

    time.sleep(REQUEST_DELAY)

    # ----------------------------------------------
    # Method 2: Exact lookup without duration
    # ----------------------------------------------

    params = {
        "track_name": title,
        "artist_name": artist
    }

    if album:
        params["album_name"] = album

    print("  Trying exact lookup without duration...")

    result = make_request(
        session,
        "get",
        params
    )

    if result and matches_title_and_artist(result, row):
        return result, "exact_without_duration"

    time.sleep(REQUEST_DELAY)

    # ----------------------------------------------
    # Method 3: Structured search
    # ----------------------------------------------

    params = {
        "track_name": title,
        "artist_name": artist
    }

    print("  Trying structured search...")

    candidates = make_request(
        session,
        "search",
        params
    )

    if candidates:

        result = select_search_result(
            candidates,
            row
        )

        if result:
            return result, "structured_search"

    time.sleep(REQUEST_DELAY)

    # ----------------------------------------------
    # Method 4: General keyword search
    # ----------------------------------------------

    params = {
        "q": f"{title} {artist}"
    }

    print("  Trying keyword search...")

    candidates = make_request(
        session,
        "search",
        params
    )

    if candidates:

        result = select_search_result(
            candidates,
            row
        )

        if result:
            return result, "keyword_search"

    return None, "not_found"


# ==================================================
# Checkpoint and output helpers
# ==================================================

def save_outputs(results):
    """Save all results and separate follow-up lists."""

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # All songs and their lyrics.
    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Songs where no verified match was found.
    not_found_df = result_df[
        result_df["status"] == "not_found"
    ].copy()

    NOT_FOUND_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    not_found_df.to_csv(
        NOT_FOUND_FILE,
        index=False
    )

    # Instrumental songs.
    instrumental_df = result_df[
        result_df["status"] == "instrumental"
    ].copy()

    instrumental_df.to_csv(
        INSTRUMENTAL_FILE,
        index=False
    )


def load_previous_results():
    """
    Load existing results to resume an interrupted run.

    Request errors are retried.
    """

    if not OUTPUT_FILE.exists():
        return [], set()

    existing_df = pd.read_csv(OUTPUT_FILE)

    results = existing_df.to_dict("records")

    completed_keys = set()

    for _, previous in existing_df.iterrows():

        if previous.get("status") == "request_error":
            continue

        completed_keys.add(
            get_song_key(previous)
        )

    return results, completed_keys


# ==================================================
# Main
# ==================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Support both CSV column names.
    if "recording" in df.columns:
        df = df.rename(
            columns={"recording": "recording_name"}
        )

    required_columns = [
        "artist",
        "recording_name"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required CSV column: {column}"
            )

    # Load existing progress.
    results, completed_keys = load_previous_results()

    print("=" * 50)
    print("LYRICS COLLECTION")
    print("=" * 50)

    print(f"Songs in input: {len(df)}")
    print(f"Already completed: {len(completed_keys)}")
    print(f"Remaining: {len(df) - len(completed_keys)}")

    with requests.Session() as session:

        for i, row in df.iterrows():

            key = get_song_key(row)

            if key in completed_keys:
                continue

            artist = row["artist"]
            title = row["recording_name"]

            print(
                f"\n[{i + 1}/{len(df)}] {artist} - {title}"
            )

            try:

                lyrics, method = find_lyrics(
                    session,
                    row
                )

                if lyrics is None:

                    print("  No verified match.")

                    output = {
                        "artist": artist,
                        "artist_mbid": get_value(row, "artist_mbid"),
                        "recording_name": title,
                        "recording_mbid": get_value(row, "recording_mbid"),
                        "release": get_value(row, "release"),
                        "release_year": get_value(row, "release_year"),

                        "found": False,
                        "instrumental": None,
                        "status": "not_found",

                        "method": method,
                        "lrclib_id": None,

                        "matched_title": None,
                        "matched_artist": None,
                        "matched_album": None,
                        "matched_duration": None,

                        "lyrics_type": None,
                        "lyrics_length": 0,
                        "lyrics_text": ""
                    }

                else:

                    output = create_result(
                        row,
                        lyrics,
                        method
                    )

                    print(
                        f"  {output['status']} via {method} "
                        f"({output['lyrics_length']} characters)"
                    )

            except requests.RequestException as error:

                print(f"  Request failed: {error}")

                output = {
                    "artist": artist,
                    "artist_mbid": get_value(row, "artist_mbid"),
                    "recording_name": title,
                    "recording_mbid": get_value(row, "recording_mbid"),
                    "release": get_value(row, "release"),
                    "release_year": get_value(row, "release_year"),

                    "found": None,
                    "instrumental": None,
                    "status": "request_error",

                    "method": None,
                    "lrclib_id": None,

                    "matched_title": None,
                    "matched_artist": None,
                    "matched_album": None,
                    "matched_duration": None,

                    "lyrics_type": None,
                    "lyrics_length": 0,
                    "lyrics_text": ""
                }

            results.append(output)

            # Mark successful lookups and not-found songs as completed.
            # Request errors will be retried next time.
            if output["status"] != "request_error":
                completed_keys.add(key)

            # Save progress regularly.
            if len(results) % CHECKPOINT_INTERVAL == 0:

                save_outputs(results)

                print("  Progress saved.")

            time.sleep(REQUEST_DELAY)

    # Final save.
    save_outputs(results)

    result_df = pd.DataFrame(results)

    # ==================================================
    # Summary
    # ==================================================

    successful = result_df["status"].eq("found").sum()
    instrumental = result_df["status"].eq("instrumental").sum()
    not_found = result_df["status"].eq("not_found").sum()
    no_lyrics = result_df["status"].eq("no_lyrics").sum()
    errors = result_df["status"].eq("request_error").sum()

    print("\n" + "=" * 50)
    print("LYRICS COLLECTION COMPLETED")
    print("=" * 50)

    print(f"Total songs processed: {len(result_df)}")
    print(f"Lyrics found: {successful}")
    print(f"Instrumental: {instrumental}")
    print(f"No lyrics text: {no_lyrics}")
    print(f"No verified match: {not_found}")
    print(f"Request errors: {errors}")

    if len(result_df) > 0:

        print(
            f"Lyric coverage: "
            f"{successful / len(result_df) * 100:.1f}%"
        )

    print("\nSaved files:")
    print(f"All lyrics: {OUTPUT_FILE}")
    print(f"Not found: {NOT_FOUND_FILE}")
    print(f"Instrumental: {INSTRUMENTAL_FILE}")

    # ==================================================
    # Successful matches by method
    # ==================================================

    print("\nSuccessful matches by method:")

    print(
        result_df[
            result_df["status"] == "found"
        ]["method"].value_counts()
    )

    # ==================================================
    # Artist coverage
    # ==================================================

    print("\nLyric coverage per artist:")

    artist_stats = result_df.groupby("artist").agg(
        total=("status", "size"),
        found=("status", lambda x: (x == "found").sum())
    )

    artist_stats["coverage"] = (
        artist_stats["found"] / artist_stats["total"] * 100
    )

    print(
        artist_stats.sort_values("coverage").to_string()
    )


if __name__ == "__main__":
    main()
