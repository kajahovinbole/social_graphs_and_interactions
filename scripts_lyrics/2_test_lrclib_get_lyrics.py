
from pathlib import Path
import re
import time
import requests
import pandas as pd
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_DIR / "data_lyrics" / "listenbrainz_test.csv"
OUTPUT_FILE = PROJECT_DIR / "data_lyrics" / "lrclib_test.csv"

API_URL = "https://lrclib.net/api/get"

HEADERS = {
    "User-Agent": "SocialGraphLyricsResearch/1.0"
}


def normalize(text):
    """Normalize text for matching."""

    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.sub(r"[^a-z0-9]", "", text.casefold())


def find_lyrics(session, row):
    """Search LRCLIB for a matching song."""

    params = {
        "track_name": row["recording"],
        "artist_name": row["artist"],
    }

    if pd.notna(row.get("release")):
        params["album_name"] = row["release"]

    if pd.notna(row.get("length_ms")):
        duration = int(row["length_ms"]) / 1000
        if 1 <= duration <= 3600:
            params["duration"] = round(duration)

    response = session.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    if response.status_code == 404:
        return None

    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", 5))
        print(f"  Rate limited. Waiting {wait} seconds.")
        time.sleep(wait)
        return find_lyrics(session, row)

    response.raise_for_status()

    result = response.json()

    # Verify that the returned record matches our song
    if normalize(result.get("trackName", "")) != normalize(
        row["recording"]
    ):
        return None

    if normalize(result.get("artistName", "")) != normalize(
        row["artist"]
    ):
        return None

    return result


def main():

    df = pd.read_csv(INPUT_FILE)

    results = []

    with requests.Session() as session:

        for i, row in df.iterrows():

            artist = row["artist"]
            title = row["recording"]

            print(f"[{i + 1}/{len(df)}] {artist} - {title}")

            try:
                lyrics = find_lyrics(session, row)

                if lyrics is None:
                    print("  No verified match.")

                    results.append({
                        "artist": artist,
                        "recording": title,
                        "found": False,
                        "instrumental": None,
                        "lyrics_length": 0,
                    })

                else:
                    plain = lyrics.get("plainLyrics") or ""

                    synced = lyrics.get("syncedLyrics") or ""

                    lyrics_text = plain if plain.strip() else synced

                    found = (
                        not lyrics.get("instrumental", False)
                        and bool(lyrics_text.strip())
                    )

                    print(
                        f"  {'Lyrics found' if found else 'No lyrics text'} "
                        f"({len(lyrics_text)} characters)"
                    )

                    results.append({
                        "artist": artist,
                        "recording": title,
                        "found": found,
                        "instrumental": lyrics.get("instrumental"),
                        "lyrics_length": len(lyrics_text),
                    })

            except requests.RequestException as error:
                print(f"  Request failed: {error}")

                results.append({
                    "artist": artist,
                    "recording": title,
                    "found": None,
                    "instrumental": None,
                    "lyrics_length": 0,
                })

            time.sleep(0.5)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)

    successful = result_df["found"].eq(True).sum()
    failed = result_df["found"].eq(False).sum()
    errors = result_df["found"].isna().sum()

    print("\n" + "=" * 45)
    print("LRCLIB TEST COMPLETED")
    print(f"Total songs: {len(result_df)}")
    print(f"Lyrics found: {successful}")
    print(f"No lyrics: {failed}")
    print(f"Request errors: {errors}")
    print(f"Coverage: {successful / len(result_df) * 100:.1f}%")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
