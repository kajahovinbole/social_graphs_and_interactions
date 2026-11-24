
import pandas as pd
from rapidfuzz import fuzz
from itertools import combinations
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path("data/data_2005_2025_network/deduplication")

INPUT_FILE = BASE_DIR / "unique_songs.csv"
REVIEW_FILE = BASE_DIR / "potential_duplicates_review.csv"

OUTPUT_FILE = BASE_DIR / "unique_songs_reviewed.csv"
MAPPING_FILE = BASE_DIR / "reviewed_song_mapping.csv"

THRESHOLD = 60

# --------------------------------------------------
# Step 1: Generate review file
# --------------------------------------------------

def generate_review_file():

    df = pd.read_csv(INPUT_FILE)
    matches = []

    # Compare songs with the same artist credits
    for artists, group in df.groupby("artists"):

        for (_, song1), (_, song2) in combinations(
            group.iterrows(), 2
        ):

            title1 = str(song1["canonical_title"])
            title2 = str(song2["canonical_title"])

            similarity = fuzz.ratio(title1, title2)

            if similarity >= THRESHOLD:

                matches.append({
                    "song_id_1": song1["song_id"],
                    "song_id_2": song2["song_id"],
                    "artists": artists,
                    "title_1": song1["title"],
                    "title_2": song2["title"],
                    "similarity": round(similarity, 1),
                    "review_status": "review",
                })

    results = pd.DataFrame(matches)

    if results.empty:
        print("No potential duplicates found.")
        return

    results = results.sort_values(
        "similarity",
        ascending=False
    )

    results.to_csv(REVIEW_FILE, index=False)

    print(f"Found {len(results)} potential duplicate pairs.")
    print(f"Review file saved to: {REVIEW_FILE}")

    print("\nReview status options:")
    print("merge  = same song, merge them")
    print("keep   = different songs, keep both")
    print("review = not yet decided")


# --------------------------------------------------
# Step 2: Apply manual decisions
# --------------------------------------------------

def apply_review_decisions():

    songs = pd.read_csv(INPUT_FILE)
    review = pd.read_csv(REVIEW_FILE)

    # Validate statuses
    valid_statuses = {"merge", "keep", "review"}

    invalid = set(
        review["review_status"].dropna().str.lower()
    ) - valid_statuses

    if invalid:
        raise ValueError(f"Invalid review statuses: {invalid}")

    # Union-find structure for merging song IDs
    parent = {
        int(song_id): int(song_id)
        for song_id in songs["song_id"]
    }

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            parent[root_y] = root_x

    # Apply only approved merges
    merge_rows = review[
        review["review_status"].str.lower() == "merge"
    ]

    for _, row in merge_rows.iterrows():

        id1 = int(row["song_id_1"])
        id2 = int(row["song_id_2"])

        if id1 not in parent or id2 not in parent:
            raise ValueError(
                f"Unknown song IDs: {id1}, {id2}"
            )

        union(id1, id2)

    # Assign each song to its final group
    songs["final_song_id"] = songs["song_id"].apply(
        lambda x: find(int(x))
    )

    # Build final unique song table
    final_songs = []

    for final_id, group in songs.groupby("final_song_id"):

        # Use the shortest title as representative
        representative = group.loc[
            group["title"].str.len().idxmin()
        ]

        final_songs.append({
            "song_id": final_id,
            "title": representative["title"],
            "canonical_title": representative["canonical_title"],
            "artists": representative["artists"],
            "first_release_date": representative["first_release_date"],
            "n_recordings": group["n_recordings"].sum(),
            "n_merged_songs": len(group),
        })

    final_df = pd.DataFrame(final_songs)

    # Create mapping from old song IDs to final song IDs
    mapping = songs[
        ["song_id", "final_song_id", "title", "artists"]
    ].copy()

    mapping = mapping.rename(columns={
        "song_id": "original_song_id",
        "final_song_id": "final_song_id",
        "title": "original_title",
    })

    final_df.to_csv(OUTPUT_FILE, index=False)
    mapping.to_csv(MAPPING_FILE, index=False)

    print("\nReview results")
    print("-------------------------")
    print(f"Original unique songs: {len(songs)}")
    print(f"Approved merge pairs: {len(merge_rows)}")
    print(f"Final unique songs: {len(final_df)}")
    print(f"Songs merged away: {len(songs) - len(final_df)}")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {MAPPING_FILE}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    if not REVIEW_FILE.exists():
        generate_review_file()

    else:
        print("Review file already exists.")
        print("Choose what to do:")
        print("1 = Regenerate potential duplicate list")
        print("2 = Apply manual review decisions")

        choice = input("Choice: ").strip()

        if choice == "1":
            generate_review_file()

        elif choice == "2":
            apply_review_decisions()

        else:
            print("Invalid choice.")
