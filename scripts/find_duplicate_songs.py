import pandas as pd
import re
import unicodedata

INPUT_FILE = "data/data_2005_2025/recordings.csv"

df = pd.read_csv(INPUT_FILE)
print(df.columns.tolist())
print(df.head())

def normalize_title(title):
    title = str(title).lower().strip()
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r"[^a-z0-9]", "", title)
    return title

def normalize_artists(artists):
    names = str(artists).split(";")
    names = [name.strip().lower() for name in names]
    return tuple(sorted(names))

df["normalized_title"] = df["title"].apply(normalize_title)
df["normalized_artists"] = df["artists"].apply(normalize_artists)

groups = (
    df.groupby(["normalized_title", "normalized_artists"])
    .filter(lambda group: len(group) > 1)
    .sort_values(["normalized_artists", "normalized_title"])
)

for (title, artists), group in groups.groupby(
    ["normalized_title", "normalized_artists"]
):
    print(f"\n{group.iloc[0]['title']} — {artists}")
    print(
        group[
            ["recording_id", "title", "first_release_date"]
        ].to_string(index=False)
    )

groups.to_csv("possible_duplicate_songs.csv", index=False)

print(f"\nFant {len(groups)} recordings i mulige duplikatgrupper.")

duplicate_groups = (
    df.groupby(["normalized_title", "normalized_artists"])
    .size()
    .reset_index(name="count")
)

duplicate_groups = duplicate_groups[
    duplicate_groups["count"] > 1
]

print("\n--- Duplicate statistics ---")
print(f"Unique duplicate groups: {len(duplicate_groups)}")
print(f"Recordings in duplicate groups: {duplicate_groups['count'].sum()}")
print(f"Potential duplicates to remove: {duplicate_groups['count'].sum() - len(duplicate_groups)}")

print("\nLargest duplicate groups:")
print(duplicate_groups.sort_values("count", ascending=False).head(20))

top_groups = (
    df.groupby(["normalized_title", "normalized_artists"])
    .size()
    .reset_index(name="count")
    .query("count > 1")
    .sort_values("count", ascending=False)
    .head(20)
)

top_duplicates = df.merge(
    top_groups[["normalized_title", "normalized_artists"]],
    on=["normalized_title", "normalized_artists"]
)

top_duplicates.to_csv("top_duplicate_groups.csv", index=False)