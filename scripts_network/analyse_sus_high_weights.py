import pandas as pd
from itertools import combinations
from collections import Counter

INPUT_FILE = "data_network/data_2005_2025/deduplicated/recordings.csv"
OUTPUT_FILE = "data_network/data_2005_2025/deduplicated/top_30_weigths_sus.csv"

df = pd.read_csv(INPUT_FILE)

# Count collaborations between artist pairs
pair_counts = Counter()

for artists in df["artists"].dropna():
    artist_list = sorted(set(artists.split("; ")))

    for pair in combinations(artist_list, 2):
        pair_counts[pair] += 1

# Get the 30 artist pairs with the highest weights
top_30_pairs = pair_counts.most_common(30)

# Collect all recordings for these pairs
results = []

for rank, (pair, weight) in enumerate(top_30_pairs, start=1):
    pair_set = set(pair)

    mask = df["artists"].apply(
        lambda x: pair_set.issubset(set(x.split("; ")))
        if isinstance(x, str) else False
    )

    recordings = df.loc[mask].copy()

    recordings["rank"] = rank
    recordings["artist_1"] = pair[0]
    recordings["artist_2"] = pair[1]
    recordings["edge_weight"] = weight

    results.append(recordings)

# Combine results
output_df = pd.concat(results, ignore_index=True)

# Reorder columns
columns = [
    "rank",
    "artist_1",
    "artist_2",
    "edge_weight",
    "recording_id",
    "title",
    "artists",
    "first_release_date",
]

output_df = output_df[columns]

# Sort by rank and title
output_df = output_df.sort_values(
    ["rank", "title"]
)

# Save to CSV
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(output_df)} recordings to {OUTPUT_FILE}")

# Print summary
print("\nTop 30 artist pairs:")
for rank, (pair, weight) in enumerate(top_30_pairs, start=1):
    print(f"{rank:2}. {pair[0]} – {pair[1]}: {weight}")