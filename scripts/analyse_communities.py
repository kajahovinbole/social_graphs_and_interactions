
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025"

BACKBONE_FILE = (
    DATA_DIR
    / "backbone_mod_alpha042"
    / "tables_communities_325_326"
    / "mod_10_0780.csv"
)

ORIGINAL_FILE = (
    DATA_DIR
    / "tables_commuties307"
    / "mod_19_0685.csv"
)

OUTPUT_DIR = DATA_DIR / "community_analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TOP_N = 10


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)

    # Normaliser kolonnenavn
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"id", "label", "modularity_class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mangler kolonner. Fant: {list(df.columns)}")

    df = df.rename(columns={
        "id": "Id",
        "label": "Label",
        "modularity_class": "community"
    })

    df["Id"] = df["Id"].astype(str)
    df["Label"] = df["Label"].fillna("Unknown")
    df["community"] = df["community"].astype(int)

    return df

# --------------------------------------------------
# Community sizes
# --------------------------------------------------

def get_community_sizes(df):
    sizes = (
        df.groupby("community")
        .agg(
            size=("Id", "count")
        )
        .reset_index()
        .sort_values(
            "size",
            ascending=False
        )
    )

    return sizes


# --------------------------------------------------
# Export artists in largest communities
# --------------------------------------------------

def export_community_members(df, community_sizes, filename):

    top_communities = (
        community_sizes.head(TOP_N)["community"]
    )

    members = df[
        df["community"].isin(top_communities)
    ].copy()

    members = members.merge(
        community_sizes,
        on="community"
    )

    members = members.rename(columns={
        "community": "Community",
        "Label": "Artist",
        "size": "Community size"
    })

    members = members.sort_values(
        ["Community size", "Community", "Artist"],
        ascending=[False, True, True]
    )

    output = members[
        ["Community", "Community size", "Artist", "Id"]
    ]

    output.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )

    return members


# --------------------------------------------------
# Compare backbone communities with original
# --------------------------------------------------

def compare_communities(backbone, original):

    merged = backbone[
        ["Id", "Label", "community"]
    ].merge(
        original[["Id", "community"]],
        on="Id",
        suffixes=("_backbone", "_original"),
        validate="one_to_one"
    )

    results = []

    top_communities = (
        backbone["community"]
        .value_counts()
        .head(TOP_N)
        .index
    )

    for backbone_id in top_communities:

        group = merged[
            merged["community_backbone"] == backbone_id
        ]

        backbone_artists = set(group["Id"])
        backbone_size = len(backbone_artists)

        # Find original community with the largest overlap
        overlaps = group["community_original"].value_counts()

        original_id = overlaps.index[0]
        intersection = int(overlaps.iloc[0])

        original_artists = set(
            original.loc[
                original["community"] == original_id,
                "Id"
            ]
        )

        union = len(
            backbone_artists | original_artists
        )

        results.append({
            "backbone_community": backbone_id,
            "backbone_size": backbone_size,
            "best_matching_original_community": original_id,
            "original_community_size": len(original_artists),
            "shared_artists": intersection,
            "backbone_group_retained_pct": round(
                100 * intersection / backbone_size, 1
            ),
            "jaccard_similarity": round(
                intersection / union, 3
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Print community members
# --------------------------------------------------

def print_community_members(members, title):

    print(f"\n{title}")
    print("=" * len(title))

    for community, group in members.groupby(
        "Community",
        sort=False
    ):

        size = group["Community size"].iloc[0]

        print(
            f"\nCommunity {community} ({size} artists)"
        )

        for artist in sorted(group["Artist"].tolist()):
            print(f"  - {artist}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    backbone = load_data(BACKBONE_FILE)
    original = load_data(ORIGINAL_FILE)

    # Overview
    print("\nNETWORK OVERVIEW")
    print("----------------")

    print(f"Artists in backbone: {len(backbone)}")
    print(f"Backbone communities: {backbone['community'].nunique()}")

    print(f"Artists in original: {len(original)}")
    print(f"Original communities: {original['community'].nunique()}")

    # Community sizes
    backbone_sizes = get_community_sizes(backbone)
    original_sizes = get_community_sizes(original)

    backbone_sizes.to_csv(
        OUTPUT_DIR / "backbone_community_sizes.csv",
        index=False
    )

    original_sizes.to_csv(
        OUTPUT_DIR / "original_community_sizes.csv",
        index=False
    )

    # Export community members
    backbone_members = export_community_members(
        backbone,
        backbone_sizes,
        "backbone_top_10_members.csv"
    )

    original_members = export_community_members(
        original,
        original_sizes,
        "original_top_10_members.csv"
    )

    # Print largest communities
    print_community_members(
        backbone_members,
        "TOP 10 BACKBONE COMMUNITIES (alpha = 0.42)"
    )

    print_community_members(
        original_members,
        "TOP 10 ORIGINAL COMMUNITIES"
    )

    # Compare communities
    comparison = compare_communities(
        backbone,
        original
    )

    comparison.to_csv(
        OUTPUT_DIR / "community_comparison.csv",
        index=False
    )

    print("\nCOMMUNITY COMPARISON")
    print("--------------------")
    print(comparison.to_string(index=False))

    # Summary of largest communities
    print("\nTOP 10 BACKBONE COMMUNITY SIZES")
    print("--------------------------------")
    print(
        backbone_sizes.head(TOP_N).to_string(index=False)
    )

    print("\nTOP 10 ORIGINAL COMMUNITY SIZES")
    print("--------------------------------")
    print(
        original_sizes.head(TOP_N).to_string(index=False)
    )

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
