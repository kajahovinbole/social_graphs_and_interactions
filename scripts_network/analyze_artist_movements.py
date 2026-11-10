
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025"

BACKBONE_FILE = (
    DATA_DIR / "backbone_mod_alpha042"
    / "tables_communities_325_326" / "mod_10_0780.csv"
)

ORIGINAL_FILE = (
    DATA_DIR / "tables_commuties307" / "mod_19_0685.csv"
)

OUTPUT_DIR = DATA_DIR / "community_analysis" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.rename(columns={
        "id": "Id",
        "label": "Artist",
        "modularity_class": "community"
    })

    required = {"Id", "Artist", "community"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {path}: {list(df.columns)}"
        )

    df["Id"] = df["Id"].astype(str)
    df["community"] = df["community"].astype(int)

    return df[["Id", "Artist", "community"]]


def compare_networks(backbone, original):
    comparison = backbone.merge(
        original[["Id", "community"]],
        on="Id",
        suffixes=("_backbone", "_original"),
        validate="one_to_one"
    )

    if len(comparison) != len(backbone) or len(comparison) != len(original):
        raise ValueError("The two networks do not contain the same artists.")

    return comparison


def create_transition_table(comparison):
    # Count artists moving between each pair of communities
    transitions = (
        comparison.groupby(
            ["community_original", "community_backbone"]
        )
        .size()
        .reset_index(name="artist_count")
    )

    # Percentage of each original community going to each backbone community
    original_sizes = (
        comparison.groupby("community_original")
        .size()
        .rename("original_size")
    )

    transitions = transitions.merge(
        original_sizes,
        on="community_original"
    )

    transitions["percentage_of_original"] = (
        transitions["artist_count"]
        / transitions["original_size"] * 100
    ).round(1)

    return transitions.sort_values(
        "artist_count", ascending=False
    )


def analyze_splits_and_merges(comparison):
    # Number of destination communities for each original community
    original_destinations = (
        comparison.groupby("community_original")["community_backbone"]
        .nunique()
    )

    # Number of source communities for each backbone community
    backbone_sources = (
        comparison.groupby("community_backbone")["community_original"]
        .nunique()
    )

    results = []

    for original_id, group in comparison.groupby("community_original"):
        destinations = group["community_backbone"].nunique()
        original_size = len(group)

        # Check whether any destination also contains artists
        # from other original communities
        destination_ids = group["community_backbone"].unique()

        has_merge = any(
            backbone_sources.loc[destination] > 1
            for destination in destination_ids
        )

        if destinations > 1 and has_merge:
            status = "Split + merge"
        elif destinations > 1:
            status = "Split"
        elif has_merge:
            status = "Merged"
        else:
            status = "Preserved"

        largest_destination = (
            group["community_backbone"].value_counts().idxmax()
        )

        largest_count = (
            group["community_backbone"].value_counts().max()
        )

        results.append({
            "original_community": original_id,
            "original_size": original_size,
            "number_of_backbone_communities": destinations,
            "largest_backbone_community": largest_destination,
            "artists_in_largest_group": largest_count,
            "largest_group_percentage": round(
                largest_count / original_size * 100, 1
            ),
            "status": status
        })

    return pd.DataFrame(results).sort_values(
        "original_size", ascending=False
    )


def create_sankey(comparison, top_n=10):
    # Select the largest original communities
    top_original = (
        comparison["community_original"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    selected = comparison[
        comparison["community_original"].isin(top_original)
    ].copy()

    # Count transitions
    flows = (
        selected.groupby(
            ["community_original", "community_backbone"]
        )
        .size()
        .reset_index(name="value")
    )

    original_ids = sorted(flows["community_original"].unique())
    backbone_ids = sorted(flows["community_backbone"].unique())

    # Create separate nodes for the two sides
    labels = (
        [f"Original {c}" for c in original_ids]
        + [f"Backbone {c}" for c in backbone_ids]
    )

    original_index = {
        community: i
        for i, community in enumerate(original_ids)
    }

    backbone_index = {
        community: i + len(original_ids)
        for i, community in enumerate(backbone_ids)
    }

    sources = [
        original_index[c]
        for c in flows["community_original"]
    ]

    targets = [
        backbone_index[c]
        for c in flows["community_backbone"]
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=18,
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=flows["value"],
            customdata=flows[
                ["community_original", "community_backbone"]
            ].values,
            hovertemplate=(
                "Original community: %{customdata[0]}<br>"
                "Backbone community: %{customdata[1]}<br>"
                "Artists: %{value}<extra></extra>"
            )
        )
    )])

    fig.update_layout(
        title_text=(
            f"Community transitions: Original to Backbone "
            f"(Top {top_n} original communities)"
        ),
        font_size=11,
        height=800
    )

    output_path = OUTPUT_DIR / "community_transitions_sankey.html"
    fig.write_html(output_path)

    print(f"Sankey diagram saved to: {output_path}")


def main():
    backbone = load_data(BACKBONE_FILE)
    original = load_data(ORIGINAL_FILE)

    comparison = compare_networks(backbone, original)

    # Transition table
    transitions = create_transition_table(comparison)
    transitions.to_csv(
        OUTPUT_DIR / "community_transitions.csv",
        index=False
    )

    # Split and merge analysis
    summary = analyze_splits_and_merges(comparison)
    summary.to_csv(
        OUTPUT_DIR / "community_split_merge_summary.csv",
        index=False
    )

    # Summary statistics
    print("\nNETWORK COMPARISON")
    print("------------------")
    print(f"Artists compared: {len(comparison)}")
    print(f"Original communities: {original['community'].nunique()}")
    print(f"Backbone communities: {backbone['community'].nunique()}")

    print("\nCOMMUNITY STRUCTURE CHANGES")
    print("---------------------------")
    print(summary["status"].value_counts().to_string())

    print("\nLARGEST ORIGINAL COMMUNITIES")
    print("----------------------------")
    print(summary.head(10).to_string(index=False))

    print("\nLARGEST COMMUNITY TRANSITIONS")
    print("-----------------------------")
    print(transitions.head(20).to_string(index=False))

    # Sankey visualization
    create_sankey(comparison, top_n=10)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
