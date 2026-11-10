
from pathlib import Path
import pandas as pd
import networkx as nx


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025" / "deduplicated"
BACKBONE_DIR = DATA_DIR / "backbone"

ARTISTS_FILE = DATA_DIR / "artists.csv"
EDGES_FILE = BACKBONE_DIR / "edges_backbone_alpha_0.42.csv"

# Original Louvain communities
ORIGINAL_COMMUNITIES_FILE = (
    PROJECT_DIR / "data" / "data_2005_2025"
    / "tables_commuties307" / "mod_19_0685.csv"
)

OUTPUT_FILE = DATA_DIR / "collaboration_network_backbone_alpha_0.42.gexf"


def main():

    # Load CSV files
    artists = pd.read_csv(ARTISTS_FILE)
    edges = pd.read_csv(EDGES_FILE)
    communities = pd.read_csv(ORIGINAL_COMMUNITIES_FILE)

    # Normalize column names
    communities.columns = (
        communities.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    communities = communities.rename(columns={
        "id": "mbid",
        "modularity_class": "original_community"
    })

    # Keep only the information we need
    communities = communities[["mbid", "original_community"]]
    communities["mbid"] = communities["mbid"].astype(str)

    # Create community lookup
    community_lookup = dict(
        zip(communities["mbid"], communities["original_community"])
    )

    # Create undirected graph
    G = nx.Graph()

    # Add all artists, including isolated artists
    for _, artist in artists.iterrows():

        mbid = str(artist["mbid"])

        community = community_lookup.get(mbid)

        G.add_node(
            mbid,
            label=str(artist["name"]),
            original_community=(
                int(community) if pd.notna(community) else -1
            )
        )

    # Add backbone edges
    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=int(edge["weight"])
        )

    # Export to GEXF
    nx.write_gexf(G, OUTPUT_FILE)

    # Report
    assigned = sum(
        1 for _, data in G.nodes(data=True)
        if data["original_community"] != -1
    )

    print("Export complete!")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Artists with original community: {assigned}")
    print(f"Artists without community: {G.number_of_nodes() - assigned}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
