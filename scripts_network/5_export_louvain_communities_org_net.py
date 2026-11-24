
from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "data_2005_2025_network"
    / "network"
)

ARTISTS_FILE = DATA_DIR / "nodes.csv"
EDGES_FILE = DATA_DIR / "edges.csv"

OUTPUT_FILE = DATA_DIR / "collaboration_network.gexf"
COMMUNITIES_FILE = DATA_DIR / "louvain_communities.csv"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    artists = pd.read_csv(ARTISTS_FILE)
    edges = pd.read_csv(EDGES_FILE)

    G = nx.Graph()

    # Add all artists, including isolates
    for _, artist in artists.iterrows():
        mbid = str(artist["mbid"])

        G.add_node(
            mbid,
            label=str(artist["name"])
        )

    # Add weighted edges
    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=int(edge["weight"])
        )

    # Louvain community detection
    communities = louvain_communities(
        G,
        weight="weight",
        resolution=1,
        seed=42
    )

    # Map each artist to a community ID
    community_lookup = {
        node: community_id
        for community_id, community in enumerate(communities)
        for node in community
    }

    nx.set_node_attributes(
        G,
        community_lookup,
        "original_community"
    )

    # Calculate modularity
    modularity_score = modularity(
        G,
        communities,
        weight="weight"
    )

    # Save community assignments
    community_df = pd.DataFrame([
        {
            "mbid": node,
            "name": G.nodes[node]["label"],
            "original_community": community_lookup[node]
        }
        for node in G.nodes
    ])

    community_df.to_csv(COMMUNITIES_FILE, index=False)

    # Export GEXF
    nx.write_gexf(G, OUTPUT_FILE)

    # Network statistics
    assigned = len(community_lookup)

    print("\nNETWORK ANALYSIS COMPLETE")
    print("=" * 40)
    print(f"Nodes:              {G.number_of_nodes()}")
    print(f"Edges:              {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Communities:        {len(communities)}")
    print(f"Modularity:         {modularity_score:.4f}")
    print(f"Isolates:           {nx.number_of_isolates(G)}")
    print(f"Artists assigned:   {assigned}")
    print(f"\nGEXF saved to:       {OUTPUT_FILE}")
    print(f"Communities saved:   {COMMUNITIES_FILE}")


if __name__ == "__main__":
    main()
