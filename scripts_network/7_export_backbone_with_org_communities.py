from pathlib import Path
import pandas as pd
import networkx as nx


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_DIR
    / "data_network"
    / "1_new_2005_2025"
    / "network"
)

NODES_FILE = DATA_DIR / "network_deduplicated/nodes.csv"
BACKBONE_EDGES_FILE = (
    DATA_DIR
    / "backbone"
    / "edges_backbone_alpha_0.42.csv"
)
COMMUNITIES_FILE = DATA_DIR / "network_deduplicated/louvain_communities.csv"

OUTPUT_FILE = DATA_DIR / "backbone" / "backbone_network.gexf"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    nodes = pd.read_csv(NODES_FILE, dtype={"mbid": str})
    edges = pd.read_csv(BACKBONE_EDGES_FILE, dtype={"source": str, "target": str})
    communities = pd.read_csv(COMMUNITIES_FILE, dtype={"mbid": str})

    # Validate required columns
    required_node_cols = {"mbid", "name"}
    required_edge_cols = {"source", "target", "weight"}
    required_community_cols = {"mbid", "original_community"}

    for filename, df, required in [
        ("nodes.csv", nodes, required_node_cols),
        ("backbone edges", edges, required_edge_cols),
        ("louvain_communities.csv", communities, required_community_cols),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    # Ensure one community assignment per artist
    if communities["mbid"].duplicated().any():
        raise ValueError("louvain_communities.csv contains duplicate MBIDs.")

    # Add the original-network community assignments to the node table
    nodes = nodes.merge(
        communities[["mbid", "original_community"]],
        on="mbid",
        how="left",
        validate="one_to_one"
    )

    if nodes["original_community"].isna().any():
        missing_count = nodes["original_community"].isna().sum()
        raise ValueError(
            f"{missing_count} artists have no original_community assignment. "
            "Check that MBIDs match between nodes.csv and louvain_communities.csv."
        )

    # Build backbone graph, retaining all nodes including isolates
    G = nx.Graph()

    for _, node in nodes.iterrows():
        G.add_node(
            str(node["mbid"]),
            label=str(node["name"]),
            original_community=int(node["original_community"])
        )

    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=float(edge["weight"])
        )

    # Export
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, OUTPUT_FILE)

    print("\nBACKBONE EXPORT COMPLETE")
    print("=" * 40)
    print(f"Nodes:                {G.number_of_nodes()}")
    print(f"Edges:                {G.number_of_edges()}")
    print(f"Isolates:             {nx.number_of_isolates(G)}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Original communities: {nodes['original_community'].nunique()}")
    print(f"Artists with community: {nodes['original_community'].notna().sum()}")
    print(f"\nGEXF saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
