
from pathlib import Path
import pandas as pd
import networkx as nx


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025" / "deduplicated"

ARTISTS_FILE = DATA_DIR / "artists.csv"
EDGES_FILE = DATA_DIR / "edges.csv"

OUTPUT_FILE = DATA_DIR / "collaboration_network.gexf"


def main():

    # Load CSV files
    artists = pd.read_csv(ARTISTS_FILE)
    edges = pd.read_csv(EDGES_FILE)

    # Create undirected graph
    G = nx.Graph()

    # Add all artists as nodes (including isolated artists)
    for _, artist in artists.iterrows():
        G.add_node(
            str(artist["mbid"]),
            label=str(artist["name"])
        )

    # Add collaboration edges
    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=int(edge["weight"])
        )

    # Export to GEXF
    nx.write_gexf(G, OUTPUT_FILE)

    print("Export complete!")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
