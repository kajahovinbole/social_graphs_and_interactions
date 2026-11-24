
from pathlib import Path

import pandas as pd
import networkx as nx

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025_network" / "network"
BACKBONE_DIR = DATA_DIR / "backbone"

ARTISTS_FILE = DATA_DIR / "nodes.csv"

def main():
    artists = pd.read_csv(ARTISTS_FILE)
    edges_original = pd.read_csv(DATA_DIR / "edges.csv")
    artist_ids = artists["mbid"].astype(str).tolist()

    files = sorted(BACKBONE_DIR.glob("edges_backbone_alpha_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No backbone threshold files found in {BACKBONE_DIR}"
        )

    print(f"Original artists: {len(artist_ids)}")
    print(f"Original edges: {len(edges_original)}\n")

    for file in files:
        edges = pd.read_csv(file)
        graph = nx.Graph()
        graph.add_nodes_from(artist_ids)

        for _, edge in edges.iterrows():
            graph.add_edge(
                str(edge["source"]),
                str(edge["target"]),
                weight=edge["weight"],
            )

        active_nodes = [node for node, degree in graph.degree() if degree > 0]
        active_graph = graph.subgraph(active_nodes)
        component_sizes = sorted(
            (len(component) for component in nx.connected_components(active_graph)),
            reverse=True,
        )

        print(f"--- {file.name} ---")
        print(f"Edges: {graph.number_of_edges()}")
        print(f"Artists with connections: {len(active_nodes)}")
        print(f"Isolated artists: {len(artist_ids) - len(active_nodes)}")
        print(f"Connected components: {len(component_sizes)}")
        print(
            "Largest component: "
            f"{component_sizes[0] if component_sizes else 0} artists"
        )
        print(f"Top 5 component sizes: {component_sizes[:5]}\n")


if __name__ == "__main__":
    main()
