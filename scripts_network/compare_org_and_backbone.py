
from pathlib import Path
import pandas as pd
import networkx as nx

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "data_2005_2025_network"
)

NODES_FILE = DATA_DIR / "nodes.csv"
ORIGINAL_EDGES = DATA_DIR / "edges.csv"
BACKBONE_EDGES = (
    DATA_DIR / "backbone"
    / "edges_backbone_alpha_0.42.csv"
)


def analyze_network(nodes_file, edges_file, name):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    G = nx.Graph()

    # Add all artists, including isolates
    for _, node in nodes.iterrows():
        G.add_node(str(node["mbid"]), label=node["name"])

    # Add edges
    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=edge["weight"]
        )

    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)

    print(f"\n{name}")
    print("=" * 40)
    print(f"Nodes:                 {G.number_of_nodes()}")
    print(f"Edges:                 {G.number_of_edges()}")
    print(f"Isolates:              {nx.number_of_isolates(G)}")
    print(f"Connected components:  {len(components)}")
    print(f"Largest component:     {len(largest_component)}")

    return G


original = analyze_network(
    NODES_FILE, ORIGINAL_EDGES, "ORIGINAL NETWORK"
)

backbone = analyze_network(
    NODES_FILE, BACKBONE_EDGES, "BACKBONE (alpha = 0.42)"
)
