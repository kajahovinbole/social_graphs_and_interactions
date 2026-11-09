
from pathlib import Path

import pandas as pd
import networkx as nx

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "data_2005_2025" / "deduplicated"
BACKBONE_DIR = DATA_DIR / "backbone"

ARTISTS_FILE = DATA_DIR / "artists.csv"

artists = pd.read_csv(ARTISTS_FILE)
artist_ids = artists["mbid"].astype(str).tolist()

files = sorted(BACKBONE_DIR.glob("edges_backbone_alpha_*.csv"))

print(f"Original artists: {len(artist_ids)}")
print(f"Original edges: 443\n")

for file in files:
    edges = pd.read_csv(file)

    G = nx.Graph()

    # Add all original artists, including isolated ones
    G.add_nodes_from(artist_ids)

    # Add retained edges
    for _, edge in edges.iterrows():
        G.add_edge(
            str(edge["source"]),
            str(edge["target"]),
            weight=edge["weight"]
        )

    # Identify connected artists
    active_nodes = [node for node, degree in G.degree() if degree > 0]
    active_graph = G.subgraph(active_nodes)

    components = list(nx.connected_components(active_graph))
    component_sizes = sorted(
        [len(c) for c in components],
        reverse=True
    )

    print(f"--- {file.name} ---")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Artists with connections: {len(active_nodes)}")
    print(f"Isolated artists: {len(artist_ids) - len(active_nodes)}")
    print(f"Connected components: {len(components)}")
    print(f"Largest component: {component_sizes[0] if component_sizes else 0} artists")
    print(f"Top 5 component sizes: {component_sizes[:5]}")
    print()
