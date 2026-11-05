
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ARTISTS_FILE = "project/data/data_2010_2020/artists.csv"
EDGES_FILE = "project/data/data_2010_2020/edges.csv"

OUTPUT_FILE = "project/data/data_2010_2020/collaboration_network.png"

TOP_N_LABELS = 25

# Set to 2 or higher to hide edges with weight 1
MIN_EDGE_WEIGHT = 1

FIGURE_SIZE = (18, 14)


# ============================================================
# LOAD DATA
# ============================================================

def load_network():

    G = nx.Graph()

    # Load all artists, including isolated nodes
    with open(ARTISTS_FILE, encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            G.add_node(
                row["mbid"],
                name=row["name"]
            )

    # Load collaboration edges
    with open(EDGES_FILE, encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            source = row["source"]
            target = row["target"]
            weight = int(row["weight"])

            if weight >= MIN_EDGE_WEIGHT:

                G.add_edge(
                    source,
                    target,
                    weight=weight
                )

    return G


# ============================================================
# CALCULATE NODE SIZES
# ============================================================

def calculate_node_sizes(G):

    weighted_degrees = dict(G.degree(weight="weight"))

    # Square root scaling prevents the largest nodes
    # from dominating the visualization
    sizes = [
        40 + 35 * np.sqrt(weighted_degrees[node])
        for node in G.nodes()
    ]

    return sizes, weighted_degrees


# ============================================================
# VISUALIZE NETWORK
# ============================================================

def visualize_network(G):

    print("\nCalculating connected components...")

    # Find all connected components
    components = [
        G.subgraph(c).copy()
        for c in nx.connected_components(G)
        if len(c) > 1
    ]

    print(f"Number of connected components: {len(components)}")

    # Sort components by size, largest first
    components.sort(key=len, reverse=True)

    # Arrange components in a grid
    cols = int(np.ceil(np.sqrt(len(components))))

    pos = {}

    for i, component in enumerate(components):

        # Calculate layout separately for each component
        component_pos = nx.spring_layout(
            component,
            seed=42,
            weight="weight",
            iterations=150,
            scale=1.0
        )

        # Grid position
        row = i // cols
        col = i % cols

        # Offset component positions
        offset_x = col * 3.5
        offset_y = -row * 3.5

        for node, (x, y) in component_pos.items():
            pos[node] = (x + offset_x, y + offset_y)

    # Remove isolated nodes from visualization
    G_connected = G.subgraph(pos.keys()).copy()

    node_sizes, weighted_degrees = calculate_node_sizes(G_connected)

    edge_widths = [
        0.4 + 0.8 * np.sqrt(G_connected[u][v]["weight"])
        for u, v in G_connected.edges()
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    nx.draw_networkx_edges(
        G_connected,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.35
    )

    nx.draw_networkx_nodes(
        G_connected,
        pos,
        ax=ax,
        node_size=node_sizes,
        alpha=0.85
    )

    # Label the artists with the highest weighted degree
    top_artists = sorted(
        weighted_degrees,
        key=weighted_degrees.get,
        reverse=True
    )[:TOP_N_LABELS]

    labels = {
        node: G_connected.nodes[node]["name"]
        for node in top_artists
    }

    nx.draw_networkx_labels(
        G_connected,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    ax.set_title(
        "Artist Collaboration Network (1970–1980)",
        fontsize=18
    )

    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nNetwork saved to {OUTPUT_FILE}")


# ============================================================
# NETWORK STATISTICS
# ============================================================

def print_statistics(G):

    print("\n========== NETWORK STATISTICS ==========")

    print(f"Number of artists: {G.number_of_nodes()}")
    print(f"Number of collaborations: {G.number_of_edges()}")

    isolated = list(nx.isolates(G))

    print(f"Isolated artists: {len(isolated)}")

    if G.number_of_nodes() > 0:

        print(
            f"Network density: {nx.density(G):.4f}"
        )

    weighted_degrees = dict(G.degree(weight="weight"))

    print("\nTop 10 artists by weighted degree:\n")

    top_artists = sorted(
        weighted_degrees,
        key=weighted_degrees.get,
        reverse=True
    )[:10]

    for node in top_artists:

        print(
            f"{G.nodes[node]['name']:<30}"
            f"Weighted degree: {weighted_degrees[node]}"
        )

        components = [
            c for c in nx.connected_components(G)
            if len(c) > 1
        ]

        print(f"Connected components: {len(components)}")
        print(f"Largest component: {max(map(len, components), default=0)} artists")


# ============================================================
# MAIN
# ============================================================

def main():

    G = load_network()

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("First 10 edges:", list(G.edges(data=True))[:10])

    print_statistics(G)

    visualize_network(G)


if __name__ == "__main__":
    main()
