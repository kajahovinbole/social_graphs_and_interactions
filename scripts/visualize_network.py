
import csv
import os
import math
import html

import networkx as nx
from pyvis.network import Network


# ============================================================
# CONFIGURATION
# ============================================================

ARTISTS_FILE = "project/data/data_2005_2025/artists.csv"
EDGES_FILE = "project/data/data_2005_2025/edges.csv"

OUTPUT_FILE = "project/data/data_2005_2025/collaboration_network.html"

NETWORK_TITLE = "Artist Collaboration Network (2005–2025)"

MIN_EDGE_WEIGHT = 1

OUTPUT_FILE_BACKBONE = (
    "project/data/data_2005_2025/collaboration_backbone.html"
)

# Visual appearance
COMPONENT_SCALE = 150
COMPONENT_SPACING = 450
NODE_SIZE_MIN = 10
NODE_SIZE_MAX = 35


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


def disparity_filter(G, alpha=0.2):

    backbone = nx.Graph()

    # Preserve all original nodes and their attributes
    backbone.add_nodes_from(G.nodes(data=True))

    # Calculate degree and strength for each node
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    for u, v, data in G.edges(data=True):

        weight = data["weight"]

        # Calculate significance for each endpoint
        p_u = weight / strengths[u]
        p_v = weight / strengths[v]

        k_u = degrees[u]
        k_v = degrees[v]

        # Probability under the disparity filter null model
        alpha_u = (1 - p_u) ** (k_u - 1)
        alpha_v = (1 - p_v) ** (k_v - 1)

        # Keep edge if significant for at least one endpoint
        if alpha_u < alpha or alpha_v < alpha:

            backbone.add_edge(
                u,
                v,
                **data,
                alpha_u=alpha_u,
                alpha_v=alpha_v
            )

    return backbone



# ============================================================
# NODE SIZES
# ============================================================

def calculate_node_sizes(G):

    weighted_degrees = dict(G.degree(weight="weight"))

    max_degree = max(weighted_degrees.values(), default=1)

    sizes = {}

    for node, degree in weighted_degrees.items():

        # Scale node size between NODE_SIZE_MIN and NODE_SIZE_MAX
        normalized = math.sqrt(degree / max_degree)

        sizes[node] = (
            NODE_SIZE_MIN
            + normalized * (NODE_SIZE_MAX - NODE_SIZE_MIN)
        )

    return sizes, weighted_degrees


# ============================================================
# CALCULATE COMPONENT POSITIONS
# ============================================================

def calculate_positions(components):

    positions = {}

    # Arrange components in a grid
    cols = math.ceil(math.sqrt(len(components)))

    for i, component in enumerate(components):

        # Calculate layout separately for each component
        component_pos = nx.spring_layout(
            component,
            seed=42,
            weight="weight",
            iterations=150,
            scale=COMPONENT_SCALE
        )

        # Grid position
        row = i // cols
        col = i % cols

        offset_x = col * COMPONENT_SPACING
        offset_y = -row * COMPONENT_SPACING

        for node, (x, y) in component_pos.items():

            positions[node] = (
                float(x + offset_x),
                float(y + offset_y)
            )

    return positions


# ============================================================
# VISUALIZE NETWORK
# ============================================================

def visualize_network(G, output_file):

    print("\nCalculating connected components...")

    # Find all connected components, excluding isolated nodes
    components = [
        G.subgraph(c).copy()
        for c in nx.connected_components(G)
        if len(c) > 1
    ]

    # Largest components first
    components.sort(key=len, reverse=True)

    print(f"Connected components: {len(components)}")

    if not components:
        print("No collaborations found.")
        return

    # Calculate positions
    positions = calculate_positions(components)

    # Only include nodes belonging to a connected component
    G_connected = G.subgraph(positions.keys()).copy()

    node_sizes, weighted_degrees = calculate_node_sizes(
        G_connected
    )

    # Create interactive network
    net = Network(
        height="900px",
        width="100%",
        directed=False,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line"
    )

    # --------------------------------------------------------
    # ADD NODES
    # --------------------------------------------------------

    for node in G_connected.nodes():

        name = G_connected.nodes[node]["name"]

        x, y = positions[node]

        net.add_node(
            node,
            label=name,
            title=(
                f"<b>{html.escape(name)}</b><br>"
                f"Weighted degree: {weighted_degrees[node]}<br>"
                f"Collaborations: {G_connected.degree(node)}"
            ),
            size=node_sizes[node],
            x=x,
            y=y,
            physics=True,
            font={
                "size": 14,
                "face": "Arial",
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            }
        )

    # --------------------------------------------------------
    # ADD EDGES
    # --------------------------------------------------------

    for u, v, data in G_connected.edges(data=True):

        weight = data["weight"]

        net.add_edge(
            u,
            v,
            width=0.5 + math.sqrt(weight),
            title=f"{weight} collaborative recordings",
            color="rgba(100, 100, 100, 0.45)"
        )

    # --------------------------------------------------------
    # INTERACTION OPTIONS
    # --------------------------------------------------------

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.04,
                "damping": 0.3,
                "avoidOverlap": 0.8
            },
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "fit": true
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true,
            "dragNodes": true,
            "multiselect": true,
            "selectConnectedEdges": true
        },
        "nodes": {
            "shape": "dot",
            "borderWidth": 1,
            "scaling": {
                "min": 8,
                "max": 25
            },
            "font": {
                "size": 14,
                "face": "Arial",
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            }
        },
        "edges": {
            "smooth": false,
            "selectionWidth": 2,
            "color": {
                "inherit": false,
                "color": "#999999",
                "highlight": "#333333"
            }
        }
    }
    """)

    # Save HTML
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    net.write_html(
        output_file,
        open_browser=False
    )

    print(f"\nInteractive network saved to: {output_file}")


# ============================================================
# NETWORK STATISTICS
# ============================================================

def print_statistics(G):

    print("\n========== NETWORK STATISTICS ==========")

    print(f"Number of artists: {G.number_of_nodes()}")
    print(f"Number of collaborations: {G.number_of_edges()}")

    isolated = list(nx.isolates(G))

    print(f"Isolated artists: {len(isolated)}")

    components = [
        c for c in nx.connected_components(G)
        if len(c) > 1
    ]

    print(f"Connected components: {len(components)}")

    if components:

        print(
            f"Largest component: "
            f"{max(map(len, components))} artists"
        )

        print("\nLargest components:")

        for i, component in enumerate(
            sorted(components, key=len, reverse=True)[:10],
            start=1
        ):
            print(f"{i}. {len(component)} artists")


# ============================================================
# MAIN
# ============================================================

def main():

    # Load original network
    G = load_network()

    print("\n========== ORIGINAL NETWORK ==========")
    print_statistics(G)

    # Apply disparity filter
    alpha = 0.2
    G_backbone = disparity_filter(G, alpha=alpha)

    print(f"\n========== BACKBONE (alpha={alpha}) ==========")
    print_statistics(G_backbone)

    # Visualize original network
    visualize_network(G, OUTPUT_FILE)

    # Visualize filtered backbone
    visualize_network(G_backbone, OUTPUT_FILE_BACKBONE)


if __name__ == "__main__":
    main()



