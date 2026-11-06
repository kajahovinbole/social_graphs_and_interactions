
import csv
import os
import math
import html
import json
import hashlib

import networkx as nx
from pyvis.network import Network

from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# Change these values when creating a new dataset
# ============================================================

# Dataset
DATASET_NAME = "data_2005_2025"

# Input / output
DATA_DIR = f"project/data/{DATASET_NAME}"

ARTISTS_FILE = os.path.join(DATA_DIR, "artists.csv")
EDGES_FILE = os.path.join(DATA_DIR, "edges.csv")

OUTPUT_FILE = os.path.join(
    DATA_DIR, "collaboration_network.html"
)

OUTPUT_FILE_BACKBONE = os.path.join(
    DATA_DIR, "collaboration_backbone.html"
)

METADATA_FILE = os.path.join(
    DATA_DIR, "visualization_metadata.json"
)

# Network filtering
MIN_EDGE_WEIGHT = 1

# Disparity filter
APPLY_BACKBONE = True
ALPHA = 0.2

# Layout
LAYOUT_SEED = 42
LAYOUT_ITERATIONS = 150
COMPONENT_SCALE = 150
COMPONENT_SPACING = 450

# Node appearance
NODE_SIZE_MIN = 10
NODE_SIZE_MAX = 35

# Edge appearance
EDGE_WIDTH_SCALE = 0.5

# Visualization
SHOW_NODE_LABELS = False
SHOW_ISOLATED_NODES = False

NETWORK_TITLE = "Artist Collaboration Network (2005–2025)"

# Physics disabled to keep node positions fixed
PHYSICS_ENABLED = False


# ============================================================
# LOAD DATA
# ============================================================

def load_network():
    """Load artists and collaboration edges into a NetworkX graph."""

    G = nx.Graph()

    # Load all selected artists, including isolated nodes
    with open(ARTISTS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in sorted(reader, key=lambda x: x["mbid"]):
            G.add_node(
                row["mbid"],
                name=row["name"]
            )

    # Load collaboration edges
    with open(EDGES_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in sorted(
            reader,
            key=lambda x: (x["source"], x["target"])
        ):

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
# DISPARITY FILTER
# ============================================================

def disparity_filter(G, alpha=ALPHA):
    """
    Extract network backbone using the disparity filter.

    An edge is retained if it is statistically significant
    for at least one of its endpoints.
    """

    backbone = nx.Graph()

    # Preserve all original nodes and their attributes
    backbone.add_nodes_from(G.nodes(data=True))

    # Calculate degree and strength
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    for u, v, data in sorted(
        G.edges(data=True),
        key=lambda edge: (edge[0], edge[1])
    ):

        weight = data["weight"]

        # Calculate normalized edge weight for each endpoint
        p_u = weight / strengths[u]
        p_v = weight / strengths[v]

        k_u = degrees[u]
        k_v = degrees[v]

        # Significance under the disparity filter null model
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
    """Calculate node sizes based on weighted degree."""

    weighted_degrees = dict(G.degree(weight="weight"))

    max_degree = max(weighted_degrees.values(), default=1)

    sizes = {}

    for node, degree in weighted_degrees.items():

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
    """Calculate deterministic positions for each component."""

    positions = {}

    if not components:
        return positions

    # Arrange components in a grid
    cols = math.ceil(math.sqrt(len(components)))

    for i, component in enumerate(components):

        # Sort nodes to ensure consistent input order
        component = component.subgraph(
            sorted(component.nodes())
        ).copy()

        # Calculate layout separately for each component
        component_pos = nx.spring_layout(
            component,
            seed=LAYOUT_SEED,
            weight="weight",
            iterations=LAYOUT_ITERATIONS,
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

def visualize_network(G, output_file, title=NETWORK_TITLE):
    """Create an interactive HTML visualization."""

    print("\nCalculating connected components...")

    if SHOW_ISOLATED_NODES:
        components = [
            G.subgraph(c).copy()
            for c in nx.connected_components(G)
        ]
    else:
        components = [
            G.subgraph(c).copy()
            for c in nx.connected_components(G)
            if len(c) > 1
        ]

    # Largest components first, with deterministic tie-breaking
    components.sort(
        key=lambda c: (-len(c), min(c.nodes()))
    )

    print(f"Connected components: {len(components)}")

    if not components:
        print("No collaborations found.")
        return

    # Calculate positions
    positions = calculate_positions(components)

    # Only include nodes belonging to selected components
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

    for node in sorted(G_connected.nodes()):

        name = G_connected.nodes[node]["name"]

        x, y = positions[node]

        # Show name on hover
        # Use a space as label to avoid displaying UUIDs
        label = name if SHOW_NODE_LABELS else " "

        net.add_node(
            node,
            label=label,
            title=(
                f"<b>{html.escape(name)}</b><br>"
                f"Weighted degree: {weighted_degrees[node]}<br>"
                f"Collaborations: {G_connected.degree(node)}"
            ),
            size=node_sizes[node],
            x=x,
            y=y,
            physics=PHYSICS_ENABLED,
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

    for u, v, data in sorted(
        G_connected.edges(data=True),
        key=lambda edge: (edge[0], edge[1])
    ):

        weight = data["weight"]

        net.add_edge(
            u,
            v,
            width=EDGE_WIDTH_SCALE + math.sqrt(weight),
            title=f"{weight} collaborative recordings",
            color="rgba(100, 100, 100, 0.45)"
        )

    # --------------------------------------------------------
    # INTERACTION OPTIONS
    # --------------------------------------------------------

    options = {
        "physics": {
            "enabled": PHYSICS_ENABLED
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "dragView": True,
            "dragNodes": True,
            "multiselect": True,
            "selectConnectedEdges": True
        },
        "nodes": {
            "shape": "dot",
            "borderWidth": 1,
            "font": {
                "size": 14,
                "face": "Arial",
                "strokeWidth": 3,
                "strokeColor": "#ffffff"
            }
        },
        "edges": {
            "smooth": False,
            "selectionWidth": 2,
            "color": {
                "inherit": False,
                "color": "#999999",
                "highlight": "#333333"
            }
        }
    }

    net.set_options(json.dumps(options))

    # Save HTML
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    net.write_html(
        output_file,
        open_browser=False
    )

    print(f"\n{title}")
    print(f"Interactive network saved to: {output_file}")


# ============================================================
# NETWORK STATISTICS
# ============================================================

def get_network_statistics(G):

    isolated = list(nx.isolates(G))

    components = [
        c for c in nx.connected_components(G)
        if len(c) > 1
    ]

    largest_component = (
        max(map(len, components))
        if components else 0
    )

    return {
        "number_of_artists": G.number_of_nodes(),
        "number_of_edges": G.number_of_edges(),
        "isolated_artists": len(isolated),
        "connected_components": len(components),
        "largest_component": largest_component
    }


def print_statistics(G):

    stats = get_network_statistics(G)

    print("\n========== NETWORK STATISTICS ==========")

    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

    components = [
        c for c in nx.connected_components(G)
        if len(c) > 1
    ]

    if components:

        print("\nLargest components:")

        largest = sorted(
            components,
            key=lambda c: (-len(c), min(c))
        )[:10]

        for i, component in enumerate(largest, start=1):
            print(f"{i}. {len(component)} artists")


# ============================================================
# SAVE METADATA
# ============================================================

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_metadata(G, G_backbone=None):
    """Save visualization configuration and network statistics."""

    metadata = {
        "dataset": DATASET_NAME,

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "input_files": {
            "artists_file": ARTISTS_FILE,
            "artists_sha256": calculate_file_hash(ARTISTS_FILE),
            "edges_file": EDGES_FILE,
            "edges_sha256": calculate_file_hash(EDGES_FILE)
        },

        "configuration": {
            "min_edge_weight": MIN_EDGE_WEIGHT,
            "apply_backbone": APPLY_BACKBONE,
            "alpha": ALPHA,
            "layout_seed": LAYOUT_SEED,
            "layout_iterations": LAYOUT_ITERATIONS,
            "component_scale": COMPONENT_SCALE,
            "component_spacing": COMPONENT_SPACING,
            "node_size_min": NODE_SIZE_MIN,
            "node_size_max": NODE_SIZE_MAX,
            "edge_width_scale": EDGE_WIDTH_SCALE,
            "show_node_labels": SHOW_NODE_LABELS,
            "show_isolated_nodes": SHOW_ISOLATED_NODES,
            "physics_enabled": PHYSICS_ENABLED
        },

        "original_network": get_network_statistics(G),

        "backbone": (
            get_network_statistics(G_backbone)
            if G_backbone is not None else None
        )
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nSaved metadata to: {METADATA_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load original network
    G = load_network()

    print("\n========== ORIGINAL NETWORK ==========")
    print_statistics(G)

    # Apply disparity filter
    G_backbone = None

    if APPLY_BACKBONE:

        G_backbone = disparity_filter(
            G,
            alpha=ALPHA
        )

        print(
            f"\n========== BACKBONE (alpha={ALPHA}) =========="
        )

        print_statistics(G_backbone)

    # Visualize original network
    visualize_network(
        G,
        OUTPUT_FILE,
        title=NETWORK_TITLE
    )

    # Visualize filtered backbone
    if G_backbone is not None:

        visualize_network(
            G_backbone,
            OUTPUT_FILE_BACKBONE,
            title=f"{NETWORK_TITLE} - Backbone"
        )

    # Save metadata
    save_metadata(G, G_backbone)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
