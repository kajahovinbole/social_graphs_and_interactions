
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "data_2005_2025_network"
    / "network"
)

BACKBONE_FILE = (
    NETWORK_DIR
    / "backbone"
    / "edges_backbone_alpha_0.42.csv"
)

OUTPUT_DIR = NETWORK_DIR / "network_properties"


# --------------------------------------------------
def main():
    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    nodes = pd.read_csv(NETWORK_DIR / "nodes.csv")
    edges = pd.read_csv(NETWORK_DIR / "edges.csv")
    backbone_edges = pd.read_csv(BACKBONE_FILE)

    print("Loaded files:")
    print(f"Nodes: {len(nodes)}")
    print(f"Original edges: {len(edges)}")
    print(f"Backbone edges: {len(backbone_edges)}")


    # --------------------------------------------------
    # BUILD NETWORKS
    # --------------------------------------------------

    def build_graph(nodes_df, edges_df):

        G = nx.Graph()

        for _, row in nodes_df.iterrows():
            G.add_node(row["mbid"])

        for _, row in edges_df.iterrows():
            G.add_edge(
                row["source"],
                row["target"],
                weight=row["weight"]
            )

        return G


    G_original = build_graph(nodes, edges)
    G_backbone = build_graph(nodes, backbone_edges)


    # --------------------------------------------------
    # CALCULATE NETWORK PROPERTIES
    # --------------------------------------------------

    def calculate_properties(G, name):

        components = list(nx.connected_components(G))
        component_sizes = sorted(
            [len(c) for c in components],
            reverse=True
        )

        degrees = dict(G.degree())
        strengths = dict(G.degree(weight="weight"))

        edge_weights = [
            data["weight"]
            for _, _, data in G.edges(data=True)
        ]

        largest_component = G.subgraph(
            max(components, key=len)
        )

        stats = {
            "Network": name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Isolates": nx.number_of_isolates(G),
            "Connected components": len(components),
            "Largest component": max(component_sizes),
            "Average degree": sum(degrees.values()) / G.number_of_nodes(),
            "Average strength": sum(strengths.values()) / G.number_of_nodes(),
            "Maximum degree": max(degrees.values()),
            "Maximum strength": max(strengths.values()),
            "Density": nx.density(G),
            "Average edge weight": (
                sum(edge_weights) / len(edge_weights)
                if edge_weights else 0
            ),
            "Median edge weight": (
                pd.Series(edge_weights).median()
                if edge_weights else 0
            ),
            "Maximum edge weight": max(edge_weights) if edge_weights else 0,
            "Total edge weight": sum(edge_weights),
            "Largest component diameter": (
                nx.diameter(largest_component)
                if len(largest_component) > 1 else 0
            )
        }

        return stats, degrees, strengths, edge_weights, component_sizes


    original_stats, degrees, strengths, weights, component_sizes = (
        calculate_properties(G_original, "Original")
    )

    backbone_stats, backbone_degrees, backbone_strengths, backbone_weights, backbone_components = (
        calculate_properties(G_backbone, "Backbone")
    )



    def plot_log_log_distribution(values, xlabel, title, filename):
        """
        Plot the raw discrete distribution using log-log axes.
        Zero values are excluded because log(0) is undefined.
        """

        counts = pd.Series(values).value_counts().sort_index()

        # Only positive values can be plotted on logarithmic axes
        counts = counts[counts.index > 0]

        plt.figure(figsize=(8, 5))

        plt.scatter(
            counts.index,
            counts.values,
            alpha=0.8
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)

        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()

        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()


    def plot_log_binned_distribution(values, xlabel, title, filename):
        """
        Plot a logarithmically binned discrete distribution.

        Bins become wider as k increases.
        The y-value is the average frequency per integer value
        within each bin, normalized by the number of observations.
        """

        values = np.asarray(values)
        values = values[values > 0]

        if len(values) == 0:
            return

        max_value = int(values.max())

        # Create geometrically increasing integer bin boundaries
        bin_edges = [1]
        boundary = 2

        while boundary <= max_value:
            bin_edges.append(boundary)
            boundary *= 2

        bin_edges.append(max_value + 1)
        bin_edges = sorted(set(bin_edges))

        probabilities = []
        bin_centers = []

        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):

            # Count observations in [lower, upper)
            bin_count = np.sum(
                (values >= lower) & (values < upper)
            )

            # Number of possible integer values in this bin
            bin_width = upper - lower

            # Average probability per integer value
            probability = bin_count / (len(values) * bin_width)

            if probability > 0:
                probabilities.append(probability)

                # Geometric midpoint of the bin
                bin_centers.append(np.sqrt(lower * upper))

        plt.figure(figsize=(8, 5))

        plt.scatter(
            bin_centers,
            probabilities,
            alpha=0.8
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(xlabel)
        plt.ylabel("Average probability per degree value")
        plt.title(title)

        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()

        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()

    # --------------------------------------------------
    # SAVE NETWORK STATISTICS
    # --------------------------------------------------

    stats_df = pd.DataFrame([
        original_stats,
        backbone_stats
    ])

    stats_df.to_csv(
        OUTPUT_DIR / "network_statistics.csv",
        index=False
    )

    print("\nNETWORK STATISTICS")
    print(stats_df.T.to_string())


    # --------------------------------------------------
    # SAVE NODE-LEVEL METRICS
    # --------------------------------------------------

    node_metrics = pd.DataFrame({
        "mbid": list(G_original.nodes()),
        "degree": [degrees[n] for n in G_original.nodes()],
        "weighted_degree": [strengths[n] for n in G_original.nodes()]
    })

    node_metrics.to_csv(
        OUTPUT_DIR / "node_metrics.csv",
        index=False
    )


    # --------------------------------------------------
    # PLOT DEGREE DISTRIBUTION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    degree_values = list(degrees.values())

    plt.hist(
        degree_values,
        bins=range(0, max(degree_values) + 2),
        align="left",
        edgecolor="black"
    )

    plt.xlabel("Degree")
    plt.ylabel("Number of artists")
    plt.title("Degree Distribution")
    plt.xticks(range(0, max(degree_values) + 1))
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "degree_distribution.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # PLOT WEIGHTED DEGREE DISTRIBUTION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        list(strengths.values()),
        bins=20,
        edgecolor="black"
    )

    plt.xlabel("Weighted degree (strength)")
    plt.ylabel("Number of artists")
    plt.title("Weighted Degree Distribution")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "weighted_degree_distribution.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # PLOT EDGE WEIGHT DISTRIBUTION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        weights,
        bins=range(1, int(max(weights)) + 2),
        align="left",
        edgecolor="black"
    )

    plt.xlabel("Edge weight")
    plt.ylabel("Number of artist pairs")
    plt.title("Edge Weight Distribution")
    plt.xticks(range(1, int(max(weights)) + 1))
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "edge_weight_distribution.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------
    # PLOT COMPONENT SIZE DISTRIBUTION
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        component_sizes,
        bins=range(1, max(component_sizes) + 2),
        edgecolor="black"
    )

    plt.xlabel("Component size (number of artists)")
    plt.ylabel("Number of components")
    plt.title("Connected Component Size Distribution")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "component_size_distribution.png",
        dpi=300
    )

    plt.close()




    # --------------------------------------------------
    # LOG-LOG DISTRIBUTIONS
    # --------------------------------------------------

    networks = {
        "original": {
            "degrees": list(degrees.values()),
            "strengths": list(strengths.values()),
            "weights": weights
        },
        "backbone": {
            "degrees": list(backbone_degrees.values()),
            "strengths": list(backbone_strengths.values()),
            "weights": backbone_weights
        }
    }

    for network_name, data in networks.items():

        # Raw log-log distributions
        plot_log_log_distribution(
            data["degrees"],
            "Degree",
            f"Degree Distribution ({network_name.capitalize()}, Log-Log)",
            f"degree_{network_name}_loglog.png"
        )

        plot_log_log_distribution(
            data["strengths"],
            "Weighted degree",
            f"Weighted Degree Distribution ({network_name.capitalize()}, Log-Log)",
            f"weighted_degree_{network_name}_loglog.png"
        )

        plot_log_log_distribution(
            data["weights"],
            "Edge weight",
            f"Edge Weight Distribution ({network_name.capitalize()}, Log-Log)",
            f"edge_weight_{network_name}_loglog.png"
        )

        # Logarithmically binned distributions
        plot_log_binned_distribution(
            data["degrees"],
            "Degree",
            f"Degree Distribution ({network_name.capitalize()}, Log-Binned)",
            f"degree_{network_name}_log_binned.png"
        )

        plot_log_binned_distribution(
            data["strengths"],
            "Weighted degree",
            f"Weighted Degree Distribution ({network_name.capitalize()}, Log-Binned)",
            f"weighted_degree_{network_name}_log_binned.png"
        )

        plot_log_binned_distribution(
            data["weights"],
            "Edge weight",
            f"Edge Weight Distribution ({network_name.capitalize()}, Log-Binned)",
            f"edge_weight_{network_name}_log_binned.png"
        )


    # --------------------------------------------------
    # FINISHED
    # --------------------------------------------------

    print("\nAnalysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
