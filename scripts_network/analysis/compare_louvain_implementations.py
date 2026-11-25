"""Compare NetworkX and python-louvain on the saved original network.

Run from the repository root:
    uv run python scripts_network/analysis/compare_louvain_implementations.py

The script is read-only with respect to the network inputs. It does not
overwrite louvain_communities.csv or any existing analysis outputs.
"""

from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score


PROJECT_DIR = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_DIR / "data" / "data_2005_2025_network" / "network"
OUTPUT_DIR = NETWORK_DIR / "louvain_implementation_comparison"

NODES_FILE = NETWORK_DIR / "nodes.csv"
EDGES_FILE = NETWORK_DIR / "edges.csv"
SAVED_COMMUNITIES_FILE = NETWORK_DIR / "louvain_communities.csv"

RESOLUTIONS = [0.5, 0.8, 1.0, 1.2, 1.5]
N_STABILITY_RUNS = 50
SEED_42 = 42
MAIN_RESOLUTION = 1.2
WEIGHT = "weight"


def load_graph() -> nx.Graph:
    """Load all original-network nodes, including isolates, and weighted edges."""
    nodes = pd.read_csv(NODES_FILE, dtype={"mbid": str})
    edges = pd.read_csv(
        EDGES_FILE,
        dtype={"source": str, "target": str},
    )

    required_nodes = {"mbid"}
    required_edges = {"source", "target", WEIGHT}
    if missing := required_nodes - set(nodes.columns):
        raise ValueError(f"nodes.csv is missing columns: {sorted(missing)}")
    if missing := required_edges - set(edges.columns):
        raise ValueError(f"edges.csv is missing columns: {sorted(missing)}")

    graph = nx.Graph()
    for mbid in nodes["mbid"].dropna():
        graph.add_node(str(mbid))

    for row in edges.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        weight = int(getattr(row, WEIGHT))
        graph.add_edge(source, target, **{WEIGHT: weight})

    if graph.number_of_nodes() != len(nodes):
        raise ValueError("nodes.csv contains missing or duplicate MBIDs.")

    return graph


def load_saved_partition(graph: nx.Graph) -> dict[str, str]:
    """Load and validate the community assignment currently used downstream."""
    saved = pd.read_csv(
        SAVED_COMMUNITIES_FILE,
        dtype={"mbid": str, "original_community": str},
    )
    required = {"mbid", "original_community"}
    if missing := required - set(saved.columns):
        raise ValueError(
            f"louvain_communities.csv is missing columns: {sorted(missing)}"
        )
    if saved["mbid"].duplicated().any():
        raise ValueError("louvain_communities.csv contains duplicate MBIDs.")

    partition = dict(zip(saved["mbid"], saved["original_community"]))
    if set(partition) != set(graph.nodes):
        raise ValueError(
            "The MBIDs in louvain_communities.csv do not match nodes.csv."
        )
    return partition


def run_networkx(
    graph: nx.Graph,
    resolution: float,
    seed: int,
) -> dict[str, int]:
    communities = nx.community.louvain_communities(
        graph,
        weight=WEIGHT,
        resolution=resolution,
        seed=seed,
    )
    return {
        node: community_id
        for community_id, community in enumerate(communities)
        for node in community
    }


def run_python_louvain(
    graph: nx.Graph,
    resolution: float,
    seed: int,
) -> dict[str, int]:
    return community_louvain.best_partition(
        graph,
        weight=WEIGHT,
        resolution=resolution,
        random_state=seed,
    )


def labels_in_graph_order(
    partition: dict[str, int] | dict[str, str],
    graph: nx.Graph,
) -> list[str | int]:
    if set(partition) != set(graph.nodes):
        missing = set(graph.nodes) - set(partition)
        extra = set(partition) - set(graph.nodes)
        raise ValueError(
            f"Partition does not cover graph nodes (missing={len(missing)}, "
            f"extra={len(extra)})."
        )
    return [partition[node] for node in graph.nodes]


def partition_modularity(
    graph: nx.Graph,
    partition: dict[str, int],
) -> float:
    """Compute standard resolution-1 modularity for comparable Q values."""
    communities_by_id: dict[int, set[str]] = {}
    for node, community_id in partition.items():
        communities_by_id.setdefault(community_id, set()).add(node)

    # Use the same standard modularity definition for both implementations
    # and for every tested partition, matching the existing sensitivity script.
    return nx.community.modularity(
        graph,
        list(communities_by_id.values()),
        weight=WEIGHT,
        resolution=1.0,
    )


def pairwise_nmi_rows(
    partitions: dict[float, dict[str, int]],
    graph: nx.Graph,
    implementation: str,
) -> list[dict[str, float | str]]:
    rows = []
    for resolution_a, resolution_b in combinations(RESOLUTIONS, 2):
        nmi = normalized_mutual_info_score(
            labels_in_graph_order(partitions[resolution_a], graph),
            labels_in_graph_order(partitions[resolution_b], graph),
        )
        rows.append({
            "implementation": implementation,
            "resolution_a": resolution_a,
            "resolution_b": resolution_b,
            "nmi": nmi,
        })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    saved_partition = load_saved_partition(graph)

    runners = {
        "NetworkX": run_networkx,
        "python-louvain": run_python_louvain,
    }
    resolution_partitions: dict[str, dict[float, dict[str, int]]] = {}
    resolution_rows = []
    resolution_nmi_rows = []
    implementation_rows = []
    seed_42_partitions = {}

    print(
        f"Loaded original network: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    # Reproduce the five-resolution sensitivity analysis with both libraries.
    for implementation, runner in runners.items():
        partitions = {}
        for resolution in RESOLUTIONS:
            partition = runner(graph, resolution, SEED_42)
            partitions[resolution] = partition
            resolution_rows.append({
                "implementation": implementation,
                "resolution": resolution,
                "communities": len(set(partition.values())),
                "standard_modularity": partition_modularity(graph, partition),
            })

            if resolution == MAIN_RESOLUTION:
                seed_42_partitions[implementation] = partition

        resolution_partitions[implementation] = partitions
        resolution_nmi_rows.extend(
            pairwise_nmi_rows(partitions, graph, implementation)
        )

    # Compare the two implementations at the same resolution and seed.
    for resolution in RESOLUTIONS:
        nx_partition = resolution_partitions["NetworkX"][resolution]
        pl_partition = resolution_partitions["python-louvain"][resolution]
        implementation_rows.append({
            "resolution": resolution,
            "seed": SEED_42,
            "networkx_communities": len(set(nx_partition.values())),
            "python_louvain_communities": len(set(pl_partition.values())),
            "nmi_between_implementations": normalized_mutual_info_score(
                labels_in_graph_order(nx_partition, graph),
                labels_in_graph_order(pl_partition, graph),
            ),
        })

    # Reproduce the 50-seed stability analysis at the selected resolution.
    stability_rows = []
    stability_pair_rows = []
    for implementation, runner in runners.items():
        partitions_by_seed = {
            seed: runner(graph, MAIN_RESOLUTION, seed)
            for seed in range(N_STABILITY_RUNS)
        }
        scores = []
        for seed_a, seed_b in combinations(range(N_STABILITY_RUNS), 2):
            nmi = normalized_mutual_info_score(
                labels_in_graph_order(partitions_by_seed[seed_a], graph),
                labels_in_graph_order(partitions_by_seed[seed_b], graph),
            )
            scores.append(nmi)
            stability_pair_rows.append({
                "implementation": implementation,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "nmi": nmi,
            })

        stability_series = pd.Series(scores)
        stability_rows.append({
            "implementation": implementation,
            "runs": N_STABILITY_RUNS,
            "pairwise_comparisons": len(scores),
            "mean_nmi": stability_series.mean(),
            "sd_nmi": stability_series.std(),
            "min_nmi": stability_series.min(),
            "max_nmi": stability_series.max(),
        })

    # Confirm whether each implementation's main-resolution, seed-42 partition
    # is the same partition as the one used by the lyric analysis.
    canonical_rows = []
    saved_labels = labels_in_graph_order(saved_partition, graph)
    for implementation, partition in seed_42_partitions.items():
        canonical_rows.append({
            "implementation": implementation,
            "resolution": MAIN_RESOLUTION,
            "seed": SEED_42,
            "communities": len(set(partition.values())),
            "nmi_vs_saved_louvain_communities": normalized_mutual_info_score(
                labels_in_graph_order(partition, graph),
                saved_labels,
            ),
        })

    outputs = {
        "resolution_summary.csv": pd.DataFrame(resolution_rows),
        "resolution_pairwise_nmi.csv": pd.DataFrame(resolution_nmi_rows),
        "implementation_comparison.csv": pd.DataFrame(implementation_rows),
        "stability_summary.csv": pd.DataFrame(stability_rows),
        "stability_pairwise_nmi.csv": pd.DataFrame(stability_pair_rows),
        "saved_partition_check.csv": pd.DataFrame(canonical_rows),
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(OUTPUT_DIR / filename, index=False)

    print("\nResolution summary:")
    print(outputs["resolution_summary.csv"].to_string(index=False))
    print("\nStability summary:")
    print(outputs["stability_summary.csv"].to_string(index=False))
    print("\nSame-seed implementation comparison:")
    print(outputs["implementation_comparison.csv"].to_string(index=False))
    print("\nComparison with the saved community assignment:")
    print(outputs["saved_partition_check.csv"].to_string(index=False))
    print(f"\nAll results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
