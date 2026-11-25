
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "data_2005_2025_network"
    / "network"
)

OUTPUT_DIR = NETWORK_DIR / "stability_runs_nmi"


# --------------------------------------------------
# Parameters
# --------------------------------------------------

N_RUNS = 50
RESOLUTION = 1.2
WEIGHT = "weight"


# --------------------------------------------------
def main():
    # --------------------------------------------------
    # Load network
    # --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    nodes = pd.read_csv(NETWORK_DIR / "nodes.csv")
    edges = pd.read_csv(NETWORK_DIR / "edges.csv")

    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(row["mbid"])

    for _, row in edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=row["weight"]
        )

    print(f"Loaded network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")


    # --------------------------------------------------
    # Run Louvain with different random seeds
    # --------------------------------------------------

    partitions = []

    for seed in range(N_RUNS):

        communities = nx.community.louvain_communities(
            G,
            weight=WEIGHT,
            resolution=RESOLUTION,
            seed=seed,
        )

        partition = {
            node: community_id
            for community_id, community in enumerate(communities)
            for node in community
        }

        labels = [
            partition[node]
            for node in G.nodes()
        ]

        partitions.append(labels)

        # Save community assignments
        results = pd.DataFrame({
            "Id": list(partition.keys()),
            "modularity_class": list(partition.values())
        })

        results.to_csv(
            OUTPUT_DIR / f"louvain_seed_{seed}.csv",
            index=False
        )


    # --------------------------------------------------
    # Calculate pairwise NMI
    # --------------------------------------------------

    nmi_scores = []

    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):

            nmi = normalized_mutual_info_score(
                partitions[i],
                partitions[j]
            )

            nmi_scores.append({
                "Run 1": i,
                "Run 2": j,
                "NMI": nmi
            })


    nmi_df = pd.DataFrame(nmi_scores)

    # Save pairwise results
    nmi_df.to_csv(
        OUTPUT_DIR / "nmi_results.csv",
        index=False
    )


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nLouvain stability results")
    print(f"Number of runs: {N_RUNS}")
    print(f"Number of comparisons: {len(nmi_df)}")
    print(f"Resolution: {RESOLUTION}")

    print(f"\nMean NMI: {nmi_df['NMI'].mean():.4f}")
    print(f"Std NMI:  {nmi_df['NMI'].std():.4f}")
    print(f"Min NMI:  {nmi_df['NMI'].min():.4f}")
    print(f"Max NMI:  {nmi_df['NMI'].max():.4f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
