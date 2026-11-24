
from pathlib import Path

import networkx as nx
import pandas as pd
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "data_2005_2025_network"
)

OUTPUT_DIR = NETWORK_DIR / "resolution_sensitivity_nmi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

RESOLUTIONS = [0.5, 0.8, 1.0, 1.2, 1.5]
RANDOM_SEED = 42
WEIGHT = "weight"


# --------------------------------------------------
# Load network
# --------------------------------------------------

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
# Run Louvain for each resolution
# --------------------------------------------------

partitions = {}
summary = []

for resolution in RESOLUTIONS:

    partition = community_louvain.best_partition(
        G,
        weight=WEIGHT,
        resolution=resolution,
        random_state=RANDOM_SEED
    )

    partitions[resolution] = partition

    # Save community assignments
    results = pd.DataFrame({
        "Id": list(partition.keys()),
        "modularity_class": list(partition.values())
    })

    results.to_csv(
        OUTPUT_DIR / f"communities_resolution_{resolution}.csv",
        index=False
    )

    num_communities = len(set(partition.values()))

    modularity = community_louvain.modularity(
        partition,
        G,
        weight=WEIGHT
    )

    summary.append({
        "Resolution": resolution,
        "Communities": num_communities,
        "Modularity": modularity
    })


# --------------------------------------------------
# Calculate pairwise NMI
# --------------------------------------------------

nmi_results = []

for i, res1 in enumerate(RESOLUTIONS):

    for res2 in RESOLUTIONS[i + 1:]:

        labels1 = [
            partitions[res1][node] for node in G.nodes()
        ]

        labels2 = [
            partitions[res2][node] for node in G.nodes()
        ]

        nmi = normalized_mutual_info_score(
            labels1,
            labels2
        )

        nmi_results.append({
            "Resolution 1": res1,
            "Resolution 2": res2,
            "NMI": nmi
        })


# --------------------------------------------------
# Save and print results
# --------------------------------------------------

summary_df = pd.DataFrame(summary)
nmi_df = pd.DataFrame(nmi_results)

summary_df.to_csv(
    OUTPUT_DIR / "resolution_summary.csv",
    index=False
)

nmi_df.to_csv(
    OUTPUT_DIR / "resolution_nmi.csv",
    index=False
)

print("\nCommunity summary:")
print(summary_df.to_string(index=False))

print("\nPairwise NMI:")
print(nmi_df.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_DIR}")
