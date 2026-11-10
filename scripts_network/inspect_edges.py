import pandas as pd

edges = pd.read_csv("project/data/data_2005_2025/edges.csv")

print(edges["weight"].describe())

print("\nDistribution of edge weights:")
print(edges["weight"].value_counts().sort_index())