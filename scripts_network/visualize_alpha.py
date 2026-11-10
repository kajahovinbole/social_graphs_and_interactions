import matplotlib.pyplot as plt

# Alpha distribution
intervals = [
    "0.0–0.1",
    "0.1–0.2",
    "0.2–0.3",
    "0.3–0.4",
    "0.4–0.5",
    "0.5–0.6",
    "0.6–1.0"
]

counts = [80, 41, 32, 29, 175, 61, 25]

plt.figure(figsize=(9, 5))

bars = plt.bar(intervals, counts, width=0.65)

plt.xlabel("Minimum alpha value", fontsize=11)
plt.ylabel("Number of edges", fontsize=11)
plt.title("Distribution of edge alpha values", fontsize=13)

plt.ylim(0, 200)
plt.grid(axis="y", alpha=0.25)
plt.gca().set_axisbelow(True)

# Add values above bars
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count + 3,
        str(count),
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.savefig("alpha_distribution.png", dpi=300, bbox_inches="tight")
plt.show()