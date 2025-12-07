import polars as pl
import matplotlib.pyplot as plt

import sys

sizes = pl.read_csv(sys.argv[1], has_header=False, new_columns=["size"])

sizes = sizes.filter(pl.col.size >= pl.col.size.quantile(0.01))
sizes = sizes.filter(pl.col.size <= pl.col.size.quantile(0.99))

plt.hist(
    sizes["size"],
    bins=100,
    edgecolor="black",
)
plt.xlabel("Chunk size (bytes)")
plt.ylabel("Frequency")
plt.title(sys.argv[2])
plt.savefig("plot.png", format="png", dpi=1200)
plt.show()