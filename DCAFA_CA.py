"""Community abundance analysis using the clusters saved by clustering.py."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import _dcafa_backend as dcafa
from helper_ftns import show_cluster_summary

# Settings
root = Path(__file__).resolve().parent
input_csv = root / "results" / "features_cluster.csv"
out_dir = root / "results" / "dcafa"
bag_col = "cell_line"
min_cluster_cells = 10
features = ["area", "elongation", "circularity", "solidity"]
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Load saved clusters
# -----------------------------
# Cell-line exclusions and QC filtering were already applied by clustering.py.
df = pd.read_csv(input_csv)
df = df.dropna(subset=["cluster", bag_col]).copy()
labels = pd.to_numeric(df["cluster"], errors="coerce")
if (labels.isna() | labels.lt(0) | labels.mod(1).ne(0)).any():
    raise ValueError("Saved cluster labels must be nonnegative integers")
df["cluster"] = labels.astype(int)
df["subtype"] = df["subtype"].astype(str).str.strip().str.lower()
# Accept the alternative spelling while using one consistent label.
df["subtype"] = df["subtype"].replace("epitheliod", "epithelioid")
df = df.loc[df["subtype"].isin(["epithelioid", "biphasic"])].copy()
df["epithelioid"] = df["subtype"].eq("epithelioid").astype(int)

# -----------------------------
# 2. Show summary and remove small clusters
# -----------------------------
show_cluster_summary(df, features, is_save=False)
cluster_counts = df["cluster"].value_counts()
small_clusters = cluster_counts[cluster_counts < min_cluster_cells].index
if len(small_clusters):
    print(f"Excluding small clusters: {list(small_clusters)}")
df = df.loc[~df["cluster"].isin(small_clusters)].copy()
cluster_ids = sorted(df["cluster"].unique())
n_clusters = len(cluster_ids)
if n_clusters < 2:
    raise ValueError("At least two clusters must remain for analysis")

# -----------------------------
# 3. Count clusters in each cell line (bag)
# -----------------------------
groups = df.groupby(bag_col)
if groups["subtype"].nunique().gt(1).any():
    raise ValueError("Each bag must have a single subtype")

bags = pd.crosstab(df[bag_col], df["cluster"]).reindex(columns=cluster_ids, fill_value=0)
# DCAFA requires c_1 ... c_K; original cluster IDs remain unchanged in df.
columns = [f"c_{k}" for k in range(1, n_clusters + 1)]
bags.columns = columns
bags = bags.join(groups.size().rename("n"))
bags = bags.join(groups[["subtype", "epithelioid"]].first()).reset_index()
bags = bags.sort_values(["epithelioid", bag_col], ascending=[False, True]).reset_index(drop=True)
bag_counts = bags.groupby("subtype").size()
if len(bag_counts) != 2 or bag_counts.min() < 2:
    raise ValueError("At least two bags per subtype are required")
if bag_counts.min() < 10:
    warnings.warn(f"Few bags per subtype ({bag_counts.to_dict()}); results are exploratory")

bags.to_csv(out_dir / "bags.csv", index=False)
source_labels = {f"k={k}": label for k, label in enumerate(cluster_ids, start=1)}

# -----------------------------
# 4. Fit community abundance models
# -----------------------------
results = dcafa.fit_ca_bag(
    df=bags, K=n_clusters, target_cols=["epithelioid"],
    covariates=[], offset_col="n", cov_type="HC1",
    plot=False, outdir=str(out_dir),
)
effects = results["epithelioid"]
effects["source_cluster"] = effects["community"].map(source_labels)

# p is the raw Wald p-value; q is adjusted across the retained clusters.
significance = effects[[
    "source_cluster", "exp_coef", "exp_ci_lo", "exp_ci_hi",
    "p", "q", "significant",
]].rename(columns={"source_cluster": "cluster"})
significance.to_csv(out_dir / "community_abundance_significance.csv", index=False)
print(significance.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

# -----------------------------
# 5. Plot abundance ratios with p-values
# -----------------------------
plot_effects = effects.copy()
plot_effects["community"] = plot_effects["source_cluster"].map(lambda label: f"Cluster {label}")
dcafa.plot_ca_bag_forest(
    effects=plot_effects, term="epithelioid",
    title="Cluster abundance: epithelioid versus biphasic",
    x_label="Abundance ratio (epithelioid / biphasic)",
    outpath=str(out_dir / "abundance_forest.png"), show_p_values=True,
)

# -----------------------------
# 6. Plot cluster percentages (epithelioid first)
# -----------------------------
percentages = bags[columns].div(bags["n"], axis=0) * 100
fig, ax = plt.subplots(figsize=(max(7, n_clusters * 1.4), max(4, len(bags) * 0.55)))
image = ax.imshow(percentages, aspect="auto", cmap="Blues", vmin=0, vmax=100)
fig.colorbar(image, ax=ax, label="Cells in cluster (%)")
ax.set_xticks(range(n_clusters), [f"Cluster {k}" for k in cluster_ids])
ax.set_yticks(range(len(bags)), [
    f"{row[bag_col]} ({row['subtype']}, n={row['n']})" for _, row in bags.iterrows()
])
for i in range(len(bags)):
    for j in range(n_clusters):
        value = percentages.iloc[i, j]
        ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                color="white" if value > 50 else "black")
ax.set_title("Morphology cluster distribution by sample")
fig.tight_layout()
fig.savefig(out_dir / "cluster_distribution.png", dpi=180, bbox_inches="tight")

# -----------------------------
# 7. Stacked area plot of mean cluster percentages by subtype
# -----------------------------
# Average percentages across cell lines, giving each cell line equal weight.
# Subtypes are categories; the connecting areas do not represent a continuous trend.
subtype_order = ["epithelioid", "biphasic"]
mean_percentages = percentages.groupby(bags["subtype"]).mean().reindex(subtype_order)
fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(
    [0, 1], mean_percentages.to_numpy().T,
    labels=[f"Cluster {cluster}" for cluster in cluster_ids],
    colors=[f"C{int(cluster) % 10}" for cluster in cluster_ids], alpha=0.85,
)
ax.set_xticks([0, 1], ["Epithelioid", "Biphasic"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.set_ylabel("Mean cells in cluster (%)")
ax.set_title("Cluster composition by subtype\nEqual weight per cell line")
ax.legend(title="Saved cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(out_dir / "ca_stacked_area.png", dpi=180, bbox_inches="tight")

print(f"Analyzed {len(df)} cells in {len(bags)} bags; results saved to {out_dir}")
plt.show()