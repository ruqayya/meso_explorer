from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap.umap_ as umap

import pandas as pd
from helper_ftns import show_extreme_cells, plot_cluster_feature_heatmap, show_cluster_summary, get_cluster_distribution, plot_PCA, plot_UMAP

# Cell lines to exclude
exclude_lines = [
    "A549", "H266",
    "Mesothelioma Patient_1-020-",
    "Mesothelioma Patient_1-021-",
    "Mesothelioma Patient_1-022-",
    "Mesothelioma Patient_1-024-"
]
features = ["area", "elongation", "circularity", "solidity"]

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("./results/features.csv")
# Preserve every original row and column for the annotated CSV.
original_features = df.copy()
df["elongation"] = df["major_axis_length"] / df["minor_axis_length"]

# Match literal substrings, ignoring case (e.g. H266 also excludes H266_1).
cell_line_names = df["cell_line"].fillna("").astype(str)
excluded = pd.Series(False, index=df.index)
for text in exclude_lines:
    if text:
        excluded |= cell_line_names.str.contains(text, case=False, regex=False)
df = df.loc[~excluded].copy()
# Apply mask quality control before scaling and clustering.
# The saved features_cluster.csv therefore contains only QC-passing cells.
if "qc_ok" not in df:
    raise ValueError("features.csv must contain qc_ok for quality filtering")
qc_pass = df["qc_ok"].astype(str).str.strip().str.lower().eq("true")
print(f"Excluded {(~qc_pass).sum()} QC-flagged cells before clustering")
df = df.loc[qc_pass].copy()

print("Remaining cell lines:")
print(df["cell_line"].unique())

# -----------------------------
# 2. Select morphology features
# -----------------------------
X = df[features].copy()

# Remove cells with missing values
valid = X.notna().all(axis=1)
df = df.loc[valid].copy()
X = X.loc[valid]

# -----------------------------
# 3. Standardize
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3.1 Show extreme cells
# -----------------------------
# selected = show_extreme_cells(
#     df,
#     group="epithelioid",
#     feature1="solidity",
#     direction1="low",
#     feature2="solidity",
#     direction2="low",
#     percentile=0.01,
#     top_n=16
# )

# -----------------------------
# 4. K-means clustering
# -----------------------------
# Start with 3 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)
show_cluster_summary(df, features, is_save = True)
df.to_csv("./results/features_cluster.csv", index=False)
print("Saved features with cluster labels to ./results/features_cluster.csv")

# -----------------------------
# 5 PCA for visualization
# -----------------------------
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# df["PC1"] = X_pca[:, 0]
# df["PC2"] = X_pca[:, 1]
# plot_PCA(df)

# -----------------------------
# 6 UMAP for visualization
# -----------------------------
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]
plot_UMAP(df, group_by="subtype", group_style="marker", cluster_style="color", show_cluster=True)    # cell_line or subtype

