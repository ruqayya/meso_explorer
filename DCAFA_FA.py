"""Feature association analysis using the cells saved by clustering.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import _dcafa_backend as dcafa

# Settings
root = Path(__file__).resolve().parent
input_csv = root / "results" / "features_cluster.csv"
out_dir = root / "results" / "dcafa"
bag_col = "cell_line"
features = ["area", "elongation", "circularity", "solidity"]
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Load cells and subtype labels
# -----------------------------
# Cell-line exclusions and QC filtering were already applied by clustering.py.
df = pd.read_csv(input_csv)
df["subtype"] = df["subtype"].astype(str).str.strip().str.lower()
df["subtype"] = df["subtype"].replace("epitheliod", "epithelioid")
df = df.loc[df["subtype"].isin(["epithelioid", "biphasic"])].copy()
df["epithelioid"] = df["subtype"].eq("epithelioid").astype(int)

# -----------------------------
# 2. Prepare and standardise features
# -----------------------------
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [bag_col]).copy()
if df["epithelioid"].nunique() != 2:
    raise ValueError("Both epithelioid and biphasic cells are required")
if df[features].nunique().lt(2).any():
    raise ValueError("Each feature must have at least two different values")

# Effects represent a one-standard-deviation increase in each feature.
df[features] = StandardScaler().fit_transform(df[features])

# -----------------------------
# 3. Fit one logistic regression per feature
# -----------------------------
# Standard errors are clustered by cell line, accounting for related cells.
results = dcafa.fit_inst_fa(
    df=df, feature_cols=features, target_cols=["epithelioid"],
    covariates=[], bag_id_col=bag_col,
    families={"epithelioid": "binomial"}, cov_type="cluster",
)
effects = results["epithelioid"]
if effects.empty:
    raise ValueError("No feature models fitted successfully")
effects.to_csv(out_dir / "feature_effects.csv", index=False)

# p is the raw Wald p-value; q is adjusted across the tested features.
print(effects[[
    "term", "effect_plot", "effect_lo", "effect_hi", "p", "q",
]].to_string(index=False, float_format=lambda value: f"{value:.6g}"))

# -----------------------------
# 4. Plot feature odds ratios with p-values
# -----------------------------
dcafa.plot_fa_inst_forest(
    effects_df=effects,
    title="Feature associations with epithelioid subtype",
    only_features=True, show_p_values=True,
    outpath=out_dir / "feature_analysis_forest.png",
)

# -----------------------------
# 5. Plot feature association heatmap
# -----------------------------
# Annotations show raw p-values; stars indicate adjusted q-value thresholds.
dcafa.plot_fa_inst_heatmap(
    results, outpath=out_dir / "feature_analysis_heatmap.png", show_p_values=True,
)
print(f"Analyzed {len(df)} cells; results saved to {out_dir}")
plt.show()