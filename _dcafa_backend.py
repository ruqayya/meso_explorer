# Upstream: https://github.com/wgrgwrght/DCAFA/blob/main/DCAFA.py
# Retrieved 2026-10-07; see DCAFA_LICENSE.txt.
from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.stats.multitest import multipletests

# Utility function for aggregating instance-level features into bag-level features
def aggregate_bag_features(
    df_instances: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,        # x_*
    membership_cols: Optional[List[str]] = None,     # m_1..m_K (optional)
    bag_id_col: str = "bag_id",
    min_count_for_mean: int = 1
) -> pd.DataFrame:
    """
    Create a bag-level DataFrame with:
        - n                 : instance count per bag
        - xg_*              : GLOBAL mean features
        - (if memberships present) xk{K}_{j} : COMMUNITY-LOCAL mean features per community
          where xk1_1 = mean of x_1 among instances with m_1>0 (weight = m_1 if soft)
        - c_k               : community "counts" (sum of m_k) if memberships present
    Notes:
        - For soft memberships, community means are weighted by m_{pk}.
        - If a bag has c_k < min_count_for_mean, we set mean to NaN (can later impute or keep NaN).
    """
    if feature_cols is None:
        feature_cols = [c for c in df_instances.columns if c.startswith("x_")]
        feature_cols = sorted(feature_cols, key=lambda s: int(s.split("_")[1]))
    has_memberships = membership_cols is not None and len(membership_cols) > 0

    # Basic groupby
    g = df_instances.groupby(bag_id_col, observed=True)
    out = g.size().rename("n").to_frame().reset_index()

    # GLOBAL means
    global_means = g[feature_cols].mean().add_prefix("xg_").reset_index()
    out = out.merge(global_means, on=bag_id_col, how="left")

    # COMMUNITY means (if available)
    if has_memberships:
        # compute community "counts" per bag (sum of m_k)
        cks = g[membership_cols].sum().reset_index()
        out = out.merge(cks, on=bag_id_col, how="left")  # these columns are m_1..m_K sums; rename later
        # rename m_k sums to c_k
        for kcol in membership_cols:
            out.rename(columns={kcol: f"c_{kcol.split('_')[1]}"}, inplace=True)

        # weighted means per community
        bag_frames = []
        for kcol in membership_cols:
            # weights w_{pk} = m_{pk}; for hard membership it is 0/1
            # compute weighted mean per bag for each feature: sum(w * x) / sum(w)
            dfw = df_instances[[bag_id_col, kcol] + feature_cols].copy()
            for f in feature_cols:
                dfw[f"{f}_wx"] = dfw[kcol] * dfw[f]
            gk = dfw.groupby(bag_id_col, observed=True).agg(
                **{f"{f}_sumwx": (f"{f}_wx", "sum") for f in feature_cols},
                **{f"{f}_sumw": (kcol, "sum") for f in feature_cols}
            )
            gk = gk.reset_index()
            # build means with guard on small weights
            for f in feature_cols:
                num = gk[f"{f}_sumwx"].values
                den = gk[f"{f}_sumw"].values
                mean = np.where(den >= min_count_for_mean, num / np.where(den == 0, np.nan, den), np.nan)
                gk[f"xk{kcol.split('_')[1]}_{f.split('_')[1]}"] = mean
                # cleanup helper cols
                del gk[f"{f}_sumwx"]; del gk[f"{f}_sumw"]
            # keep only the new xk* columns and bag_id
            keep_cols = [bag_id_col] + [c for c in gk.columns if c.startswith("xk")]
            bag_frames.append(gk[keep_cols])

        # merge all community-mean blocks
        for gf in bag_frames:
            out = out.merge(gf, on=bag_id_col, how="left")

    return out

# clustering function
def cluster_data(df, feature_cols, n_clusters=5, norm=False, sname=False):
    """
    Cluster the data based on the feature columns using KMeans clustering.
    
    Parameters:
    - df: DataFrame containing the data to cluster
    - feature_cols: List of feature column names to use for clustering
    - n_clusters: Number of clusters for KMeans
    - norm: If True, normalize the features before clustering
    - sname: If True, return cluster names instead of indices
    
    Returns:
    - DataFrame with an additional column 'cluster' indicating cluster assignments
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X = df[feature_cols].values
    
    if norm:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    if sname:
        df['cluster'] = df['cluster'].apply(lambda x: f'Cluster_{x}')
    
    return df


# Build a summary dataframe with counts of each cluster per bag, label per bag, and total instances per bag
def build_cluster_summary(df, bag_col='PID', cluster_col='cluster', label_col='target'):
    """
    Build a summary dataframe with:
    - counts of each cluster per bag
    - label per bag
    - total instances per bag (n)

    Parameters:
    df (pd.DataFrame): Input dataframe
    bag_col (str): Column representing bag IDs (e.g., PID)
    cluster_col (str): Column representing cluster IDs
    label_col (str): Column representing labels

    Returns:
    pd.DataFrame: Aggregated dataframe indexed by bag_col
    """

    # Crosstab: counts of clusters per bag
    counts = pd.crosstab(df[bag_col], df[cluster_col])

    # Rename columns (c_1, c_2, ...)
    counts.columns = [f'c_{int(col)}' for col in counts.columns]

    # Reset index for merging
    counts = counts.reset_index()

    # Get one label per bag
    labels = df.groupby(bag_col)[label_col].first().reset_index()

    # Merge counts + labels
    grouped = counts.merge(labels, on=bag_col, how='left')

    # Drop missing labels
    grouped = grouped.dropna(subset=[label_col])

    # Ensure label is numeric
    grouped[label_col] = grouped[label_col].astype(float)

    # Total instances per bag
    count_df = df.groupby(bag_col).size().reset_index(name='n')

    # Merge total counts
    grouped = grouped.merge(count_df, on=bag_col, how='left')

    # Set index
    grouped = grouped.set_index(bag_col)

    return grouped


# Feature analysis (FA) for instance-level outcomes
def fit_inst_fa(
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,    # e.g., ['x_1', ... 'x_d']
        target_cols: Optional[List[str]] = None,     # e.g., ['t_r1','t_r2',...]
        covariates: Optional[List[str]] = None,      # e.g., ['site','z_num']
        bag_id_col: str = "bag_id",
        families: Optional[Dict[str, str]] = None,   # map target -> 'binomial'|'gaussian'|'poisson'
        cov_type: str = "cluster"                    # clustered SEs by bag
    ) -> Dict[str, pd.DataFrame]:
    """
    Fit FA1‑style *univariate* GLMs:
        each target column is modelled separately as:
            target ~ x_j
        for each feature x_j, plus optional covariates if provided.

    Returns a dict of tidy tables (one per target) with BH‑FDR across *all* features,
    not just within each target.

    Includes robust error handling and skips degenerate cases.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("x_")]
        feature_cols = sorted(feature_cols, key=lambda s: int(s.split("_")[1]))
    if target_cols is None:
        target_cols = [c for c in df.columns if c.startswith("t_r")]
        target_cols = sorted(target_cols, key=lambda s: int(s.split("t_r")[1]))
    if covariates is None:
        covariates = []

    def _fam(name: str):
        return {"binomial": Binomial(), "gaussian": Gaussian(), "poisson": Poisson()}[name]

    # 1) Collect all rows globally (for FDR across all feature‑target pairs)
    all_rows = []

    # Covariate string (same for all models)
    cov_rhs = " + ".join(covariates) if covariates else "1"

    for tgt in target_cols:
        fam_name = families[tgt] if families and tgt in families else "gaussian"
        fam = _fam(fam_name)

        rows = []

        for xcol in feature_cols:
            # formula: target ~ x + covariates
            if covariates:
                formula = f"{tgt} ~ {xcol} + {cov_rhs}"
            else:
                formula = f"{tgt} ~ {xcol}"

            # --- PRE‑FIT SANITY CHECKS ---
            df_pair = df[[bag_id_col, tgt, xcol]].dropna()
            if len(df_pair) == 0:
                print(f"⚠️  No valid data for {tgt} ~ {xcol}")
                continue

            y_vals = df_pair[tgt].values
            x_vals = df_pair[xcol].values

            if np.allclose(y_vals, 0.0) or np.var(y_vals) < 1e-10:
                print(f"⚠️  {tgt} is all 0 or near‑constant for {xcol} — skipping")
                continue

            if np.allclose(x_vals, 0.0) or np.var(x_vals) < 1e-10:
                print(f"⚠️  {xcol} is all 0 or near‑constant for {tgt} — skipping")
                continue

            # --- TRY FIT WITH EXCEPTION CATCHING ---
            try:
                model = smf.glm(formula=formula, data=df_pair, family=fam)
                res = model.fit(
                    cov_type="cluster" if cov_type == "cluster" else cov_type,
                    cov_kwds={"groups": df_pair[bag_id_col].values} if cov_type == "cluster" else None
                )

                # Skip if patsy / glm somehow failed to give a normal parameter
                if xcol not in res.params.index:
                    print(f"⚠️  {xcol} not in params for {tgt} — skipped")
                    continue

                coef = float(res.params[xcol])
                se = float(res.bse[xcol])
                p = float(res.pvalues[xcol])
                lo, hi = res.conf_int().loc[xcol].values

                if np.any(np.isnan([coef, se, p, lo, hi])):
                    print(f"⚠️  NaN coef/SE/p/CI for {tgt} ~ {xcol} — skipped")
                    continue

            except Exception as e:
                print(f"⚠️  Failed to fit {tgt} ~ {xcol}: {str(e)}")
                continue

            # --- TRANSFORM TO EFFECT PLOT SCALE ---
            if fam_name in ("binomial", "poisson"):
                eff = np.exp(coef)
                eff_lo = np.exp(lo)
                eff_hi = np.exp(hi)
            else:
                eff = coef
                eff_lo = lo
                eff_hi = hi

            row = {
                "target": tgt,
                "term": xcol,
                "is_feature": True,
                "coef": coef,
                "se": se,
                "p": p,
                "ci_lo": lo,
                "ci_hi": hi,
                "effect_plot": eff,
                "effect_lo": eff_lo,
                "effect_hi": eff_hi,
                "family": fam_name,
                "nobs": res.nobs,
            }
            rows.append(row)
            all_rows.append(row)

        # Make eff_df even if rows is empty
        eff_df = pd.DataFrame(rows)
        eff_df["q"] = np.nan
        eff_df["significant"] = False
        eff_df = eff_df.sort_values("term").reset_index(drop=True)

    # 2) Global FDR across all feature‑target pairs
    df_all = pd.DataFrame(all_rows)

    if not df_all.empty:
        rej, qvals, _, _ = multipletests(df_all["p"].values, method="fdr_bh")
        df_all["q"] = qvals
        df_all["significant"] = rej
    else:
        df_all["q"] = np.nan
        df_all["significant"] = False

    # 3) Rebuild per‑target dict with global q
    results: Dict[str, pd.DataFrame] = {}
    for tgt in target_cols:
        tgt_df = df_all[df_all["target"] == tgt].copy()
        tgt_df = tgt_df.sort_values("term").reset_index(drop=True)
        results[tgt] = tgt_df

    return results

# Feature analysis (FA) for bag-level outcomes
def fit_fa_aggregate(
    df_bag_design: pd.DataFrame,
    df_bags_targets_covs: pd.DataFrame,  # must include bag_id + y_* + optional v_*
    target_cols: List[str],
    covariate_cols: Optional[List[str]] = None,  # bag-level covariates already in df_bags_targets_covs
    use_global_means: bool = True,
    use_community_means: bool = False,
    family_by_target: Optional[Dict[str, str]] = None,  # map target -> 'binomial'|'gaussian'|'poisson'
    bag_id_col: str = "bag_id",
    cov_type: str = "HC1"
) -> Dict[str, pd.DataFrame]:
    """
    Fit FA2 on bag-level design:
      - If use_global_means:   y ~ xg_* + covariates
      - If use_community_means:y ~ sum_k xk{k}_* + covariates
    Returns dict[target] -> tidy effects table with BH-FDR across features (or features-by-community).
    """
    if covariate_cols is None:
        covariate_cols = []

    # Join targets/covariates onto design
    df = df_bag_design.merge(df_bags_targets_covs[[bag_id_col] + target_cols + covariate_cols],
                             on=bag_id_col, how="left", validate="one_to_one")

    results: Dict[str, pd.DataFrame] = {}

    # Prepare feature sets
    global_feats = [c for c in df.columns if c.startswith("xg_")] if use_global_means else []
    community_feats = [c for c in df.columns if c.startswith("xk")] if use_community_means else []

    def _fam(name: str):
        return {"binomial": Binomial(), "gaussian": Gaussian(), "poisson": Poisson()}[name]

    for tgt in target_cols:
        fam_name = family_by_target[tgt] if family_by_target and tgt in family_by_target else "gaussian"
        fam = _fam(fam_name)

        rhs_terms = (global_feats + community_feats + covariate_cols)
        rhs = " + ".join(rhs_terms) if rhs_terms else "1"
        formula = f"{tgt} ~ {rhs}"

        model = smf.glm(formula=formula, data=df, family=fam)
        res = model.fit(cov_type=cov_type)

        rows = []
        for term in res.params.index:
            if term == "Intercept":
                continue
            coef = float(res.params[term]); se = float(res.bse[term]); p = float(res.pvalues[term])
            lo, hi = res.conf_int().loc[term].values

            if fam_name in ("binomial", "poisson"):
                eff, eff_lo, eff_hi = np.exp(coef), np.exp(lo), np.exp(hi)
            else:
                eff, eff_lo, eff_hi = coef, lo, hi

            rows.append({
                "target": tgt, "term": term,
                "is_global_feat": term in global_feats,
                "is_comm_feat": term in community_feats,
                "coef": coef, "se": se, "p": p, "q": np.nan,
                "ci_lo": lo, "ci_hi": hi,
                "effect_plot": eff, "effect_lo": eff_lo, "effect_hi": eff_hi,
                "family": fam_name, "nobs": res.nobs
            })
        eff_df = pd.DataFrame(rows)

        # BH-FDR: adjust p-values within each feature family separately
        for mask_name, mask in {
            "global": eff_df["is_global_feat"].values,
            "community": eff_df["is_comm_feat"].values
        }.items():
            if mask.any():
                rej, qvals, _, _ = multipletests(eff_df.loc[mask, "p"].values, method="fdr_bh")
                eff_df.loc[mask, "q"] = qvals
                eff_df.loc[mask, "significant"] = rej

        results[tgt] = eff_df.sort_values("term").reset_index(drop=True)

    return results


def _fit_nb_glm_one_formula(df: pd.DataFrame,
                            formula: str,
                            offset_col: str = "n",
                            cov_type: str = "HC1"):
    """
    Internal helper: fit NB GLM with a Patsy formula and log(offset_col) offset.
    Returns the statsmodels results object.
    """
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=1.0),   # NB2 variance by default
        offset=np.log(df[offset_col].values)     # log(n) offset makes it "relative abundance"
    )
    res = model.fit(cov_type=cov_type)
    return res


def fit_ca_bag(df: pd.DataFrame,
                          K: int,
                          target_cols: List[str],
                          covariates: Optional[List[str]] = None,
                          offset_col: str = "n",
                          cov_type: str = "HC1",
                          outdir: Optional[str] = None,
                          plot: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fit CA1 for *multiple* bag-level targets with optional covariates.
    For each target, fit K per-community NB GLMs and produce a tidy table and forest plot.

    Parameters
    ----------
    df : wide bag-level DataFrame (see schema above)
    K : number of communities (expects c_1..c_K columns present)
    target_cols : list of target column names to regress on
    covariates : list of additional covariate column names (can be categorical or numeric)
    offset_col : name of the total bag size column to use in offset log(n)
    cov_type : covariance type for SEs (e.g., "nonrobust", "HC1")
    outdir : if provided, CSV and PNGs are written here
    plot : if True, generate forest plots per target

    Returns
    -------
    results_by_target : dict mapping target -> tidy effects DataFrame with:
        community, response, term, coef_log, se, p, q, exp_coef, exp_ci_lo, exp_ci_hi, deviance, pearson_chi2, nobs
    """
    if covariates is None:
        covariates = []

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    results_by_target: Dict[str, pd.DataFrame] = {}

    # Build the RHS once per target: "y + cov1 + cov2 + ..."
    for target in target_cols:
        rhs_terms = [target] + list(covariates)
        rhs = " + ".join(rhs_terms) if rhs_terms else "1"

        rows = []
        # Fit K separate models: c_k ~ target + covariates
        for k in range(1, K + 1):
            response = f"c_{k}"
            formula = f"{response} ~ {rhs}"

            res = _fit_nb_glm_one_formula(df, formula, offset_col, cov_type)

            # Loop over all *non-intercept* terms so we can report covariates too
            for term in res.params.index:
                if term == "Intercept":
                    continue
                coef = res.params.get(term, np.nan)
                se = res.bse.get(term, np.nan)
                pval = res.pvalues.get(term, np.nan)
                lo, hi = res.conf_int().loc[term].values if term in res.params.index else (np.nan, np.nan)

                rows.append({
                    "target": target,          # which outcome is used on RHS
                    "community": f"k={k}",     # which c_k is on LHS
                    "response": response,      # column name c_k
                    "term": term,              # which predictor term (target or covariate)
                    "coef_log": coef,          # log-scale coefficient
                    "se": se,
                    "p": pval,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "exp_coef": np.exp(coef) if np.isfinite(coef) else np.nan,
                    "exp_ci_lo": np.exp(lo) if np.isfinite(lo) else np.nan,
                    "exp_ci_hi": np.exp(hi) if np.isfinite(hi) else np.nan,
                    "deviance": res.deviance,
                    "pearson_chi2": res.pearson_chi2,
                    "nobs": res.nobs
                })

        effects = pd.DataFrame(rows)

        # --- Multiple testing: BH-FDR *per term* across communities ---
        # This means for each predictor (e.g., y_resp, age, site[T.B], ...),
        # we adjust p-values across all K communities.
        effects["q"] = np.nan
        effects["significant"] = False
        for term_name, grp in effects.groupby("term", sort=False):
            rej, qvals, _, _ = multipletests(grp["p"].values, method="fdr_bh")
            effects.loc[grp.index, "q"] = qvals
            effects.loc[grp.index, "significant"] = rej

        # Save per-target CSV
        if outdir:
            csv_path = os.path.join(outdir, f"ca1_effects_{target}.csv")
            effects.to_csv(csv_path, index=False)

        # Forest plots for the *target term only* (not covariates)
        if plot:
            plot_ca_bag_forest(
                effects=effects,
                term=target,
                title=f"CA1: exp(beta) for '{target}' by community",
                x_label="Multiplicative change in expected abundance (exp(beta))",
                outpath=os.path.join(outdir, f"ca1_forest_{target}.png") if outdir else None
            )

        results_by_target[target] = effects

    return results_by_target








# Community Abundance (CA) for instance-level outcomes
def fit_ca_inst(
    df: pd.DataFrame,
    target_col: str = "t",
    membership_cols: Optional[List[str]] = None,  # e.g., ['m_1','m_2',...,'m_K']
    covariates: Optional[List[str]] = None,       # e.g., ['x_num','site']
    bag_id_col: str = "bag_id",
    family: str = "binomial",                     # 'binomial' | 'gaussian' | 'poisson'
    drop_reference_if_exclusive: bool = True,     # drop last membership to avoid collinearity
    cov_type: str = "cluster"                     # clustered SEs by bag
) -> pd.DataFrame:
    """
    Fit CA2 using statsmodels GLM with Patsy formulas:
        t ~ m_1 + ... + m_K + covariates
    - Uses bag-clustered SEs (cov_type='cluster') to account for within-bag dependence.
    - If memberships are exclusive (sum to 1), you should drop a reference column
      (set drop_reference_if_exclusive=True and ensure membership_cols are one-hot).
    Returns a tidy DataFrame with coefficients (focus on membership terms), CIs, p-values and BH-FDR.
    """
    if membership_cols is None:
        # auto-detect 'm_1'...'m_K' style columns
        membership_cols = [c for c in df.columns if c.startswith("m_")]
        membership_cols = sorted(membership_cols, key=lambda s: int(s.split("_")[1]))
    if covariates is None:
        covariates = []

    # Optionally drop a reference community to avoid collinearity for exclusive memberships
    used_memberships = membership_cols.copy()
    if drop_reference_if_exclusive and len(membership_cols) >= 2:
        used_memberships = membership_cols[:-1]  # drop last as reference

    # Build formula RHS: memberships + covariates
    rhs_terms = used_memberships + covariates
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"

    # Choose family object
    fam = {"binomial": Binomial(), "gaussian": Gaussian(), "poisson": Poisson()}[family]

    # Fit GLM
    formula = f"{target_col} ~ {rhs}"
    model = smf.glm(formula=formula, data=df, family=fam)
    # Cluster-robust SEs by bag
    res = model.fit(
        cov_type="cluster" if cov_type == "cluster" else cov_type,
        cov_kwds={"groups": df[bag_id_col].values} if cov_type == "cluster" else None
    )

    # Collect coefficients for membership terms (and optionally covariates)
    rows = []
    for term in res.params.index:
        if term == "Intercept":
            continue
        coef = float(res.params[term])
        se = float(res.bse[term])
        p = float(res.pvalues[term])
        lo, hi = res.conf_int().loc[term].values

        # For forest plot scale: exp for logit/log links; identity for Gaussian
        if family in ("binomial", "poisson"):
            effect = np.exp(coef)
            effect_lo = np.exp(lo)
            effect_hi = np.exp(hi)
        else:
            effect = coef
            effect_lo = lo
            effect_hi = hi

        rows.append({
            "term": term,
            "is_membership": term in membership_cols,
            "coef": coef,
            "se": se,
            "p": p,
            "ci_lo": lo,
            "ci_hi": hi,
            "effect_plot": effect,       # exp(coef) for logit/log; raw coef for identity
            "effect_lo": effect_lo,
            "effect_hi": effect_hi,
            "nobs": res.nobs
        })

    out = pd.DataFrame(rows)

    # Multiple testing: BH-FDR across *membership terms only*
    mem_mask = out["is_membership"].values
    if mem_mask.any():
        rej, qvals, _, _ = multipletests(out.loc[mem_mask, "p"].values, method="fdr_bh")
        out.loc[mem_mask, "q"] = qvals
        out.loc[mem_mask, "significant"] = rej
    else:
        out["q"] = np.nan
        out["significant"] = False

    # Attach model diagnostics (optional; same for all rows)
    out["deviance"] = res.deviance
    out["pearson_chi2"] = res.pearson_chi2 if hasattr(res, "pearson_chi2") else np.nan

    return out.sort_values(["is_membership", "term"], ascending=[False, True]).reset_index(drop=True)




# =============================================================================
# PLOTTING: Forest plot per target
# =============================================================================

def plot_fa_inst_forest(
        effects_df: pd.DataFrame,
        title: str = "FA: Feature effects on instance outcome",
        only_features: bool = True,
        outpath=None,
        show_p_values=False
    ) -> None:
        """
        Forest plot for one target's effects:
        - Uses exp(coef) for binomial/poisson targets (odds/rate ratios)
        - Uses raw coef for gaussian targets
        """
        df = effects_df.copy()
        fam = df["family"].iloc[0] if "family" in df.columns and not df.empty else "gaussian"
        if only_features:
            df = df[df["is_feature"]]

        if df.empty:
            print("[warn] No rows to plot.")
            return

        # Sort for consistent top-down ordering
        df = df.sort_values("term", ascending=False).reset_index(drop=True)
        y = np.arange(len(df))

        plt.figure(figsize=(7, 4.5))

        # CIs
        for i, r in enumerate(df.itertuples(index=False)):
            plt.plot([r.effect_lo, r.effect_hi], [y[i], y[i]], lw=2)

        # Points
        plt.scatter(df["effect_plot"].values, y, zorder=3)

        # Reference line
        if fam in ("binomial", "poisson"):
            plt.axvline(1.0, linestyle="--")
            plt.xlabel("Effect (exp(coef))")
        else:
            plt.axvline(0.0, linestyle="--")
            plt.xlabel("Effect (coef, identity link)")

        labels = df["term"].astype(str).tolist()
        if show_p_values:
            labels = [f"{label}  (p = {p:.3g})" for label, p in zip(labels, df["p"])]
        plt.yticks(y, labels)
        plt.title(title)
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.show()


def plot_fa_aggregate_forest(effects_df: pd.DataFrame, title: str, highlight: str = "global"):
    """
    Forest plot for bag-aggregation FA2:
      - highlight='global'  plots global features only (xg_*)
      - highlight='community' plots community-localized features only (xk*)
    """
    df = effects_df.copy()
    fam = df["family"].iloc[0]
    if highlight == "global":
        df = df[df["is_global_feat"]]
    elif highlight == "community":
        df = df[df["is_comm_feat"]]
    if df.empty:
        print(f"[warn] No rows to plot for highlight={highlight}."); return

    df = df.sort_values("term", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    plt.figure(figsize=(8, 5))
    for i, r in enumerate(df.itertuples(index=False)):
        plt.plot([r.effect_lo, r.effect_hi], [y[i], y[i]], lw=2)
    plt.scatter(df["effect_plot"].values, y, zorder=3)

    if fam in ("binomial", "poisson"):
        plt.axvline(1.0, linestyle="--"); plt.xlabel("Effect (exp(coef))")
    else:
        plt.axvline(0.0, linestyle="--"); plt.xlabel("Effect (coef)")
    plt.yticks(y, df["term"].values)
    plt.title(title)
    plt.tight_layout(); plt.show()



def plot_ca_bag_forest(effects: pd.DataFrame,
                          term: str,
                          title: str,
                          x_label: str,
                          outpath: Optional[str] = None,
                          show_p_values: bool = False):
    """
    Forest plot of exp(coef) with 95% CI across communities for a single predictor term.
    """
    term_df = effects.loc[effects["term"] == term].copy()
    if term_df.empty:
        print(f"[warn] No rows found for term '{term}' — skipping forest plot.")
        return

    # Order communities top-to-bottom
    term_df = term_df.sort_values("community", ascending=False)
    y_pos = np.arange(len(term_df))

    plt.figure(figsize=(7, 4.5))

    # Confidence interval segments
    for i, row in enumerate(term_df.itertuples(index=False)):
        plt.plot([row.exp_ci_lo, row.exp_ci_hi], [y_pos[i], y_pos[i]], lw=2)

    # Point estimates
    plt.scatter(term_df["exp_coef"].values, y_pos, zorder=3)

    # Reference line at no-effect = 1
    plt.axvline(1.0, linestyle="--")

    labels = term_df["community"].astype(str).tolist()
    if show_p_values:
        labels = [
            f"{label}  (p = {p_value:.3g})" if np.isfinite(p_value)
            else f"{label}  (p unavailable)"
            for label, p_value in zip(labels, term_df["p"])
        ]
    plt.yticks(y_pos, labels)
    plt.xlabel(x_label)
    plt.title(title)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=160)
        print(f"[saved] {outpath}")
    plt.show()




def plot_ca_inst_forest(
    effects_df: pd.DataFrame,
    family: str = "binomial",
    title: str = "CA: Community effects on instance outcome",
    x_label_binom_poiss: str = "Effect (exp(coef))",
    x_label_gauss: str = "Effect (coef, identity link)",
    only_memberships: bool = True
) -> None:
    """
    CA2
    Draw a forest plot of membership effects with 95% CI.
    - For binomial/poisson, plot exp(coef) and CI on the exp scale (odds/rate ratio).
    - For gaussian, plot raw coef and CI.
    """
    df = effects_df.copy()
    if only_memberships:
        df = df[df["is_membership"]]

    if df.empty:
        print("[warn] No rows to plot.")
        return

    # Order terms top-to-bottom
    df = df.sort_values("term", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df))

    plt.figure(figsize=(7, 4.5))

    # CI segments
    for i, r in enumerate(df.itertuples(index=False)):
        plt.plot([r.effect_lo, r.effect_hi], [y_pos[i], y_pos[i]], lw=2)

    # Point estimates
    plt.scatter(df["effect_plot"].values, y_pos, zorder=3)

    # Reference line at "no effect"
    if family in ("binomial", "poisson"):
        plt.axvline(1.0, linestyle="--")  # odds/rate ratio of 1
        plt.xlabel(x_label_binom_poiss)
    else:
        plt.axvline(0.0, linestyle="--")
        plt.xlabel(x_label_gauss)

    plt.yticks(y_pos, df["term"].values)
    plt.title(title)
    plt.tight_layout()
    plt.show()







def plot_fa_inst_heatmap(
    dcafa_results,
    outpath=None,
    show_p_values=False
    ):
    # Prepare data using YOUR exact reverse mapping and cleaning
    all_data = []

    for tgt, eff_df in dcafa_results.items():
        # Apply YOUR exact cleaning steps
        eff_df_clean = eff_df.copy()
        eff_df_clean["term"] = eff_df_clean["term"]
        eff_df_clean["term"] = eff_df_clean["term"]
        
        for _, row in eff_df_clean.iterrows():
            all_data.append({
                'target': tgt,
                'term': row['term'],
                'effect_plot': row['effect_plot'],
                'p': row.get('p', np.nan),
                'q': row.get('q', 1.0)
            })

    df_heatmap = pd.DataFrame(all_data)

    # Pivot matrices
    pivot_coef = df_heatmap.pivot(index='term', columns='target', values='effect_plot').fillna(0)
    pivot_p = df_heatmap.pivot(index='term', columns='target', values='p')
    pivot_q = df_heatmap.pivot(index='term', columns='target', values='q').fillna(1.0)

    # Limit to top 20 features
    top_features = list(pivot_coef.index)[:30]
    pivot_subset = pivot_coef.loc[top_features]
    pivot_q_subset = pivot_q.loc[top_features]

    # Stars using YOUR q-thresholds
    annot_matrix = np.full(pivot_subset.shape, '', dtype='<U10')
    for i in range(pivot_subset.shape[0]):
        for j in range(pivot_subset.shape[1]):
            qval = pivot_q_subset.iloc[i, j]
            if qval < 0.001:
                annot_matrix[i, j] = '***'
            elif qval < 0.01:
                annot_matrix[i, j] = '**'
            elif qval < 0.05:
                annot_matrix[i, j] = '*'

    # EXACT style match to your reference plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_subset, cmap='coolwarm', center=0, annot=False, 
                    linewidths=0.5, linecolor='white')

    # Top x-axis labels (like your reference)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # Stars for significance (like your reference)
    for i in range(pivot_subset.shape[0]):
        for j in range(pivot_subset.shape[1]):
            annotation = annot_matrix[i, j]
            if show_p_values:
                p_value = pivot_p.loc[top_features].iloc[i, j]
                annotation = f"p = {p_value:.3g}" + (f"\n{annotation}" if annotation else "")
            if annotation:
                ax.text(j + 0.5, i + 0.5, annotation,
                        ha='center', va='center', color='black', fontsize=12)

    plt.xlabel('Targets')
    plt.ylabel('Features')
    plt.title('FA effects on instance target\n***q<0.001, **q<0.01, *q<0.05')
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()

def plot_fa_aggregate_heatmap(results):

    # Apply YOUR exact cleaning (reverse mapping + properties removal)
    all_data = []

    for tgt, eff_df in results.items():
        # Clean terms like your FA1 code
        eff_df_clean = eff_df.copy()
        eff_df_clean["term"] = eff_df_clean["term"]
        eff_df_clean["term"] = eff_df_clean["term"]
        
        # Filter to global features only (xg_*)
        global_feats = eff_df_clean[eff_df_clean["is_global_feat"]]
        
        for _, row in global_feats.iterrows():
            all_data.append({
                'target': tgt,
                'term': row['term'],
                'coef': row['coef'],
                'q': row.get('q', 1.0)
            })

    df_heatmap = pd.DataFrame(all_data)

    # Pivot for heatmap
    pivot_coef = df_heatmap.pivot(index='term', columns='target', values='coef').fillna(0)
    pivot_q = df_heatmap.pivot(index='term', columns='target', values='q').fillna(1.0)
    
    # Top 20 features
    top_features = list(pivot_coef.index)[:20]
    pivot_subset = pivot_coef.loc[top_features]
    pivot_q_subset = pivot_q.loc[top_features]
    
    # Stars matrix
    annot_matrix = np.full(pivot_subset.shape, '', dtype='<U10')
    for i in range(pivot_subset.shape[0]):
        for j in range(pivot_subset.shape[1]):
            qval = pivot_q_subset.iloc[i, j]
            if qval < 0.001:
                annot_matrix[i, j] = '***'
            elif qval < 0.01:
                annot_matrix[i, j] = '**'
            elif qval < 0.05:
                annot_matrix[i, j] = '*'
    
    # EXACT same style as your reference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_subset, cmap='coolwarm', center=0, annot=False, 
                    linewidths=0.5, linecolor='white')
    
    # Top x-axis labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # Stars
    for i in range(pivot_subset.shape[0]):
        for j in range(pivot_subset.shape[1]):
            if annot_matrix[i, j]:
                ax.text(j + 0.5, i + 0.5, annot_matrix[i, j], 
                    ha='center', va='center', color='black', 
                    fontsize=12, fontweight='bold')
    
    plt.xlabel('Targets')
    plt.ylabel('Features (Bag-Level Means)')
    plt.title('FA Aggregation \n***q<0.001, **q<0.01, *q<0.05')
    plt.tight_layout()
    plt.show()


def ca_stacked_area_plot(
    df1: pd.DataFrame,
    target_col: str = "y_resp",
    cluster_cols: list = ['c_1'],
    nbins: int = 10
):
    df = df1.copy()

    # Convert cluster counts to percentages
    df[cluster_cols] = df[cluster_cols].div(df['n'], axis=0) * 100

    # Detect if continuous
    n_unique = df[target_col].nunique()
    n_total = len(df[target_col])

    if pd.api.types.is_numeric_dtype(df[target_col]) and (n_unique / n_total) > 0.05:
        # Bin continuous variable and keep intervals
        df['_bin'] = pd.cut(df[target_col], bins=nbins)

        # Use bin midpoints for plotting (keeps original scale feel)
        df['y_resp'] = df['_bin'].apply(lambda x: x.mid)

        # Optional: keep labels for nicer xticks
        bin_labels = (
            df[['_bin', 'y_resp']]
            .drop_duplicates()
            .sort_values('y_resp')
        )
    else:
        df['y_resp'] = df[target_col]
        bin_labels = None

    # Aggregate
    grouped_counts = (
        df.groupby('y_resp')[cluster_cols + ['n']]
        .sum()
        .reset_index()
        .sort_values('y_resp')
    )

    # Normalize to 100% within each group
    grouped_counts[cluster_cols] = grouped_counts[cluster_cols].div(
        grouped_counts[cluster_cols].sum(axis=1), axis=0
    ) * 100

    # Plot
    plt.figure(figsize=(12, 7))

    colors = sns.color_palette('tab10', len(cluster_cols))
    bottom = np.zeros(len(grouped_counts))

    x = grouped_counts['y_resp'].values

    for i, cluster in enumerate(cluster_cols):
        cluster_data = grouped_counts[cluster].values
        plt.fill_between(
            x,
            bottom,
            bottom + cluster_data,
            color=colors[i],
            alpha=0.8,
            label=cluster
        )
        bottom += cluster_data

    plt.ylabel('Proportion (%)', fontsize=12)
    plt.xlabel(target_col, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Use interval labels if continuous
    if bin_labels is not None:
        plt.xticks(
            bin_labels['y_resp'],
            [str(interval) for interval in bin_labels['_bin']],
            rotation=45
        )
    else:
        plt.xticks(x)

    plt.tight_layout()
    plt.show()