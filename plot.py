"""UMAPs, boxplots, group summaries and median feature heatmaps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from common import write_csv
from features import FEATURE_COLUMNS, FEATURE_FAMILIES


def prepare_matrix(frame):
    columns = [
        c for c in FEATURE_COLUMNS if c in frame and frame[c].replace([np.inf, -np.inf], np.nan).nunique() > 1
    ]
    if not columns:
        return np.empty((len(frame), 0)), columns
    data = frame[columns].replace([np.inf, -np.inf], np.nan)
    return StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data)), columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["marker_group", "subtype", "cell_line"],
        help="Metadata columns used to color/group plots",
    )
    parser.add_argument("--meso-only", action="store_true")
    parser.add_argument(
        "--include-flagged", action="store_true", help="Include measured cells that failed mask QC"
    )
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="Optional separate plot directory")
    args = parser.parse_args()
    if args.neighbors < 2:
        parser.error("neighbors must be >=2")
    run = args.run.resolve()
    out = args.out or run / ("plots_meso" if args.meso_only else "plots")
    frame = pd.read_csv(run / "features.csv")
    total = len(frame)
    frame = frame[frame.feature_status == "ok"].copy()
    if not args.include_flagged:
        frame = frame[frame.qc_ok.astype(str).str.lower() == "true"].copy()
    if args.meso_only:
        frame = frame[frame.disease_group == "mesothelioma"].copy()
    if frame.empty:
        parser.error("No cells remain after filtering. Inspect features.csv and qc/.")
    for group in args.group_by:
        if group not in {
            "source_folder",
            "marker_label",
            "marker_group",
            "subtype",
            "cell_line",
            "disease_group",
        }:
            parser.error(f"Unknown grouping: {group}")
        frame[group] = frame[group].fillna("unknown")
    matrix, columns = prepare_matrix(frame)
    out.mkdir(parents=True, exist_ok=True)
    old_summary = out / "analysis.json"
    old_files = json.loads(old_summary.read_text()).get("output_files", []) if old_summary.exists() else []
    output_files = []

    def output_path(name):
        output_files.append(name)
        return out / name

    sns.set_theme(style="whitegrid", context="notebook")
    summary = {
        "input_cells": total,
        "plotted_cells": len(frame),
        "meso_only": args.meso_only,
        "include_flagged": args.include_flagged,
        "group_by": args.group_by,
        "seed": args.seed,
        "feature_columns": columns,
        "imputation": "median",
        "scaling": "z-score",
        "umap": "skipped: at least 4 cells and 2 variable features required",
    }
    embedding = None
    if len(frame) >= 4 and len(columns) >= 2:
        neighbors = min(args.neighbors, len(frame) - 1)
        coords = UMAP(
            n_neighbors=neighbors, min_dist=0.1, random_state=args.seed, n_jobs=1, init="random"
        ).fit_transform(matrix)
        embedding = frame[["image_id"] + list(dict.fromkeys(args.group_by))].copy()
        embedding[["UMAP1", "UMAP2"]] = coords
        write_csv(embedding, output_path("umap.csv"))
        summary["umap"] = {"neighbors": neighbors, "min_dist": 0.1, "init": "random"}
    write_csv(frame[["image_id"] + columns], output_path("analysis_cells.csv"))
    for group in args.group_by:
        counts = frame.groupby(group, dropna=False).size().rename("cells")
        write_csv(counts.reset_index(), output_path(f"counts_by_{group}.csv"))
        if columns:
            stats = frame.groupby(group)[columns].agg(["count", "median", "mean", "std"])
            stats.columns = [f"{feature}_{stat}" for feature, stat in stats.columns]
            write_csv(stats.reset_index(), output_path(f"summary_by_{group}.csv"))
        palette = dict(zip(counts.index, sns.color_palette("husl", len(counts))))
        if embedding is not None:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.scatterplot(
                data=embedding,
                x="UMAP1",
                y="UMAP2",
                hue=group,
                hue_order=list(counts.index),
                palette=palette,
                s=25,
                alpha=0.75,
                linewidth=0,
                ax=ax,
            )
            ax.set_title(f"Cell morphology by {group.replace('_', ' ')} (n={len(frame)})")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            fig.savefig(output_path(f"umap_by_{group}.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)
        for family, names in FEATURE_FAMILIES.items():
            names = [n for n in names if n in frame and frame[n].notna().any()]
            if not names:
                continue
            nrows = (len(names) + 2) // 3
            fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.5 * nrows), squeeze=False)
            for ax, name in zip(axes.flat, names):
                sns.boxplot(
                    data=frame,
                    x=group,
                    y=name,
                    hue=group,
                    order=list(counts.index),
                    hue_order=list(counts.index),
                    palette=palette,
                    legend=False,
                    showfliers=False,
                    ax=ax,
                )
                ax.set_xlabel("")
                ax.set_ylabel(name.replace("_", " "))
                ax.tick_params(axis="x", rotation=50, labelsize=8)
            for ax in list(axes.flat)[len(names) :]:
                ax.set_visible(False)
            fig.suptitle(
                f"{family.title()} by {group.replace('_', ' ')}; "
                + ", ".join(f"{k}: n={v}" for k, v in counts.items()),
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(output_path(f"boxplots_{family}_by_{group}.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)
        if columns:
            z = pd.DataFrame(matrix, index=frame.index, columns=columns)
            z[group] = frame[group]
            medians = z.groupby(group)[columns].median()
            fig, ax = plt.subplots(figsize=(max(10, len(columns) * 0.38), max(2.5, len(medians) * 0.55)))
            sns.heatmap(
                medians, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Median standardized feature"}
            )
            ax.set_title(f"Morphology by {group.replace('_', ' ')}")
            fig.savefig(output_path(f"heatmap_by_{group}.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)
    # Remove only artifacts listed by the previous successful run in this directory.
    for name in set(old_files) - set(output_files):
        if Path(name).name == name and Path(name).suffix in {".png", ".csv"}:
            (out / name).unlink(missing_ok=True)
    summary["output_files"] = output_files
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Plotted {len(frame)}/{total} cells -> {out}")
    if embedding is None:
        print(summary["umap"])


if __name__ == "__main__":
    main()
