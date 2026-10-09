"""Plot area, solidity, elongation and circularity distributions by cell line."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Exclude cell lines containing any of these strings (case-insensitive).
# Spaces and underscores are treated alike. Use [] to exclude none.
EXCLUDED_CELL_LINE_STRINGS = [
    "mesothelioma_patient",
    "a549",
    "h266",
    # "biphasic",
    # "epithelioid",
    # "msto"
]

# Square-root scaling makes low-count bars easier to see; use "linear" for raw heights.
Y_AXIS_SCALE = "sqrt"

FEATURE_LABELS = {
    "area": "Area (pixels²)",
    "solidity": "Solidity",
    "elongation": "Elongation (major / minor axis)",
    "circularity": "Circularity",
}


def read_features_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot_distributions(frame, out_dir, bins=25, include_flagged=False, group_by="cell_line"):
    """Plot separate group rows with counts and shared axes per feature."""
    if group_by not in {"subtype", "cell_line"}:
        raise ValueError("group_by must be subtype or cell_line")
    required = {group_by, "cell_line", "area", "solidity", "circularity"}
    if "elongation" not in frame:
        required.update({"major_axis_length", "minor_axis_length"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if bins < 1:
        raise ValueError("bins must be positive")
    frame = frame.copy()
    # Treat spaces and underscores alike: the CSV uses "Mesothelioma Patient".
    cell_line_names = frame.cell_line.fillna("").astype(str).str.lower().str.replace(
        r"[\s_]+", "_", regex=True
    )
    excluded = pd.Series(False, index=frame.index)
    for text in EXCLUDED_CELL_LINE_STRINGS:
        normalized = "_".join(text.lower().replace("_", " ").split())
        if normalized:
            excluded |= cell_line_names.str.contains(normalized, regex=False)
    frame = frame.loc[~excluded].copy()
    if "feature_status" in frame:
        frame = frame.loc[frame.feature_status == "ok"].copy()
    if not include_flagged and "qc_ok" in frame:
        frame = frame.loc[frame.qc_ok.astype(str).str.lower() == "true"].copy()
    if frame.empty:
        raise ValueError("No cells remain after filtering; inspect features.csv or use --include-flagged")
    frame[group_by] = frame[group_by].fillna("unknown").replace("", "unknown").astype(str)
    if group_by == "subtype":
        frame[group_by] = frame[group_by].str.strip().str.lower()
        groups = ["epithelioid", "biphasic"]
        frame = frame.loc[frame[group_by].isin(groups)].copy()
        absent = set(groups) - set(frame[group_by])
        if absent:
            raise ValueError(f"No eligible cells for subtype(s): {', '.join(sorted(absent))}")
    else:
        groups = sorted(frame[group_by].unique())
    if "elongation" not in frame:
        major = pd.to_numeric(frame.major_axis_length, errors="coerce")
        minor = pd.to_numeric(frame.minor_axis_length, errors="coerce")
        frame["elongation"] = major / minor.where(minor > 0)
    for name in FEATURE_LABELS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    group_colors = {}
    for group in groups:
        group_frame = frame.loc[frame[group_by] == group]
        if "subtype" in group_frame:
            subtypes = group_frame.subtype.fillna("").astype(str).str.strip().str.lower()
            epithelioid = subtypes.eq("epithelioid").all()
        else:
            epithelioid = "epithelioid" in group.lower()
        group_colors[group] = "#397da8" if epithelioid else "#d97732"
    # Keep epithelioid rows together, followed by all other cell lines.
    # The existing alphabetical order is preserved within each category.
    groups.sort(key=lambda group: group_colors[group] != "#397da8")
    fig, axes = plt.subplots(
        len(groups), 4,
        figsize=(18, 3 * len(groups)),
        squeeze=False, sharex="col", sharey="col",
    )
    for column, (name, label) in enumerate(FEATURE_LABELS.items()):
        all_values = frame[name].dropna()
        edges = np.histogram_bin_edges(all_values, bins=bins) if len(all_values) else None
        max_count = 0
        for row, group in enumerate(groups):
            ax = axes[row, column]
            values = frame.loc[frame[group_by] == group, name].dropna()
            color = group_colors[group]
            if len(values):
                counts, _, _ = ax.hist(
                    values, bins=edges,
                    color=color, edgecolor="white", linewidth=0.5,
                    label=f"{group.title()} (n={len(values)})",
                )
                max_count = max(max_count, float(counts.max()))
                ax.axvline(values.median(), color="#333333", linestyle="--", linewidth=1.3)
            else:
                ax.plot([], [], color=color, label=f"{group.title()}: no valid values")
            ax.set_title(f"{group}\nn = {len(values)}", fontsize=10)
            ax.set_xlabel(label)
            if Y_AXIS_SCALE == "sqrt":
                ax.set_yscale("function", functions=(
                    lambda values: np.sqrt(np.maximum(values, 0)),
                    lambda values: np.square(values),
                ))
                ax.set_ylabel("Cell count (square-root scale)")
            elif Y_AXIS_SCALE == "linear":
                ax.set_ylabel("Cell count")
            else:
                raise ValueError('Y_AXIS_SCALE must be sqrt or linear')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis="x", labelbottom=True)
            if edges is not None:
                ax.set_xlim(edges[0], edges[-1])
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
        # Set the shared range only after every group's histogram is drawn.
        # Setting it inside the loop disables autoscaling for subsequent rows.
        axes[0, column].set_ylim(0, max(1, np.ceil(max_count * 1.15)))
    # selection = "including QC-flagged cells" if include_flagged else "QC-passing cells when QC is available"
    fig.suptitle(f"Feature distributions by {group_by.replace('_', ' ')} — dashed lines show group medians", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"feature_distributions_by_{group_by}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = frame.groupby(group_by)[list(FEATURE_LABELS)].agg(["count", "median", "mean", "std"])
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary.to_csv(out_dir / f"feature_summary_by_{group_by}.csv")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--csv", type=Path, help="Input CSV; defaults to <run>/features.csv")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to <run>/plots_features")
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--group-by", choices=["subtype", "cell_line"], default="cell_line",
                        help="Separate rows for each cell line (default), or epithelioid/biphasic subtypes")
    parser.add_argument("--include-flagged", action="store_true", help="Include measured cells that failed mask QC")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    try:
        output = plot_distributions(
            read_features_csv(args.csv or args.run / "features.csv"),
            args.out or args.run / "plots_features", args.bins, args.include_flagged, args.group_by,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved feature distributions: {output.resolve()}")


if __name__ == "__main__":
    main()
