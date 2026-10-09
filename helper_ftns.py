import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D


def get_image_sizes(folder, group):
    records = []

    for path in folder.glob("*"):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            with Image.open(path) as img:
                width, height = img.size

            records.append({
                "filename": path.name,
                "group": group,
                "width": width,
                "height": height,
                "area_pixels": width * height
            })

    return records


def compare_folder_images(folder1, folder2):
    data = (
            get_image_sizes(folder1, "Folder 1") +
            get_image_sizes(folder2, "Folder 2")
    )

    df = pd.DataFrame(data)

    print(df.groupby("group")[["width", "height", "area_pixels"]].describe())

    for group in df["group"].unique():
        subset = df[df["group"] == group]

        plt.hist(
            subset["area_pixels"],
            bins=30,
            alpha=0.5,
            label=group
        )

    plt.xlabel("Image size (width × height pixels)")
    plt.ylabel("Number of images")
    plt.legend()
    plt.show()


def select_extreme_cells_sequential(
    df,
    group_by="subtype",
    group="epithelioid",
    feature1="elongation",
    direction1="high",
    n1=50,
    feature2="circularity",
    direction2="low",
    n2=16,
    path_col="processed_path",
    show_images=True,
    ncols=4
):
    """
    Sequentially select cells (set1) which are in direction 1 for feature1 and then
    select cells (set2) from set1 which are in direction 2 for feature2,
    and optionally display the selected images.

    Step 1:
        Select n1 cells with high/low feature1.

    Step 2:
        From those cells, select n2 cells with
        high/low feature2.
    """

    # -----------------------------
    # Validation
    # -----------------------------
    if direction1 not in ["high", "low"]:
        raise ValueError("direction1 must be 'high' or 'low'")

    if direction2 not in ["high", "low"]:
        raise ValueError("direction2 must be 'high' or 'low'")

    # display overall feature ranges
    print("\nOverall feature ranges (all cells):")
    print(
        f"{feature1}: min = {df[feature1].min():.3f}, "
        f"max = {df[feature1].max():.3f}"
    )
    print(
        f"{feature2}: min = {df[feature2].min():.3f}, "
        f"max = {df[feature2].max():.3f}"
    )

    # -----------------------------
    # Select group
    # -----------------------------
    cells = df[df[group_by] == group].copy()

    print(f"\n{group} feature ranges:")
    print(
        f"{feature1}: min = {cells[feature1].min():.3f}, "
        f"max = {cells[feature1].max():.3f}"
    )
    print(
        f"{feature2}: min = {cells[feature2].min():.3f}, "
        f"max = {cells[feature2].max():.3f}"
    )


    if cells.empty:
        print(f"No cells found for {group_by} = {group}")
        return cells

    # -----------------------------
    # First feature selection
    # -----------------------------
    cells_feature1 = cells.sort_values(
        feature1,
        ascending=(direction1 == "low")
    ).head(n1)

    # -----------------------------
    # Second feature selection
    # -----------------------------
    selected = cells_feature1.sort_values(
        feature2,
        ascending=(direction2 == "low")
    ).head(n2)

    # Order images for display
    selected = selected.sort_values(
        by=["elongation", "circularity"],
        ascending=[False, True]
    )

    # -----------------------------
    # Display images
    # -----------------------------
    if show_images and not selected.empty:

        nrows = (len(selected) + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3 * ncols, 3 * nrows)
        )

        # Makes axes iterable even if only one image
        axes = [axes] if len(selected) == 1 else axes.flatten()
        run_dir = Path(__file__).resolve().parent / "results"

        for ax, (_, row) in zip(axes, selected.iterrows()):

            image_path = run_dir / row[path_col]

            with Image.open(image_path) as image:
                ax.imshow(image)

            ax.set_title(
                f"{row['cell_line']}\n"
                f"{feature1}: {row[feature1]:.3f} | "
                f"{feature2}: {row[feature2]:.3f}",
                fontsize=9
            )

            ax.axis("off")

        # Hide unused panels
        for ax in axes[len(selected):]:
            ax.axis("off")

        plt.suptitle(
            f"{group}: {direction1} {feature1} → "
            f"{direction2} {feature2}",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()

    return selected


def show_extreme_cells(
    df,
    group="Epithelioid", group_by="subtype",
    feature1="elongation", direction1="high",
    feature2="circularity", direction2="low",
    percentile=0.30,
    top_n=20, path_col="processed_path", ncols=4
):
    cells = df[df[group_by] == group].copy()

    def get_cutoff(feature, direction):
        q = 1 - percentile if direction == "high" else percentile
        return cells[feature].quantile(q)

    cutoff1 = get_cutoff(feature1, direction1)
    cutoff2 = get_cutoff(feature2, direction2)

    # Apply BOTH conditions
    condition1 = (
        cells[feature1] >= cutoff1 if direction1 == "high"
        else cells[feature1] <= cutoff1
    )

    condition2 = (
        cells[feature2] >= cutoff2 if direction2 == "high"
        else cells[feature2] <= cutoff2
    )

    selected = cells[condition1 & condition2].copy()

    print(f"{feature1} cutoff ({direction1}): {cutoff1:.3f}")
    print(f"{feature2} cutoff ({direction2}): {cutoff2:.3f}")
    print(f"Cells satisfying BOTH: {len(selected)}")

    # Normalize only for ordering the selected cells
    for f, d, name in [
        (feature1, direction1, "score1"),
        (feature2, direction2, "score2")
    ]:
        mn, mx = df[f].min(), df[f].max()
        selected[name] = (selected[f] - mn) / (mx - mn)

        if d == "low":
            selected[name] = 1 - selected[name]

    selected["combined_score"] = (
        selected["score1"] + selected["score2"]
    ) / 2

    selected = selected.nlargest(top_n, "combined_score")

    # Sort selected cells by elongation for display
    selected = selected.sort_values(
        "elongation",
        ascending=False
    )

    if selected.empty:
        print("No cells satisfy both criteria.")
        return selected

    # Results directory relative to this Python script
    run_dir = Path(__file__).resolve().parent / "results"

    # Display images
    nrows = (len(selected) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False
    )
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, selected.iterrows()):
        image_path = Path(row[path_col])

        # Adjust path to point to the overlays directory
        image_path = Path("overlays") / image_path.relative_to("images")
        # Change .png.png -> .png
        if image_path.name.endswith(".png.png"):
            image_path = image_path.with_name(
                image_path.name[:-4]
            )

        with Image.open(run_dir / image_path) as img:
            ax.imshow(img)

        # Display original feature values
        ax.set_title(
            f"{row['cell_line']}\n"
            f"{feature1}: {row[feature1]:.3f} | "
            f"{feature2}: {row[feature2]:.3f}",
            fontsize=8
        )

        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.suptitle(
        f"{group}: {direction1} {feature1} + "
        f"{direction2} {feature2}"
    )

    plt.tight_layout()
    plt.show()

    return selected

def _epithelioid_first_groups(df, group_by, groups):
    """Order epithelioid groups first, alphabetically within each category."""
    def is_epithelioid(group):
        if group_by == "cell_line" and "subtype" in df:
            subtypes = df.loc[df[group_by] == group, "subtype"]
            normalized = subtypes.fillna("").astype(str).str.strip().str.lower()
            return normalized.isin(["epithelioid", "epitheliod"]).all()
        return str(group).strip().lower() in {"epithelioid", "epitheliod"}

    return sorted(groups, key=lambda group: (not is_epithelioid(group), str(group).lower()))

def get_cluster_distribution(
    df,
    group_by="cell_line",
    cluster_col="cluster",
    percentage=True,
    decimals=1,
    show_heatmap=True,
    outpath=None,
    is_save=False
):
    """
    Calculate and optionally plot cluster distribution for each group.
    """

    # Calculate distribution
    if percentage:
        table = pd.crosstab(
            df[group_by],
            df[cluster_col],
            normalize="index"
        ) * 100

        label = "Percentage of cells (%)"

    else:
        table = pd.crosstab(
            df[group_by],
            df[cluster_col]
        )

        label = "Number of cells"

    # Display epithelioid cell lines first, then the remaining cell lines.
    table = table.reindex(_epithelioid_first_groups(df, group_by, table.index))
    table = table.round(decimals)

    # -------------------------
    # Heatmap
    # -------------------------
    if show_heatmap:

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(
            table.values,
            aspect="auto",
            cmap="coolwarm",
            vmin=0,
            vmax=100 if percentage else table.values.max()
        )

        plt.colorbar(im, ax=ax, label=label)

        # Cluster labels
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(
            [f"Cluster {c}" for c in table.columns]
        )

        # Group labels
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)

        # Add values inside heatmap
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):

                value = table.iloc[i, j]

                text = (
                    f"{value:.{decimals}f}%"
                    if percentage
                    else f"{int(value)}"
                )

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center"
                )

        ax.set_xlabel("Cluster")
        ax.set_ylabel(
            group_by.replace("_", " ").title()
        )

        ax.set_title(
            f"Cluster Distribution by "
            f"{group_by.replace('_', ' ').title()}"
        )

        plt.tight_layout()
        if outpath is not None and is_save:
            Path(outpath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.show()

    return table

def plot_cluster_feature_heatmap(
    df,
    features,
    summary="mean",
    outpath=None,
    is_save = False
):
    """
    Plot standardized feature values for each K-means cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and a 'cluster' column.

    features : list
        Features to include in the heatmap.

    summary : str
        Summary statistic to use: 'mean' or 'median'.
        Default is 'mean'.
    """

    # Validate summary option
    if summary not in ["mean", "median"]:
        raise ValueError("summary must be either 'mean' or 'median'")

    # --------------------------------
    # Calculate cluster summary
    # --------------------------------
    if summary == "mean":
        cluster_summary = (
            df.groupby("cluster")[features]
            .mean()
        )

    else:
        cluster_summary = (
            df.groupby("cluster")[features]
            .median()
        )

    # --------------------------------
    # Standardize across clusters
    # for each feature
    # --------------------------------
    cluster_summary_z = (
        cluster_summary - cluster_summary.mean()
    ) / cluster_summary.std()

    # Features = rows
    # Clusters = columns
    cluster_summary_z = cluster_summary_z.T

    # --------------------------------
    # Plot heatmap
    # --------------------------------
    plt.figure(figsize=(8, 5))

    plt.imshow(
        cluster_summary_z,
        aspect="auto",
        cmap="coolwarm"
    )

    plt.colorbar(label="Relative feature value (z-score)")

    plt.xticks(
        range(len(cluster_summary_z.columns)),
        [f"Cluster {c}" for c in cluster_summary_z.columns]
    )

    plt.yticks(
        range(len(cluster_summary_z.index)),
        cluster_summary_z.index
    )

    # Add z-score values
    for i in range(cluster_summary_z.shape[0]):
        for j in range(cluster_summary_z.shape[1]):

            plt.text(
                j,
                i,
                f"{cluster_summary_z.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title(
        f"Cluster Morphology ({summary.capitalize()} Feature Values)"
    )

    plt.tight_layout()
    if outpath is not None and is_save:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()

    return cluster_summary

def plot_PCA(df):
    plt.figure(figsize=(9, 7))

    # Plot each cluster/subtype combination
    for cluster in sorted(df["cluster"].unique()):
        for subtype in df["subtype"].unique():
            mask = (
                    (df["cluster"] == cluster) &
                    (df["subtype"] == subtype)
            )

            plt.scatter(
                df.loc[mask, "PC1"],
                df.loc[mask, "PC2"],
                c=[f"C{cluster}"],  # colour = cluster
                marker=subtype_markers[subtype],  # shape = subtype
                alpha=0.6,
                s=35
            )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )

    plt.title("PCA of Cell Morphology")

    # Create separate legends
    from matplotlib.lines import Line2D

    # Cluster colour legend
    cluster_legend = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=f"C{cluster}",
            markeredgecolor=f"C{cluster}",
            label=f"Cluster {cluster}"
        )
        for cluster in sorted(df["cluster"].unique())
    ]

    # Subtype shape legend
    subtype_legend = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="",
            color="black",
            markerfacecolor="black",
            label=subtype
        )
        for subtype, marker in subtype_markers.items()
    ]

    legend1 = plt.legend(
        handles=cluster_legend,
        title="Cluster",
        loc="upper right"
    )

    plt.gca().add_artist(legend1)

    plt.legend(
        handles=subtype_legend,
        title="Subtype",
        loc="lower right"
    )

    plt.tight_layout()
    plt.show()

def plot_UMAP(
    df,
    group_by="cell_line",
    group_style="color",
    cluster_style="marker",
    show_cluster=True
):

    valid_options = {"color", "marker"}

    if group_style not in valid_options:
        raise ValueError("group_style must be 'color' or 'marker'")

    if show_cluster:
        if cluster_style not in valid_options:
            raise ValueError("cluster_style must be 'color' or 'marker'")

        if group_style == cluster_style:
            raise ValueError(
                "group_style and cluster_style must be different."
            )

    groups = _epithelioid_first_groups(df, group_by, df[group_by].dropna().unique())

    markers = [
        "o", "^", "s", "D", "P", "X",
        "v", "<", ">", "*"
    ]

    group_colors = {
        group: f"C{i}"
        for i, group in enumerate(groups)
    }

    group_markers = {
        group: markers[i % len(markers)]
        for i, group in enumerate(groups)
    }

    plt.figure(figsize=(9, 7))

    # ==================================================
    # Show clusters
    # ==================================================
    if show_cluster:

        clusters = sorted(df["cluster"].dropna().unique())

        cluster_colors = {
            cluster: f"C{i}"
            for i, cluster in enumerate(clusters)
        }

        cluster_markers = {
            cluster: markers[i % len(markers)]
            for i, cluster in enumerate(clusters)
        }

        for group in groups:
            for cluster in clusters:

                mask = (
                    (df[group_by] == group)
                    & (df["cluster"] == cluster)
                )

                # Determine colour
                if group_style == "color":
                    color = group_colors[group]
                else:
                    color = cluster_colors[cluster]

                # Determine marker
                if group_style == "marker":
                    marker = group_markers[group]
                else:
                    marker = cluster_markers[cluster]

                plt.scatter(
                    df.loc[mask, "UMAP1"],
                    df.loc[mask, "UMAP2"],
                    color=color,
                    marker=marker,
                    s=35,
                    alpha=0.65
                )

    # ==================================================
    # Do NOT show clusters
    # ==================================================
    else:

        for group in groups:

            mask = df[group_by] == group

            if group_style == "color":
                color = group_colors[group]
                marker = "o"
            else:
                color = "black"
                marker = group_markers[group]

            plt.scatter(
                df.loc[mask, "UMAP1"],
                df.loc[mask, "UMAP2"],
                color=color,
                marker=marker,
                s=35,
                alpha=0.65
            )

    # ==================================================
    # Group legend
    # ==================================================

    if group_style == "color":

        group_legend = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color=group_colors[group],
                label=group
            )
            for group in groups
        ]

    else:

        group_legend = [
            Line2D(
                [0], [0],
                marker=group_markers[group],
                linestyle="",
                color="black",
                label=group
            )
            for group in groups
        ]

    legend1 = plt.legend(
        handles=group_legend,
        title=group_by.replace("_", " ").title(),
        loc="upper right"
    )

    # ==================================================
    # Cluster legend
    # ==================================================

    if show_cluster:

        plt.gca().add_artist(legend1)

        if cluster_style == "color":

            cluster_legend = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color=cluster_colors[cluster],
                    label=f"Cluster {cluster}"
                )
                for cluster in clusters
            ]

        else:

            cluster_legend = [
                Line2D(
                    [0], [0],
                    marker=cluster_markers[cluster],
                    linestyle="",
                    color="black",
                    label=f"Cluster {cluster}"
                )
                for cluster in clusters
            ]

        plt.legend(
            handles=cluster_legend,
            title="Cluster",
            loc="upper left"
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of Cell Morphology")

    plt.tight_layout()
    plt.show()


def show_cluster_summary(df, features, is_save = False):
    output_dir = Path(__file__).resolve().parent / "results" / "cluster"
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_summary = (
        df.groupby("cluster")[features]
          .agg(["mean", "median", "std"])
          .round(3)
    )

    cluster_summary = cluster_summary.T
    plot_cluster_feature_heatmap(
        df, features, summary="mean",
        outpath=output_dir / "cluster_feature_summary.png", is_save=is_save
    )
    print(cluster_summary)
    cluster_summary.to_csv("./results/cluster_summary.csv")

    cluster_by_line = get_cluster_distribution(
        df,
        group_by="cell_line",
        percentage=True,
        show_heatmap=True,
        outpath=output_dir / "cluster_distribution_by_cell_line.png", is_save=is_save
    )

    # cluster0_images = df.loc[
    #     df["cluster"] == 0,
    #     "processed_path"
    # ]
    # print(cluster0_images.to_list())
