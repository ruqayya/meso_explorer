"""Measure one dominant cell per SAM mask and save visual quality checks."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy
from skimage import color, feature, filters, measure, morphology, segmentation
from tqdm import tqdm

from common import fingerprint, read_csv, settings_hash, write_csv

FEATURE_FAMILIES = {
    "shape": [
        "area",
        "perimeter",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "solidity",
        "extent",
        "circularity",
    ],
    "intensity": [
        "intensity_cell_mean",
        "intensity_cell_std",
        "intensity_cell_p05",
        "intensity_cell_p50",
        "intensity_cell_p95",
        "intensity_background_ring_mean",
        "intensity_cell_minus_ring_mean",
        "intensity_cell_to_ring_std_ratio",
    ],
    "texture": [
        "texture_entropy",
        "texture_grad_mean",
        "texture_grad_std",
        "texture_edge_density",
        "texture_lbp_uniformity",
        "texture_lbp_entropy",
        "texture_glcm_contrast",
        "texture_glcm_homogeneity",
        "texture_glcm_energy",
        "texture_glcm_correlation",
    ],
}
FEATURE_COLUMNS = [name for group in FEATURE_FAMILIES.values() for name in group]


def masked_glcm(gray, mask):
    """Count only pairs with BOTH pixels inside the cell; no filled background."""
    quant = np.minimum((gray * 32).astype(int), 31)
    matrices = []
    for distance in (1, 2, 4):
        for dy, dx in ((0, distance), (distance, distance), (distance, 0), (distance, -distance)):
            h, w = mask.shape
            y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
            if y1 <= y0 or x1 <= x0:
                continue
            valid = mask[y0:y1, x0:x1] & mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            if not valid.any():
                continue
            a = quant[y0:y1, x0:x1][valid]
            b = quant[y0 + dy : y1 + dy, x0 + dx : x1 + dx][valid]
            matrix = np.bincount(a * 32 + b, minlength=1024).reshape(32, 32).astype(float)
            matrix += matrix.T.copy()
            matrices.append(matrix / matrix.sum())
    if not matrices:
        return {f"texture_glcm_{p}": np.nan for p in ("contrast", "homogeneity", "energy", "correlation")}
    glcm = np.stack(matrices, axis=-1)[:, :, None, :]
    return {
        f"texture_glcm_{p}": float(feature.graycoprops(glcm, p).mean())
        for p in ("contrast", "homogeneity", "energy", "correlation")
    }


def has_long_border_contact(mask, min_pixels=14):
    """Require a continuous run on one edge; separate contacts are not added."""
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        changes = np.diff(np.pad(edge.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if np.any(ends - starts >= min_pixels):
            return True
    return False


def measure_cell(rgb, raw_mask):
    gray = color.rgb2gray(rgb)
    labels = measure.label(raw_mask)
    regions = measure.regionprops(labels)
    values = {name: np.nan for name in FEATURE_COLUMNS}
    if not regions:
        return values | {"qc_ok": False, "qc_flags": "empty_mask", "component_count": 0}, raw_mask
    region = max(regions, key=lambda r: r.area)
    mask = labels == region.label
    area, perimeter = float(region.area), float(region.perimeter)
    area_fraction = area / mask.size
    selected_fraction = area / raw_mask.sum()
    border = has_long_border_contact(mask)
    flags = []
    if area_fraction < 0.012:
        flags.append("tiny_mask")
    if area_fraction > 0.78:
        flags.append("large_mask")
    if selected_fraction < 0.9:
        flags.append("multiple_components")
    if border:
        flags.append("touches_border")
    values.update(
        area=area,
        perimeter=perimeter,
        equivalent_diameter_area=region.equivalent_diameter_area,
        major_axis_length=region.axis_major_length,
        minor_axis_length=region.axis_minor_length,
        eccentricity=region.eccentricity,
        solidity=region.solidity,
        extent=region.extent,
        circularity=4 * np.pi * area / perimeter**2 if perimeter else np.nan,
        qc_ok=not flags,
        qc_flags=";".join(flags) or "ok",
        component_count=len(regions),
        area_fraction=area_fraction,
        selected_fraction=selected_fraction,
    )
    inside = gray[mask]
    ring = morphology.binary_dilation(mask, morphology.disk(5)) & ~raw_mask
    outside = gray[ring]
    ring_mean, ring_std = (float(outside.mean()), float(outside.std())) if outside.size else (np.nan, np.nan)
    values.update(
        intensity_cell_mean=inside.mean(),
        intensity_cell_std=inside.std(),
        intensity_cell_p05=np.percentile(inside, 5),
        intensity_cell_p50=np.median(inside),
        intensity_cell_p95=np.percentile(inside, 95),
        intensity_background_ring_mean=ring_mean,
        intensity_cell_minus_ring_mean=inside.mean() - ring_mean,
        intensity_cell_to_ring_std_ratio=inside.std() / ring_std if ring_std > 0 else np.nan,
    )
    if area >= 16:
        gray8 = np.round(gray * 255).astype(np.uint8)
        hist = np.bincount(gray8[mask], minlength=256)
        grad = filters.sobel(gray)
        # Erode by one pixel so neighboring background does not dominate local texture.
        interior = morphology.binary_erosion(mask, morphology.disk(1))
        if interior.any():
            g = grad[interior]
            threshold = filters.threshold_otsu(g) if np.ptp(g) else g[0]
            lbp = feature.local_binary_pattern(gray8, P=8, R=1, method="uniform")[interior]
            lbp_hist = np.bincount(lbp.astype(int), minlength=10) / len(lbp)
            values.update(
                texture_grad_mean=g.mean(),
                texture_grad_std=g.std(),
                texture_edge_density=np.mean(g > threshold),
                texture_lbp_uniformity=np.sum(lbp_hist**2),
                texture_lbp_entropy=entropy(lbp_hist, base=2),
            )
        values["texture_entropy"] = entropy(hist, base=2)
        values.update(masked_glcm(gray, mask))
    return values, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--qc-sample", type=int, default=36)
    args = parser.parse_args()
    if args.qc_sample < 0:
        parser.error("qc-sample must be nonnegative")
    run = args.run.resolve()
    images = read_csv(run / "images.csv")
    masks = read_csv(run / "masks.csv").set_index("image_id")
    signature = settings_hash(json.loads((run / "segment_settings.json").read_text()))
    if masks.index.duplicated().any():
        parser.error("Duplicate image IDs in masks.csv")
    rows = []
    for row in tqdm(images.to_dict("records"), desc="Features"):
        record = row | {name: np.nan for name in FEATURE_COLUMNS}
        record.update(qc_ok=False, qc_flags="missing_mask", feature_status="error")
        try:
            if row["preprocess_status"] != "ok":
                raise ValueError("Preprocessing failed")
            if row["image_id"] not in masks.index:
                raise ValueError("No segmentation record")
            seg = masks.loc[row["image_id"]]
            if seg["status"] not in {"ok", "empty"}:
                raise ValueError(f"Segmentation status: {seg['status']}")
            if seg["settings_hash"] != signature:
                raise ValueError("Segmentation settings changed; rerun segment.py")
            if fingerprint(run / row["processed_path"]) != seg["image_sha256"]:
                raise ValueError("Image changed since segmentation; rerun segment.py")
            if fingerprint(run / seg["mask_path"]) != seg["mask_sha256"]:
                raise ValueError("Mask changed since segmentation; rerun segment.py")
            with Image.open(run / row["processed_path"]) as image:
                rgb = np.asarray(image.convert("RGB")) / 255.0
            with Image.open(run / seg["mask_path"]) as image:
                raw_mask = np.asarray(image.convert("L")) > 127
            if raw_mask.shape != rgb.shape[:2]:
                raise ValueError("Mask and image dimensions differ")
            values, mask = measure_cell(rgb, raw_mask)
            dest = Path("cell_masks") / Path(row["processed_path"]).relative_to("images")
            (run / dest).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(run / dest)
            record.update(
                values,
                **{key: value for key, value in seg.items() if key.startswith("external_mask_")},
                raw_mask_path=seg.get("raw_mask_path", seg["mask_path"]),
                mask_path=seg["mask_path"],
                cell_mask_path=dest.as_posix(),
                feature_status="ok",
                error="",
            )
        except (ValueError, OSError) as exc:
            record.update(error=str(exc))
        rows.append(record)
    frame = pd.DataFrame(rows)
    write_csv(frame, run / "features.csv")
    (run / "feature_columns.json").write_text(json.dumps(FEATURE_FAMILIES, indent=2) + "\n")
    # Include flagged cells as well as random examples across marker/source groups.
    available = frame[frame.feature_status == "ok"]
    flagged = available[~available.qc_ok].sample(frac=1, random_state=42)
    shuffled = available.sample(frac=1, random_state=42)
    group_examples = shuffled.groupby(["source_folder", "marker_group"], sort=False).head(1)
    selected = (
        pd.concat([flagged.head(args.qc_sample // 3), group_examples, shuffled])
        .drop_duplicates("image_id")
        .head(args.qc_sample)
    )
    qc = run / "qc"
    qc.mkdir(exist_ok=True)
    write_csv(selected, qc / "sample.csv")
    for page in range(0, len(selected), 12):
        subset = selected.iloc[page : page + 12]
        fig, axes = plt.subplots(len(subset), 3, figsize=(9, 2.3 * len(subset)), squeeze=False)
        for axes_row, (_, row) in zip(axes, subset.iterrows()):
            rgb = np.asarray(Image.open(run / row.processed_path).convert("RGB")) / 255.0
            raw = np.asarray(Image.open(run / row.mask_path)) > 127
            mask = np.asarray(Image.open(run / row.cell_mask_path)) > 127
            for ax, arr in zip(
                axes_row, [rgb, raw, segmentation.mark_boundaries(rgb, mask, color=(1, 0.2, 0))]
            ):
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            axes_row[0].set_title(row.image_id, fontsize=7, wrap=True)
            axes_row[1].set_title("Segmentation mask", fontsize=8)
            axes_row[2].set_title("Measured cell: " + row.qc_flags, fontsize=8)
        fig.tight_layout()
        fig.savefig(qc / f"masks_{page // 12 + 1:02d}.png", dpi=120)
        plt.close(fig)
    errors = (frame.feature_status != "ok").sum()
    print(f"Features: {len(frame) - errors}/{len(frame)}; QC passed: {frame.qc_ok.sum()}. See {run / 'qc'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
