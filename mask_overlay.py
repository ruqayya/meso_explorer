"""Save original crops beside their segmentation overlays from a pipeline run."""

import argparse
from pathlib import Path
import cv2
import numpy as np

from common import read_csv, ensure_dir


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Overlays a binary mask onto a brightfield image with transparency."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError("Mask and image dimensions differ")
    # Ensure image is 3-channel BGR
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    # Create a colored mask overlay
    mask_boolean = mask > 0
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask_boolean] = color

    # Blend original image and colored mask where mask exists
    blended = cv2.addWeighted(overlay, 1.0 - alpha, colored_mask, alpha, 0)

    # Keep the original image intact outside the mask
    overlay[mask_boolean] = blended[mask_boolean]

    return overlay


def process_overlays(run_dir: Path, alpha: float = 0.4, color: tuple = (0, 255, 0)) -> None:
    """Save comparisons with the original crop on the left and overlay on the right."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    run_dir = run_dir.resolve()

    images_csv = run_dir / "images.csv"
    masks_csv = run_dir / "masks.csv"

    if not images_csv.exists() or not masks_csv.exists():
        raise FileNotFoundError(f"Missing images.csv or masks.csv in {run_dir}")

    # Read CSVs and set index for matching
    images_df = read_csv(run_dir / "images.csv")
    masks_df = read_csv(run_dir / "masks.csv").set_index("image_id")

    # Join metadata
    df = images_df.join(masks_df, on="image_id", rsuffix="_mask")

    output_dir = ensure_dir(run_dir / "overlays")
    count = 0
    skipped = 0

    # Iterate through matched image and mask metadata
    for _, row in df.iterrows():
        if row["preprocess_status"] != "ok" or row["status"] not in {"ok", "empty"}:
            skipped += 1
            continue
        # Construct full file paths
        rel_path = Path(row["processed_path"]).relative_to("images")

        # there are two .png, remove last .png from rel_path
        if rel_path.suffix == ".png" and rel_path.name.endswith(".png"):
            rel_path = rel_path.with_name(rel_path.stem)

        img_path = run_dir / row["processed_path"]
        mask_path = run_dir / row["mask_path"]

        if not img_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        # Read image and binary mask
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Could not decode image or mask for {row['image_id']}")

        # Generate overlay
        overlay = create_overlay(image, mask, alpha=alpha, color=color)
        comparison = np.concatenate((image, overlay), axis=1)

        # Save result mirroring directory structure
        out_path = output_dir / rel_path
        ensure_dir(out_path.parent)
        ok, encoded = cv2.imencode(".png", comparison)
        if not ok:
            raise OSError(f"PNG encoding failed for {out_path}")
        encoded.tofile(out_path)
        count += 1

    print(f"Successfully generated {count} overlay images in {output_dir}; skipped {skipped}")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Alpha transparency for the mask overlay (0.0 to 1.0, default: 0.4).")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be in [0, 1]")

    process_overlays(args.run, alpha=args.alpha)


if __name__ == "__main__":
    main()
