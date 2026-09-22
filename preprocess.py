"""Remove the metadata bar and crop the cell, without resizing its pixels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import IMAGE_EXTENSIONS, metadata, read_csv, save_settings, write_csv


def detect_top_bar_h(gray: np.ndarray, mode: str) -> int:
    """Return pixel height of the top metadata bar.

    These images have a *black* header bar with white text (e.g. "Ch01").
    A naive mean-intensity threshold fails because the white text increases the mean.

    Strategy (auto): look for a contiguous prefix of rows that are mostly dark.
    Fall back to the strongest horizontal edge in the top quarter.
    """

    if mode != "auto":
        return max(0, int(mode))

    h = gray.shape[0]
    max_h = int(min(h * 0.30, 220))
    if max_h <= 4:
        return 0

    top = gray[:max_h]

    # Fraction of pixels in each row that are "dark".
    # Header bar is predominantly dark even with white text.
    dark_th = 55
    dark_frac = (top < dark_th).mean(axis=1)

    thr = 0.70
    bar_end = 0
    # extend while rows are mostly dark
    for i in range(max_h):
        if dark_frac[i] >= thr:
            bar_end = i + 1
        else:
            # once we have a plausible bar, stop at first sustained drop
            if bar_end >= 8:
                break

    # Edge-based fallback / refinement: find strongest horizontal boundary.
    means = top.mean(axis=1)
    grad = np.abs(np.diff(means, prepend=means[0]))
    peak = int(np.argmax(grad[1:]) + 1)  # ignore row0

    # If the peak is plausible and above is mostly dark, trust it.
    if 6 <= peak <= max_h - 2 and float(dark_frac[:peak].mean()) >= 0.55:
        bar_end = max(bar_end, peak)

    return int(np.clip(bar_end, 0, max_h))


def find_cell_bbox(gray: np.ndarray, ignore_top_frac: float = 0.12) -> tuple[int, int, int, int] | None:
    """Find a cell-ish bounding box in a brightfield crop.

    We ignore the very top of the image to avoid selecting the embedded image-id digits.
    We also prefer components nearer the image centre (single-cell crops are typically centred).
    """

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Emphasize cell-like structure with background subtraction.
    bg = cv2.GaussianBlur(blur, (0, 0), sigmaX=21, sigmaY=21)
    fg = cv2.absdiff(blur, bg)

    _, th = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        return None

    img_area = h * w
    y_ignore = int(h * ignore_top_frac)
    cx0, cy0 = w / 2.0, h / 2.0

    best_idx = -1
    best_score = -1.0
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        if area < img_area * 0.004 or area > img_area * 0.70:
            continue
        if y < y_ignore:
            continue

        aspect = bw / max(1, bh)
        if aspect > 6 or aspect < 0.15:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        dist = ((cx - cx0) / max(1.0, w)) ** 2 + ((cy - cy0) / max(1.0, h)) ** 2

        # Prefer large components near centre.
        score = float(area) / (1.0 + 6.0 * dist)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None

    x, y, bw, bh, _ = stats[best_idx]
    return int(x), int(y), int(x + bw), int(y + bh)


def process_image(path, root, run, top_bar, pad, ignore_top_frac):
    rel = path.relative_to(root)
    record = {"image_id": rel.as_posix(), "raw_path": str(path), **metadata(rel)}
    try:
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError("Image could not be decoded")
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bar = min(detect_top_bar_h(gray, top_bar), h - 1)
        cropped = raw[bar:]
        bbox = find_cell_bbox(gray[bar:], ignore_top_frac)
        x1, y1, x2, y2 = (0, 0, w, h - bar) if bbox is None else bbox
        if bbox is not None:
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h - bar, y2 + pad)
        image = cropped[y1:y2, x1:x2]
        # Keep the original extension in the basename to avoid a.jpg/a.png collisions.
        dest_rel = Path("images") / rel.parent / (rel.name + ".png")
        dest = run / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError("PNG encoding failed")
        encoded.tofile(dest)
        record.update(
            processed_path=dest_rel.as_posix(),
            preprocess_status="ok",
            error="",
            raw_width=w,
            raw_height=h,
            top_bar_px=bar,
            roi_used=bbox is not None,
            crop_x=x1,
            crop_y=bar + y1,
            width=x2 - x1,
            height=y2 - y1,
        )
    except (ValueError, OSError, cv2.error) as exc:
        record.update(preprocess_status="error", error=str(exc))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw image directory")
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--crop-top-bar", default="auto", help="auto or pixel height (0 disables)")
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--ignore-top-frac", type=float, default=0.12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: round-robin sample across folders")
    parser.add_argument("--metadata", type=Path, help="Optional CSV: image_id plus label columns to override")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.pad < 0 or args.limit < 0 or args.workers < 1 or not 0 <= args.ignore_top_frac < 1:
        parser.error("pad/limit must be nonnegative, workers positive, ignore-top-frac in [0,1)")
    if args.crop_top_bar != "auto" and not args.crop_top_bar.isdigit():
        parser.error("--crop-top-bar must be auto or a nonnegative integer")
    root, run = args.input.resolve(), args.run.resolve()
    if not root.is_dir():
        parser.error(f"Raw directory does not exist: {root}")
    if run == root or run.is_relative_to(root):
        parser.error("--run must be outside --input to avoid processing generated images")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if args.limit:
        groups = {}
        for path in files:
            groups.setdefault(path.parent, []).append(path)
        files = [p for row in zip_longest(*groups.values()) for p in row if p is not None][: args.limit]
    if not files:
        parser.error("No raw images found")
    save_settings(
        run,
        "preprocess",
        {
            "input": str(root),
            "top_bar": args.crop_top_bar,
            "pad": args.pad,
            "ignore_top_frac": args.ignore_top_frac,
            "resize": False,
        },
        args.overwrite,
    )
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(
            tqdm(
                pool.map(
                    lambda p: process_image(p, root, run, args.crop_top_bar, args.pad, args.ignore_top_frac),
                    files,
                ),
                total=len(files),
                desc="Preprocess",
            )
        )
    frame = pd.DataFrame(rows)
    if args.metadata:
        labels = read_csv(args.metadata)
        allowed = {
            "image_id",
            "source_folder",
            "cell_line",
            "marker_label",
            "marker_group",
            "subtype",
            "disease_group",
        }
        if "image_id" not in labels or not set(labels.columns) <= allowed:
            parser.error(f"Metadata columns must be a subset of {sorted(allowed)} including image_id")
        labels["image_id"] = labels.image_id.str.replace("\\", "/", regex=False)
        if labels.image_id.duplicated().any():
            parser.error("Duplicate image_id in metadata")
        labels = labels.set_index("image_id")
        frame = frame.set_index("image_id")
        frame.update(labels.replace("", np.nan))
        frame = frame.reset_index()
    write_csv(frame, run / "images.csv")
    errors = (frame.preprocess_status != "ok").sum()
    print(f"{len(frame) - errors}/{len(frame)} images ready. Manifest: {run / 'images.csv'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
