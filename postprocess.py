"""Guarded light morphology from meso_explorer.pipeline.clean_external_mask."""

import math
from dataclasses import dataclass

import numpy as np
from skimage import measure, morphology


@dataclass(frozen=True)
class ExternalMaskCleanupConfig:
    """Conservative cleanup for externally generated binary masks.

    The default ``light`` mode is intentionally small: robust thresholding happens
    during mask load, then this cleanup removes only tiny speckles/holes, applies
    radius-5 opening/closing, and keeps a dominant central/largest component only
    when fragments are clearly minor. Area-change guards prevent materially
    reshaping SAM masks.
    """

    mode: str = "light"
    min_object_fraction: float = 0.001
    min_hole_fraction: float = 0.01
    max_relative_area_change: float = 0.08
    max_absolute_area_change_fraction: float = 0.010

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


def _component_count(mask: np.ndarray) -> int:
    return int(measure.label(mask).max())


def _area_change_ok(
    original_area: int, new_area: int, n_pixels: int, config: ExternalMaskCleanupConfig
) -> bool:
    if original_area <= 0:
        return new_area == 0
    abs_change = abs(new_area - original_area)
    return (
        abs_change / float(original_area) <= config.max_relative_area_change
        or abs_change / float(n_pixels) <= config.max_absolute_area_change_fraction
    )


def _dominant_central_component(mask: np.ndarray, min_object_area: int) -> tuple[np.ndarray, bool]:
    labels = measure.label(mask)
    if labels.max() <= 1:
        return mask, False
    h, w = mask.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    diag = math.hypot(h, w)
    regions = measure.regionprops(labels)
    total_area = float(sum(r.area for r in regions))
    best = max(
        regions,
        key=lambda r: (
            r.area
            * (
                0.35
                + 0.65 * (1.0 - min(1.0, math.hypot(r.centroid[0] - cy, r.centroid[1] - cx) / (diag / 2.0)))
            )
        ),
    )
    largest = max(regions, key=lambda r: r.area)
    minor_area = total_area - float(best.area)
    keep_best = best.area / max(1.0, total_area) >= 0.85 or (
        best.label == largest.label
        and best.area / max(1.0, total_area) >= 0.72
        and minor_area <= max(min_object_area * 4, 0.04 * mask.size)
    )
    if keep_best:
        return labels == best.label, True
    return mask, False


def clean_external_mask(
    mask: np.ndarray,
    config: ExternalMaskCleanupConfig,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    """Apply guarded, light morphology to an external mask without material distortion."""
    mask = mask.astype(bool)
    n_pixels = int(mask.size)
    original_area = int(mask.sum())
    before_components = _component_count(mask)
    if not config.enabled:
        return mask, {
            "external_mask_cleanup": "none",
            "external_mask_area_before_cleanup": original_area,
            "external_mask_area_after_cleanup": original_area,
            "external_mask_cleanup_area_delta_fraction": 0.0,
            "external_mask_components_before_cleanup": before_components,
            "external_mask_components_after_cleanup": before_components,
            "external_mask_component_reduced": False,
            "external_mask_cleanup_guard_reverted": False,
        }

    min_object_area = max(4, round(config.min_object_fraction * n_pixels))
    min_hole_area = max(6, round(config.min_hole_fraction * n_pixels))

    base = morphology.remove_small_objects(mask, min_size=min_object_area)
    base = morphology.remove_small_holes(base, area_threshold=min_hole_area)
    if original_area > 0 and not base.any():
        base = mask.copy()

    morphed = morphology.opening(base, morphology.disk(5))
    morphed = morphology.closing(morphed, morphology.disk(5))
    morphed = morphology.remove_small_objects(morphed, min_size=min_object_area)
    morphed = morphology.remove_small_holes(morphed, area_threshold=min_hole_area)
    if (
        original_area > 0
        and morphed.any()
        and _area_change_ok(original_area, int(morphed.sum()), n_pixels, config)
    ):
        cleaned = morphed
        guard_reverted = False
    else:
        cleaned = base if base.any() or original_area == 0 else mask.copy()
        guard_reverted = bool(original_area > 0 and not np.array_equal(cleaned, morphed))

    component_reduced = False
    component_candidate, component_reduced = _dominant_central_component(cleaned, min_object_area)
    if component_reduced and _area_change_ok(original_area, int(component_candidate.sum()), n_pixels, config):
        cleaned = component_candidate
    elif component_reduced:
        component_reduced = False
        guard_reverted = True

    after_area = int(cleaned.sum())
    area_delta_fraction = (after_area - original_area) / float(n_pixels) if n_pixels else 0.0
    return cleaned.astype(bool), {
        "external_mask_cleanup": config.mode,
        "external_mask_cleanup_min_object_area": min_object_area,
        "external_mask_cleanup_min_hole_area": min_hole_area,
        "external_mask_area_before_cleanup": original_area,
        "external_mask_area_after_cleanup": after_area,
        "external_mask_cleanup_area_delta_fraction": area_delta_fraction,
        "external_mask_components_before_cleanup": before_components,
        "external_mask_components_after_cleanup": _component_count(cleaned),
        "external_mask_component_reduced": component_reduced,
        "external_mask_cleanup_guard_reverted": guard_reverted,
    }
