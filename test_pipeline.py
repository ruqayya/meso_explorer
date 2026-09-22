"""Small numerical/data-integrity tests; run: python -m unittest -v."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import disk

from common import fingerprint, metadata, save_settings
from features import masked_glcm, measure_cell
from postprocess import ExternalMaskCleanupConfig, clean_external_mask
from preprocess import detect_top_bar_h, process_image


class PipelineTests(unittest.TestCase):
    def test_metadata(self):
        epi = metadata(Path("Epithelioid Cells 23/23-WT1+CAL/1.png"))
        self.assertEqual(epi["marker_group"], "WT1+CAL")
        self.assertEqual(epi["subtype"], "epithelioid")
        self.assertEqual(
            metadata(Path("A549 Lung cancer cells (NSCLC)/1.png"))["disease_group"], "nsclc_control"
        )
        self.assertEqual(metadata(Path("mystery/1.png"))["disease_group"], "unknown")

    def test_shape_and_intensity_use_same_component(self):
        mask = np.zeros((100, 100), bool)
        rr, cc = disk((50, 50), 15, shape=mask.shape)
        mask[rr, cc] = True
        expected_area = int(mask.sum())
        rgb = np.ones((100, 100, 3)) * 0.8
        rgb[mask] = 0.3
        mask[5:8, 5:8] = True
        rgb[5:8, 5:8] = 1
        values, measured = measure_cell(rgb, mask)
        self.assertEqual(values["area"], expected_area)
        self.assertEqual(measured.sum(), expected_area)
        self.assertEqual(values["component_count"], 2)
        self.assertAlmostEqual(values["intensity_cell_mean"], 0.3)
        self.assertGreater(values["circularity"], 0.85)
        self.assertTrue(values["qc_ok"])

    def test_empty_and_border_masks_are_flagged(self):
        rgb = np.ones((50, 50, 3)) * 0.5
        values, _ = measure_cell(rgb, np.zeros((50, 50), bool))
        self.assertFalse(values["qc_ok"])
        self.assertTrue(np.isnan(values["area"]))
        mask = np.zeros((50, 50), bool)
        mask[:20, :20] = True
        values, _ = measure_cell(rgb, mask)
        self.assertIn("touches_border", values["qc_flags"])

    def test_border_qc_requires_ten_consecutive_pixels_on_one_edge(self):
        rgb = np.ones((50, 50, 3)) * 0.5
        cases = []
        for length in (0, 1, 9, 10, 11):
            mask = np.zeros((50, 50), bool)
            mask[1:20, 10:40] = True
            mask[0, 15 : 15 + length] = True
            cases.append((f"run_{length}", mask, length >= 10))
        # Twelve border pixels split into two short contacts still pass.
        mask = np.zeros((50, 50), bool)
        mask[1:20, 10:40] = True
        mask[0, 10:16] = mask[0, 25:31] = True
        cases.append(("separate_contacts", mask, False))
        # Contacts on different sides of a corner are checked independently.
        mask = np.zeros((50, 50), bool)
        mask[:6, :6] = True
        cases.append(("corner", mask, False))
        for name, mask, expected in cases:
            for rotation in range(4):
                with self.subTest(case=name, rotation=rotation):
                    values, _ = measure_cell(rgb, np.rot90(mask, rotation))
                    self.assertEqual("touches_border" in values["qc_flags"], expected)
                    self.assertEqual(values["qc_ok"], not expected)

    def test_glcm_ignores_background(self):
        rng = np.random.default_rng(42)
        gray = rng.random((40, 40))
        mask = np.zeros(gray.shape, bool)
        mask[10:30, 10:30] = True
        before = masked_glcm(gray, mask)
        gray[~mask] = 1
        self.assertEqual(before, masked_glcm(gray, mask))

    def test_preprocess_preserves_pixels_and_unique_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw, run = base / "raw", base / "run"
            raw.mkdir()
            arr = np.full((100, 80, 3), 150, np.uint8)
            arr[:15] = 0
            self.assertEqual(detect_top_bar_h(arr[:, :, 0], "auto"), 15)
            Image.fromarray(arr).save(raw / "cell.png")
            Image.fromarray(arr).save(raw / "cell.bmp")
            a = process_image(raw / "cell.png", raw, run, "15", 16, 0.12)
            b = process_image(raw / "cell.bmp", raw, run, "15", 16, 0.12)
            self.assertNotEqual(a["processed_path"], b["processed_path"])
            actual = np.asarray(Image.open(run / a["processed_path"]))
            np.testing.assert_array_equal(actual, arr[15:])
            self.assertEqual((a["crop_x"], a["crop_y"]), (0, 15))

    def test_settings_protect_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            save_settings(directory, "segment", {"threshold": 0.6})
            save_settings(directory, "segment", {"threshold": 0.6})
            with self.assertRaises(ValueError):
                save_settings(directory, "segment", {"threshold": 0.7})

    def test_resume_rejects_stale_settings_and_changed_files(self):
        from segment import can_resume

        with tempfile.TemporaryDirectory() as directory:
            mask = Path(directory) / "mask.png"
            mask.write_bytes(b"test mask")
            raw_mask = Path(directory) / "raw.png"
            raw_mask.write_bytes(b"raw mask")
            record = {
                "status": "ok",
                "settings_hash": "config-a",
                "image_sha256": "image-a",
                "mask_sha256": fingerprint(mask),
                "raw_mask_sha256": fingerprint(raw_mask),
            }
            self.assertTrue(can_resume(record, "image-a", "config-a", mask, raw_mask))
            self.assertFalse(can_resume(record, "image-a", "config-b", mask, raw_mask))
            self.assertFalse(can_resume(record, "image-b", "config-a", mask, raw_mask))
            mask.write_bytes(b"changed mask")
            self.assertFalse(can_resume(record, "image-a", "config-a", mask, raw_mask))
            mask.write_bytes(b"test mask")
            raw_mask.write_bytes(b"changed raw mask")
            self.assertFalse(can_resume(record, "image-a", "config-a", mask, raw_mask))

    def test_postprocess_removes_speckles_and_fills_tiny_holes(self):
        mask = np.zeros((120, 120), bool)
        rr, cc = disk((60, 60), 25, shape=mask.shape)
        mask[rr, cc] = True
        mask[5, 5] = True
        mask[60, 60] = False
        original = mask.copy()
        cleaned, metrics = clean_external_mask(mask, ExternalMaskCleanupConfig())
        self.assertFalse(cleaned[5, 5])
        self.assertTrue(cleaned[60, 60])
        self.assertEqual(metrics["external_mask_cleanup"], "light")
        self.assertEqual(metrics["external_mask_components_after_cleanup"], 1)
        np.testing.assert_array_equal(mask, original)

    def test_postprocess_disabled_preserves_every_pixel(self):
        mask = np.random.default_rng(7).random((40, 50)) > 0.7
        cleaned, metrics = clean_external_mask(mask, ExternalMaskCleanupConfig(mode="none"))
        np.testing.assert_array_equal(mask, cleaned)
        self.assertEqual(metrics["external_mask_cleanup"], "none")
        self.assertEqual(metrics["external_mask_cleanup_area_delta_fraction"], 0)

    def test_postprocess_guard_preserves_thin_and_empty_masks(self):
        mask = np.zeros((80, 80), bool)
        mask[38:41, 20:60] = True
        cleaned, metrics = clean_external_mask(mask, ExternalMaskCleanupConfig())
        np.testing.assert_array_equal(mask, cleaned)
        self.assertTrue(metrics["external_mask_cleanup_guard_reverted"])
        empty = np.zeros_like(mask)
        cleaned, _ = clean_external_mask(empty, ExternalMaskCleanupConfig())
        self.assertFalse(cleaned.any())

    def test_segmentation_cli_defaults_and_disabled_postprocessing(self):
        from segment import main

        mask = np.zeros((80, 80), bool)
        mask[20:60, 20:60] = True
        mask[5, 5] = True
        predictor = Mock()
        predictor.predict.return_value = (mask, {"text_score": 0.9})
        fake_sam = SimpleNamespace(SAM_REVISION="test", Sam31=Mock(return_value=predictor))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sam31": fake_sam}):
            run = Path(directory)
            (run / "images").mkdir()
            Image.fromarray(np.full((80, 80, 3), 128, np.uint8)).save(run / "images/cell.png")
            pd.DataFrame(
                [{"image_id": "cell.png", "processed_path": "images/cell.png", "preprocess_status": "ok"}]
            ).to_csv(run / "images.csv", index=False)
            checkpoint = run / "fake-checkpoint"
            checkpoint.write_bytes(b"test")
            argv = ["segment.py", "--run", str(run), "--checkpoint", str(checkpoint), "--device", "cpu"]
            with patch.object(sys, "argv", argv):
                main()
                main()  # Resume without calling SAM again.
            self.assertEqual(predictor.predict.call_count, 1)
            manifest = pd.read_csv(run / "masks.csv")
            self.assertEqual(manifest.loc[0, "external_mask_cleanup"], "light")
            raw = np.asarray(Image.open(run / manifest.loc[0, "raw_mask_path"])) > 127
            clean = np.asarray(Image.open(run / manifest.loc[0, "mask_path"])) > 127
            np.testing.assert_array_equal(raw, mask)
            self.assertFalse(clean[5, 5])
            with patch.object(sys, "argv", argv + ["--no-postproc"]), self.assertRaises(ValueError):
                main()  # A changed mode cannot silently resume old masks.
            with patch.object(sys, "argv", argv + ["--no_postproc", "--overwrite"]):
                main()
            self.assertEqual(predictor.predict.call_count, 2)
            manifest = pd.read_csv(run / "masks.csv")
            self.assertEqual(manifest.loc[0, "external_mask_cleanup"], "none")
            actual = np.asarray(Image.open(run / manifest.loc[0, "mask_path"])) > 127
            np.testing.assert_array_equal(actual, mask)

    def test_umap_excludes_labels_and_constant_missing_features(self):
        from plot import prepare_matrix

        frame = pd.DataFrame(
            {
                "area": [1, 2, 3, np.nan],
                "perimeter": [2, 4, 6, 8],
                "solidity": [1, 1, 1, 1],
                "eccentricity": [np.nan] * 4,
                "crop_x": [0, 100, 200, 300],
                "subtype": [1, 2, 3, 4],
            }
        )
        matrix, cols = prepare_matrix(frame)
        self.assertEqual(cols, ["area", "perimeter"])
        self.assertEqual(matrix.shape, (4, 2))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
