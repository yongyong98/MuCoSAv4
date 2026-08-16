from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pairbst.data_checks import (
    CENTER_BOX_XYXY,
    verify_images_and_centers,
    verify_release_files_against_manifest,
)
from pairbst.datasets import ROIRecord


def _record(path: Path) -> ROIRecord:
    return ROIRecord(
        roi_path=path,
        relative_path="part_01/Sarcoma_0001_roi_1.png",
        slide_name="Sarcoma_0001.svs_roi_1.png",
        wsi_name="Sarcoma_0001.svs",
        patient_idx="1",
        roi_idx="1",
        diagnosis="Diagnosis A",
        differentiation="Uncertain",
        growth_pattern="Spindle",
    )


class ImageDataChecksTests(unittest.TestCase):
    def test_release_integrity_hashes_every_manifest_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi = root / "Sarcoma_0001_roi_1.png"
            metadata = root / "metadata.csv"
            manifest = root / "required.tsv"
            output = root / "qa"
            roi.write_bytes(b"synthetic-png-payload")
            metadata.write_text("diagnosis,patient_idx\nDiagnosis A,1\n", encoding="utf-8")

            def row(file_id: int, path: Path) -> str:
                payload = path.read_bytes()
                digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                return (
                    f"1\t{file_id}\t{path.name}\t{len(payload)}\t{digest}\t"
                    f"https://example.invalid/{path.name}"
                )

            manifest.write_text(
                "\n".join((row(10, roi), row(11, metadata))) + "\n",
                encoding="utf-8",
            )
            result = verify_release_files_against_manifest(
                [_record(roi)],
                metadata_csv=metadata,
                required_manifest=manifest,
                output_directory=output,
                workers=1,
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["matched_files"], 2)
        self.assertEqual(result["failure_count"], 0)

    def test_omitted_legacy_root_is_explicitly_not_compared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roi = Path(tmp) / "Sarcoma_0001_roi_1.png"
            Image.new("RGB", (4096, 4096), (12, 34, 56)).save(roi)
            result = verify_images_and_centers([_record(roi)])
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["comparison_performed"])
        self.assertEqual(result["center_equivalence_status"], "NOT_PERFORMED")
        self.assertEqual(result["legacy_centers_found"], 0)
        self.assertEqual(result["legacy_centers_pixel_exact"], 0)

    def test_supplied_empty_legacy_root_fails_instead_of_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi = root / "Sarcoma_0001_roi_1.png"
            legacy = root / "legacy"
            legacy.mkdir()
            Image.new("RGB", (4096, 4096), (12, 34, 56)).save(roi)
            result = verify_images_and_centers(
                [_record(roi)], legacy_center_root=legacy
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["comparison_performed"])
        self.assertEqual(result["center_equivalence_status"], "FAIL")

    def test_supplied_legacy_center_is_pixel_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi = root / "Sarcoma_0001_roi_1.png"
            legacy = root / "legacy"
            legacy.mkdir()
            with Image.new("RGB", (4096, 4096), (12, 34, 56)) as source:
                source.save(roi)
                with source.crop(CENTER_BOX_XYXY) as center:
                    center.save(legacy / "SMC_Sarcoma_0001.svs_roi_1.png")
            result = verify_images_and_centers(
                [_record(roi)], legacy_center_root=legacy
            )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["comparison_performed"])
        self.assertEqual(result["center_equivalence_status"], "PASS")
        self.assertEqual(result["legacy_centers_pixel_exact"], 1)


if __name__ == "__main__":
    unittest.main()
