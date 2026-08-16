from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pairbst.manifest import (
    ManifestValidationError,
    build_manifest,
    make_patient_uid,
    validate_manifest,
)
from pairbst.splits import (
    SplitValidationError,
    apply_patient_folds,
    import_legacy_split,
    make_patient_folds,
    summarize_cv_folds,
    summarize_legacy_split,
)


WORKSPACE = Path(__file__).resolve().parents[2]
PUBLIC_METADATA = (
    WORKSPACE
    / "data"
    / "figshare"
    / "PAIR_BST_8223469_v1"
    / "metadata"
    / "Sarcoma_WSI_and_ROI_Metadata.csv"
)
PUBLIC_ROI_ROOT = PUBLIC_METADATA.parent.parent / "roi_4096"
RECOVERY_LEGACY_SPLIT = Path(
    os.environ.get("PAIRBST_RECOVERY_LEGACY_SPLIT", "__not_configured__")
)


def _synthetic_dataset(root: Path) -> tuple[Path, Path]:
    metadata_path = root / "metadata" / "Sarcoma_WSI_and_ROI_Metadata.csv"
    roi_root = root / "roi_4096"
    part = roi_root / "part_01"
    part.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    diagnoses = (
        "Diagnosis A",
        "Diagnosis B",
        "Diagnosis C",
        "Solitaty Fibrous Tumor",
    )
    wsi_counter = 1
    for diagnosis_number, diagnosis in enumerate(diagnoses):
        for patient_idx in range(1, 4):
            # One patient has two WSI to exercise patient grouping above slide level.
            wsi_count = 2 if diagnosis_number == 0 and patient_idx == 1 else 1
            for local_wsi in range(wsi_count):
                wsi = f"Sarcoma_{wsi_counter:04d}"
                wsi_counter += 1
                roi_count = 1 + ((diagnosis_number + patient_idx + local_wsi) % 3)
                for roi_idx in range(1, roi_count + 1):
                    roi_file = f"{wsi}_roi_{roi_idx}.png"
                    (part / roi_file).touch()
                    rows.append(
                        {
                            "slide_name": f"{wsi}.svs_roi_{roi_idx}.png",
                            "patient_idx": patient_idx,
                            "roi_idx": roi_idx,
                            "diagnosis_raw": diagnosis,
                            "diagnosis": diagnosis,
                            "differentiation": (
                                "Uncertain" if roi_idx % 2 else "Myogenic"
                            ),
                            "growth_pattern": (
                                "Spindle" if (patient_idx + roi_idx) % 2 else "Epithelioid"
                            ),
                            "top_left_x": 0,
                            "top_left_y": 0,
                            "top_right_x": 4095,
                            "top_right_y": 0,
                            "bottom_left_x": 0,
                            "bottom_left_y": 4095,
                            "bottom_right_x": 4095,
                            "bottom_right_y": 4095,
                        }
                    )
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path, roi_root


class ManifestTests(unittest.TestCase):
    def test_manifest_maps_public_names_and_preserves_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, roi_root = _synthetic_dataset(Path(temporary))
            manifest = build_manifest(metadata, roi_root)
            summary = validate_manifest(manifest)

            self.assertEqual(summary.n_patient, 12)
            self.assertEqual(summary.n_diagnosis, 4)
            self.assertEqual(summary.n_wsi, 13)
            self.assertIn("Solitaty Fibrous Tumor", set(manifest["diagnosis"]))
            typo_row = manifest[manifest["diagnosis"] == "Solitaty Fibrous Tumor"].iloc[0]
            self.assertEqual(
                typo_row["patient_uid"],
                "Solitaty Fibrous Tumor::patient_idx=1",
            )
            self.assertTrue(typo_row["roi_path"].startswith("roi_4096/part_01/"))
            self.assertEqual(typo_row["wsi_id"].split(".")[-1], "svs")

    def test_manifest_rejects_unmatched_roi_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, roi_root = _synthetic_dataset(Path(temporary))
            extra = roi_root / "part_01" / "Sarcoma_9999_roi_1.png"
            extra.touch()
            with self.assertRaisesRegex(ManifestValidationError, "absent from metadata"):
                build_manifest(metadata, roi_root)

    def test_patient_uid_is_diagnosis_scoped(self) -> None:
        self.assertNotEqual(
            make_patient_uid("Diagnosis A", 1),
            make_patient_uid("Diagnosis B", 1),
        )

    def test_manifest_rejects_rowwise_roi_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, roi_root = _synthetic_dataset(Path(temporary))
            manifest = build_manifest(metadata, roi_root)
            tampered = manifest.copy()
            tampered.loc[tampered.index[:2], "roi_path"] = list(
                reversed(tampered.loc[tampered.index[:2], "roi_path"].tolist())
            )
            with self.assertRaisesRegex(
                ManifestValidationError, "row identities are inconsistent"
            ):
                validate_manifest(tampered)


class SplitTests(unittest.TestCase):
    def test_three_fold_is_deterministic_grouped_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, roi_root = _synthetic_dataset(Path(temporary))
            manifest = build_manifest(metadata, roi_root)
            first = make_patient_folds(manifest, seed=1234)
            second = make_patient_folds(manifest, seed=1234)
            pd.testing.assert_frame_equal(first, second)

            split_manifest = apply_patient_folds(manifest, first)
            summaries = summarize_cv_folds(split_manifest)
            self.assertEqual([summary.n_patient for summary in summaries], [4, 4, 4])
            self.assertTrue(all(summary.n_diagnosis == 4 for summary in summaries))
            self.assertEqual(
                split_manifest.groupby("patient_uid")["fold"].nunique().max(), 1
            )
            multi_wsi_patient = "Diagnosis A::patient_idx=1"
            self.assertEqual(
                split_manifest.loc[
                    split_manifest["patient_uid"] == multi_wsi_patient, "fold"
                ].nunique(),
                1,
            )

    def test_split_rejects_diagnosis_with_fewer_than_three_patients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, roi_root = _synthetic_dataset(Path(temporary))
            manifest = build_manifest(metadata, roi_root)
            removed_patient = "Diagnosis A::patient_idx=3"
            reduced = manifest[manifest["patient_uid"] != removed_patient].copy()
            with self.assertRaisesRegex(SplitValidationError, "Diagnosis A=2"):
                make_patient_folds(reduced)

    def test_legacy_split_maps_wsi_names_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, roi_root = _synthetic_dataset(root)
            manifest = build_manifest(metadata, roi_root)
            patient_folds = make_patient_folds(manifest, seed=55)
            synthetic = apply_patient_folds(manifest, patient_folds)
            legacy = synthetic.drop(
                columns=[
                    "patient_uid",
                    "wsi_id",
                    "roi_uid",
                    "roi_file",
                    "roi_path",
                    "roi_part",
                    "fold",
                ]
            ).copy()
            wsi_by_slide = synthetic["wsi_id"].map(lambda value: f"SMC_{value}")
            legacy["slide_name"] = wsi_by_slide
            split_by_patient = patient_folds.set_index("patient_uid")["fold"].map(
                {0: "train", 1: "val", 2: "test"}
            )
            legacy["split"] = synthetic["patient_uid"].map(split_by_patient)
            legacy_path = root / "legacy.csv"
            legacy.to_csv(legacy_path, index=False)

            mapped = import_legacy_split(
                manifest,
                legacy_path,
                verify_historical_counts=False,
            )
            summary = summarize_legacy_split(mapped, verify_historical_counts=False)
            self.assertEqual(len(mapped), len(manifest))
            self.assertEqual(set(mapped["legacy_split"]), {"train", "val", "test"})
            self.assertEqual(
                sum(value["n_patient"] for value in summary["counts"].values()), 12
            )
            self.assertEqual(mapped.groupby("patient_uid")["legacy_split"].nunique().max(), 1)


@unittest.skipUnless(
    PUBLIC_METADATA.is_file() and PUBLIC_ROI_ROOT.is_dir(),
    "public PAIR-BST dataset is not available in this workspace",
)
class PublicDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_manifest(PUBLIC_METADATA, PUBLIC_ROI_ROOT)

    def test_public_manifest_and_cv3_contract(self) -> None:
        summary = validate_manifest(self.manifest)
        self.assertEqual(summary.n_roi, 2252)
        self.assertEqual(summary.n_wsi, 470)
        self.assertEqual(summary.n_patient, 268)
        self.assertEqual(summary.n_diagnosis, 33)
        self.assertIn("Solitaty Fibrous Tumor", set(self.manifest["diagnosis"]))

        folds = make_patient_folds(self.manifest)
        split_manifest = apply_patient_folds(self.manifest, folds)
        summaries = summarize_cv_folds(split_manifest)
        self.assertEqual([summary.n_patient for summary in summaries], [90, 89, 89])
        self.assertTrue(all(summary.n_diagnosis == 33 for summary in summaries))
        self.assertEqual(split_manifest.groupby("patient_uid")["fold"].nunique().max(), 1)

    @unittest.skipUnless(
        RECOVERY_LEGACY_SPLIT.is_file(),
        "recovered historical split CSV is not available",
    )
    def test_recovered_legacy_split_exact_counts(self) -> None:
        mapped = import_legacy_split(self.manifest, RECOVERY_LEGACY_SPLIT)
        summary = summarize_legacy_split(mapped)
        self.assertEqual(summary["counts"]["train"]["n_patient"], 133)
        self.assertEqual(summary["counts"]["val"]["n_patient"], 27)
        self.assertEqual(summary["counts"]["test"]["n_patient"], 108)
        self.assertEqual(summary["counts"]["train"]["n_roi"], 1161)
        self.assertEqual(summary["counts"]["val"]["n_roi"], 199)
        self.assertEqual(summary["counts"]["test"]["n_roi"], 892)


if __name__ == "__main__":
    unittest.main()
