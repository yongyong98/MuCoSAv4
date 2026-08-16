from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import h5py

from pairbst.cli import main
from pairbst.datasets import ROIRecord
from pairbst.features.h5store import ResumableFeatureStore
from pairbst.models.registry import get_model_spec, get_transform_spec
from pairbst.pipeline import (
    ExecutionHoldError,
    PipelineContext,
    align_feature_file,
    build_report,
    classification_bootstrap_ci_by_seed,
    extract_features,
    file_identity,
    load_feature_matrix,
    require_execution_permission,
    require_exact_stage_manifest_inputs,
    run_statistics,
    verify_compatible_feature_lineage,
    verify_stage_manifest,
)


def _record(index: int, root: Path, *, label: str) -> ROIRecord:
    return ROIRecord(
        roi_path=root / f"roi_{index}.png",
        relative_path=f"roi_{index}.png",
        slide_name=f"slide_{index}.svs_roi_{index}.png",
        wsi_name=f"slide_{index}.svs",
        patient_idx=str(index),
        roi_idx=str(index),
        diagnosis=label,
        differentiation="D",
        growth_pattern="G",
    )


class PipelineSafetyTests(unittest.TestCase):
    def test_primary_overrides_and_sensitivity_report_collision_are_blocked(self) -> None:
        benchmark_root = Path(__file__).resolve().parents[1]
        context = PipelineContext.load(
            benchmark_root / "configs" / "paths.example.yaml",
            benchmark_root / "configs" / "protocol_cv3_independent_seed_oof_v1.yaml",
            benchmark_root / "configs" / "models.yaml",
            benchmark_root / "configs" / "comparisons.yaml",
        )
        with self.assertRaisesRegex(ValueError, "frozen per-model batch sizes"):
            extract_features(context, batch_size=1, dry_run=True)
        with self.assertRaisesRegex(ValueError, "exactly 10000"):
            run_statistics(context, n_bootstrap=10, dry_run=True)
        report_plan = build_report(
            context,
            profile="legacy_common_224",
            dry_run=True,
        )
        self.assertEqual(report_plan["profile"], "legacy_common_224")
        self.assertEqual(
            Path(report_plan["output_directory"]),
            context.final_dir / "sensitivity" / "legacy_common_224",
        )

    def test_cross_run_stage_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction_a = root / "extract_a.json"
            extraction_b = root / "extract_b.json"
            feature_a = root / "feature_a.h5"
            feature_b = root / "feature_b.h5"
            context_file = root / "protocol.yaml"
            classification_stage = root / "classification.json"
            retrieval_stage = root / "retrieval.json"
            for path, payload in (
                (extraction_a, "extract-a"),
                (extraction_b, "extract-b"),
                (feature_a, "feature-a"),
                (feature_b, "feature-b"),
                (context_file, "protocol"),
                (classification_stage, "classification"),
                (retrieval_stage, "retrieval"),
            ):
                path.write_text(payload, encoding="utf-8")
            classification = {
                "input_stage_manifests": [file_identity(extraction_a)],
                "input_identities": [
                    file_identity(feature_a),
                    file_identity(context_file),
                ],
            }
            compatible_retrieval = {
                "input_stage_manifests": [file_identity(extraction_a)],
                "input_identities": [
                    file_identity(feature_a),
                    file_identity(context_file),
                ],
            }
            verify_compatible_feature_lineage(classification, compatible_retrieval)
            mixed_retrieval = {
                "input_stage_manifests": [file_identity(extraction_b)],
                "input_identities": [
                    file_identity(feature_b),
                    file_identity(context_file),
                ],
            }
            with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                verify_compatible_feature_lineage(classification, mixed_retrieval)
            statistics = {
                "input_stage_manifests": [file_identity(classification_stage)]
            }
            with self.assertRaisesRegex(ValueError, "selected upstream runs"):
                require_exact_stage_manifest_inputs(
                    statistics,
                    [classification_stage, retrieval_stage],
                    label="statistics manifest",
                )

    def test_stage_manifest_rejects_mutated_bound_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.csv"
            output.write_text("value\n1\n", encoding="utf-8")
            manifest = root / "stage.json"
            manifest.write_text(
                json.dumps(
                    {
                        "action": "synthetic.run",
                        "profile": "primary",
                        "identity_schema": "pairbst.stage_identity.v1",
                        "input_identities": [],
                        "input_stage_manifests": [],
                        "output_identities": [file_identity(output)],
                    }
                ),
                encoding="utf-8",
            )
            verify_stage_manifest(
                manifest,
                expected_action="synthetic.run",
                expected_profile="primary",
            )
            output.write_text("value\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_stage_manifest(
                    manifest,
                    expected_action="synthetic.run",
                    expected_profile="primary",
                )

    def test_execution_hold_blocks_real_work_but_override_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hold = Path(directory) / "EXECUTION_HOLD.json"
            hold.write_text(json.dumps({"hold": True, "reason": "review"}), encoding="utf-8")
            with self.assertRaises(ExecutionHoldError):
                require_execution_permission(hold, action="training")
            result = require_execution_permission(
                hold, action="training", override_hold=True
            )
            self.assertTrue(result["allowed"])
            self.assertTrue(result["override"])

    def test_feature_alignment_uses_roi_uid_not_h5_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # H5 order is deliberately B, A; split order is A, B.
            records = [_record(2, root, label="B"), _record(1, root, label="A")]
            feature_path = root / "features.h5"
            spec = get_model_spec("resnet50_v2")
            provenance = {
                "model_name": spec.name,
                "architecture": spec.architecture,
                "embedding_dim": spec.embedding_dim,
                "checkpoint_sha256": spec.checkpoint.sha256,
                "transform_profile": "official_model_specific",
                "transform": get_transform_spec(
                    spec, "official_model_specific"
                ).as_dict(),
                "pooling_dtype": "float32",
                "encoder_autocast_dtype": None,
                "deterministic_algorithms": True,
                "tf32_allowed": False,
                "dataset_manifest_sha256": "SYNTHETIC",
            }
            store = ResumableFeatureStore(
                feature_path,
                records,
                spec.embedding_dim,
                provenance,
            )
            with store:
                store.write_row(
                    0,
                    center=np.full(spec.embedding_dim, 20, dtype=np.float32),
                    mean=np.full(spec.embedding_dim, 22, dtype=np.float32),
                    max_=np.full(spec.embedding_dim, 24, dtype=np.float32),
                )
                store.write_row(
                    1,
                    center=np.full(spec.embedding_dim, 10, dtype=np.float32),
                    mean=np.full(spec.embedding_dim, 12, dtype=np.float32),
                    max_=np.full(spec.embedding_dim, 14, dtype=np.float32),
                )
            store.finalize()
            rows = []
            for fold, record in enumerate(reversed(records)):
                rows.append(
                    {
                        "roi_uid": record.roi_uid,
                        "patient_uid": record.patient_uid,
                        "diagnosis": record.diagnosis,
                        "differentiation": record.differentiation,
                        "growth_pattern": record.growth_pattern,
                        "fold": fold,
                    }
                )
            # Add a third synthetic row/fold by duplicating neither ROI nor H5 is
            # impossible; alignment validation itself only needs valid fold IDs.
            # Use folds 0/1 then patch the fold check to focus on row alignment.
            split = root / "split.csv"
            pd.DataFrame(rows).to_csv(split, index=False)
            with patch("pairbst.pipeline._load_split_frame") as loader:
                frame = pd.read_csv(split, keep_default_na=False)
                frame["fold"] = frame["fold"].astype(int)
                loader.return_value = frame
                alignment = align_feature_file(
                    feature_path,
                    split,
                    expected_model_name="resnet50_v2",
                    expected_transform_profile="official_model_specific",
                    expected_manifest_sha256="SYNTHETIC",
                )
            matrix = load_feature_matrix(alignment, "center")
            self.assertEqual(matrix.shape, (2, spec.embedding_dim))
            np.testing.assert_array_equal(matrix[:, 0], np.asarray([10, 20]))

            with h5py.File(feature_path, "r+") as handle:
                changed = json.loads(str(handle.attrs["provenance_json"]))
                changed["encoder_autocast_dtype"] = "float16"
                handle.attrs["provenance_json"] = json.dumps(
                    changed, sort_keys=True, separators=(",", ":")
                )
            with patch("pairbst.pipeline._load_split_frame") as loader:
                loader.return_value = frame
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    align_feature_file(
                        feature_path,
                        split,
                        expected_model_name="resnet50_v2",
                        expected_transform_profile="official_model_specific",
                        expected_manifest_sha256="SYNTHETIC",
                    )

    def test_secondary_task_bootstrap_uses_patient_stable_diagnosis_strata(self) -> None:
        # Each patient has two different secondary labels. Stratifying by the
        # task label would be invalid, while diagnosis remains patient-stable.
        rows = []
        for patient, diagnosis in (("p1", "DxA"), ("p2", "DxA"), ("p3", "DxB"), ("p4", "DxB")):
            for roi_index, task_label in enumerate(("low", "high")):
                rows.append(
                    {
                        "protocol_id": "cv3_independent_seed_oof_v1",
                        "model_id": "resnet50_v2",
                        "model": "ResNet",
                        "strategy": "center",
                        "task": "differentiation",
                        "seed": 101,
                        "roi_uid": f"{patient}:{roi_index}",
                        "patient_uid": patient,
                        "diagnosis_stratum": diagnosis,
                        "true_label": task_label,
                        "predicted_label": task_label if roi_index == 0 else "low",
                    }
                )
        summary, distributions = classification_bootstrap_ci_by_seed(
            pd.DataFrame(rows),
            n_bootstrap=40,
            confidence_level=0.95,
            seed=7,
        )
        self.assertEqual(
            set(summary["strata_definition"]),
            {"patient_unique_task_label_signature"},
        )
        self.assertIn("macro_f1", set(summary["metric"]))
        self.assertEqual(set(summary["seed"]), {101})
        self.assertTrue(distributions)

    def test_cli_experiment_dry_run_never_calls_extractor(self) -> None:
        benchmark_root = Path(__file__).resolve().parents[1]
        argv = [
            "features", "extract", "--dry-run", "--model", "uni2_h",
            "--config", str(benchmark_root / "configs" / "paths.example.yaml"),
            "--protocol", str(
                benchmark_root / "configs" / "protocol_cv3_independent_seed_oof_v1.yaml"
            ),
            "--models-config", str(benchmark_root / "configs" / "models.yaml"),
            "--comparisons-config", str(benchmark_root / "configs" / "comparisons.yaml"),
        ]
        with patch("pairbst.pipeline.extract_models_sequentially") as extractor:
            code = main(argv)
        self.assertEqual(code, 0)
        extractor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
