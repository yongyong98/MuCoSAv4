from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pairbst.audit import (
    AuditCheck,
    audit_cv_split_artifacts,
    audit_dataset,
    audit_deterministic_pilot,
    audit_models,
    audit_release_integrity_artifacts,
    summarize_checks,
)


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def test_release_gate_rehashes_current_bytes_even_if_size_and_mtime_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "roi.png"
            required_manifest = root / "required.tsv"
            summary = root / "release_file_integrity_summary.json"
            rows = root / "release_file_integrity.csv"
            lock_path = root / "release.lock.json"
            target.write_bytes(b"ABCD")
            original_stat = target.stat()
            expected_md5 = hashlib.md5(
                target.read_bytes(), usedforsecurity=False
            ).hexdigest()
            required_manifest.write_text(
                f"1\t1\t{target.name}\t4\t{expected_md5}\t"
                f"https://example.invalid/{target.name}\n",
                encoding="utf-8",
            )
            required_sha = hashlib.sha256(required_manifest.read_bytes()).hexdigest().upper()
            summary.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "file_count": 1,
                        "matched_files": 1,
                        "failure_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            rows.write_text(
                "filename,path,expected_bytes,expected_md5,matches\n"
                f"{target.name},{target},4,{expected_md5},True\n",
                encoding="utf-8",
            )

            def sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest().upper()

            lock_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "required_manifest_sha256": required_sha,
                        "file_count": 1,
                        "matched_files": 1,
                        "summary_sha256": sha(summary),
                        "rows_sha256": sha(rows),
                    }
                ),
                encoding="utf-8",
            )
            dataset_lock = {
                "required_manifest_sha256": required_sha,
                "required_manifest_rows": 1,
            }
            initial = audit_release_integrity_artifacts(
                required_manifest_path=required_manifest,
                dataset_lock=dataset_lock,
                summary_path=summary,
                lock_path=lock_path,
                workers=1,
            )
            self.assertTrue(all(check.status == "PASS" for check in initial))

            target.write_bytes(b"WXYZ")
            os.utime(
                target,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            changed = audit_release_integrity_artifacts(
                required_manifest_path=required_manifest,
                dataset_lock=dataset_lock,
                summary_path=summary,
                lock_path=lock_path,
                workers=1,
            )
        current = next(
            check for check in changed if check.name == "dataset.release_file_md5_integrity"
        )
        self.assertEqual(current.status, "FAIL")

    def test_pilot_status_string_alone_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot.json"
            path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            result = audit_deterministic_pilot(
                path,
                {"models": {}},
                expected_dataset_manifest_sha256="SYNTHETIC",
                expected_roi_uids=["roi-a", "roi-b"],
                expected_records_fingerprint="SYNTHETIC",
                expected_gpu_name="synthetic-gpu",
                expected_cuda_version="0.0",
            )
        self.assertEqual(result.status, "FAIL")

    def test_deferred_blocking_gate_is_not_ready(self) -> None:
        result = summarize_checks([AuditCheck("decode", "DEFERRED", "not run")])
        self.assertEqual(result["status"], "NOT_READY")
        self.assertEqual(result["blocking_failures"], ["decode"])

    def test_nonblocking_warning_does_not_block(self) -> None:
        result = summarize_checks([AuditCheck("note", "WARN", "legacy mismatch", blocking=False)])
        self.assertEqual(result["status"], "READY")

    def test_size_only_model_audit_is_not_an_integrity_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "RetCCL"
            source_dir.mkdir()
            checkpoint = source_dir / "best_ckpt.pth"
            architecture = source_dir / "ResNet.py"
            checkpoint.write_bytes(b"checkpoint")
            architecture.write_bytes(b"architecture")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest().upper()
            paths = {
                "_source_dir": str(root),
                "weights": {"root": str(root), "retccl": str(checkpoint)},
            }
            lock = {
                "models": {
                    "retccl_resnet50": {
                        "expected_bytes": checkpoint.stat().st_size,
                        "expected_sha256": digest(checkpoint),
                        "architecture_source_sha256": digest(architecture),
                    }
                }
            }
            deferred = audit_models(paths, lock, hash_contents=False)
            verified = audit_models(paths, lock, hash_contents=True)
        self.assertEqual([check.status for check in deferred], ["DEFERRED", "DEFERRED"])
        self.assertEqual(summarize_checks(deferred)["status"], "NOT_READY")
        self.assertEqual([check.status for check in verified], ["PASS", "PASS"])

    def test_real_frozen_split_and_all_hash_bindings_pass(self) -> None:
        locks = BENCHMARK_ROOT / "locks"
        required = (
            locks / "folds_cv3_v1.csv",
            locks / "folds_cv3_v1.lock.json",
            locks / "folds_cv3_v1_qa" / "split_qa.json",
            locks / "folds_cv3_v1_qa" / "class_coverage_patients.csv",
            locks / "folds_cv3_v1_qa" / "class_coverage_rois.csv",
            locks / "folds_cv3_v1_qa.lock.json",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("controlled split-audit fixtures are not in the public code release")
        checks = audit_cv_split_artifacts(
            split_csv_path=locks / "folds_cv3_v1.csv",
            split_lock_path=locks / "folds_cv3_v1.lock.json",
            split_qa_path=locks / "folds_cv3_v1_qa" / "split_qa.json",
            split_qa_lock_path=locks / "folds_cv3_v1_qa.lock.json",
        )
        self.assertTrue(checks)
        self.assertEqual(
            [(check.name, check.status) for check in checks if check.status != "PASS"], []
        )

    def test_tampered_split_cannot_pass_source_hash_gates(self) -> None:
        source = BENCHMARK_ROOT / "locks"
        required = (
            source / "folds_cv3_v1.csv",
            source / "folds_cv3_v1.lock.json",
            source / "folds_cv3_v1_qa" / "split_qa.json",
            source / "folds_cv3_v1_qa" / "class_coverage_patients.csv",
            source / "folds_cv3_v1_qa" / "class_coverage_rois.csv",
            source / "folds_cv3_v1_qa.lock.json",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("controlled split-audit fixtures are not in the public code release")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_dir = root / "folds_cv3_v1_qa"
            qa_dir.mkdir()
            for name in ("folds_cv3_v1.csv", "folds_cv3_v1.lock.json", "folds_cv3_v1_qa.lock.json"):
                shutil.copy2(source / name, root / name)
            for name in ("split_qa.json", "class_coverage_patients.csv", "class_coverage_rois.csv"):
                shutil.copy2(source / "folds_cv3_v1_qa" / name, qa_dir / name)
            split = root / "folds_cv3_v1.csv"
            split.write_bytes(split.read_bytes() + b"\n")
            checks = audit_cv_split_artifacts(
                split_csv_path=split,
                split_lock_path=root / "folds_cv3_v1.lock.json",
                split_qa_path=qa_dir / "split_qa.json",
                split_qa_lock_path=root / "folds_cv3_v1_qa.lock.json",
            )
        statuses = {check.name: check.status for check in checks}
        self.assertEqual(statuses["split.qa_source_split"], "FAIL")
        self.assertEqual(statuses["split.primary_lock_source"], "FAIL")
        self.assertEqual(statuses["split.source_lock_binding"], "FAIL")

    def test_not_performed_center_comparison_is_deferred_by_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi_root = root / "roi" / "part_01"
            roi_root.mkdir(parents=True)
            (roi_root / "Sarcoma_0001_roi_1.png").write_bytes(b"placeholder")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "slide_name,patient_idx,roi_idx,diagnosis,differentiation,growth_pattern\n"
                "Sarcoma_0001.svs_roi_1.png,1,1,A,D,G\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            manifest.write_text("roi_uid\nr1\n", encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest().upper()
            metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest().upper()
            manifest_lock = root / "manifest.lock.json"
            manifest_lock.write_text(
                json.dumps(
                    {
                        "manifest_sha256": manifest_hash,
                        "metadata_sha256": metadata_hash,
                    }
                ),
                encoding="utf-8",
            )
            qa_dir = root / "image_qa"
            qa_dir.mkdir()
            qa_summary = qa_dir / "image_center_summary.json"
            qa_summary.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "comparison_performed": False,
                        "center_equivalence_status": "NOT_PERFORMED",
                        "roi_checked": 1,
                        "fully_decoded": 1,
                        "rgb_4096": 1,
                        "legacy_centers_found": 0,
                        "legacy_centers_pixel_exact": 0,
                    }
                ),
                encoding="utf-8",
            )
            qa_rows = qa_dir / "image_center_checks.csv"
            qa_rows.write_text("roi_uid,comparison_performed\nr1,false\n", encoding="utf-8")
            qa_lock = root / "image_qa.lock.json"
            qa_lock.write_text(
                json.dumps(
                    {
                        "dataset_manifest_sha256": manifest_hash,
                        "image_center_summary_sha256": hashlib.sha256(
                            qa_summary.read_bytes()
                        ).hexdigest().upper(),
                        "image_center_checks_sha256": hashlib.sha256(
                            qa_rows.read_bytes()
                        ).hexdigest().upper(),
                    }
                ),
                encoding="utf-8",
            )
            dataset_lock = {
                "metadata_rows": 1,
                "metadata_sha256": metadata_hash,
                "expected_patient_count": 1,
                "expected_slide_count": 1,
                "class_counts": {"diagnosis": 1, "differentiation": 1, "growth_pattern": 1},
                "roi_png_count": 1,
                "roi_shape": [4096, 4096, 3],
            }
            checks = audit_dataset(
                {
                    "_source_dir": str(root),
                    "dataset": {"metadata_csv": str(metadata), "roi_root": str(root / "roi")},
                },
                dataset_lock,
                image_qa_path=qa_summary,
                image_qa_lock_path=qa_lock,
                dataset_manifest_path=manifest,
                dataset_manifest_lock_path=manifest_lock,
            )
        statuses = {check.name: check.status for check in checks}
        self.assertEqual(statuses["dataset.full_decode"], "PASS")
        self.assertEqual(statuses["dataset.center_equivalence"], "DEFERRED")


if __name__ == "__main__":
    unittest.main()
