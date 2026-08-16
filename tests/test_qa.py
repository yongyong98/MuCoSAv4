from __future__ import annotations

import unittest

import pandas as pd

from pairbst.qa import audit_cv_split


class SplitQaTests(unittest.TestCase):
    def test_complete_patient_grouped_split_passes(self) -> None:
        rows = []
        for fold in range(3):
            for label in ("A", "B"):
                patient = f"{label}-{fold}"
                rows.append(
                    {
                        "patient_uid": patient,
                        "wsi_id": f"wsi-{patient}",
                        "roi_uid": f"roi-{patient}",
                        "fold": fold,
                        "diagnosis": label,
                        "differentiation": label,
                        "growth_pattern": label,
                    }
                )
        expected_counts = {
            fold: {"patients": 2, "wsi": 2, "roi": 2} for fold in range(3)
        }
        result = audit_cv_split(
            pd.DataFrame(rows), expected_fold_counts=expected_counts
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["patients_in_multiple_folds"], 0)
        self.assertTrue(result["fold_ids_match"])
        self.assertTrue(result["fold_counts_match"])

    def test_patient_leakage_fails(self) -> None:
        frame = pd.DataFrame(
            [
                {"patient_uid": "p", "wsi_id": "w0", "roi_uid": "r0", "fold": 0, "diagnosis": "A", "differentiation": "A", "growth_pattern": "A"},
                {"patient_uid": "p", "wsi_id": "w1", "roi_uid": "r1", "fold": 1, "diagnosis": "A", "differentiation": "A", "growth_pattern": "A"},
            ]
        )
        result = audit_cv_split(frame)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["patients_in_multiple_folds"], 1)

    def test_missing_fold_fails_even_when_observed_classes_are_covered(self) -> None:
        rows = []
        for fold in (0, 1):
            rows.append(
                {
                    "patient_uid": f"p{fold}",
                    "wsi_id": f"w{fold}",
                    "roi_uid": f"r{fold}",
                    "fold": fold,
                    "diagnosis": "A",
                    "differentiation": "A",
                    "growth_pattern": "A",
                }
            )
        result = audit_cv_split(pd.DataFrame(rows), expected_fold_counts=None)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["fold_ids_match"])
        self.assertIn({"task": "diagnosis", "class": "A"}, result["missing_class_fold_pairs"])

    def test_wrong_fold_counts_fail(self) -> None:
        rows = []
        for fold in range(3):
            rows.append(
                {
                    "patient_uid": f"p{fold}",
                    "wsi_id": f"w{fold}",
                    "roi_uid": f"r{fold}",
                    "fold": fold,
                    "diagnosis": "A",
                    "differentiation": "A",
                    "growth_pattern": "A",
                }
            )
        expected = {fold: {"patients": 2, "wsi": 1, "roi": 1} for fold in range(3)}
        result = audit_cv_split(pd.DataFrame(rows), expected_fold_counts=expected)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["fold_counts_match"])


if __name__ == "__main__":
    unittest.main()
