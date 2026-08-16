from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pairbst.classification.linear_probe import (
    LinearProbeConfig,
    assemble_seed_oof_probabilities,
    run_outer_cv_linear_probe,
)
from pairbst.classification.metrics import classification_metrics
from pairbst.reporting.tables import (
    MANUSCRIPT_TABLE_COLUMNS,
    build_classification_ci_table,
    build_legacy_seed_probability_ensemble_table,
    build_manuscript_table,
    build_retrieval_companion_table,
    write_final_results_bundle,
    write_table_bundle,
)
from pairbst.retrieval.search import (
    exact_cosine_topk,
    run_patient_disjoint_cv_retrieval,
)
from pairbst.statistics.bootstrap import (
    cluster_bootstrap_classification,
    cluster_bootstrap_mean,
    paired_cluster_bootstrap_classification,
    paired_cluster_bootstrap_mean,
)
from pairbst.statistics.comparisons import apply_holm_correction


class ClassificationTests(unittest.TestCase):
    def test_metrics_include_confusions_and_per_class_rows(self) -> None:
        result = classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            labels=np.asarray([0, 1]),
        )
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 2)
        np.testing.assert_array_equal(
            result["confusion_matrix"], np.asarray([[1, 1], [0, 2]])
        )
        np.testing.assert_allclose(
            result["confusion_matrix_row_normalized"],
            np.asarray([[0.5, 0.5], [0.0, 1.0]]),
        )
        self.assertEqual([row["support"] for row in result["per_class"]], [2, 2])

    def test_linear_probe_is_deterministic_and_writes_audit_artifacts(self) -> None:
        features: list[list[float]] = []
        labels: list[str] = []
        folds: list[int] = []
        patients: list[str] = []
        sample_ids: list[str] = []
        for fold in range(3):
            for class_index, label in enumerate(("A", "B")):
                for replicate in range(2):
                    features.append(
                        [
                            -2.0 if class_index == 0 else 2.0,
                            fold * 0.1 + replicate * 0.01,
                        ]
                    )
                    labels.append(label)
                    folds.append(fold)
                    patients.append(f"p{fold}_{label}_{replicate}")
                    sample_ids.append(f"roi{fold}_{label}_{replicate}")
        config = LinearProbeConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.01,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            first = run_outer_cv_linear_probe(
                np.asarray(features, dtype=np.float32),
                labels,
                folds,
                patients,
                sample_ids=sample_ids,
                config=config,
                output_dir=directory,
            )
            second = run_outer_cv_linear_probe(
                np.asarray(features, dtype=np.float32),
                labels,
                folds,
                patients,
                sample_ids=sample_ids,
                config=config,
            )
            np.testing.assert_array_equal(
                first.seed_oof_probabilities, second.seed_oof_probabilities
            )
            np.testing.assert_array_equal(
                first.seed_oof_predictions_encoded,
                second.seed_oof_predictions_encoded,
            )
            self.assertEqual(first.seed_oof_probabilities.shape, (5, 12, 2))
            self.assertEqual(first.seed_fold_metrics.shape[0], 15)
            self.assertEqual(first.seed_oof_metrics.shape[0], 5)
            self.assertEqual(len(first.seed_pooled_metrics), 5)
            self.assertEqual(
                first.provenance["config"]["seeds"], (101, 202, 303, 404, 505)
            )
            self.assertFalse(first.provenance["probability_ensemble_across_seeds"])
            self.assertEqual(first.provenance["seed_sd_ddof"], 1)
            required = [
                "label_encoder.json",
                "seed_oof_predictions.csv",
                "seed_oof_probabilities.npz",
                "seed_fold_metrics.csv",
                "seed_oof_metrics.csv",
                "seed_metric_mean_sd.csv",
                "seed_oof_per_class_metrics.csv",
                "seed_oof_confusion_matrices.npz",
                "provenance.json",
                "fold_0/standard_scaler.npz",
                "fold_0/classifier_seed_101.pt",
                "fold_0/seed_probabilities.npz",
            ]
            for relative in required:
                self.assertTrue((Path(directory) / relative).is_file(), relative)
            with np.load(Path(directory) / "seed_oof_probabilities.npz") as stored:
                self.assertEqual(stored["probabilities"].shape, (5, 12, 2))
                self.assertNotIn("mean_probabilities", stored.files)
            predictions = pd.read_csv(Path(directory) / "seed_oof_predictions.csv")
            self.assertEqual(len(predictions), 60)
            self.assertTrue(
                (predictions.groupby(["seed", "sample_id"]).size() == 1).all()
            )
            per_class = pd.read_csv(
                Path(directory) / "seed_oof_per_class_metrics.csv"
            )
            self.assertEqual(len(per_class), 10)
            self.assertTrue((per_class.groupby("seed").size() == 2).all())
            with np.load(
                Path(directory) / "seed_oof_confusion_matrices.npz"
            ) as confusions:
                self.assertEqual(confusions["raw"].shape, (5, 2, 2))
                self.assertEqual(confusions["row_normalized"].shape, (5, 2, 2))
                self.assertNotIn("ensemble", confusions.files)

    def test_frozen_default_protocol(self) -> None:
        config = LinearProbeConfig()
        self.assertEqual(config.seeds, (101, 202, 303, 404, 505))
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-2)
        self.assertEqual(config.protocol_id, "cv3_independent_seed_oof_v1")
        self.assertEqual(config.primary_metric_unit, "complete_oof_per_seed")
        self.assertEqual(config.seed_aggregation, "metric_mean_sd")
        self.assertFalse(config.probability_ensemble_across_seeds)
        self.assertEqual(config.seed_sd_ddof, 1)

    def test_probability_ensemble_configuration_is_rejected(self) -> None:
        config = LinearProbeConfig(probability_ensemble_across_seeds=True)
        with self.assertRaisesRegex(ValueError, "Probability ensembling"):
            config.validate()
        with self.assertRaisesRegex(ValueError, "Canonical seed order"):
            LinearProbeConfig(seeds=(11, 12)).validate()

    def test_seed_oof_assembly_is_independent_and_requires_exact_partition(self) -> None:
        fold_zero = np.asarray(
            [
                [[0.9, 0.1], [0.8, 0.2]],
                [[0.1, 0.9], [0.2, 0.8]],
            ],
            dtype=np.float32,
        )
        fold_one = np.asarray(
            [
                [[0.7, 0.3], [0.6, 0.4]],
                [[0.3, 0.7], [0.4, 0.6]],
            ],
            dtype=np.float32,
        )
        assembled = assemble_seed_oof_probabilities(
            [fold_zero, fold_one],
            [np.asarray([0, 2]), np.asarray([1, 3])],
            n_samples=4,
            seeds=(101, 202),
        )
        self.assertEqual(assembled.shape, (2, 4, 2))
        np.testing.assert_array_equal(assembled[0, [0, 2]], fold_zero[0])
        np.testing.assert_array_equal(assembled[1, [1, 3]], fold_one[1])
        altered = fold_zero.copy()
        altered[1, 0] = np.asarray([0.95, 0.05], dtype=np.float32)
        changed = assemble_seed_oof_probabilities(
            [altered, fold_one],
            [np.asarray([0, 2]), np.asarray([1, 3])],
            n_samples=4,
            seeds=(101, 202),
        )
        np.testing.assert_array_equal(assembled[0], changed[0])
        self.assertFalse(np.array_equal(assembled[1], changed[1]))
        with self.assertRaisesRegex(ValueError, "partition"):
            assemble_seed_oof_probabilities(
                [fold_zero, fold_one],
                [np.asarray([0, 2]), np.asarray([2, 3])],
                n_samples=4,
                seeds=(101, 202),
            )


class RetrievalTests(unittest.TestCase):
    def test_exact_cosine_excludes_patient_and_breaks_ties_by_id(self) -> None:
        result = exact_cosine_topk(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            ["same"],
            ["same", "other_b", "other_a"],
            gallery_ids=["z", "b", "a"],
            k=2,
        )
        np.testing.assert_array_equal(result.indices, np.asarray([[2, 1]]))
        self.assertEqual(result.eligible_counts.tolist(), [2])

    def test_three_fold_retrieval_outputs_standard_metrics(self) -> None:
        features = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]] * 3,
            dtype=np.float32,
        )
        labels = np.asarray(["A", "B"] * 3)
        folds = np.repeat(np.arange(3), 2)
        patients = np.asarray([f"patient_{index}" for index in range(6)])
        ids = np.asarray([f"roi_{index}" for index in range(6)])
        result = run_patient_disjoint_cv_retrieval(
            features,
            labels,
            folds,
            patients,
            sample_ids=ids,
            ks=(1, 2),
            query_chunk_size=2,
        )
        self.assertEqual(result.fold_metrics.shape[0], 6)
        self.assertTrue((result.neighbors["query_patient_id"] != result.neighbors["gallery_patient_id"]).all())
        top_one = result.pooled_metrics[result.pooled_metrics["k"] == 1].iloc[0]
        self.assertEqual(top_one["hit_at_k"], 1.0)
        self.assertEqual(top_one["average_precision_at_k"], 1.0)


class StatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y_true = np.repeat(np.asarray(["A", "B", "C"]), 6)
        self.patients = np.repeat(
            np.asarray([f"{label}{index}" for label in "ABC" for index in range(3)]),
            2,
        )
        self.good = self.y_true.copy()
        self.good[0] = "B"
        self.bad = np.roll(self.y_true, 6)

    def test_cluster_bootstrap_is_stratified_finite_and_reproducible(self) -> None:
        first = cluster_bootstrap_classification(
            self.y_true,
            self.good,
            self.patients,
            strata=self.y_true,
            n_bootstrap=200,
            seed=7,
            chunk_size=50,
        )
        second = cluster_bootstrap_classification(
            self.y_true,
            self.good,
            self.patients,
            strata=self.y_true,
            n_bootstrap=200,
            seed=7,
            chunk_size=50,
        )
        np.testing.assert_array_equal(
            first.distributions["macro_f1"], second.distributions["macro_f1"]
        )
        self.assertTrue(np.isfinite(first.distributions["balanced_accuracy"]).all())
        self.assertEqual(first.n_patients, 9)
        self.assertTrue(first.stratified)
        self.assertLessEqual(
            first.confidence_intervals["macro_f1"][0],
            first.point_estimates["macro_f1"],
        )

    def test_paired_bootstrap_and_cluster_mean(self) -> None:
        comparison = paired_cluster_bootstrap_classification(
            self.y_true,
            self.good,
            self.bad,
            self.patients,
            strata=self.y_true,
            n_bootstrap=200,
            seed=3,
            chunk_size=40,
        )
        self.assertGreater(comparison.point_differences["macro_f1"], 0)
        self.assertGreater(comparison.probability_a_better["macro_f1"], 0.9)
        self.assertEqual(comparison.p_value_method, "patient_cluster_random_swap")
        means = cluster_bootstrap_mean(
            np.arange(18, dtype=float),
            self.patients,
            strata=self.y_true,
            n_bootstrap=100,
            seed=2,
            chunk_size=25,
        )
        self.assertAlmostEqual(means["estimate"], 8.5)
        self.assertEqual(means["n_patients"], 9)
        paired_mean = paired_cluster_bootstrap_mean(
            np.ones(18),
            np.zeros(18),
            self.patients,
            strata=self.y_true,
            n_bootstrap=100,
            seed=2,
            chunk_size=25,
        )
        self.assertEqual(paired_mean["difference_a_minus_b"], 1.0)
        self.assertEqual(paired_mean["probability_a_better"], 1.0)
        self.assertEqual(paired_mean["p_value_method"], "patient_cluster_random_swap")

    def test_safe_default_supports_patients_with_multiple_task_labels(self) -> None:
        y_true = np.asarray(["A", "B", "A", "B"])
        y_pred = y_true.copy()
        patients = np.asarray(["p1", "p1", "p2", "p2"])
        result = cluster_bootstrap_classification(
            y_true,
            y_pred,
            patients,
            n_bootstrap=20,
            seed=5,
            chunk_size=10,
        )
        self.assertFalse(result.stratified)

    def test_explicit_class_order_is_preserved(self) -> None:
        result = cluster_bootstrap_classification(
            self.y_true,
            self.good,
            self.patients,
            labels=["C", "A", "B"],
            n_bootstrap=20,
            seed=1,
            chunk_size=10,
        )
        self.assertTrue(np.isfinite(result.point_estimates["macro_f1"]))

    def test_holm_correction_is_monotone_in_sorted_p_values(self) -> None:
        frame = pd.DataFrame(
            {"family": ["primary"] * 3, "p_value": [0.01, 0.04, 0.03]}
        )
        corrected = apply_holm_correction(frame)
        np.testing.assert_allclose(corrected["p_value_holm"], [0.03, 0.06, 0.06])


class ReportingTests(unittest.TestCase):
    @staticmethod
    def _classification_rows() -> pd.DataFrame:
        rows = []
        for task in ("diagnosis", "differentiation", "growth_pattern"):
            for seed, value in zip(
                (101, 202, 303, 404, 505),
                (0.60, 0.65, 0.70, 0.75, 0.80),
                strict=True,
            ):
                rows.append(
                    {
                        "protocol_id": "cv3_independent_seed_oof_v1",
                        "model_id": "resnet50_v2",
                        "model": "resnet50",
                        "strategy": "center",
                        "task": task,
                        "seed": seed,
                        "balanced_accuracy": value,
                        "macro_f1": value - 0.1,
                        "accuracy": value,
                        "weighted_f1": value - 0.05,
                        "n_samples": 12,
                        "n_patients": 12,
                        "n_classes": 2,
                    }
                )
        return pd.DataFrame(rows)

    def test_table5_exact_columns_and_three_output_formats(self) -> None:
        table = build_manuscript_table(
            self._classification_rows(),
            model_order=("ResNet",),
            strategy_order=("Center Crop",),
        )
        self.assertEqual(tuple(table.columns), MANUSCRIPT_TABLE_COLUMNS)
        self.assertEqual(table.iloc[0]["Diagnosis B.Acc"], "0.700 ± 0.079")
        self.assertEqual(table.iloc[0]["Growth Pattern Macro-F1"], "0.600 ± 0.079")
        expected_sd = np.std([0.60, 0.65, 0.70, 0.75, 0.80], ddof=1)
        self.assertAlmostEqual(expected_sd, 0.0790569415042095)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_table_bundle(table, directory, stem="table5_manuscript")
            self.assertEqual(set(paths), {"csv", "markdown", "latex"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            latex = paths["latex"].read_text(encoding="utf-8")
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("$\\pm$", latex)
            self.assertIn("\\textbf{", latex)
            self.assertIn("independently trained linear probes", latex)
            self.assertIn("**0.700 ± 0.079**", markdown)

    def test_table5_rejects_legacy_protocol_fold_rows_and_incomplete_seeds(self) -> None:
        source = self._classification_rows()
        legacy = source.assign(protocol_id="PAIRBST-REV-CV3-v1", held_fold=0)
        with self.assertRaisesRegex(ValueError, "Legacy fold metrics"):
            build_manuscript_table(
                legacy,
                model_order=("ResNet",),
                strategy_order=("Center Crop",),
            )
        with self.assertRaisesRegex(ValueError, "has seeds"):
            build_manuscript_table(
                source[source["seed"] != 505],
                model_order=("ResNet",),
                strategy_order=("Center Crop",),
            )
        with self.assertRaisesRegex(RuntimeError, "not available"):
            build_legacy_seed_probability_ensemble_table(pd.DataFrame())

    def test_changing_one_seed_changes_only_that_seed_before_summary(self) -> None:
        source = self._classification_rows()
        changed = source.copy()
        target = (changed["task"] == "diagnosis") & (changed["seed"] == 505)
        changed.loc[target, "balanced_accuracy"] = 0.90
        pd.testing.assert_frame_equal(
            source[~target].reset_index(drop=True),
            changed[~target].reset_index(drop=True),
        )
        original_table = build_manuscript_table(
            source,
            model_order=("ResNet",),
            strategy_order=("Center Crop",),
        )
        changed_table = build_manuscript_table(
            changed,
            model_order=("ResNet",),
            strategy_order=("Center Crop",),
        )
        self.assertNotEqual(
            original_table.iloc[0]["Diagnosis B.Acc"],
            changed_table.iloc[0]["Diagnosis B.Acc"],
        )
        self.assertEqual(
            original_table.iloc[0]["Differentiation B.Acc"],
            changed_table.iloc[0]["Differentiation B.Acc"],
        )

    def test_best_cell_uses_unrounded_mean_not_display_tie(self) -> None:
        resnet = self._classification_rows()
        for metric in ("balanced_accuracy", "macro_f1"):
            resnet[metric] = resnet[metric] + 0.0004
        uni = self._classification_rows().assign(
            model_id="uni",
            model="uni",
        )
        for metric in ("balanced_accuracy", "macro_f1"):
            uni[metric] = uni[metric] + 0.0003
        table = build_manuscript_table(
            pd.concat([resnet, uni], ignore_index=True),
            model_order=("ResNet", "UNI"),
            strategy_order=("Center Crop",),
        )
        self.assertEqual(
            table.iloc[0]["Diagnosis B.Acc"], table.iloc[1]["Diagnosis B.Acc"]
        )
        best_cells = table.attrs["best_cells"]
        self.assertIn((0, "Diagnosis B.Acc"), best_cells)
        self.assertNotIn((1, "Diagnosis B.Acc"), best_cells)

    def test_companion_ci_and_retrieval_tables(self) -> None:
        ci_source = pd.DataFrame(
                [
                    {
                        "model": "uni2-h",
                        "strategy": "mean",
                        "task": "diagnosis",
                        "metric": "macro_f1",
                        "estimate": 0.7,
                        "ci_low": 0.6,
                        "ci_high": 0.8,
                        "n_bootstrap": 10_000,
                    }
                ]
            )
        ci = build_classification_ci_table(
            ci_source,
            model_order=("UNI-2",),
            strategy_order=("Mean Pooling",),
            task_order=("diagnosis",),
            metric_order=("macro_f1",),
        )
        self.assertEqual(ci.iloc[0]["Model"], "UNI-2")
        self.assertEqual(ci.iloc[0]["95% CI"], "0.600–0.800")

        rows = []
        for fold, value in enumerate((0.6, 0.7, 0.8)):
            rows.append(
                {
                    "model": "uni2-h",
                    "strategy": "mean",
                    "task": "diagnosis",
                    "held_fold": fold,
                    "k": 5,
                    "precision_at_k": value,
                    "recall_at_k": value,
                    "hit_at_k": value,
                    "average_precision_at_k": value,
                    "majority_vote_correct": value,
                }
            )
        retrieval = build_retrieval_companion_table(
            pd.DataFrame(rows),
            model_order=("UNI-2",),
            strategy_order=("Mean Pooling",),
            task_order=("diagnosis",),
            k_order=(5,),
        )
        self.assertEqual(retrieval.iloc[0]["mAP@5"], "0.700 ± 0.100")
        with tempfile.TemporaryDirectory() as directory:
            bundles = write_final_results_bundle(
                self._classification_rows(),
                pd.DataFrame(rows).assign(model="resnet50", strategy="center"),
                directory,
                model_order=("ResNet",),
                strategy_order=("Center Crop",),
                retrieval_task_order=("diagnosis",),
                retrieval_k_order=(5,),
            )
            self.assertEqual(set(bundles), {"table5", "retrieval"})
            self.assertTrue(all(path.is_file() for group in bundles.values() for path in group.values()))


if __name__ == "__main__":
    unittest.main()
