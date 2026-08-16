from __future__ import annotations

import tempfile
import unittest
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

from pairbst.datasets import ROIRecord, load_roi_records
from pairbst.features.h5store import (
    ResumableFeatureStore,
    partial_path_for,
    validate_feature_file,
)
from pairbst.features.import_legacy import RecoveryFeatureAdapter, RecoverySourceContract
from pairbst.hashing import sha256_file
from pairbst.features.pooling import OnlineMeanMax, pool_grid_features
from pairbst.geometry import (
    CENTER_START,
    CENTER_STOP,
    GRID_LOCATIONS,
    center_box,
    center_crop_224,
    iter_grid_patches,
)
from pairbst.models.registry import MODEL_REGISTRY, get_model_spec, get_transform_spec
from pairbst.models.adapters import build_transform, forward_embeddings, pool_virchow2_tokens


def _record(root: Path, number: int) -> ROIRecord:
    wsi = f"Sarcoma_{number:04d}.svs"
    return ROIRecord(
        roi_path=root / f"Sarcoma_{number:04d}_roi_1.png",
        relative_path=f"part_01/Sarcoma_{number:04d}_roi_1.png",
        slide_name=f"Sarcoma_{number:04d}.svs_roi_1.png",
        wsi_name=wsi,
        patient_idx=str(number),
        roi_idx="1",
        diagnosis=f"Diagnosis {number}",
        differentiation="Uncertain",
        growth_pattern="Spindle",
    )


class GeometryTests(unittest.TestCase):
    def test_center_box_is_exact_and_grid_is_row_major(self) -> None:
        self.assertEqual(center_box(), (1936, 1936, 2160, 2160))
        self.assertEqual(CENTER_STOP - CENTER_START, 224)
        self.assertEqual(len(GRID_LOCATIONS), 256)
        self.assertEqual((GRID_LOCATIONS[0].row, GRID_LOCATIONS[0].column), (0, 0))
        self.assertEqual((GRID_LOCATIONS[1].row, GRID_LOCATIONS[1].column), (0, 1))
        self.assertEqual((GRID_LOCATIONS[16].row, GRID_LOCATIONS[16].column), (1, 0))
        self.assertEqual(GRID_LOCATIONS[-1].box, (3840, 3840, 4096, 4096))

        image = Image.new("RGB", (4096, 4096), "black")
        image.putpixel((1936, 1936), (1, 2, 3))
        image.putpixel((2159, 2159), (4, 5, 6))
        center = center_crop_224(image)
        self.assertEqual(center.size, (224, 224))
        self.assertEqual(center.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(center.getpixel((223, 223)), (4, 5, 6))
        sampled = list(islice(iter_grid_patches(image), 17))
        self.assertEqual((sampled[1].row, sampled[1].column), (0, 1))
        self.assertEqual((sampled[16].row, sampled[16].column), (1, 0))
        for patch in sampled:
            self.assertEqual(patch.image.size, (256, 256))
            patch.image.close()
        center.close()
        image.close()


class DatasetTests(unittest.TestCase):
    def test_public_metadata_filename_resolves_to_released_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roi_root = root / "roi_4096" / "part_01"
            roi_root.mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(roi_root / "Sarcoma_0003_roi_1.png")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "slide_name,patient_idx,roi_idx,diagnosis,differentiation,growth_pattern\n"
                "Sarcoma_0003.svs_roi_1.png,1,1,Diagnosis A,Uncertain,Spindle\n",
                encoding="utf-8",
            )
            records = load_roi_records(metadata, roi_root.parent)
            self.assertEqual(records[0].roi_path.name, "Sarcoma_0003_roi_1.png")
            self.assertEqual(records[0].wsi_name, "Sarcoma_0003.svs")
            self.assertEqual(records[0].patient_uid, "Diagnosis A::patient_idx=1")
            self.assertEqual(records[0].roi_uid, "Sarcoma_0003.svs::roi_idx=1")


class PoolingTests(unittest.TestCase):
    def test_float32_mean_max_are_batch_size_independent(self) -> None:
        values = np.arange(256 * 5, dtype=np.float32).reshape(256, 5) / np.float32(7)
        expected_mean, expected_max = pool_grid_features(values)
        for batch_size in (1, 7, 64, 256):
            accumulator = OnlineMeanMax(5)
            for start in range(0, len(values), batch_size):
                accumulator.update(values[start : start + batch_size])
            actual_mean, actual_max = accumulator.finalize(expected_count=256)
            np.testing.assert_array_equal(actual_mean, expected_mean)
            np.testing.assert_array_equal(actual_max, expected_max)
            self.assertEqual(actual_mean.dtype, np.dtype("float32"))


class RegistryTests(unittest.TestCase):
    def test_seven_frozen_models_and_transform_profiles(self) -> None:
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "resnet50_v2",
                "swin_t",
                "retccl",
                "uni",
                "uni2_h",
                "prov_gigapath",
                "virchow2",
            },
        )
        self.assertEqual(get_model_spec("Swin-T").name, "swin_t")
        self.assertNotIn("base", get_model_spec("Swin-T").architecture.lower())
        self.assertEqual(get_model_spec("UNI-2").embedding_dim, 1536)
        self.assertEqual(get_model_spec("Prov-GigaPath").embedding_dim, 1536)
        self.assertEqual(get_model_spec("Virchow2").embedding_dim, 2560)
        self.assertEqual(get_transform_spec("resnet50_v2").resize, 232)
        self.assertEqual(get_transform_spec("retccl").resize, 256)
        self.assertEqual(get_transform_spec("uni2_h").resize, 224)
        self.assertEqual(get_transform_spec("prov_gigapath").crop, 224)
        self.assertEqual(get_transform_spec("virchow2").resize, 224)
        self.assertEqual(get_transform_spec("virchow2").crop, 224)
        self.assertEqual(get_transform_spec("uni2_h", "legacy_common_224").resize, (224, 224))

    def test_official_and_legacy_transforms_have_frozen_output_shapes(self) -> None:
        image = Image.new("RGB", (256, 256), "white")
        self.assertEqual(tuple(build_transform("resnet50_v2")(image).shape), (3, 224, 224))
        self.assertEqual(tuple(build_transform("swin_t")(image).shape), (3, 224, 224))
        self.assertEqual(tuple(build_transform("retccl")(image).shape), (3, 256, 256))
        self.assertEqual(tuple(build_transform("uni")(image).shape), (3, 224, 224))
        self.assertEqual(tuple(build_transform("uni2_h")(image).shape), (3, 224, 224))
        self.assertEqual(tuple(build_transform("prov_gigapath")(image).shape), (3, 224, 224))
        self.assertEqual(tuple(build_transform("virchow2")(image).shape), (3, 224, 224))
        self.assertEqual(
            tuple(build_transform("retccl", "legacy_common_224")(image).shape),
            (3, 224, 224),
        )
        image.close()

    def test_virchow2_pooling_excludes_register_tokens(self) -> None:
        torch = __import__("torch")
        tokens = torch.zeros((2, 261, 1280), dtype=torch.float32)
        tokens[:, 0] = 2.0
        tokens[:, 1:5] = 1000.0
        tokens[:, 5:] = 3.0
        embedded = pool_virchow2_tokens(tokens)
        self.assertEqual(tuple(embedded.shape), (2, 2560))
        torch.testing.assert_close(embedded[:, :1280], torch.full((2, 1280), 2.0))
        torch.testing.assert_close(embedded[:, 1280:], torch.full((2, 1280), 3.0))
        with self.assertRaisesRegex(ValueError, "261"):
            pool_virchow2_tokens(tokens[:, :-1])

    def test_unpooled_token_outputs_are_rejected(self) -> None:
        torch = __import__("torch")

        class TokenModel:
            def __call__(self, batch):
                return torch.zeros((batch.shape[0], 4, 8))

        with self.assertRaisesRegex(ValueError, "model-specific pooling"):
            forward_embeddings(TokenModel(), torch.zeros((2, 3)))

    def test_registry_yaml_and_model_lock_are_consistent(self) -> None:
        project = Path(__file__).resolve().parents[1]
        configured = yaml.safe_load((project / "configs" / "models.yaml").read_text())[
            "models"
        ]
        locked = __import__("json").loads(
            (project / "locks" / "models.expected.json").read_text()
        )["models"]
        lock_key = {
            "resnet50_v2": "resnet50_in1k_v2",
            "swin_t": "swin_t_in1k_v1",
            "retccl": "retccl_resnet50",
            "uni": "uni_vitl16",
            "uni2_h": "uni2_h",
            "prov_gigapath": "prov_gigapath",
            "virchow2": "virchow2",
        }
        self.assertEqual(set(configured), set(MODEL_REGISTRY))
        self.assertEqual(set(lock_key.values()), set(locked))
        for model_id, spec in MODEL_REGISTRY.items():
            config = configured[model_id]
            lock = locked[lock_key[model_id]]
            self.assertEqual(config["feature_dim"], spec.embedding_dim)
            self.assertEqual(config["expected_bytes"], spec.checkpoint.size_bytes)
            self.assertEqual(config["expected_sha256"], spec.checkpoint.sha256)
            self.assertEqual(lock["feature_dim"], spec.embedding_dim)
            self.assertEqual(lock["expected_bytes"], spec.checkpoint.size_bytes)
            self.assertEqual(lock["expected_sha256"], spec.checkpoint.sha256)


class H5StoreTests(unittest.TestCase):
    def test_resume_and_atomic_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(root, 1), _record(root, 2)]
            output = root / "features.h5"
            provenance = {"checkpoint_sha256": "ABC", "model_name": "synthetic"}

            with ResumableFeatureStore(output, records, 3, provenance) as store:
                store.write_row(
                    0,
                    center=np.array([1, 2, 3], dtype=np.float32),
                    mean=np.array([4, 5, 6], dtype=np.float32),
                    max_=np.array([7, 8, 9], dtype=np.float32),
                )
                np.testing.assert_array_equal(store.incomplete_indices(), np.array([1]))
            self.assertTrue(partial_path_for(output).is_file())
            self.assertFalse(output.exists())

            resumed = ResumableFeatureStore(output, records, 3, provenance)
            with resumed:
                np.testing.assert_array_equal(resumed.incomplete_indices(), np.array([1]))
                resumed.write_row(
                    1,
                    center=np.array([10, 11, 12], dtype=np.float32),
                    mean=np.array([13, 14, 15], dtype=np.float32),
                    max_=np.array([16, 17, 18], dtype=np.float32),
                )
                self.assertTrue(resumed.is_complete())
            finalized = resumed.finalize()
            self.assertEqual(finalized, output.resolve())
            self.assertTrue(output.is_file())
            self.assertFalse(partial_path_for(output).exists())
            result = validate_feature_file(
                output,
                expected_rows=2,
                expected_dim=3,
                expected_metadata_rows=records,
                expected_provenance=provenance,
                require_complete=True,
            )
            self.assertEqual(result["completed_rows"], 2)
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["status"], "complete")
                np.testing.assert_array_equal(
                    handle["features/center"][:],
                    np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32),
                )

    def test_completed_file_rejects_partial_status_and_wrong_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(root, 1)]
            output = root / "features.h5"
            store = ResumableFeatureStore(output, records, 2, {"run": "A"})
            with store:
                store.write_row(
                    0,
                    center=np.asarray([1, 2], dtype=np.float32),
                    mean=np.asarray([1, 2], dtype=np.float32),
                    max_=np.asarray([1, 2], dtype=np.float32),
                )
            store.finalize()
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_feature_file(
                    output,
                    expected_provenance={"run": "B"},
                    require_complete=True,
                )
            with h5py.File(output, "r+") as handle:
                handle.attrs["status"] = "partial"
            with self.assertRaisesRegex(ValueError, "completed feature file"):
                validate_feature_file(output, require_complete=True)


class RecoveryAdapterTests(unittest.TestCase):
    @staticmethod
    def _contract(path: Path, strategy: str) -> RecoverySourceContract:
        return RecoverySourceContract(
            model_name="synthetic",
            strategy=strategy,
            feature_dim=2,
            source_sha256=sha256_file(path),
            checkpoint_sha256="A" * 64,
            transform_profile="unit_test",
            allowed_use="inspection",
        )

    @staticmethod
    def _legacy_file(path: Path, records: list[ROIRecord], strategy: str) -> None:
        order = [1, 0]
        string_dtype = h5py.string_dtype("utf-8")
        with h5py.File(path, "w") as handle:
            if strategy in {"mean", "max"}:
                handle.attrs["pooling_method"] = strategy
            handle.create_dataset(
                "features",
                data=np.asarray([[20, 21], [10, 11]], dtype=np.float32),
            )
            handle.create_dataset(
                "diagnosis",
                data=np.asarray([records[index].diagnosis for index in order], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "patient_idx",
                data=np.asarray([records[index].patient_idx for index in order], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "slide_name",
                data=np.asarray([f"SMC_{records[index].wsi_name}" for index in order], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "roi_idx",
                data=np.asarray([records[index].roi_idx for index in order], dtype=object),
                dtype=string_dtype,
            )

    def test_read_only_adapter_reorders_to_canonical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(root, 1), _record(root, 2)]
            center = root / "center.h5"
            mean = root / "mean.h5"
            self._legacy_file(center, records, "center")
            self._legacy_file(mean, records, "mean")
            contracts = {
                "center": self._contract(center, "center"),
                "mean": self._contract(mean, "mean"),
            }
            with RecoveryFeatureAdapter(
                records, {"center": center, "mean": mean}, contracts
            ) as adapter:
                np.testing.assert_array_equal(
                    adapter.read("center"), np.asarray([[10, 11], [20, 21]], dtype=np.float32)
                )
                self.assertEqual(adapter.feature_dim("mean"), 2)

    def test_adapter_rejects_known_pseudo_grid_and_alternate_center_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(root, 1), _record(root, 2)]
            for blocked in (
                "center_repooling_not_genuine_grid",
                "alternate_center_features",
            ):
                path = root / blocked / "source.h5"
                path.parent.mkdir()
                path.touch()
                with self.assertRaisesRegex(ValueError, "Refusing invalid recovery feature source"):
                    RecoveryFeatureAdapter(
                        records,
                        {"center": path},
                        {
                            "center": RecoverySourceContract(
                                model_name="synthetic",
                                strategy="center",
                                feature_dim=2,
                                source_sha256="A" * 64,
                                checkpoint_sha256=None,
                                transform_profile="unit_test",
                            )
                        },
                    )

    def test_primary_reuse_requires_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(root, 1), _record(root, 2)]
            path = root / "mean.h5"
            self._legacy_file(path, records, "mean")
            contract = RecoverySourceContract(
                model_name="uni2_h",
                strategy="mean",
                feature_dim=2,
                source_sha256=sha256_file(path),
                checkpoint_sha256="A" * 64,
                transform_profile="unverified_recovery",
                allowed_use="primary_after_anchor",
            )
            with self.assertRaisesRegex(ValueError, "verified anchor pilot"):
                RecoveryFeatureAdapter(
                    records,
                    {"mean": path},
                    {"mean": contract},
                    intended_use="primary_after_anchor",
                )


if __name__ == "__main__":
    unittest.main()
