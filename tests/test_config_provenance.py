from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pairbst.config import get_path, load_yaml
from pairbst.provenance import write_json_atomic


class ConfigAndProvenanceTests(unittest.TestCase):
    def test_relative_paths_resolve_from_yaml_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.yaml"
            config_file.write_text("data:\n  file: ../input.csv\n", encoding="utf-8")
            config = load_yaml(config_file)
            self.assertEqual(get_path(config, "data", "file"), (root / "../input.csv").resolve())

    def test_atomic_json_writer_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            write_json_atomic({"status": "PASS"}, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "PASS")
            self.assertFalse(output.with_suffix(".json.partial").exists())


if __name__ == "__main__":
    unittest.main()

