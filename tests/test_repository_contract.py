"""Dependency-free checks for the TissueLoom repository contract."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"


class RepositoryContractTests(unittest.TestCase):
    def test_all_python_sources_compile(self) -> None:
        python_files = sorted(SOURCE.rglob("*.py"))
        self.assertTrue(python_files, "No Python source files were found")

        for path in python_files:
            with self.subTest(path=path.relative_to(ROOT)):
                source_text = path.read_text(encoding="utf-8")
                compile(source_text, str(path), "exec")

    def test_tissueloom_entry_points_exist(self) -> None:
        expected_paths = (
            SOURCE / "__main__.py",
            SOURCE / "models" / "tissueformer" / "ta_bnt_final.py",
            SOURCE / "conf" / "model" / "ta_bnt_final_configs.yaml",
        )

        for path in expected_paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())

    def test_nonredistributable_baseline_ports_are_not_in_release_tree(self) -> None:
        excluded_paths = (
            SOURCE / "models" / "brainnetmlp.py",
            SOURCE / "models" / "ComBrainTF",
            SOURCE / "models" / "DHGFormer",
            SOURCE / "models" / "LRBGT",
            SOURCE / "conf" / "model" / "brainnetmlp.yaml",
            SOURCE / "conf" / "model" / "comtf.yaml",
            SOURCE / "conf" / "model" / "dhgformer.yaml",
            SOURCE / "conf" / "model" / "lrbgt.yaml",
        )

        for path in excluded_paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
