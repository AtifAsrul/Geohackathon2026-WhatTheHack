"""Well-folder regressions using temporary, fictional file layouts only."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from copilot import pipeline, server, wells


class WellImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "challenge"
        self.stack = self.data / "stack"
        self.stack.mkdir(parents=True)
        (self.stack / "full.sgy").touch()
        self.well_dir = self.data / "well_data"
        (self.well_dir / "las").mkdir(parents=True)
        (self.well_dir / "las" / "fictional-A.LAS").touch()
        self.project = self.root / "project.json"
        self.old_project = {"data_dir": str(self.stack), "wells_dir": str(self.stack),
                            "sources": [str(self.stack / "full.sgy")]}
        self.project.write_text(json.dumps(self.old_project))
        context = patch.object(pipeline, "PROJECT", self.project)
        context.start()
        self.addCleanup(context.stop)

    def import_data(self, **body):
        req = SimpleNamespace(json=AsyncMock(return_value=body))
        return asyncio.run(server.do_import(req))

    def test_browsing_stack_finds_sibling_well_data(self):
        result = server.browse(str(self.stack))
        self.assertEqual(result["segy_found"], 1)
        self.assertEqual(result["wells_guess"], str(self.well_dir))
        self.assertEqual(wells.scan(result["wells_guess"])["las"], 1)

    def test_browsing_root_prefers_its_well_data_child(self):
        self.assertEqual(server.browse(str(self.data))["wells_guess"], str(self.well_dir))

    def test_existing_well_selection_wins_over_neighbour(self):
        custom = self.data / "custom_logs"
        custom.mkdir()
        (custom / "fictional-B.las").touch()
        self.assertEqual(wells.guess_folder(custom), str(custom))
        self.import_data(path=str(self.stack), wells_path=str(custom))
        self.assertEqual(json.loads(self.project.read_text())["wells_dir"], str(custom))

    def test_no_guess_when_no_well_files_exist(self):
        (self.well_dir / "las" / "fictional-A.LAS").unlink()
        self.assertEqual(server.browse(str(self.stack))["wells_guess"], "")

    def test_empty_selected_well_folder_is_rejected_without_changing_project(self):
        with self.assertRaises(HTTPException) as error:
            self.import_data(path=str(self.stack), wells_path=str(self.stack))
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn(str(self.well_dir), error.exception.detail)
        self.assertEqual(json.loads(self.project.read_text()), self.old_project)

    def test_wells_only_requires_a_folder(self):
        with self.assertRaises(HTTPException):
            self.import_data(wells_only=True, wells_path="")
        self.assertEqual(json.loads(self.project.read_text()), self.old_project)

    def test_wells_only_preserves_seismic_selection_and_cached_outputs(self):
        cached = self.root / "rgt.npy"
        cached.write_bytes(b"existing seismic interpretation")
        self.import_data(wells_only=True, wells_path=str(self.well_dir))
        proj = json.loads(self.project.read_text())
        self.assertEqual(proj["sources"], self.old_project["sources"])
        self.assertEqual(proj["data_dir"], self.old_project["data_dir"])
        self.assertEqual(proj["wells_dir"], str(self.well_dir))
        self.assertEqual(cached.read_bytes(), b"existing seismic interpretation")

    def test_invalid_saved_folder_fails_instead_of_publishing_zero_wells(self):
        saved = self.root / "wells.json"
        original = '{"wells": [{"name": "previous well"}]}'
        saved.write_text(original)
        with patch.object(wells, "WELLS_JSON", saved):
            with self.assertRaisesRegex(RuntimeError, "No well files found"):
                pipeline.stage_wells()
        self.assertEqual(saved.read_text(), original)

    def test_missing_saved_folder_is_reported(self):
        self.old_project["wells_dir"] = str(self.root / "missing")
        self.project.write_text(json.dumps(self.old_project))
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            pipeline.stage_wells()

    def test_legacy_project_without_well_path_finds_sibling(self):
        self.old_project.pop("wells_dir")
        self.project.write_text(json.dumps(self.old_project))
        with patch.object(wells, "load_wells") as load, patch.object(pipeline.store, "reference_volume", return_value=None):
            pipeline.stage_wells()
        self.assertEqual(load.call_args.args[0], str(self.well_dir))

    def test_reader_error_does_not_replace_previous_wells_with_empty_result(self):
        self.old_project["wells_dir"] = str(self.well_dir)
        self.project.write_text(json.dumps(self.old_project))
        saved = self.root / "wells.json"
        original = '{"wells": [{"name": "previous well"}]}'
        saved.write_text(original)
        with patch.object(wells, "WELLS_JSON", saved), \
                patch.object(wells, "load_wells", side_effect=ValueError("unreadable file")), \
                patch.object(pipeline.store, "reference_volume", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "unreadable file"):
                pipeline.stage_wells()
        self.assertEqual(saved.read_text(), original)

    def test_unreadable_well_list_does_not_publish_empty_results(self):
        empty = self.root / "broken_wells"
        empty.mkdir()
        (empty / "well_list.xlsx").write_bytes(b"not a workbook")
        saved = self.root / "wells.json"
        saved.write_text('{"wells": [{"name": "previous well"}]}')
        with patch.object(wells, "WELLS_JSON", saved):
            with self.assertRaisesRegex(ValueError, "no well names"):
                wells.load_wells(empty)
        self.assertEqual(json.loads(saved.read_text())["wells"][0]["name"], "previous well")


if __name__ == "__main__":
    unittest.main()
