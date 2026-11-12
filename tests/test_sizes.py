"""Size/rate measurement without reading scientific data or changing files."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sizes
import terminal


class SizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_sizes_use_metadata_without_opening_contents(self):
        (self.root / "folder").mkdir()
        (self.root / "one.vds").write_bytes(b"a" * 100)
        (self.root / "folder" / "two.part").write_bytes(b"b" * 200)
        with patch.object(Path, "open", side_effect=AssertionError("must not read contents")):
            snapshot = sizes.scan_tree(self.root, budget_seconds=1)
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["logical"], 300)
        self.assertEqual(snapshot["files"], 2)

    def test_missing_directory_is_not_created(self):
        path = self.root / "missing"
        self.assertEqual(sizes.scan_tree(path)["logical"], 0)
        self.assertFalse(path.exists())

    def test_entry_budget_produces_lower_bound(self):
        (self.root / "one.vds").write_bytes(b"a" * 100)
        snapshot = sizes.scan_tree(self.root, max_entries=0)
        self.assertFalse(snapshot["complete"])
        self.assertEqual(snapshot["logical"], 0)

    def test_symlink_root_is_never_scanned(self):
        with patch.object(Path, "is_symlink", return_value=True), \
                patch.object(sizes.os, "scandir", side_effect=AssertionError("must not traverse symlink")):
            self.assertFalse(sizes.scan_tree(self.root)["complete"])

    def test_net_growth_and_allocation_rates_are_distinct(self):
        lines = []
        monitor = sizes.SizeMonitor(self.root, self.root / "models", lines.append, interval=5)
        first = {"logical": 10000000, "allocated": 1000000, "files": 1, "complete": True}
        second = {"logical": 10000000, "allocated": 6000000, "files": 1, "complete": True}
        with patch.object(sizes, "scan_tree", return_value=first), patch.object(sizes.time, "monotonic", return_value=0):
            monitor.poll()
        with patch.object(sizes, "scan_tree", return_value=second), patch.object(sizes.time, "monotonic", return_value=5):
            monitor.poll()
        self.assertIn("+0.00 MB/s net file-size growth", lines[-1])
        self.assertIn("+1.00 MB/s net allocation growth", lines[-1])

    def test_partial_scans_never_report_misleading_rates(self):
        lines = []
        monitor = sizes.SizeMonitor(self.root, self.root / "models", lines.append, interval=5)
        partial = {"logical": 10000000, "allocated": None, "files": 4, "complete": False}
        with patch.object(sizes, "scan_tree", return_value=partial), patch.object(sizes.time, "monotonic", return_value=0):
            monitor.poll()
        self.assertIn(">=10.00 MB", lines[-1])
        self.assertIn("partial scan, rate unavailable", lines[-1])
        self.assertNotIn("MB/s", lines[-1])

    def test_sampling_interval_avoids_repeated_scans(self):
        monitor = sizes.SizeMonitor(self.root, self.root / "models", lambda _: None, interval=30)
        with patch.object(sizes, "scan_tree", return_value={"logical": 0, "allocated": None, "files": 0, "complete": True}) as scan, \
                patch.object(sizes.time, "monotonic", return_value=0):
            monitor.poll()
            monitor.poll()
            self.assertEqual(scan.call_count, 4)

    def test_size_observer_failure_does_not_interrupt_child(self):
        class BrokenSizes:
            def poll(self):
                raise OSError("metadata unavailable")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = terminal.run_command([sys.executable, "-c", "pass"], "child", sizes=BrokenSizes())
        self.assertEqual(result, 0)
        self.assertIn("SIZE WARNING", output.getvalue())


if __name__ == "__main__":
    unittest.main()
