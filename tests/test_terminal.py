"""Monitoring tests: no HPC data, downloads, package installs, or GPU needed."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


terminal = load_script("terminal")


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.logs = self.workspace / "logs"
        self.logs.mkdir()
        self.lines = []

    def status(self, **changes):
        status = {"stages": [{"id": "convert", "label": "Convert"}, {"id": "rgt", "label": "RGT"}],
                  "stage": "rgt", "state": "running", "message": "waiting for a GPU", "started": 1}
        status.update(changes)
        (self.workspace / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return status

    def test_stage_position_done_and_error(self):
        status = self.status()
        self.assertIn("[#>] Stage 2/2", terminal.stage_line(status))
        status["state"] = "error"
        self.assertIn("[#!]", terminal.stage_line(status))
        status.update(state="done", stage="done")
        self.assertIn("[##] 2/2 stages finished", terminal.stage_line(status))

    def test_monitor_does_not_write_any_files(self):
        self.status()
        path = self.logs / "pipeline.log"
        path.write_text("existing output\n")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        monitor.poll()
        monitor.poll()
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(self.lines), 1)
        self.assertNotIn("existing output", "\n".join(self.lines))

    def test_attach_does_not_create_workspace(self):
        missing = self.workspace / "not-created"
        monitor = terminal.Monitor(missing, self.lines.append)
        monitor.poll()
        self.assertFalse(missing.exists())

    def test_live_pipeline_gpu_and_converter_logs(self):
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        for name in ("pipeline.log", "gpu_job_123.log", "openvds_full.log"):
            (self.logs / name).write_text("first update\n")
        monitor.poll()
        self.assertEqual(len(self.lines), 3)
        monitor.poll()
        self.assertEqual(len(self.lines), 3)
        with (self.logs / "gpu_job_123.log").open("a") as stream:
            stream.write("second update\n")
        monitor.poll()
        self.assertIn("[gpu_job_123.log] second update", self.lines[-1])

    def test_truncated_or_replaced_log_is_followed(self):
        path = self.logs / "pipeline.log"
        path.write_text("old output\n")
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        path.write_text("new output\n")  # Same size, new content.
        monitor.poll()
        self.assertEqual(self.lines, ["[pipeline.log] new output"])
        path.write_text("x\n")
        monitor.poll()
        self.assertEqual(self.lines[-1], "[pipeline.log] x")

    def test_log_reads_are_bounded(self):
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        (self.logs / "pipeline.log").write_text("x" * 100000 + "\nlast line\n")
        monitor.poll()
        self.assertIn("older output omitted", self.lines[0])
        self.assertIn("last line", self.lines[-1])
        self.assertLess(len("\n".join(self.lines)), 34000)

    def test_bad_or_missing_status_is_tolerated(self):
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        for text in ("{incomplete", "[]", '{"stage":"convert","stages":null}', '{"stage":"convert","stages":[null,4]}'):
            (self.workspace / "status.json").write_text(text)
            monitor.poll()
        self.assertIn("convert", self.lines[-1])

    def test_new_run_same_log_prefix_replays_current_output(self):
        self.status(started=1)
        path = self.logs / "pipeline.log"
        path.write_text("same prefix\n")
        monitor = terminal.Monitor(self.workspace, self.lines.append)
        monitor.poll()
        self.status(started=2)
        path.write_text("same prefix\n")
        monitor.poll()
        self.assertIn("[pipeline.log] same prefix", self.lines)


class CommandTests(unittest.TestCase):
    def command(self, source):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "terminal.py"), "run",
                               "--label", "test child", "--heartbeat", "0.05", "--",
                               sys.executable, "-u", "-c", source], capture_output=True, text=True, timeout=10)

    def test_child_output_heartbeat_and_success(self):
        result = self.command("import time; print('child output', flush=True); time.sleep(0.2)")
        self.assertEqual(result.returncode, 0)
        self.assertIn("child output", result.stdout)
        self.assertIn("[ACTIVE", result.stdout)
        self.assertIn("[OK]", result.stdout)

    def test_error_code_and_stderr_preserved(self):
        result = self.command("import sys; print('actual error', file=sys.stderr); sys.exit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertIn("actual error", result.stderr)
        self.assertIn("[FAILED]", result.stdout)
        self.assertNotIn("[OK]", result.stdout)

    def test_broken_observer_does_not_stop_child(self):
        class BrokenMonitor:
            def poll(self):
                raise ValueError("bad status")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = terminal.run_command([sys.executable, "-c", "pass"], "child", monitor=BrokenMonitor())
        self.assertEqual(result, 0)
        self.assertIn("MONITOR WARNING", output.getvalue())

    def test_missing_executable_is_clear_failure(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = terminal.run_command([str(ROOT / "does-not-exist")], "missing child")
        self.assertEqual(result, 1)
        self.assertIn("[FAILED]", output.getvalue())


class DownloadTests(unittest.TestCase):
    def test_rgt_bar_uses_downloaded_bytes(self):
        downloader = load_script("download_models")
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with patch.object(downloader, "MODELS", Path(temp)), patch.object(downloader, "RGT_SIZE", 8), \
                    patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"12345678")), \
                    contextlib.redirect_stdout(output):
                self.assertTrue(downloader.get_rgt())
            self.assertIn("[####################] 100%", output.getvalue())
            self.assertEqual((Path(temp) / "rgt" / "RGT-Est_CIG-Benchmark.pt").read_bytes(), b"12345678")


if __name__ == "__main__":
    unittest.main()
