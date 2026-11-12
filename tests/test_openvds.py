"""Real lossless SEG-Y/VDS integration; all data is generated in temporary folders."""
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("openvds", "segyio"))


@unittest.skipUnless(HAS_DEPS, "Install openvds==3.4.9 and segyio to run real conversion tests")
class OpenVDSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.dict(os.environ, {"COPILOT_WORKSPACE": str(self.root)}):
            from copilot import store, vds
        self.store, self.vds = store, vds
        if not vds.importer_bin():
            self.skipTest("OpenVDS SEGYImport executable is unavailable")
        self.logs, self.stores = self.root / "logs", self.root / "stores"
        self.logs.mkdir(exist_ok=True)
        self.stores.mkdir(exist_ok=True)
        for name, value in (("LOGS", self.logs), ("STORES", self.stores)):
            context = patch.object(store.config, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.messages = []
        self.il = np.arange(10, 15)
        self.xl = np.arange(30, 44, 2)
        self.times = np.arange(17) * 4 + 1000
        self.data = np.arange(5 * 7 * 17, dtype=np.float32).reshape(5, 7, 17) - 200

    def make_segy(self, sorting=2, il_byte=189, xl_byte=193):
        import segyio
        source = self.root / "Full_Stack.sgy"
        spec = segyio.spec()
        spec.ilines, spec.xlines, spec.samples = self.il, self.xl, self.times
        spec.iline, spec.xline, spec.sorting, spec.format = il_byte, xl_byte, sorting, 5
        with segyio.create(str(source), spec) as handle:
            handle.bin[segyio.BinField.Interval] = 4000
            positions = [(i, x) for i in range(len(self.il)) for x in range(len(self.xl))] if sorting == 2 \
                else [(i, x) for x in range(len(self.xl)) for i in range(len(self.il))]
            for trace, (i, x) in enumerate(positions):
                handle.header[trace] = {il_byte: int(self.il[i]), xl_byte: int(self.xl[x]), 109: 1000,
                                        181: 10000 + x * 25, 185: 20000 + i * 25, 71: 1}
                handle.trace[trace] = self.data[i, x]
        return source

    def ingest(self, source):
        with patch.dict(os.environ, {"COPILOT_NO_OPENVDS": "0"}):
            meta = self.store.ingest(source, log=self.messages.append)
        self.assertEqual(meta["kind"], "vds", "\n".join(self.messages))
        return meta

    def test_lossless_values_geometry_and_conversion_metrics(self):
        source = self.make_segy()
        before = source.read_bytes()
        meta = self.ingest(source)
        volume = self.store.Volume(meta)
        np.testing.assert_array_equal(volume.lazy()[:], self.data)
        np.testing.assert_array_equal(volume.samples, self.times)
        np.testing.assert_array_equal(volume.ilines, self.il)
        np.testing.assert_array_equal(volume.xlines, self.xl)
        np.testing.assert_allclose(volume.xy(12, 36), (10075, 20050), atol=1e-6)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(meta["conversion"]["input_bytes"], len(before))
        self.assertEqual(meta["conversion"]["output_bytes"], Path(meta["path"]).stat().st_size)
        self.assertGreater(meta["conversion"]["source_MB_s"], 0)
        self.assertIn("100.00 % Done", next(self.logs.glob("openvds_*.log")).read_text())

    def test_viewer_slices_values_and_interpretation_grid(self):
        volume = self.store.Volume(self.ingest(self.make_segy()))
        with patch.object(volume, "load_async"):
            for axis, index, expected in (("i", 2, self.data[2]), ("x", 3, self.data[:, 3]), ("t", 7, self.data[:, :, 7])):
                np.testing.assert_array_equal(volume.slice(axis, index), expected)
            self.assertEqual(volume.value(2, 3, 4), float(self.data[2, 3, 4]))
        np.testing.assert_array_equal(volume.decimated([0, 2, 4], [1, 3], [2, 5, 8]), self.data[[0, 2, 4]][:, [1, 3]][:, :, [2, 5, 8]])
        np.testing.assert_array_equal(volume.lazy()[::-1, ::2, 8:1:-2], self.data[::-1, ::2, 8:1:-2])
        self.assertEqual(volume.lazy()[0:0].shape, (0, 7, 17))
        with self.assertRaises(IndexError):
            volume.lazy()[99]

    def test_crossline_sort_mismatch_falls_back_safely(self):
        # 3.4.9's importer infers incorrect axis steps for this crossline-sorted
        # fixture. Geometry validation must reject it instead of misplacing data.
        meta = self.store.ingest(self.make_segy(sorting=1), self.messages.append)
        self.assertEqual(meta["kind"], "npy")
        self.assertTrue(any("axes do not match" in message for message in self.messages))
        np.testing.assert_array_equal(self.store.Volume(meta).lazy()[:], self.data)

    def test_nonstandard_header_locations(self):
        meta = self.ingest(self.make_segy(il_byte=9, xl_byte=21))
        np.testing.assert_array_equal(self.store.Volume(meta).lazy()[:], self.data)

    def test_existing_conversion_is_reused(self):
        source = self.make_segy()
        meta = self.ingest(source)
        with patch.object(self.store, "_convert_openvds", side_effect=AssertionError("should not reconvert")):
            self.assertEqual(self.store.ingest(source, self.messages.append), meta)

    def test_disabled_converter_falls_back_without_data_changes(self):
        source = self.make_segy()
        with patch.dict(os.environ, {"COPILOT_NO_OPENVDS": "1"}):
            meta = self.store.ingest(source, self.messages.append)
        self.assertEqual(meta["kind"], "npy")
        np.testing.assert_array_equal(self.store.Volume(meta).lazy()[:], self.data)

    def test_irregular_axis_spacing_falls_back(self):
        self.il = np.array([10, 11, 14, 20, 25])
        meta = self.store.ingest(self.make_segy(), self.messages.append)
        self.assertEqual(meta["kind"], "npy")
        np.testing.assert_array_equal(self.store.Volume(meta).lazy()[:], self.data)

    def test_import_failure_keeps_existing_output_and_live_diagnostics(self):
        source = self.make_segy()
        output = self.root / "existing.vds"
        output.write_bytes(b"existing store must survive")
        log_path = self.logs / "openvds_failure.log"
        def fail(command, **kwargs):
            Path(command[command.index("--vdsfile") + 1]).write_bytes(b"partial")
            kwargs["stdout"].write("live failure diagnostic\n")
            kwargs["stdout"].flush()
            self.assertIn("live failure", log_path.read_text())
            return SimpleNamespace(returncode=7)
        with patch.object(self.vds.subprocess, "run", side_effect=fail), self.assertRaisesRegex(RuntimeError, "exit 7"):
            self.vds.import_segy(source, output, (self.il, self.xl, self.times, 189, 193, False), log_path, self.messages.append)
        self.assertEqual(output.read_bytes(), b"existing store must survive")
        self.assertFalse(list(self.root.glob("*.partial-*.vds")))

    def test_timeout_keeps_log_and_cleans_only_its_partial_file(self):
        source = self.make_segy()
        log_path = self.logs / "timeout.log"
        def timeout(command, **kwargs):
            Path(command[command.index("--vdsfile") + 1]).write_bytes(b"partial")
            kwargs["stdout"].write("last update\n")
            raise subprocess.TimeoutExpired(command, 1)
        with patch.object(self.vds.subprocess, "run", side_effect=timeout), self.assertRaises(subprocess.TimeoutExpired):
            self.vds.import_segy(source, self.root / "out.vds", (self.il, self.xl, self.times, 189, 193, False), log_path)
        self.assertIn("last update", log_path.read_text())
        self.assertFalse(list(self.root.glob("*.partial-*.vds")))

    def test_legacy_mdio_remains_readable(self):
        meta = self.ingest(self.make_segy())
        meta["kind"] = "mdio"
        with patch.object(self.store, "_zarr_array", return_value=self.data):
            np.testing.assert_array_equal(self.store.Volume(meta).lazy()[:], self.data)


if __name__ == "__main__":
    unittest.main()
