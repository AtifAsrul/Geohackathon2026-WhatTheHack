"""Fault cutoffs, tiled-model orientation, deterministic sticks and cached refreshes."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from copilot import ai_pass, config, faults, faultssl, llm, pipeline, server, sticks, tools


class CutoffTests(unittest.TestCase):
    def test_strict_cutoff_and_off_restore_the_original_scores(self):
        base = np.array([0.2, 0.4, 0.8, 1], dtype=np.float32)
        coherence = np.array([0.1, 0.49, 0.5, 0.9], dtype=np.float32)
        np.testing.assert_allclose(faults.apply_cutoff(base, coherence, 0.5), [0.2, 0.4, 0, 0])
        np.testing.assert_allclose(faults.apply_cutoff(base, coherence, 1), base)
        np.testing.assert_equal(faults.apply_cutoff(base, coherence, 0), 0)
        np.testing.assert_allclose(base, [0.2, 0.4, 0.8, 1])

    def test_invalid_cutoffs_and_empty_traces(self):
        for bad in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                faults.validate_cutoff(bad)
        result = faults.apply_cutoff(np.ones((2, 3, 4)), np.zeros((2, 3, 4)), 0.5,
                                     live=np.array([[True, False, True], [False, True, False]]))
        self.assertEqual(result.sum(), 12)
        self.assertEqual(faults.apply_cutoff(np.array([1, np.nan]), np.array([np.nan, 0]), 1).sum(), 0)


class TilingTests(unittest.TestCase):
    def test_axis_order_padding_and_overlaps_preserve_a_known_probability_cube(self):
        expected = np.random.default_rng(1).uniform(0, 1, (19, 29, 35)).astype(np.float32)
        shapes = []
        def predict(a):
            shapes.append(a.shape)
            self.assertTrue(all(s % 16 == 0 for s in a.shape))
            return a
        actual = faultssl.predict_tiles(expected, predict, tile=16, overlap=4, log=lambda _: None)
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        self.assertGreater(len(shapes), 1)

    def test_invalid_model_output_is_rejected(self):
        a = np.zeros((17, 19, 21), dtype=np.float32)
        for predict in (lambda p: np.ones((1, 2, 3)), lambda p: np.full(p.shape, np.nan), lambda p: np.full(p.shape, 2)):
            with self.assertRaises(ValueError):
                faultssl.predict_tiles(a, predict, log=lambda _: None)


class CachedFaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.grid = {"ii": [0, 1], "xi": [0, 1, 2], "ti": [0, 1, 2, 3], "ilines": [100, 101],
                     "xlines": [200, 201, 202], "times": [1000, 1004, 1008, 1012], "dt_ms": 4, "shape": [2, 3, 4]}
        (self.root / "grid.json").write_text(json.dumps(self.grid))
        np.save(self.root / "seis.npy", np.zeros((2, 3, 4), dtype=np.float32))
        self.coherence = np.broadcast_to(np.array([0.1, 0.49, 0.5, 0.9], dtype=np.float32), (2, 3, 4)).copy()
        np.save(self.root / "coherence.npy", self.coherence)
        np.save(self.root / "faultssl.npy", np.full((2, 3, 4), 255, dtype=np.uint8))
        np.save(self.root / "faults_base.npy", np.full((2, 3, 4), 255, dtype=np.uint8))
        np.save(self.root / "faults.npy", np.zeros((2, 3, 4), dtype=np.uint8))
        (self.root / "faults.json").write_text(json.dumps({"source": "faultssl", "coherence_max": 1}))
        for owner, name, value in ((tools, "I", self.root), (pipeline, "I", self.root), (config, "INTERP", self.root)):
            ctx = patch.object(owner, name, value)
            ctx.start()
            self.addCleanup(ctx.stop)
        ctx = patch.dict(tools._c, {}, clear=True)
        ctx.start()
        self.addCleanup(ctx.stop)

    def test_preview_filters_each_axis_without_writing_files(self):
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.root.iterdir()}
        for axis, number, shape in (("i", 100, (4, 3)), ("x", 201, (4, 2)), ("t", 1008, (2, 3))):
            data, _ = tools.overlay("faults", axis, number, fault_coherence_max=0.5)
            self.assertEqual(data.shape, shape)
            if axis == "t":
                self.assertEqual(data.max(), 0)  # coherence == 0.5 is excluded
            else:
                self.assertEqual(int(data[:2].min()), 255)
                self.assertEqual(int(data[2:].max()), 0)
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.root.iterdir()})

    def test_apply_reuses_model_probabilities_without_inference_or_gpu_job(self):
        cache = self.root / "faultssl.npy"
        before = (cache.stat().st_mtime_ns, cache.read_bytes(), (self.root / "seis.npy").read_bytes())
        with patch.object(config, "FAULT_METHOD", "faultssl"), patch.object(config, "FAULT_COHERENCE_MAX", 0.5), \
                patch.object(pipeline, "get_live", return_value=np.ones((2, 3), dtype=bool)), \
                patch.object(pipeline, "load_grid", return_value=self.grid), patch.object(pipeline, "log"), \
                patch.object(pipeline.store, "register_derived"), patch.object(sticks, "build"), \
                patch.object(faultssl, "infer", side_effect=AssertionError("must reuse cached prediction")), \
                patch.object(pipeline, "_slurm_job", side_effect=AssertionError("must not queue a GPU")):
            pipeline.stage_faults(force=True)
        result = np.load(self.root / "faults.npy")
        self.assertEqual(int(result[..., :2].min()), 255)
        self.assertEqual(int(result[..., 2:].max()), 0)
        self.assertEqual(before, (cache.stat().st_mtime_ns, cache.read_bytes(), (self.root / "seis.npy").read_bytes()))
        self.assertEqual(json.loads((self.root / "faults.json").read_text())["coherence_max"], 0.5)

    def test_apply_endpoint_selects_only_fault_dependent_stages(self):
        project = self.root / "project.json"
        project.write_text(json.dumps({"sources": ["fictional.sgy"]}))
        req = SimpleNamespace(json=AsyncMock(return_value={"coherence_max": 0.5}))
        with patch.object(pipeline, "PROJECT", project), patch.object(config, "LOGS", self.root), \
                patch.object(config, "set_status"), patch.dict(server._proc, {"p": None}), \
                patch.object(server.subprocess, "Popen") as start:
            self.assertTrue(asyncio.run(server.apply_fault_settings(req))["ok"])
        cmd = start.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--stages") + 1], "faults,leads,knowledge,dossier")
        self.assertEqual(cmd[cmd.index("--fault-method") + 1], "faultssl")
        self.assertIn("--no-slurm", cmd)

    def test_apply_never_restarts_model_when_prediction_cache_is_missing(self):
        (self.root / "faultssl.npy").unlink()
        req = SimpleNamespace(json=AsyncMock(return_value={"coherence_max": 0.5}))
        with patch.object(server.subprocess, "Popen") as start:
            with self.assertRaisesRegex(Exception, "Cached FaultSSL probabilities are missing"):
                asyncio.run(server.apply_fault_settings(req))
        start.assert_not_called()

    def test_gpu_worker_only_prepares_prediction_and_parent_updates_results(self):
        before = (self.root / "faults.npy").read_bytes()
        with patch.object(config, "FAULT_METHOD", "faultssl"), patch.object(config, "FAULT_COHERENCE_MAX", 0.5), \
                patch.object(pipeline, "get_live", return_value=np.ones((2, 3), dtype=bool)), \
                patch.object(sticks, "build", side_effect=AssertionError("parent generates sticks")):
            pipeline.stage_faults(force=True, use_slurm=False, predict_only=True)
        self.assertEqual((self.root / "faults.npy").read_bytes(), before)


class AlgorithmicStickTests(unittest.TestCase):
    def test_fault_lookup_labels_algorithmic_checks_without_gemma_review(self):
        named = [{"id": "F1", "sticks_reviewed": 0}]
        saved = {"method": "algorithmic", "sticks": [{"fault": "F1", "axis": "i", "line": 100,
                 "verdict": "fault", "reviewed_by": "algorithm", "throw_fit": 0.9}]}
        with patch.object(tools, "faults_named", return_value=named), patch.object(tools, "sticks_file", return_value=saved):
            result = tools.fault("F1")
        self.assertEqual(result["sticks_reviewed"], 0)
        self.assertEqual(result["stick_checks"][0]["reviewed_by"], "algorithm")

    def test_sticks_use_measured_offsets_without_any_gemma_calls(self):
        rng = np.random.default_rng(7)
        trace = rng.standard_normal(160).astype(np.float32)
        seis = np.broadcast_to(trace, (24, 40, 160)).copy()
        seis[:, 20:] = np.roll(seis[:, 20:], 4, axis=2)
        fl = np.zeros_like(seis)
        fl[:, 20, 25:140] = 0.95
        grid = {"ilines": list(range(24)), "xlines": list(range(40)), "times": list(range(160)), "dt_ms": 4}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(llm, "ask_json", side_effect=AssertionError("no Gemma")), \
                patch.object(llm, "ask_many", side_effect=AssertionError("no Gemma")):
            result = sticks.build(grid, seis, fl, np.ones((24, 40), dtype=bool), Path(tmp), log=lambda _: None)
            self.assertGreater(result["accepted"], 0)
            self.assertFalse(result["gemma_used"])
            saved = json.loads((Path(tmp) / "sticks.json").read_text())
            self.assertTrue(all(s["reviewed_by"] == "algorithm" for s in saved["sticks"]))


if __name__ == "__main__":
    unittest.main()
