"""Regression tests for the stage-5 OpenBLAS crash and held Slurm jobs."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from copilot.runtime import cpu_threads


class ThreadLimitTests(unittest.TestCase):
    def test_login_and_allocated_cpu_limits(self):
        self.assertEqual(cpu_threads({}), 4)
        self.assertEqual(cpu_threads({"SLURM_JOB_ID": "123"}), 1)
        self.assertEqual(cpu_threads({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2", "COPILOT_CPU_THREADS": "8"}), 2)
        self.assertEqual(cpu_threads({"COPILOT_CPU_THREADS": "bad"}), 4)
        self.assertEqual(cpu_threads({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "bad"}), 1)

    def test_fresh_app_overrides_inherited_excessive_threads(self):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="128", OMP_NUM_THREADS="128", MKL_NUM_THREADS="128",
                   SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="1")
        result = subprocess.run([sys.executable, "-c",
            "import copilot,os; import numpy; from threadpoolctl import threadpool_info; "
            "assert os.environ['OPENBLAS_NUM_THREADS']=='1'; "
            "assert os.environ['OMP_NUM_THREADS']=='1'; "
            "assert all(p['num_threads']==1 for p in threadpool_info() if p['user_api']=='blas'); "
            "print('thread limits active')"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("thread limits active", result.stdout)

    def test_real_facies_clustering_with_bounded_blas(self):
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_info
        from copilot.sfm import reduce_and_cluster
        features = np.random.default_rng(1).normal(size=(6, 7, 8, 12)).astype(np.float32)
        original = MiniBatchKMeans.fit
        observed = []
        def fit(model, data, *args, **kwargs):
            observed.extend(p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas")
            return original(model, data, *args, **kwargs)
        with patch.object(MiniBatchKMeans, "fit", fit):
            reduced, labels = reduce_and_cluster(features, k=3, dims=4, log=lambda _: None)
        self.assertTrue(observed)
        self.assertTrue(all(n == 1 for n in observed))
        self.assertEqual(reduced.shape, (6, 7, 8, 4))
        self.assertEqual(labels.shape, (6, 7, 8))
        np.testing.assert_allclose(np.linalg.norm(reduced, axis=-1), 1, atol=1e-5)
        self.assertEqual(len(np.unique(labels)), 3)


class SlurmStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        from copilot import pipeline
        self.pipeline = pipeline
        self.root = Path(self.temp.name)
        self.calls = []
        for owner, name, value in ((pipeline, "I", self.root), (pipeline.config, "LOGS", self.root)):
            context = patch.object(owner, name, value)
            context.start()
            self.addCleanup(context.stop)

    def simulate(self, states, cancel_error=None):
        states = iter(states)
        def command(args, **kwargs):
            self.calls.append(args)
            if args[0] == "sbatch":
                return SimpleNamespace(returncode=0, stdout="123;cluster\n", stderr="")
            if args[0] == "squeue":
                return SimpleNamespace(returncode=0, stdout=next(states), stderr="")
            if args[0] == "scancel":
                if cancel_error:
                    raise cancel_error
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(args)
        with patch.object(self.pipeline.subprocess, "run", side_effect=command), \
                patch.object(self.pipeline.time, "sleep"), patch.object(self.pipeline, "set_status"):
            self.pipeline._slurm_job()

    def test_special_exit_fails_promptly_and_cancels_only_own_job(self):
        with self.assertRaisesRegex(RuntimeError, "123 stopped in SPECIAL_EXIT"):
            self.simulate(["SPECIAL_EXIT\n"])
        self.assertEqual([c for c in self.calls if c[0] == "scancel"], [["scancel", "123"]])
        self.assertEqual(len([c for c in self.calls if c[0] == "squeue"]), 1)

    def test_failed_states_do_not_wait_indefinitely(self):
        for state in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "REQUEUE_HOLD"):
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, state):
                self.simulate([state + "\n"])

    def test_running_job_then_disappearance_keeps_completed_outputs(self):
        (self.root / "rgt.npy").touch()
        (self.root / "facies.npy").touch()
        self.simulate(["RUNNING\n", "", ""])
        self.assertFalse(any(c[0] == "scancel" for c in self.calls))

    def test_completed_state_checks_outputs(self):
        with self.assertRaisesRegex(RuntimeError, "ended without results"):
            self.simulate(["COMPLETED\n"])

    def test_worker_script_limits_threads_before_python(self):
        text = self.pipeline.JOB_TEMPLATE
        self.assertLess(text.index("export OPENBLAS_NUM_THREADS=1"), text.index('"{python}"'))
        self.assertIn("export OMP_NUM_THREADS=1", text)

    def test_unconfirmed_stop_does_not_start_local_fallback(self):
        with patch.object(self.pipeline.shutil, "which", return_value="sbatch"), \
                patch.dict(os.environ, {"COPILOT_NO_SLURM": "0"}), \
                patch.object(self.pipeline, "_slurm_job", side_effect=self.pipeline.SlurmStopError("cancel failed")), \
                patch.object(self.pipeline, "stage_sfm", side_effect=AssertionError("must not start fallback")), \
                self.assertRaisesRegex(self.pipeline.SlurmStopError, "cancel failed"):
            self.pipeline.run_gpu_stages(want=("sfm",))

    def test_missing_scancel_stops_instead_of_falling_back(self):
        with self.assertRaisesRegex(self.pipeline.SlurmStopError, "could not run scancel"):
            self.simulate(["SPECIAL_EXIT\n"], cancel_error=OSError("command unavailable"))


if __name__ == "__main__":
    unittest.main()
