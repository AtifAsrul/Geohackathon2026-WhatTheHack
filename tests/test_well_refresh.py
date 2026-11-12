"""Exercise the HPC repair command on fictional data and verify seismic caches survive."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class WellRefreshTests(unittest.TestCase):
    def test_cli_repair_updates_wells_without_reinterpreting_seismic(self):
        with tempfile.TemporaryDirectory(prefix="copilot_well_refresh_") as tmp:
            base = Path(tmp)
            data, workspace = base / "data", base / "workspace"
            env = dict(os.environ, COPILOT_WORKSPACE=str(workspace), COPILOT_MODELS=str(base / "models"),
                       COPILOT_LLM_URL="mock", COPILOT_NO_OPENVDS="1", COPILOT_NO_SLURM="1", COPILOT_FAULT_METHOD="coherence")

            def run(*args):
                result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-3000:])

            print("\nPreparing fictional survey and initial interpretation...", flush=True)
            run("tests/make_synthetic.py", str(data), "--dead", "--wells", "6")
            run("-m", "copilot.pipeline", "--no-slurm", "--data", str(data), "--stages",
                "convert,wells,grid,rgt,sfm,horizons,faults,calibrate,leads,knowledge")
            interp = workspace / "interp"
            preserved = [interp / name for name in ("grid.json", "seis.npy", "live.npy", "rgt.npy", "rgt.json",
                         "facies.npy", "horizons.npz", "faults.npy", "semb.npy")]
            preserved += list(interp.glob("vol_*.npy"))
            preserved += [p for p in (workspace / "stores").rglob("*") if p.is_file()]

            def snapshot():
                result = {}
                for p in preserved:
                    with p.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    result[str(p)] = (p.stat().st_mtime_ns, digest)
                return result

            before = snapshot()
            # Reproduce the saved input/result state reported by the HPC user.
            project_file = workspace / "project.json"
            project = json.loads(project_file.read_text())
            project.update(data_dir=str(data / "stack"), wells_dir=str(data / "stack"))
            project_file.write_text(json.dumps(project))
            (workspace / "wells.json").write_text(json.dumps({"wells": [], "data_dir": str(data / "stack")}))

            print("Refreshing only wells, calibration, leads and evidence...", flush=True)
            stages = "wells,calibrate,leads,knowledge,dossier"
            run("-m", "copilot.pipeline", "--wells", str(data / "well_data"), "--force", "--stages", stages)
            self.assertEqual(snapshot(), before, "A seismic output was changed or recalculated")
            summary = json.loads((workspace / "wells.json").read_text())
            self.assertEqual(len(summary["wells"]), 6)
            self.assertEqual(summary["counts"]["las"], 6)
            self.assertGreater(len(json.loads((interp / "calib.json").read_text())["wells_used"]), 0)
            self.assertEqual(len(json.loads((workspace / "knowledge.json").read_text())["wells"]), 6)
            updated = json.loads(project_file.read_text())
            self.assertEqual(updated["wells_dir"], str(data / "well_data"))
            self.assertEqual(updated["data_dir"], project["data_dir"])
            self.assertEqual(updated["sources"], project["sources"])
            status = json.loads((workspace / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertEqual([s["id"] for s in status["stages"]], stages.split(","))
            self.assertEqual(json.loads((workspace / "soft_fail.json").read_text()), [])
            print(f"Verified 6 wells and {len(preserved)} unchanged seismic cache files.", flush=True)

            # A cached model-probability stand-in exercises the new fault-only refresh.
            # Real TorchScript execution and tiling are covered separately.
            scientific = [p for p in preserved if p.name not in ("faults.npy", "semb.npy")]
            def scientific_snapshot():
                result = {}
                for p in scientific:
                    with p.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    result[str(p)] = (p.stat().st_mtime_ns, digest)
                return result
            old = scientific_snapshot()
            (interp / "faultssl.npy").write_bytes((interp / "faults_base.npy").read_bytes())
            model_cache = (interp / "faultssl.npy").stat().st_mtime_ns
            print("Applying a 0.5 coherence cutoff from cached model probabilities...", flush=True)
            run("-m", "copilot.pipeline", "--fault-method", "faultssl", "--fault-coherence-max", "0.5",
                "--force", "--stages", "faults,leads,knowledge,dossier")
            self.assertEqual(old, scientific_snapshot())
            self.assertEqual((interp / "faultssl.npy").stat().st_mtime_ns, model_cache)
            import numpy as np
            self.assertEqual(int(np.load(interp / "faults.npy")[np.load(interp / "coherence.npy") >= 0.5].max()), 0)
            self.assertFalse(json.loads((interp / "sticks.json").read_text())["gemma_used"])
            knowledge = json.loads((workspace / "knowledge.json").read_text())
            self.assertIn("FaultSSL", knowledge["methods"]["faults"])
            self.assertEqual(json.loads((workspace / "soft_fail.json").read_text()), [])
            print("Fault-only refresh preserved converted seismic, RGT, facies and horizons; no model rerun.", flush=True)


if __name__ == "__main__":
    unittest.main()
