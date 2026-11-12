"""End-to-end checks of the version-4 additions on the small synthetic survey (no HPC, no real model).

    python -m unittest tests.test_v4

The survey is built with an empty wedge at its edge, one fault with a 36 ms throw, a gas sand on a dome,
six wells (one with a checkshot that is 16 ms slow). The whole pipeline is run with the stand-in model,
then the results are compared with the known truth.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(ws, args, **env):
    e = dict(os.environ, COPILOT_WORKSPACE=str(ws), COPILOT_LLM_URL="mock", COPILOT_NO_OPENVDS="1", COPILOT_NO_SLURM="1",
             COPILOT_FAULT_METHOD="coherence")
    e.pop("COPILOT_MOCK_MODE", None)
    e.update(env)
    r = subprocess.run([sys.executable, "-m", "copilot.pipeline", "--no-slurm"] + args, cwd=str(ROOT), env=e,
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


class V4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="copilot_v4_"))
        cls.data, cls.ws = cls.tmp / "demo", cls.tmp / "ws"
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic.py"), str(cls.data), "--dead", "--wells", "6"],
                       check=True, capture_output=True, timeout=600)
        run_pipeline(cls.ws, ["--data", str(cls.data), "--stages", "convert,wells,grid,rgt,sfm,horizons,faults,calibrate,leads,knowledge"])
        cls.k_first = json.loads((cls.ws / "knowledge.json").read_text())
        run_pipeline(cls.ws, ["--stages", "ai,dossier"])
        cls.k = json.loads((cls.ws / "knowledge.json").read_text())
        cls.I = cls.ws / "interp"
        cls.truth = np.load(cls.data / "truth.npz")
        cls.grid = json.loads((cls.I / "grid.json").read_text())
        cls.live = np.load(cls.I / "live.npy")
        cls.dt, cls.t0 = cls.grid["dt_ms"], cls.grid["times"][0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- empty survey edge
    def test_empty_edge_is_found_and_left_out(self):
        dead = self.truth["dead"]
        self.assertEqual(int((dead & self.live).sum()), 0, "an empty trace was treated as live")
        self.assertLess((~self.live).mean(), dead.mean() + 0.06, "too much live area was thrown away")
        self.assertEqual(int(((np.load(self.I / "lead_map.npy") > 0) & ~self.live[None]).sum()), 0, "a lead sits on empty traces")
        self.assertEqual(int(((np.load(self.I / "faults.npy") > 90) & ~self.live[:, :, None]).sum()), 0, "a fault was found on empty traces")

    def test_horizons_are_flat_over_the_empty_edge(self):
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~self.live, return_distances=False, return_indices=True)
        d = ~self.live
        for name in ("horizons.npz", "horizons_gemma.npz"):
            for z in np.load(self.I / name)["z"]:
                self.assertTrue(np.allclose(z[d], z[idx[0][d], idx[1][d]]), f"{name}: a horizon bends inside the empty area")

    # ---------------------------------------------------------------- faults
    def test_one_fault_with_the_right_throw(self):
        f = json.loads((self.I / "fault_surfaces.json").read_text())
        self.assertEqual(len(f), 1, f"expected the single true fault, got {[q['id'] for q in f]}")
        self.assertAlmostEqual(f[0]["throw_ms"], 36.0, delta=4.0)
        self.assertTrue(5196 <= f[0]["where"]["xl"] <= 5224, f[0]["where"])
        s = json.loads((self.I / "sticks.json").read_text())
        self.assertEqual(s["decoys"]["shown"], s["decoys"]["rejected"])
        self.assertTrue(any(q["verdict"] == "not_fault" and not q["accepted"] for q in s["sticks"]), "sand-body edges should be rejected")

    def test_coherence_is_a_viewable_volume_and_low_on_the_fault(self):
        meta = json.loads((self.ws / "stores" / "Coherence.json").read_text())
        self.assertTrue(meta["derived"])
        semb = np.load(self.I / "semb.npy").astype(float) / 255.0
        fault = self.truth["fault"]
        edge = (fault[:, 1:] != fault[:, :-1]) & self.live[:, 1:] & self.live[:, :-1]
        self.assertLess(semb[:, 1:][edge].mean(), semb[self.live].mean() - 0.15)

    # ---------------------------------------------------------------- tracked horizons
    def test_tracked_horizon_follows_the_sand_top_across_the_fault(self):
        d = np.load(self.I / "horizons_gemma.npz")
        truth = (self.truth["sand_top_ms"] - self.t0) / self.dt
        best = None
        for z in d["z"]:
            e = (z - truth)[self.live]
            off = float(np.median(e))
            if abs(off) < 3 and (best is None or abs(off) < abs(best[0])):
                best = (off, e - off, z)
        self.assertIsNotNone(best, "no tracked horizon lies on the sand top")
        self.assertLess(float(np.median(np.abs(best[1]))), 1.0)
        down = self.live & self.truth["fault"]
        self.assertLess(float(np.median(np.abs((best[2] - truth - best[0])[down]))), 1.5, "the horizon was not carried across the fault")
        picks = json.loads((self.I / "picks_gemma.json").read_text())
        tied = sum(h["ties"]["tied"] for h in picks["horizons"])
        mis = sum(h["ties"]["mistied"] for h in picks["horizons"])
        self.assertGreater(tied / max(1, tied + mis), 0.95)

    def test_comparison_with_rgt(self):
        c = json.loads((self.I / "hz_compare.json").read_text())
        same = [p for p in c["pairs"] if p["same_reflector"]]
        self.assertGreaterEqual(len(same), 3)
        for p in same:
            self.assertLessEqual(p["median_abs_ms"], 4.0)
        self.assertTrue(any(k.endswith("__diff") for k in np.load(self.I / "maps_gemma.npz").files))

    # ---------------------------------------------------------------- wells
    def test_well_tie_recovers_the_slow_checkshot(self):
        c = json.loads((self.I / "calib.json").read_text())
        self.assertTrue(c["shifts"]["w-5"]["clear"])
        self.assertAlmostEqual(c["shifts"]["w-5"]["shift_ms"], -16.0, delta=4.0)
        self.assertFalse(c["shifts"]["w-1"]["clear"] and abs(c["shifts"]["w-1"]["shift_ms"]) > 4)

    def test_rock_certainty_is_checked_and_honest(self):
        from copilot import calibrate
        c = json.loads((self.I / "calib.json").read_text())
        self.assertIn(c["model"], ("isotonic", "logistic"))
        self.assertLess(c["error_score"][c["model"]], c["error_score"]["baseline"])
        rel = c["reliability"]
        self.assertLess(rel[0]["observed"], rel[-1]["observed"])
        hi, lo = calibrate.p_sand(c, {"sand": 0.95}), calibrate.p_sand(c, {"sand": 0.08})
        self.assertEqual((hi["rock"], lo["rock"]), ("sand", "shale"))
        self.assertLess(hi["certainty"], 1.0)
        self.assertIn("wells", hi["basis"])
        self.assertIsNone(calibrate.p_sand(c, {}))

    def test_log_zones_and_extrapolated_time_depth(self):
        from copilot import wells
        z = json.loads((self.I / "well_zones.json").read_text())
        self.assertGreater(z["w-1"]["summary"]["net_pay_m"], 20.0)       # the gas sand on the dome
        self.assertEqual(z["w-6"]["summary"]["net_pay_m"], 0.0)
        t, n = wells.td_time(np.array([50.0, 150.0, 400.0]), np.array([100.0, 200.0, 300.0]), np.array([80.0, 160.0, 240.0]))
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(t[2]), 320.0)                       # carried on at the last interval velocity
        self.assertAlmostEqual(float(t[0]), 40.0)

    # ---------------------------------------------------------------- facies, evidence, briefing
    def test_every_facies_has_a_name(self):
        f = self.k["facies"]
        self.assertTrue(all(q.get("name") for q in f))
        self.assertEqual(sum(1 for q in f if q.get("no_data")), 1)
        self.assertTrue(all(q.get("reading") and q.get("lithology") for q in f if not q.get("no_data")))

    def test_evidence_ids_do_not_change_when_the_assistant_pass_is_added(self):
        a, b = self.k_first["evidence"], self.k["evidence"]
        self.assertGreater(len(b), len(a))
        self.assertEqual([(e["id"], e["text"]) for e in a], [(e["id"], e["text"]) for e in b[:len(a)]])
        src = {e["source"] for e in b}
        self.assertTrue({"measured", "well-log", "well-check", "assistant-visual", "comparison"} <= src, src)

    def test_leads_carry_the_tied_evidence(self):
        l = self.k["leads"][0]
        et = {e["id"]: e for e in self.k["evidence"]}
        kinds = {et[e]["source"] for e in l["for"] + l["against"] + l["unknown"]}
        self.assertTrue({"measured", "well-check", "comparison"} <= kinds, kinds)
        self.assertIn("ROCK CERTAINTY", self.k["briefing"])
        self.assertIn("NAMED FAULTS", self.k["briefing"])
        self.assertIn("SECOND-OPINION HORIZONS", self.k["briefing"])

    # ---------------------------------------------------------------- the assistant: cache and bad answers
    def test_second_pass_asks_nothing_new(self):
        ws = self.tmp / "ws_cache"
        shutil.copytree(self.ws, ws)
        run_pipeline(ws, ["--stages", "ai,dossier", "--force"])
        d = json.loads((ws / "interp" / "ai_done.json").read_text())
        self.assertEqual(d["requests"]["asked"], 0)
        self.assertGreater(d["requests"]["cached"], 20)

    def test_bad_answers_never_stop_the_run(self):
        for mode in ("garbage", "wrongids", "timeout"):
            ws = self.tmp / f"ws_{mode}"
            shutil.copytree(self.ws, ws)
            shutil.rmtree(ws / "llm_cache")
            run_pipeline(ws, ["--stages", "ai,dossier", "--force"], COPILOT_MOCK_MODE=mode)
            I = ws / "interp"
            f = json.loads((I / "fault_surfaces.json").read_text())
            self.assertEqual(len(f), 1, f"{mode}: the app's own fault test should still find the fault")
            self.assertEqual(f[0]["sticks_reviewed"], 0, mode)
            self.assertTrue((I / "horizons_gemma.npz").exists(), mode)
            p = json.loads((I / "picks_gemma.json").read_text())
            self.assertFalse(p["selection_votes_used"], mode)
            k = json.loads((ws / "knowledge.json").read_text())
            self.assertTrue(any("no usable answer" in q or "not reviewed" in q for q in k["data_quality"]), f"{mode}: the limit is not reported")

    # ---------------------------------------------------------------- chat plumbing
    def test_chat_tools_and_pictures(self):
        env = dict(COPILOT_WORKSPACE=str(self.ws), COPILOT_LLM_URL="mock")
        code = (
            "import json\n"
            "from copilot import llm, tools\n"
            "v = {'il': 2055, 'xl': 5120, 'twt': 1390, 'axis': 'i'}\n"
            "out = {}\n"
            "for name, args in [('look_slice', {'twt': 1390}), ('look_well', {'name': 'w-1'}), ('fault', {'id': 'F1'}),\n"
            "                   ('compare_horizons', {}), ('dossier', {'lead': 'L1'}), ('dossier', {'il': 2055, 'xl': 5120, 'twt': 1390})]:\n"
            "    r, png, act = llm.run_tool(name, args, v)\n"
            "    out[name + str(len(out))] = [r[:60], bool(png and png[:4] == b'\\x89PNG')]\n"
            "out['view'] = tools.render_view(v)[:4] == b'\\x89PNG'\n"
            "ev = list(llm.chat([{'role': 'user', 'content': 'Explain lead L1 and fault F1 near well w-1'}], v, True))\n"
            "out['steps'] = [e['text'] for e in ev if e['type'] == 'step']\n"
            "p = tools.probe(2055, 5120, 1390)\n"
            "out['probe'] = [p.get('facies_name'), (p.get('sand_certainty') or {}).get('rock'), bool(p.get('nearest_well_logs_here'))]\n"
            "print('RESULT' + json.dumps(out))\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ, **env), capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = json.loads(r.stdout.split("RESULT")[-1])
        self.assertTrue(out["look_slice0"][1] and out["look_well1"][1] and out["view"])
        self.assertFalse(any("failed" in v[0] for k, v in out.items() if isinstance(v, list) and k[-1].isdigit()), out)
        self.assertTrue(any(s.startswith("Located:") for s in out["steps"]))
        self.assertTrue(any("L1" in s for s in out["steps"]) and any("F1" in s for s in out["steps"]) and any("w-1" in s for s in out["steps"]))
        self.assertEqual(out["probe"][1], "sand")
        self.assertTrue(out["probe"][0] and out["probe"][2])


if __name__ == "__main__":
    unittest.main()
