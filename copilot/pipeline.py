"""The interpretation pipeline, start to finish.

    python -m copilot.pipeline --data /path/to/challenge/folder

Steps (each one saves its result, so a re-run only redoes what is missing):
  convert  SEG-Y -> local OpenVDS stores (or segyio if OpenVDS cannot)
  wells    well list, deviations, checkshots, LAS logs
  grid     thin all volumes onto one interpretation grid
  rgt      Relative Geologic Time with RGT-Est            (GPU through Slurm if possible)
  sfm      foundation-model features -> seismic facies    (GPU through Slurm if possible)
  horizons key horizons as iso-RGT surfaces
  faults   coherence and fault likelihood
  calibrate  well ties, log zones, rock certainty checked against the wells
  leads    maps, traps, ranked leads, evidence
  knowledge  the briefing the chat assistant reads      (the app is usable from here on)
  ai       the assistant's own pass: it looks at pictures and reviews horizons, faults, slices, wells, facies
  dossier  merges that pass into the evidence
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import config, store
from .config import read_json, set_status, write_json

I = config.INTERP
STAGES = ["convert", "wells", "grid", "rgt", "sfm", "horizons", "faults", "calibrate", "leads", "knowledge",
          "ai", "dossier"]
STAGE_LABEL = {"convert": "Convert volumes", "wells": "Read wells", "grid": "Prepare grid",
               "rgt": "Relative geologic time (RGT-Est)", "sfm": "Foundation model facies",
               "horizons": "Horizons", "faults": "Faults", "calibrate": "Well ties and rock certainty",
               "leads": "Traps and leads", "knowledge": "Briefing", "ai": "Assistant's own interpretation",
               "dossier": "Final evidence"}
SOFT = {"calibrate", "ai", "dossier"}       # a failure in these never stops the run; the limit is reported instead
PROJECT = config.WORKSPACE / "project.json"


def log(msg):
    set_status(message=str(msg))


def find_segy(data_dir):
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in (".sgy", ".segy") and p.is_file())


# ------------------------------------------------------------------ stages
def stage_convert(force=False):
    proj = read_json(PROJECT, {})
    files = [Path(p) for p in proj.get("sources", [])]
    if not files:
        raise RuntimeError("No SEG-Y files selected. Import a data folder first.")
    for n, f in enumerate(files):
        set_status(message=f"Volume {n + 1} of {len(files)}: {f.name}")
        try:
            store.ingest(f, log=log, force=force)
        except Exception as e:
            log(f"!! {f.name} could not be converted: {type(e).__name__}: {e}")
    if not store.list_volumes():
        raise RuntimeError("None of the SEG-Y files could be converted.")


def stage_wells(force=False):
    from . import wells
    proj = read_json(PROJECT, {})
    explicit = proj.get("wells_dir")
    d = explicit or wells.guess_folder(proj.get("data_dir"))
    if not d or not Path(d).exists():
        if explicit:
            raise RuntimeError(f"The selected well folder does not exist: {d}. Choose the well data folder again.")
        log("No well folder given - continuing without wells")
        write_json(wells.WELLS_JSON, {"wells": [], "notes": ["no well data folder"]})
        return
    if not wells.has_files(d):
        guess = wells.guess_folder(d)
        hint = f" Use --wells {guess}." if guess else " Choose the folder containing the well files."
        raise RuntimeError(f"No well files found in {d}." + hint)
    try:
        wells.load_wells(d, store.reference_volume(), log=log)
    except Exception as e:
        raise RuntimeError(f"Well data could not be read from {d}: {type(e).__name__}: {e}") from e


def stage_grid(force=False):
    if (I / "grid.json").exists() and (I / "seis.npy").exists() and not force:
        return
    ref = store.reference_volume()
    ni, nx, nt = ref.shape
    si, sx = math.ceil(ni / config.GRID_MAX_LATERAL), math.ceil(nx / config.GRID_MAX_LATERAL)
    st = math.ceil(nt / config.GRID_MAX_TIME)
    ii, xi, ti = np.arange(0, ni, si), np.arange(0, nx, sx), np.arange(0, nt, st)
    grid = {"ref": ref.name, "ii": ii, "xi": xi, "ti": ti, "ilines": ref.ilines[ii], "xlines": ref.xlines[xi],
            "times": ref.samples[ti], "dt_ms": float(np.median(np.diff(ref.samples[ti]))) if len(ti) > 1 else 4.0,
            "stride": [si, sx, st], "shape": [len(ii), len(xi), len(ti)], "cell_m": None}
    a = ref.meta.get("affine")
    if a:
        b = ref.bin_size()
        grid["cell_m"] = [b[0] * si, b[1] * sx]
        sgi = 1.0 if ref.ilines[-1] >= ref.ilines[0] else -1.0
        sgx = 1.0 if ref.xlines[-1] >= ref.xlines[0] else -1.0
        grid["u_il"] = [sgi * a["xi"] / math.hypot(a["xi"], a["yi"]), sgi * a["yi"] / math.hypot(a["xi"], a["yi"])]
        grid["u_xl"] = [sgx * a["xx"] / math.hypot(a["xx"], a["yx"]), sgx * a["yx"] / math.hypot(a["xx"], a["yx"])]
    log(f"Interpretation grid: {len(ii)} x {len(xi)} x {len(ti)} (every {si} inline, {sx} crossline, {st} sample)")
    np.save(I / "seis.npy", ref.decimated(ii, xi, ti, log))
    (I / "live.npy").unlink(missing_ok=True)
    roles = {}
    for v in store.list_volumes():
        role = v.meta["role"]
        if v.name == ref.name or role in roles or role == "other" or v.meta.get("derived"):
            continue
        # match by inline / crossline number and time, so differently-cropped volumes still line up
        vi = np.abs(v.ilines[None, :] - ref.ilines[ii][:, None]).argmin(1)
        vx = np.abs(v.xlines[None, :] - ref.xlines[xi][:, None]).argmin(1)
        vt = np.abs(v.samples[None, :] - ref.samples[ti][:, None]).argmin(1)
        np.save(I / f"vol_{role}.npy", v.decimated(vi, vx, vt, log))
        roles[role] = v.name
    grid["roles"] = dict(roles, **{ref.meta["role"]: ref.name})
    write_json(I / "grid.json", grid)


def load_grid():
    g = read_json(I / "grid.json")
    if g:
        for k in ("ii", "xi", "ti", "ilines", "xlines", "times"):
            g[k] = np.asarray(g[k])
    return g


def get_live():
    """(inline, crossline) mask of traces that hold real data on the interpretation grid."""
    p = I / "live.npy"
    if p.exists():
        return np.load(p)
    from . import rgt
    live = rgt.live_mask(np.load(I / "seis.npy", mmap_mode="r"))
    np.save(p, live)
    if not live.all():
        log(f"{100 * (1 - live.mean()):.1f}% of the survey area has no traces (empty edge): it is left out of the "
            f"analysis and horizons are carried flat across it")
    return live


def stage_rgt(force=False, allow_classical=True):
    """RGT-Est if its weights and PyTorch are available; classical estimate otherwise."""
    from . import rgt
    if (I / "rgt.npy").exists() and not force:
        return
    live = get_live()
    seis = rgt.fill_dead(np.load(I / "seis.npy"), live)      # empty traces borrow their nearest neighbour
    method = None
    if config.RGT_WEIGHTS.exists():
        try:
            out, used = rgt.rgt_est(seis, config.RGT_WEIGHTS, log=log)
            method = "RGT-Est (pretrained deep network, Dou et al. 2026)"
            if used != (400, 512, 512):
                method += f", run on a reduced {used[0]}x{used[1]}x{used[2]} grid"
        except Exception as e:
            log(f"RGT-Est could not run here: {type(e).__name__}: {str(e)[:300]}")
            if not allow_classical:
                raise
    else:
        log("RGT-Est weights are not downloaded")
        if not allow_classical:
            raise RuntimeError("RGT-Est weights missing")
    if method is None:
        out = rgt.rgt_classical(seis, log=log)
        method = "classical dip-steered RGT (RGT-Est was not available)"
    np.save(I / "rgt.npy", rgt.fill_dead(out, live).astype(np.float32))
    write_json(I / "rgt.json", {"method": method})
    log(f"RGT ready: {method}")


def stage_sfm(force=False):
    from . import sfm
    if (I / "facies.npy").exists() and not force:
        return
    g = load_grid()
    seis = np.load(I / "seis.npy")
    extra = [np.load(I / f"vol_{r}.npy") for r in ("sand", "ai", "vpvs") if (I / f"vol_{r}.npy").exists()]
    sfm.compute(seis, g["dt_ms"], I, extra=extra, log=log)


def stage_horizons(force=False):
    from . import rgt
    if (I / "horizons.npz").exists() and not force:
        return
    live = get_live()
    # empty traces take their nearest live neighbour, so every surface carries on flat across an empty edge
    seis = rgt.fill_dead(np.load(I / "seis.npy"), live)
    r = rgt.fill_dead(np.load(I / "rgt.npy"), live)
    hz, _, _ = rgt.pick_key_horizons(seis, r, log=log)
    for h in hz:
        h["z"] = rgt.fill_dead(h["z"], live)
    np.savez_compressed(I / "horizons.npz", names=np.array([h["name"] for h in hz]),
                        z=np.stack([h["z"] for h in hz]).astype(np.float32),
                        rgt=np.array([h["rgt"] for h in hz]), polarity=np.array([h["polarity"] for h in hz]),
                        strength=np.array([h["strength"] for h in hz]))
    log(f"{len(hz)} key horizons picked")


def load_horizons(source="rgt"):
    """source 'rgt' = surfaces from the RGT cube; 'gemma' = the assistant's tracked picks (same layout)."""
    p = I / ("horizons_gemma.npz" if source == "gemma" else "horizons.npz")
    if not p.exists():
        return []
    d = np.load(p)
    return [{"name": str(n), "z": d["z"][k], "rgt": float(d["rgt"][k]), "polarity": str(d["polarity"][k]),
             "strength": float(d["strength"][k])} for k, n in enumerate(d["names"])]


def stage_faults(force=False):
    from . import faults
    if (I / "faults.npy").exists() and not force:
        return
    from . import rgt
    live = get_live()
    fl, semb = faults.fault_likelihood(rgt.fill_dead(np.load(I / "seis.npy"), live), log=log, live=live)
    np.save(I / "faults.npy", (fl * 255).astype(np.uint8))
    np.save(I / "semb.npy", (semb * 255).astype(np.uint8))
    try:        # show the coherence cube in the viewer's Volume list (black = neighbouring traces differ)
        store.register_derived("Coherence", "Coherence (black = break)", "coh", semb, load_grid(), cmap="gray",
                               lo=float(np.percentile(semb[::4, ::4, ::4], 2)), hi=1.0)
    except Exception as e:
        log(f"(the coherence cube could not be added to the viewer: {e})")


def stage_leads(force=False):
    from . import attributes, wells
    if (I / "interp.json").exists() and not force:
        return
    g = load_grid()
    seis = np.load(I / "seis.npy")
    vols = {r: np.load(I / f"vol_{r}.npy", mmap_mode="r") for r in store.ROLE_ORDER if (I / f"vol_{r}.npy").exists()}
    vols[next((k for k, v in (g.get("roles") or {}).items() if v == g["ref"]), "full")] = seis
    fl = np.load(I / "faults.npy").astype(np.float32) / 255.0
    semb = np.load(I / "semb.npy").astype(np.float32) / 255.0
    fac = np.load(I / "facies.npy") if (I / "facies.npy").exists() else None
    parts, maps, lead_map = attributes.build(g, seis, vols, load_horizons(), fl, semb, fac, wells.wells_summary(), log=log,
                                             live=get_live(), calib=read_json(I / "calib.json"))
    np.savez_compressed(I / "maps.npz", **maps)
    np.save(I / "lead_map.npy", lead_map)
    write_json(I / "interp.json", parts)
    log(f"{len(parts['leads'])} leads ranked")


def stage_calibrate(force=False):
    from . import calibrate
    if (I / "calib.json").exists() and not force:
        return
    calibrate.run(log=log)


def stage_knowledge(force=False):
    from . import knowledge
    knowledge.build()
    log("Briefing for the assistant is ready - the app can be used from now on")


def stage_ai(force=False):
    from . import ai_pass
    if (I / "ai_done.json").exists() and not force:
        return
    ai_pass.run(log=log)


def stage_dossier(force=False):
    from . import knowledge
    knowledge.build()
    log("Evidence file updated with the assistant's own interpretation")


FUNCS = {"convert": stage_convert, "wells": stage_wells, "grid": stage_grid, "rgt": stage_rgt, "sfm": stage_sfm,
         "horizons": stage_horizons, "faults": stage_faults, "calibrate": stage_calibrate, "leads": stage_leads,
         "knowledge": stage_knowledge, "ai": stage_ai, "dossier": stage_dossier}


# ------------------------------------------------------------------ GPU through Slurm
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --partition=petronas_apgce
#SBATCH --constraint=nodetype-h200.B-3072
#SBATCH --mcs-label=apgce
#SBATCH --mem={mem}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --job-name=copilot
#SBATCH --output={log}

cd "{root}"
export COPILOT_WORKSPACE="{ws}"
export COPILOT_MODELS="{models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1
"{python}" -u -m copilot.pipeline --stages {stages} --in-job --strict
"""


def gpu_outputs_ready():
    return (I / "rgt.npy").exists() and (I / "facies.npy").exists()


def _ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 0.0


class SlurmStopError(RuntimeError):
    """The previous worker could not be stopped safely; do not start a fallback."""


def run_gpu_stages(force=False, use_slurm=True, want=("rgt", "sfm")):
    """Run RGT-Est and/or the foundation model on a GPU node when one can be had, else here."""
    out = {"rgt": I / "rgt.npy", "sfm": I / "facies.npy"}
    if force:
        for s in want:
            out[s].unlink(missing_ok=True)
    todo = [s for s in want if not out[s].exists()]
    if not todo:
        return
    if use_slurm and shutil.which("sbatch") and os.environ.get("COPILOT_NO_SLURM") != "1":
        try:
            _slurm_job(todo)
        except SlurmStopError:
            raise
        except Exception as e:
            log(f"GPU job did not work out ({e}) - continuing on this machine")
    # whatever is still missing is done locally (GPU if this machine has one, else CPU)
    if "rgt" in todo and not out["rgt"].exists():
        set_status(stage="rgt")
        done = False
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            pass
        if config.RGT_WEIGHTS.exists() and (has_gpu or _ram_gb() >= 48 or os.environ.get("COPILOT_RGT_FORCE_CPU") == "1"):
            limit = int(os.environ.get("COPILOT_RGT_CPU_MIN", "60"))
            log(f"Trying RGT-Est on this machine (up to {limit} minutes)")
            try:
                r = subprocess.run([sys.executable, "-u", "-m", "copilot.pipeline", "--stages", "rgt", "--in-job", "--strict"],
                                   cwd=str(config.ROOT), timeout=limit * 60)
                done = r.returncode == 0 and out["rgt"].exists()
            except subprocess.TimeoutExpired:
                log("RGT-Est took too long on this machine")
        elif config.RGT_WEIGHTS.exists():
            log("This machine has too little memory for RGT-Est without a GPU")
        if not done:
            _classical_rgt()
    if "sfm" in todo and not out["sfm"].exists():
        set_status(stage="sfm")
        stage_sfm()


def _classical_rgt():
    from . import rgt
    live = get_live()
    out = rgt.rgt_classical(rgt.fill_dead(np.load(I / "seis.npy"), live), log=log)
    np.save(I / "rgt.npy", rgt.fill_dead(out, live).astype(np.float32))
    write_json(I / "rgt.json", {"method": "classical dip-steered RGT (RGT-Est could not run)"})


def _slurm_job(todo=("rgt", "sfm")):
    set_status(stage=todo[0])
    script = config.LOGS / "gpu_job.sh"
    joblog = config.LOGS / "gpu_job_%j.log"
    jid = None
    for mem in (32000, 16000):
        script.write_text(JOB_TEMPLATE.format(mem=mem, log=joblog, root=config.ROOT, ws=config.WORKSPACE,
                                              models=config.MODELS, python=sys.executable, stages=",".join(todo)))
        r = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            jid = r.stdout.strip().split(";")[0]
            break
        log(f"sbatch refused the job ({(r.stderr or r.stdout).strip()[:200]})")
    if jid is None:
        raise RuntimeError("could not submit a GPU job")
    log(f"GPU job {jid} submitted - waiting for a GPU")
    wait_min = float(os.environ.get("COPILOT_SLURM_WAIT_MIN", "20"))
    t0, last, gone = time.time(), None, 0
    while True:
        time.sleep(5)
        set_status()                                   # heartbeat so the page knows we are still alive
        r = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True)
        state = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if not state:
            gone += 1                                  # one empty answer can be a hiccup; two means finished
            if gone >= 2:
                break
            continue
        gone = 0
        if state != last:
            set_status(message=f"GPU job {jid}: {state.lower()}")
            last = state
        if state in {"SPECIAL_EXIT", "REQUEUE_HOLD", "REVOKED", "FAILED", "CANCELLED", "TIMEOUT",
                     "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}:
            # A held/crashed job will not make progress while it stays in squeue.
            # Do not fall back while an automatic requeue could restart the same worker.
            # Cancel only this job, which this process just submitted.
            try:
                stopped = subprocess.run(["scancel", jid], capture_output=True, text=True)
            except OSError as exc:
                raise SlurmStopError(f"GPU job {jid} is {state}; could not run scancel: {exc}") from exc
            if stopped.returncode != 0 and "Invalid job id" not in stopped.stderr:
                raise SlurmStopError(f"GPU job {jid} is {state}; could not cancel it: {stopped.stderr.strip()[:300]}")
            raise RuntimeError(f"GPU job {jid} stopped in {state}; see {str(joblog).replace('%j', jid)}")
        if state == "COMPLETED":
            break
        if state == "PENDING" and time.time() - t0 > wait_min * 60:
            subprocess.run(["scancel", jid])
            raise RuntimeError(f"no GPU became free within {wait_min:.0f} minutes")
    if not all({"rgt": I / "rgt.npy", "sfm": I / "facies.npy"}[s].exists() for s in todo):
        tail = ""
        try:
            tail = Path(str(joblog).replace("%j", jid)).read_text()[-600:]
        except Exception:
            pass
        raise RuntimeError(f"the GPU job ended without results. Last lines of its log: {tail}")


# ------------------------------------------------------------------ orchestration
def run(stages=None, force=False, use_slurm=True, in_job=False, strict=False):
    stages = stages or STAGES
    from .runtime import cpu_threads
    print(f"[runtime] Process-local thread limits: BLAS=1, OpenMP={cpu_threads()}", flush=True)
    if not in_job:
        set_status(stage=stages[0], message="Starting", state="running", error=None, log=[],
                   stages=[{"id": s, "label": STAGE_LABEL[s]} for s in stages], started=time.time())
        write_json(config.WORKSPACE / "soft_fail.json", [q for q in (read_json(config.WORKSPACE / "soft_fail.json", []) or [])
                                                         if q.get("stage") not in stages])
    failed = False
    try:
        k = 0
        while k < len(stages):
            s = stages[k]
            if in_job:
                set_status(stage=s)
                try:
                    if s == "rgt":
                        stage_rgt(force, allow_classical=not strict)
                    else:
                        FUNCS[s](force)
                except Exception as e:      # keep going: the other GPU step may still succeed
                    traceback.print_exc()
                    log(f"!! {STAGE_LABEL[s]} failed here: {type(e).__name__}: {str(e)[:300]}")
                    failed = True
            elif s in ("rgt", "sfm"):
                want = [s]
                while k + 1 < len(stages) and stages[k + 1] in ("rgt", "sfm"):
                    k += 1
                    want.append(stages[k])
                run_gpu_stages(force, use_slurm, want)
            elif s in SOFT:
                set_status(stage=s, message=STAGE_LABEL[s])
                try:
                    FUNCS[s](force)
                except Exception as e:      # optional step: report it and carry on with what we have
                    traceback.print_exc()
                    log(f"!! {STAGE_LABEL[s]} did not finish ({type(e).__name__}: {str(e)[:200]}) - continuing without it")
                    sf = read_json(config.WORKSPACE / "soft_fail.json", []) or []
                    write_json(config.WORKSPACE / "soft_fail.json",
                               sf + [{"stage": s, "label": STAGE_LABEL[s], "error": f"{type(e).__name__}: {str(e)[:200]}"}])
            else:
                set_status(stage=s, message=STAGE_LABEL[s])
                FUNCS[s](force)
            k += 1
        if in_job and failed:
            sys.exit(1)
        if not in_job:
            set_status(stage="done", message="Interpretation finished", state="done")
    except Exception as e:
        traceback.print_exc()
        if in_job:
            log(f"!! {type(e).__name__}: {e}")
            sys.exit(1)
        set_status(message=f"Stopped: {type(e).__name__}: {e}", state="error", error=str(e))
        sys.exit(1)


def set_project(data_dir=None, sources=None, wells_dir=None):
    proj = read_json(PROJECT, {})
    if data_dir:
        proj["data_dir"] = str(data_dir)
        if not sources:
            sources = [str(p) for p in find_segy(data_dir)]
    if sources:
        proj["sources"] = [str(s) for s in sources]
    if wells_dir is not None:
        proj["wells_dir"] = str(wells_dir) if wells_dir else None
    write_json(PROJECT, proj)
    return proj


def reset_interpretation():
    """Forget every interpretation product (the converted volumes are kept)."""
    for p in I.glob("*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    (config.WORKSPACE / "knowledge.json").unlink(missing_ok=True)
    (config.WORKSPACE / "soft_fail.json").unlink(missing_ok=True)
    store.remove_derived()


def main():
    ap = argparse.ArgumentParser(description="Exploration Copilot interpretation pipeline")
    ap.add_argument("--data", help="folder holding the SEG-Y volumes and the well data")
    ap.add_argument("--wells", help="folder holding the well data, if it is not inside --data")
    ap.add_argument("--stages", help="comma-separated subset of: " + ",".join(STAGES))
    ap.add_argument("--force", action="store_true", help="redo steps even if their results exist")
    ap.add_argument("--no-slurm", action="store_true", help="never submit a GPU job, run everything here")
    ap.add_argument("--in-job", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.data or a.wells:
        set_project(a.data, wells_dir=a.wells)
    stages = [s.strip() for s in a.stages.split(",")] if a.stages else None
    if a.force and not a.stages and not a.in_job:
        reset_interpretation()
    run(stages, a.force and bool(a.stages), not a.no_slurm, a.in_job, a.strict)


if __name__ == "__main__":
    main()
