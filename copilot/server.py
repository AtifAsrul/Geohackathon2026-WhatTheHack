"""Web server: serves the page, the slices, the overlays and the chat.

    python -m copilot.server        (then open http://127.0.0.1:8050 in Firefox on the HPC desktop)

It listens on this machine only (127.0.0.1): nothing is reachable from outside,
and the only place data is sent to is the Gemma model hosted on the same HPC.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, knowledge, llm, pipeline, store, tools, wells
from .config import read_json

app = FastAPI(title="Exploration Copilot")
app.mount("/static", StaticFiles(directory=str(config.WEB)), name="static")
_proc = {"p": None}


@app.middleware("http")
async def no_cache(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-store"       # always serve the current page files
    return resp


def _clean(o):
    """Make numpy things JSON-safe."""
    return json.loads(json.dumps(o, default=config._json_default))


@app.get("/")
def index():
    return FileResponse(str(config.WEB / "index.html"), headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ state
def _status():
    s = read_json(config.STATUS_FILE, {}) or {}
    p = _proc["p"]
    if s.get("state") == "running":
        if p is not None and p.poll() is not None and p.returncode != 0:
            s["state"], s["error"] = "error", s.get("error") or "The run stopped unexpectedly. See workspace/logs/pipeline.log"
        elif p is None and time.time() - s.get("updated", 0) > 90 * 60:
            s["state"], s["error"] = "error", "The run seems to have stopped (no progress for a long time)."
    return s


@app.get("/api/state")
def state():
    vols = []
    for v in store.list_volumes():
        m = v.meta
        vols.append({"name": v.name, "label": m["label"], "role": m["role"], "shape": m["shape"],
                     "ilines": m["ilines"], "xlines": m["xlines"], "samples": m["samples"],
                     "stats": m["stats"], "converter": m.get("converter"), "affine": m.get("affine")})
    ref = store.reference_volume()
    k = knowledge.load() if tools.ready() else None
    interp = {"ready": bool(k)}
    if k:
        h = tools.horizons()
        interp.update({
            "horizons": [{"name": n, "color": tools.HCOLORS[i % len(tools.HCOLORS)]} for i, n in enumerate(h["names"])] if h else [],
            "leads": [{"id": l["id"], "kind": l["kind"], "horizon": l["horizon"], "where": l["where"],
                       "area_km2": l.get("area_km2"), "relief_ms": l["relief_ms"], "scores": l["scores"],
                       "rating": l["rating"], "tested": l["tested"], "wells_inside": l.get("wells_inside", []),
                       "for": l["for"], "against": l["against"], "unknown": l["unknown"]} for l in k.get("leads", [])],
            "methods": k.get("methods"), "facies": k.get("facies"), "facies_colors": tools.FACIES_COLORS,
            "built": k.get("built"), "data_quality": k.get("data_quality", []),
            "map_attrs": sorted({key.split("__")[1] for key in (tools.maps() or {})}),
            "map_labels": tools.MAP_LABEL,
        })
    ws = wells.wells_summary()
    ok, msg = llm.check()
    proj = read_json(pipeline.PROJECT, {})
    return _clean({"volumes": vols, "ref": ref.name if ref else None, "interp": interp,
                   "wells": [{k2: w.get(k2) for k2 in ("name", "il", "xl", "path", "has", "curves", "td_source", "info", "inside_survey", "notes")}
                             for w in ws.get("wells", [])],
                   "status": _status(), "llm": {"ok": ok, "msg": msg, "model": config.LLM_MODEL},
                   "project": proj, "default_data_dir": config.DEFAULT_DATA_DIR})


@app.get("/api/status")
def status():
    return _clean(_status())


@app.get("/api/knowledge")
def get_knowledge():
    k = knowledge.load() or {}
    return _clean({"evidence": k.get("evidence", []), "briefing": k.get("briefing", ""), "leads": k.get("leads", []),
                   "horizons": k.get("horizons", []), "facies": k.get("facies", []), "calibration": k.get("calibration", []),
                   "methods": k.get("methods", {}), "data_quality": k.get("data_quality", [])})


# ------------------------------------------------------------------ data import + run
@app.get("/api/browse")
def browse(path: str = ""):
    p = Path(path or config.DEFAULT_DATA_DIR).expanduser()
    if not p.exists():
        p = Path.home()
    if p.is_file():
        p = p.parent
    dirs, files = [], []
    try:
        for c in sorted(p.iterdir(), key=lambda q: q.name.lower()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                dirs.append(c.name)
            elif c.suffix.lower() in (".sgy", ".segy", ".las", ".txt", ".xlsx", ".csv"):
                files.append({"name": c.name, "mb": round(c.stat().st_size / 1e6, 1)})
    except PermissionError:
        pass
    segy = [str(q) for q in pipeline.find_segy(p)][:200] if p.exists() else []
    n_las = sum(1 for _ in p.rglob("*.las")) if p.exists() else 0
    return {"path": str(p), "parent": str(p.parent), "dirs": dirs, "files": files, "segy_found": len(segy), "las_found": n_las}


@app.post("/api/import")
async def do_import(req: Request):
    body = await req.json()
    p = Path(body.get("path", ""))
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, "That folder does not exist on this machine.")
    segy = pipeline.find_segy(p)
    if not segy:
        raise HTTPException(400, "No .sgy / .segy files were found in that folder (or the folders inside it).")
    proj = pipeline.set_project(p, [str(s) for s in segy])
    return {"ok": True, "segy": [Path(s).name for s in proj["sources"]], "data_dir": proj["data_dir"]}


@app.post("/api/run")
async def run(req: Request):
    body = await req.json()
    p = _proc["p"]
    if p is not None and p.poll() is None:
        return {"ok": False, "msg": "A run is already in progress."}
    if not read_json(pipeline.PROJECT, {}).get("sources"):
        raise HTTPException(400, "Import a data folder first.")
    cmd = [sys.executable, "-u", "-m", "copilot.pipeline"]
    if body.get("force"):
        cmd.append("--force")
    if body.get("no_slurm"):
        cmd.append("--no-slurm")
    config.set_status(stage="convert", message="Starting", state="running", error=None, log=[],
                      stages=[{"id": s, "label": pipeline.STAGE_LABEL[s]} for s in pipeline.STAGES], started=time.time())
    logf = open(config.LOGS / "pipeline.log", "w")
    _proc["p"] = subprocess.Popen(cmd, cwd=str(config.ROOT), stdout=logf, stderr=subprocess.STDOUT)
    return {"ok": True}


# ------------------------------------------------------------------ slices
def _vol(name):
    v = store.get_volume(name)
    if v is None:
        raise HTTPException(404, "unknown volume")
    return v


@app.get("/api/slice")
def get_slice(vol: str, axis: str, idx: int, lo: float = None, hi: float = None):
    v = _vol(vol)
    s = v.slice(axis, idx)
    st = v.meta["stats"]
    lo = st["lo"] if lo is None else lo
    hi = st["hi"] if hi is None else hi
    if axis in ("i", "x"):
        s = s.T                                               # rows = time, columns = lateral position
    q = (np.clip((s - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)
    return Response(np.ascontiguousarray(q).tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{q.shape[0]},{q.shape[1]}", "Cache-Control": "no-store"})


@app.get("/api/hlines")
def hlines(axis: str, number: float):
    return tools.horizon_lines(axis, number) if tools.horizons() is not None else []


@app.get("/api/overlay")
def get_overlay(layer: str, axis: str, number: float):
    r = tools.overlay(layer, axis, number)
    if r is None:
        raise HTTPException(404, "not available")
    data, ext = r
    return Response(data.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{data.shape[0]},{data.shape[1]}", "X-Extent": json.dumps(ext), "Cache-Control": "no-store"})


@app.get("/api/map")
def get_map(horizon: str, attr: str = "time"):
    mp, g = tools.maps(), tools.grid()
    key = f"{horizon}__{attr}"
    if mp is None or key not in mp:
        raise HTTPException(404, "not available")
    a = np.ascontiguousarray(mp[key], dtype=np.float32)
    lo, hi = (float(v) for v in np.percentile(a, [2, 98]))
    if attr == "facies":
        lo, hi = 0.0, float(a.max())
    ext = {"h": [int(g["xlines"][0]), int(g["xlines"][-1])], "v": [int(g["ilines"][0]), int(g["ilines"][-1])]}
    return Response(a.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{a.shape[0]},{a.shape[1]}", "X-Range": f"{lo},{hi}", "X-Extent": json.dumps(ext),
                             "Cache-Control": "no-store"})


@app.get("/api/outlines")
def outlines(horizon: str):
    return tools.lead_outlines(horizon)


@app.get("/api/probe")
def probe(il: float, xl: float, twt: float):
    return _clean(tools.probe(il, xl, twt))


@app.get("/api/lookalike")
def lookalike(il: float, xl: float, twt: float):
    return _clean(tools.lookalike(il, xl, twt))


@app.get("/api/well")
def well(name: str):
    d = tools.well(name)
    if "error" in d:
        raise HTTPException(404, d["error"])
    a = wells.well_arrays(wells.norm(name))
    logs = {}
    if "twt" in a and np.isfinite(a["twt"]).any():
        step = max(1, len(a["twt"]) // 1500)
        logs["twt"] = a["twt"][::step]
        for c in ("gr", "vsh", "rt", "phi", "hc", "sand"):
            if f"c_{c}" in a:
                logs[c] = np.where(np.isfinite(a[f"c_{c}"][::step]), a[f"c_{c}"][::step], None)
    d["logs"] = logs
    return _clean(d)


# ------------------------------------------------------------------ chat
@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()

    def gen():
        try:
            for ev in llm.chat(body.get("messages", []), body.get("view"), bool(body.get("attach_view"))):
                yield json.dumps(ev, default=config._json_default) + "\n"
        except Exception as e:  # never leave the page hanging
            yield json.dumps({"type": "error", "text": f"{type(e).__name__}: {e}"}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def main():
    import uvicorn
    port = config.PORT
    for _ in range(20):
        with socket.socket() as s:
            try:
                s.bind((config.HOST, port))
                break
            except OSError:
                port += 1
    print("\n" + "=" * 62)
    print("  Exploration Copilot is running.")
    print(f"  Open this address in Firefox:   http://{config.HOST}:{port}")
    print("  Keep this window open. To stop the app, press Ctrl+C here.")
    print("=" * 62 + "\n", flush=True)
    import os
    if os.environ.get("COPILOT_OPEN_BROWSER") == "1" and os.environ.get("DISPLAY"):
        import threading
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{config.HOST}:{port}")).start()
    uvicorn.run(app, host=config.HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
