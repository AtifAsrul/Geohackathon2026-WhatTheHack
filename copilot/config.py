"""Central settings. Everything can be overridden with environment variables."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # the repository folder
WORKSPACE = Path(os.environ.get("COPILOT_WORKSPACE", ROOT / "workspace"))
STORES = WORKSPACE / "stores"        # AI-ready copies of the SEG-Y volumes
INTERP = WORKSPACE / "interp"        # interpretation products (RGT, horizons, faults, facies...)
MODELS = Path(os.environ.get("COPILOT_MODELS", ROOT / "models"))
LOGS = WORKSPACE / "logs"
WEB = ROOT / "web"

# Shared Gemma model on the HPC (OpenAI-compatible API served by vLLM).
LLM_URL = os.environ.get("COPILOT_LLM_URL", "http://172.28.34.1:8000/v1")
LLM_MODEL = os.environ.get("COPILOT_LLM_MODEL", "google/gemma-4-26B-A4B-it")

DEFAULT_DATA_DIR = os.environ.get(
    "COPILOT_DATA_DIR", "/data/epic20/APGCE/input_data/challenge1_challenge2")

HOST = os.environ.get("COPILOT_HOST", "127.0.0.1")
PORT = int(os.environ.get("COPILOT_PORT", "8050"))

# How much memory the viewer may use to keep volumes in RAM for instant slicing.
RAM_BUDGET_GB = float(os.environ.get("COPILOT_RAM_GB", "48"))

# Interpretation grid limits (the volume is thinned to at most this many points per axis).
GRID_MAX_LATERAL = int(os.environ.get("COPILOT_GRID_LATERAL", "512"))
GRID_MAX_TIME = int(os.environ.get("COPILOT_GRID_TIME", "800"))

RGT_WEIGHTS = MODELS / "rgt" / "RGT-Est_CIG-Benchmark.pt"
NCS_MODEL_DIRS = [MODELS / "ncs" / "NCS-v1-2.5d-base", MODELS / "ncs" / "NCS-v1-2d-base"]
SFM_WEIGHTS_DIR = MODELS / "sfm"     # optional: Sheng et al. SFM checkpoint (*.pth) placed by hand
FAULTSSL_WEIGHTS = Path(os.environ.get("COPILOT_FAULTSSL_WEIGHTS", MODELS / "faultssl" / "PrecisionFirst.pt"))
FAULT_METHOD = os.environ.get("COPILOT_FAULT_METHOD", "auto")  # auto, faultssl, coherence
FAULT_COHERENCE_MAX = os.environ.get("COPILOT_FAULT_COHERENCE_MAX")  # None preserves the saved setting

# The assistant's own interpretation pass (pictures sent to the shared Gemma model).
LLM_CACHE = WORKSPACE / "llm_cache"  # every answer is kept, so a stopped run resumes without asking again
AI_MINUTES = float(os.environ.get("COPILOT_AI_MINUTES", "30"))     # time budget for that pass
AI_WORKERS = int(os.environ.get("COPILOT_AI_WORKERS", "3"))        # requests sent at the same time (shared model)
PROBE_FILE = WORKSPACE / "gemma_probe.json"

for _d in (WORKSPACE, STORES, INTERP, LOGS, LLM_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    for attempt in range(40):               # another process may be reading the file right now
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------- progress reporting
STATUS_FILE = WORKSPACE / "status.json"


def set_status(stage=None, message=None, state=None, **extra):
    """Progress file read by the web page while the interpretation is running."""
    s = read_json(STATUS_FILE, {}) or {}
    s.setdefault("log", [])
    if stage is not None:
        s["stage"] = stage
    if message is not None:
        s["message"] = message
        s["log"] = (s["log"] + [{"t": time.strftime("%H:%M:%S"), "stage": s.get("stage"), "msg": message}])[-200:]
        print(f"[{time.strftime('%H:%M:%S')}] {s.get('stage', '')}: {message}", flush=True)
    if state is not None:
        s["state"] = state
    s.update(extra)
    s["updated"] = time.time()
    try:
        write_json(STATUS_FILE, s)
    except Exception as e:                  # progress reporting must never stop the real work
        print(f"(could not update the progress file: {e})", flush=True)
    return s
