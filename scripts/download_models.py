"""Download the pretrained models the app uses (run once, needs internet).

    python scripts/download_models.py

  * RGT-Est weights (about 310 MB) from Zenodo, record 20118902
  * NCS seismic foundation models (about 350 MB each) from Hugging Face

Safe to run again: files that are already complete are skipped.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("COPILOT_MODELS", ROOT / "models"))

RGT_URL = "https://zenodo.org/api/records/20118902/files/RGT-Est_CIG-Benchmark.pt/content"
RGT_SIZE = 308817715
NCS = ["NorskRegnesentralSTI/NCS-v1-2.5d-base", "NorskRegnesentralSTI/NCS-v1-2d-base"]


def get_rgt():
    out = MODELS / "rgt" / "RGT-Est_CIG-Benchmark.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == RGT_SIZE:
        print("RGT-Est weights: already here")
        return True
    print("RGT-Est weights: downloading (about 310 MB)...")
    tmp = out.with_suffix(".part")
    try:
        req = urllib.request.Request(RGT_URL, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            done = 0
            started = last = time.monotonic()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last >= 2 or done >= RGT_SIZE:
                    fraction = min(done / RGT_SIZE, 1)
                    filled = int(fraction * 20)
                    rate = done / max(now - started, 0.001)
                    eta = max(0, RGT_SIZE - done) / max(rate, 1)
                    print(f"   [{'#' * filled}{'-' * (20 - filled)}] {fraction:.0%} | "
                          f"{done / 1e6:.1f}/{RGT_SIZE / 1e6:.1f} MB | {rate / 1e6:.1f} MB/s | ETA {eta:.0f}s", flush=True)
                    last = now
        if tmp.stat().st_size != RGT_SIZE:
            print(f"!! RGT-Est download is incomplete ({tmp.stat().st_size} of {RGT_SIZE} bytes)")
            return False
        tmp.replace(out)
        print("RGT-Est weights: done")
        return True
    except Exception as e:
        print(f"!! RGT-Est weights could not be downloaded: {e}")
        return False


def get_ncs():
    ok = True
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"!! huggingface_hub is not installed: {e}")
        return False
    for repo in NCS:
        dest = MODELS / "ncs" / repo.split("/")[1]
        if (dest / "config.json").exists() and (dest / "model.safetensors").exists():
            print(f"{repo}: already here")
            continue
        print(f"{repo}: downloading (about 350 MB)...")
        try:
            snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.md"])
            print(f"{repo}: done")
        except Exception as e:
            print(f"!! {repo} could not be downloaded: {e}")
            ok = False
    return ok


if __name__ == "__main__":
    a = get_rgt() if "--skip-rgt" not in sys.argv else True
    b = get_ncs() if "--skip-ncs" not in sys.argv else True
    print("\nAll models are ready." if a and b else
          "\nSome models are missing. The app still works: it falls back to simpler methods for the missing ones.")
