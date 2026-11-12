"""Local FaultSSL Precision First TorchScript inference; no data leaves the HPC.

Model and input convention: https://github.com/douyimin/FaultSSL (Apache-2.0).
The upstream predictor uses (time, inline, crossline), multiples of 16, and
clipped z-score/min-max normalisation. Overlapping lateral tiles retain the
full time axis and blend predictions, keeping memory bounded on the HPC GPU.
"""
from pathlib import Path

import numpy as np


def _starts(size, tile, overlap):
    if size <= tile:
        return [0]
    return sorted(set(range(0, size - tile + 1, tile - overlap)) | {size - tile})


def predict_tiles(seis, predict, tile=96, overlap=24, log=print):
    """predict receives a padded (time, inline, crossline) array and returns probabilities."""
    if seis.ndim != 3 or min(seis.shape) < 1:
        raise ValueError("FaultSSL needs a nonempty (inline, crossline, time) cube")
    if tile < 16 or not 0 <= overlap < tile:
        raise ValueError("FaultSSL requires tile >= 16 and 0 <= overlap < tile")
    ni, nx, nt = seis.shape
    out = np.zeros(seis.shape, dtype=np.float32)
    weight = np.zeros((ni, nx), dtype=np.float32)
    jobs = [(i, x) for i in _starts(ni, tile, overlap) for x in _starts(nx, tile, overlap)]
    for n, (i, x) in enumerate(jobs, 1):
        a = np.asarray(seis[i:i + tile, x:x + tile], dtype=np.float32).transpose(2, 0, 1)
        shape = a.shape
        pad = [(0, (-s) % 16) for s in shape]
        padded = np.pad(a, pad, mode="reflect" if min(shape) > 1 else "edge")
        pred = np.asarray(predict(padded), dtype=np.float32)
        if pred.shape != padded.shape or not np.isfinite(pred).all() or pred.min() < -1e-5 or pred.max() > 1.00001:
            raise ValueError("FaultSSL returned invalid probabilities or a mismatched cube shape")
        pred = np.clip(pred[:nt, :shape[1], :shape[2]], 0, 1).transpose(1, 2, 0)
        w = (np.hanning(shape[1] + 2)[1:-1, None] * np.hanning(shape[2] + 2)[None, 1:-1]).astype(np.float32)
        out[i:i + shape[1], x:x + shape[2]] += pred * w[..., None]
        weight[i:i + shape[1], x:x + shape[2]] += w
        log(f"FaultSSL: tile {n}/{len(jobs)}")
    out /= weight[..., None]
    return out


def infer(seis, weights, log=print, tile=96):
    """Load the official TorchScript checkpoint and return probabilities on the input grid."""
    import torch
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"FaultSSL checkpoint missing: {weights}. Run scripts/download_faultssl.py first.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"FaultSSL Precision First: {weights.name}, running on {device}")
    model = torch.jit.load(str(weights), map_location="cpu").to(device).eval()

    def predict(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
        t = (t - t.mean()) / t.std().clamp_min(1e-6)
        t = t.clamp(-3.2, 3.2)
        t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-6)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            return model(t.to(device))[0, 0].float().cpu().numpy()

    return predict_tiles(seis, predict, tile=tile, overlap=tile // 4, log=log)
