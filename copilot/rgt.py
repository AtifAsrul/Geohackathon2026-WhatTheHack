"""Relative Geologic Time (RGT) and horizons.

Primary engine : RGT-Est (Dou, Wu, Gao, Bi 2026 - github.com/douyimin/RGT-Est),
                 a pretrained 3-D network that turns a seismic cube into an RGT cube.
Fallback engine: a classical dip-steered estimate (structure-tensor slopes +
                 least-squares surface fitting) used only if RGT-Est cannot run.

Horizons are iso-surfaces of the RGT cube. Array convention: (inline, crossline, time).
"""
import os

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, median_filter


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-6)


def z_score_clip(a, clip=3.0):
    z = (a - a.mean()) / (a.std() + 1e-8)
    return _norm01(np.clip(z, -clip, clip))


# ------------------------------------------------------------------ empty traces at the survey edge
def live_mask(seis, erode=2):
    """True where a trace holds real data. Traces that are all zero (or nearly: tapered edge) are 'dead'.

    The rim next to a dead area is also dropped (erode), because low-fold edge traces are unreliable.
    """
    ni = seis.shape[0]
    rms = np.empty(seis.shape[:2], dtype=np.float32)
    for b in range(0, ni, 32):
        blk = np.nan_to_num(np.asarray(seis[b:b + 32], dtype=np.float32))
        rms[b:b + 32] = np.sqrt(np.mean(blk * blk, axis=2))
    pos = rms[rms > 0]
    if pos.size == 0:
        return np.ones(rms.shape, dtype=bool)
    live = rms > 0.02 * float(np.median(pos))
    if live.all() or not live.any():
        return np.ones(rms.shape, dtype=bool) if not live.any() else live
    if erode:
        shrunk = binary_erosion(live, iterations=int(erode), border_value=1)
        if shrunk.any():
            live = shrunk
    return live


def fill_dead(a, live):
    """Copy the nearest live trace (or map value) into every dead position. Works on cubes and on maps.

    For a horizon map this is what makes the surface carry on flat, at constant time, into an empty area.
    """
    if live is None or live.all():
        return a
    idx = distance_transform_edt(~live, return_distances=False, return_indices=True)
    dead = ~live
    out = np.array(a, copy=True)
    out[dead] = np.asarray(a)[idx[0][dead], idx[1][dead]]
    return out


# ------------------------------------------------------------------ RGT-Est
def rgt_est(seis, weights, device=None, log=print):
    """Run the pretrained RGT-Est model as in the authors' demo.

    seis: float32 (inline, crossline, time). Returns (RGT in [0, 1] with the same shape,
    increasing with time, and the inference grid that was used).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"RGT-Est: loading model on {device}")
    model = torch.jit.load(str(weights), map_location=device).eval()

    ni, nx, nt = seis.shape
    x = (z_score_clip(seis.astype(np.float32), 2.0) * 2 - 1).astype(np.float32)
    src = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None, None]    # (1,1,T,H,W)
    # (400, 512, 512) is the grid the authors recommend. Smaller grids are tried only if the
    # GPU runs out of memory; COPILOT_RGT_GRID overrides the list (used for quick tests).
    if os.environ.get("COPILOT_RGT_GRID"):
        sizes = [tuple(int(v) for v in os.environ["COPILOT_RGT_GRID"].split(","))]
    else:
        sizes = [(400, 512, 512)] + ([(320, 416, 416), (256, 320, 320)] if device.startswith("cuda") else [])
    out, used, last = None, None, None
    for size in sizes:
        try:
            log(f"RGT-Est: running inference on a {size[0]} x {size[1]} x {size[2]} grid (the heavy step)")
            with torch.no_grad():
                t = F.interpolate(src, size, mode="trilinear")
                inp = torch.cat([t, torch.zeros_like(t), torch.zeros_like(t)], dim=1).to(device)   # no horizon constraints
                padded = nn.ReplicationPad3d(8)(inp)
                if device.startswith("cuda"):
                    with torch.autocast(device_type="cuda"):
                        o = model(padded)
                else:
                    o = model(padded)
                o = o[:, :, 8:-8, 8:-8, 8:-8].float().cpu()
                out = F.interpolate(o, (nt, ni, nx), mode="trilinear").numpy()[0, 0]              # (T,H,W)
            used = size
            break
        except RuntimeError as e:                                   # includes CUDA out-of-memory
            last = e
            if "out of memory" not in str(e).lower():
                raise
            log("RGT-Est: not enough GPU memory at this size, trying a smaller one")
            inp = padded = o = t = None
            torch.cuda.empty_cache()
    if out is None:
        raise RuntimeError(f"RGT-Est ran out of memory at every size: {last}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    rgt = z_score_clip(out).transpose(1, 2, 0)                                        # (inline, crossline, time)
    if rgt[..., -8:].mean() < rgt[..., :8].mean():
        rgt = 1.0 - rgt
    return np.ascontiguousarray(rgt, dtype=np.float32), used


# ------------------------------------------------------------------ classical fallback
def local_slopes(seis, sigma=(2.0, 2.0, 4.0)):
    """Reflector slope (samples per trace) along inline and crossline from the structure tensor."""
    s = gaussian_filter(seis.astype(np.float32), 1.0)
    gi, gx, gt = np.gradient(s)
    jtt = gaussian_filter(gt * gt, sigma) + 1e-12
    p = -gaussian_filter(gi * gt, sigma) / jtt
    q = -gaussian_filter(gx * gt, sigma) / jtt
    return np.clip(p, -4, 4).astype(np.float32), np.clip(q, -4, 4).astype(np.float32)


def _sample(vol, z):
    """Linear interpolation of vol (ni, nx, nt) at fractional time index z (..., ni, nx)."""
    nt = vol.shape[2]
    z = np.clip(z, 0, nt - 1.001)
    k = np.floor(z).astype(np.int64)
    f = (z - k).astype(np.float32)
    if z.ndim == 2:
        a = np.take_along_axis(vol, k[..., None], axis=2)[..., 0]
        b = np.take_along_axis(vol, (k + 1)[..., None], axis=2)[..., 0]
        return a * (1 - f) + b * f
    out = np.empty(z.shape, dtype=np.float32)
    for h in range(z.shape[0]):
        out[h] = _sample(vol, z[h])
    return out


def _poisson_neumann(f):
    """Solve laplacian(z) = f on a grid with zero-gradient edges (mean of z = 0)."""
    n, m = f.shape[-2:]
    F = sfft.dctn(f, type=2, axes=(-2, -1), norm="ortho")
    wi = 2 * np.cos(np.pi * np.arange(n) / n) - 2
    wx = 2 * np.cos(np.pi * np.arange(m) / m) - 2
    den = wi[:, None] + wx[None, :]
    den[0, 0] = 1.0
    F = F / den
    F[..., 0, 0] = 0.0
    return sfft.idctn(F, type=2, axes=(-2, -1), norm="ortho")


def rgt_classical(seis, n_levels=72, iters=6, log=print):
    """Dip-steered RGT: fit surfaces to the local slopes, then stack them into an RGT cube."""
    ni, nx, nt = seis.shape
    log("Classical RGT: estimating reflector dips")
    p, q = local_slopes(seis)
    levels = np.linspace(4, nt - 5, n_levels).astype(np.float32)
    z = np.broadcast_to(levels[:, None, None], (n_levels, ni, nx)).copy()
    for it in range(iters):
        log(f"Classical RGT: fitting surfaces, pass {it + 1}/{iters}")
        for h in range(n_levels):
            ph, qh = _sample(p, z[h]), _sample(q, z[h])
            fi = 0.5 * (ph[1:, :] + ph[:-1, :])          # forward-difference slopes
            fx = 0.5 * (qh[:, 1:] + qh[:, :-1])
            div = np.zeros((ni, nx), dtype=np.float64)
            div[:-1, :] += fi
            div[1:, :] -= fi
            div[:, :-1] += fx
            div[:, 1:] -= fx
            zz = _poisson_neumann(div)
            z[h] = np.clip(levels[h] + zz - zz.mean(), 0, nt - 1)
    z = np.maximum.accumulate(z, axis=0) + np.arange(n_levels, dtype=np.float32)[:, None, None] * 1e-3
    log("Classical RGT: building the RGT cube")
    rgt = np.empty((ni, nx, nt), dtype=np.float32)
    tt = np.arange(nt, dtype=np.float32)
    val = np.linspace(0.02, 0.98, n_levels)
    for i in range(ni):
        zi = z[:, i, :]
        for x in range(nx):
            rgt[i, x] = np.interp(tt, zi[:, x], val, left=0.0, right=1.0)
    # fill the top/bottom ramps so that RGT is strictly increasing
    ramp = (tt / (nt - 1))[None, None, :] * 0.02
    rgt = rgt + ramp
    return _norm01(rgt).astype(np.float32)


# ------------------------------------------------------------------ horizons from RGT
def horizons_from_rgt(rgt, values, smooth=1.0):
    """Iso-RGT surfaces. Returns float32 (n, inline, crossline) of fractional time-sample index.

    Same idea as the Wu / Dou helper shipped with RGT-Est: make RGT monotonic along
    time, then interpolate the time at which each requested RGT value is reached.
    """
    m = np.maximum.accumulate(rgt, axis=2)
    nt = m.shape[2]
    out = np.empty((len(values),) + m.shape[:2], dtype=np.float32)
    for h, v in enumerate(values):
        k = (m < v).sum(axis=2)
        k = np.clip(k, 1, nt - 1)
        u0 = np.take_along_axis(m, (k - 1)[..., None], axis=2)[..., 0]
        u1 = np.take_along_axis(m, k[..., None], axis=2)[..., 0]
        du = u1 - u0
        f = np.where(du > 1e-9, (v - u0) / np.where(du > 1e-9, du, 1.0), 0.0)
        out[h] = (k - 1) + np.clip(f, 0, 1)
    if smooth and smooth > 0:
        for h in range(len(values)):
            out[h] = gaussian_filter(out[h], smooth, mode="nearest")
    return out


def pick_key_horizons(seis, rgt, n_keep=8, n_candidates=90, log=print):
    """Choose the strongest, most continuous reflectors among many iso-RGT surfaces.

    Returns list of dict(name, rgt, z (inline, crossline) in fractional samples, polarity, strength).
    """
    ni, nx, nt = seis.shape
    vals = np.linspace(0.04, 0.96, n_candidates)
    log(f"Horizons: tracing {n_candidates} candidate surfaces from the RGT cube")
    zs = horizons_from_rgt(rgt, vals, smooth=1.0)
    score, pol = np.zeros(n_candidates), np.zeros(n_candidates)
    for h in range(n_candidates):
        amp = _sample(seis, zs[h])
        mean_abs = float(np.mean(np.abs(amp))) + 1e-9
        score[h] = abs(float(np.mean(amp)))               # strong AND same polarity everywhere
        pol[h] = np.sign(np.mean(amp)) or 1.0
        score[h] *= 0.5 + 0.5 * abs(float(np.mean(amp))) / mean_abs
    order = np.argsort(-score)
    zmean = zs.reshape(n_candidates, -1).mean(axis=1)
    chosen = []
    # keep the strongest events, but spread out in time so every part of the section gets a horizon
    for gap in (nt / (n_keep * 2.5), nt / (n_keep * 5.0), 2.0):
        for h in order:
            if len(chosen) == n_keep:
                break
            if all(abs(zmean[h] - zmean[c]) >= gap for c in chosen):
                chosen.append(int(h))
    chosen.sort()
    out = []
    for n, h in enumerate(chosen):
        z = snap_to_event(seis, zs[h], pol[h])
        out.append({"name": f"H{n + 1}", "rgt": float(vals[h]), "z": z, "polarity": "peak" if pol[h] > 0 else "trough",
                    "strength": float(score[h])})
    # keep the stack ordered in time everywhere (surfaces must not cross)
    for n in range(1, len(out)):
        out[n]["z"] = np.maximum(out[n]["z"], out[n - 1]["z"] + 1.0)
    return out, zs, vals


def snap_to_event(seis, z, polarity, half=3):
    """Move each pick to the nearest peak (or trough) within +-half samples, then de-spike."""
    nt = seis.shape[2]
    base = np.rint(z).astype(np.int64)
    best = np.full(z.shape, -np.inf, dtype=np.float32)
    off = np.zeros(z.shape, dtype=np.float32)
    for d in range(-half, half + 1):
        k = np.clip(base + d, 0, nt - 1)
        a = np.take_along_axis(seis, k[..., None], axis=2)[..., 0] * polarity - 0.02 * abs(d) * np.abs(seis).mean()
        better = a > best
        best[better] = a[better]
        off[better] = d
    return np.clip(median_filter(base + off, size=3, mode="nearest"), 0, nt - 1).astype(np.float32)
