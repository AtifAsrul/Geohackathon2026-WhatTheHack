"""Fault likelihood from seismic discontinuity, and a plain-language fault inventory."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label, uniform_filter


def validate_cutoff(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Coherence cutoff must be a finite number between 0 and 1")
    return value


def apply_cutoff(likelihood, coherence, cutoff=1.0, live=None):
    """Keep candidates only where coherence < cutoff; 1 disables this optional filter.

    Coherence is semblance (0..1), not Pearson correlation or model probability.
    The strict comparison means a cutoff of 0.5 excludes coherence == 0.5.
    """
    cutoff = validate_cutoff(cutoff)
    out = np.clip(np.asarray(likelihood, dtype=np.float32), 0, 1).copy()
    if coherence.shape != out.shape:
        raise ValueError("Fault likelihood and coherence must have matching shapes")
    out[~np.isfinite(out) | ~np.isfinite(coherence)] = 0
    if cutoff < 1:
        out[coherence >= cutoff] = 0
    if live is not None:
        out[~live] = 0
    return out


def semblance(seis, lateral=3, vertical=9):
    """Coherence cube, 0..1: how alike a trace is to its neighbours (3 x 3 traces, 9 samples).

    1 = neighbouring traces are identical (unbroken reflectors); low = they differ (faults, channel edges, noise).
    """
    s = np.asarray(seis, dtype=np.float32)
    num = uniform_filter(s, size=(lateral, lateral, 1)) ** 2
    den = uniform_filter(s * s, size=(lateral, lateral, 1))
    num = uniform_filter(num, size=(1, 1, vertical))
    den = uniform_filter(den, size=(1, 1, vertical)) + 1e-12
    return np.clip(num / den, 0, 1).astype(np.float32)


def fault_likelihood(seis, log=print, live=None):
    """0..1 cube: high where neighbouring traces stop looking alike over a tall vertical window.

    Coherence (low at breaks), smoothed vertically so that steep, tall features (faults) are kept and
    bed-parallel changes are suppressed. `live` (inline, crossline) marks traces with real data: empty
    traces never count as faults and are left out of the scaling.
    """
    log("Faults: measuring trace-to-trace similarity (coherence)")
    semb = semblance(seis)
    fl = gaussian_filter1d(1.0 - semb, sigma=8, axis=2)
    # remove the lateral background so dipping / noisy zones do not light up everywhere
    bg = uniform_filter(fl, size=(15, 15, 1))
    fl = fl - bg
    sub = fl[::3, ::3, ::3]
    if live is not None and not live.all():
        fl[~live] = 0.0
        semb[~live] = 1.0
        keep = live[::3, ::3]
        sub = sub[keep] if keep.any() else sub
    hi = np.percentile(sub, 99.7)
    fl = np.clip(fl / (hi + 1e-9), 0, 1)
    return fl.astype(np.float32), semb.astype(np.float32)


def fault_segments(fl, z, cell_m=(25.0, 25.0), u_il=(0.0, 1.0), u_xl=(1.0, 0.0), thresh=0.35, min_cells=25):
    """Fault traces on one horizon: list of dict(strike_deg, length_km, throw_samples, centre).

    u_il / u_xl: unit vectors (east, north) of the inline-index and crossline-index axes,
    so strikes are true compass directions.
    """
    ni, nx, nt = fl.shape
    k = np.clip(np.rint(z).astype(int), 0, nt - 1)
    fmap = np.zeros((ni, nx), dtype=np.float32)
    for d in range(-6, 7, 2):
        kk = np.clip(k + d, 0, nt - 1)
        fmap = np.maximum(fmap, np.take_along_axis(fl, kk[..., None], axis=2)[..., 0])
    lab, n = label(fmap > thresh, structure=np.ones((3, 3)))
    segs = []
    for c in range(1, n + 1):
        ii, xx = np.nonzero(lab == c)
        if ii.size < min_cells:
            continue
        px, py = xx * cell_m[1], ii * cell_m[0]
        cov = np.cov(np.stack([px - px.mean(), py - py.mean()]))
        w, v = np.linalg.eigh(cov)
        if w[1] < 1e-6 or w[1] / max(w[0], 1e-6) < 3.0:
            continue                                   # blob, not a line
        major = v[:, 1]
        length = float(4 * np.sqrt(w[1]))              # ~ full extent of a line
        east = major[1] * u_il[0] + major[0] * u_xl[0]
        north = major[1] * u_il[1] + major[0] * u_xl[1]
        strike = np.degrees(np.arctan2(east, north)) % 180.0
        # throw: time difference of the horizon either side of the trace
        minor = v[:, 0]
        step = 4
        oi = np.clip(np.rint(ii + step * minor[1]).astype(int), 0, ni - 1)
        ox = np.clip(np.rint(xx + step * minor[0]).astype(int), 0, nx - 1)
        pi = np.clip(np.rint(ii - step * minor[1]).astype(int), 0, ni - 1)
        px2 = np.clip(np.rint(xx - step * minor[0]).astype(int), 0, nx - 1)
        throw = float(np.median(np.abs(z[oi, ox] - z[pi, px2])))
        segs.append({"strike_deg": round(float(strike), 0), "length_km": round(length / 1000.0, 2),
                     "throw_samples": round(throw, 1), "ci": float(ii.mean()), "cx": float(xx.mean()),
                     "cells": int(ii.size)})
    segs.sort(key=lambda s: -s["length_km"])
    return segs, fmap


def strike_summary(segs):
    """Dominant strike directions weighted by fault length."""
    if not segs:
        return []
    bins = np.zeros(6)
    for s in segs:
        bins[int(s["strike_deg"] // 30) % 6] += s["length_km"]
    names = ["N-S to NNE-SSW", "NE-SW", "ENE-WSW to E-W", "E-W to ESE-WNW", "NW-SE", "NNW-SSE to N-S"]
    total = bins.sum() + 1e-9
    order = np.argsort(-bins)
    return [{"trend": names[b], "share": round(float(bins[b] / total), 2)} for b in order if bins[b] / total > 0.12]
