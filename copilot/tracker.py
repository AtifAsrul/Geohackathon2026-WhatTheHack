"""Horizon tracking the way an interpreter does it: follow one reflector along a line, trace to trace, by
matching the shape of the wiggle; do that on a sparse net of inlines and crosslines so the lines tie where
they cross; then fill in between the lines.

The assistant chooses WHICH reflectors to follow and how they continue across faults; this module places
the picks. Units: trace index and (fractional) time-sample index. Sections are (lateral, time).
"""
from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _ncc(a, b):
    den = np.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-12
    return float((a * b).sum()) / den


def _refine(tr, k, polarity, reach=1):
    """Nearest peak (polarity +1) or trough (-1) within +-reach samples, then a parabola fit for sub-sample position."""
    nt = len(tr)
    k = int(np.clip(k, 1, nt - 2))
    lo, hi = max(1, k - reach), min(nt - 2, k + reach)
    seg = tr[lo:hi + 1] * polarity
    k = lo + int(np.argmax(seg))
    a, b, c = tr[k - 1] * polarity, tr[k] * polarity, tr[k + 1] * polarity
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den < -1e-12 else 0.0
    return float(k + np.clip(off, -0.5, 0.5))


def track_line(sec, col, z0, polarity, fl_sec=None, live=None, min_corr=0.55, half=8, step=2, fl_stop=0.6):
    """Follow one reflector along a section, both ways from trace `col`.

    Returns (z, q): fractional sample index per trace (NaN where it could not be followed) and the match
    quality 0..1. Tracking stops where the wiggle stops matching, at a likely fault, or at empty traces.
    """
    nlat, nt = sec.shape
    z = np.full(nlat, np.nan, dtype=np.float32)
    q = np.zeros(nlat, dtype=np.float32)
    k0 = int(round(z0))
    if not (half <= k0 < nt - half) or (live is not None and not live[col]):
        return z, q
    zs = _refine(sec[col], k0, polarity, 2)
    z[col], q[col] = zs, 1.0
    for d in (1, -1):
        k = int(round(zs))
        if not (half <= k < nt - half):
            continue
        ref = sec[col, k - half:k + half + 1].astype(np.float64)
        zc, j = zs, col + d
        while 0 <= j < nlat:
            if live is not None and not live[j]:
                break
            k = int(round(zc))
            best, bk = -2.0, k
            for s in range(-step, step + 1):
                kk = k + s
                if kk - half < 0 or kk + half >= nt:
                    continue
                c = _ncc(ref, sec[j, kk - half:kk + half + 1]) - 0.02 * abs(s)
                if c > best:
                    best, bk = c, kk
            if best < min_corr:
                break
            if fl_sec is not None and fl_sec[j, bk] > fl_stop:
                break
            zn = _refine(sec[j], bk, polarity, 1)
            kn = int(round(zn))
            if not (half <= kn < nt - half):
                break
            z[j], q[j] = zn, best
            ref = 0.7 * ref + 0.3 * sec[j, kn - half:kn + half + 1]
            zc = zn
            j += d
    return z, q


def candidates_on_line(sec, live=None, fl_sec=None, n=14, col=None, min_gap=6):
    """The strongest, most continuous reflectors on one section, found from the seismic alone.

    Returns a list (shallow to deep) of {"col", "z0", "polarity", "z", "score", "coverage", "strength"}.
    """
    nlat, nt = sec.shape
    lv = np.ones(nlat, dtype=bool) if live is None else live
    if col is None:
        idx = np.flatnonzero(lv)
        col = int(idx[len(idx) // 2]) if idx.size else nlat // 2
    tr = sec[max(0, col - 2):col + 3].mean(axis=0)
    rms = float(np.sqrt(np.mean(sec[lv] ** 2))) + 1e-12
    cands = []
    for pol in (1, -1):
        v = tr * pol
        pk = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > 0.5 * rms)) + 1
        cands += [(float(v[k]), int(k), pol) for k in pk if 10 <= k < nt - 10]
    cands.sort(reverse=True)
    out = []
    for amp, k, pol in cands[:48]:
        z, q = track_line(sec, col, k, pol, fl_sec, lv)
        ok = np.isfinite(z)
        cov = float(ok.sum() / max(1, lv.sum()))
        kk = np.clip(np.rint(z[ok]).astype(int), 0, nt - 1)
        strength = float(np.mean(np.abs(sec[np.flatnonzero(ok), kk]))) / rms
        out.append({"col": col, "z0": float(z[col]) if np.isfinite(z[col]) else float(k), "polarity": pol, "z": z, "q": q,
                    "coverage": round(cov, 3), "strength": round(strength, 2), "score": round(cov * strength, 3)})
    out.sort(key=lambda c: -c["score"])
    chosen = []
    for c in out:
        if len(chosen) == n:
            break
        if all(abs(c["z0"] - o["z0"]) >= min_gap for o in chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c["z0"])
    return chosen


def decoy_line(cand, nlat, nt, seed=0):
    """A line that deliberately cuts across the reflectors (it starts on one and drifts off it)."""
    rng = np.random.default_rng(seed)
    z = np.array(cand["z"], dtype=np.float32)
    ok = np.isfinite(z)
    drift = np.linspace(-1, 1, nlat) * float(rng.choice([-1, 1])) * float(rng.uniform(18, 30))
    z = np.where(ok, z + drift + 7.0, np.nan)
    return {"col": cand["col"], "z0": float(cand["z0"] + 7.0), "polarity": cand["polarity"], "z": np.clip(z, 2, nt - 3),
            "coverage": cand["coverage"], "strength": 0.0, "score": 0.0, "decoy": True}


def continuations(sec, z, polarity, fl_sec=None, live=None, gap=6, reach=60, half=18, max_n=4):
    """Where a tracked line stops (usually at a fault): the best-matching places to pick it up again on the
    far side. Returns a list of ends, each {"side": +1/-1, "from": trace, "col": trace, "options": [{"z", "corr"}]}."""
    nlat, nt = sec.shape
    ok = np.isfinite(z)
    if not ok.any():
        return []
    ends = []
    for side in (1, -1):
        idx = np.flatnonzero(ok)
        je = int(idx[-1] if side == 1 else idx[0])
        jf = je + side * gap
        while 0 <= jf < nlat and ((live is not None and not live[jf]) or (fl_sec is not None and fl_sec[jf].max() > 0.5 and
                                                                         fl_sec[jf, int(np.clip(round(float(z[je])), 0, nt - 1))] > 0.3)):
            jf += side
        if not (0 <= jf < nlat) or (live is not None and not live[jf]):
            continue
        if abs(jf - je) > 40 or np.isfinite(z[jf]):
            continue
        jr = int(np.clip(je - side * 2, 0, nlat - 1))
        if not np.isfinite(z[jr]):
            jr = je
        kr = int(round(float(z[jr])))
        if not (half <= kr < nt - half):
            continue
        ref = sec[jr, kr - half:kr + half + 1].astype(np.float64)
        v = sec[jf] * polarity
        lo, hi = max(half, kr - reach), min(nt - half - 1, kr + reach)
        pk = [k for k in range(lo + 1, hi) if v[k] > v[k - 1] and v[k] >= v[k + 1] and v[k] > 0]
        opts = sorted(({"z": float(k), "corr": round(_ncc(ref, sec[jf, k - half:k + half + 1]), 3)} for k in pk),
                      key=lambda o: -o["corr"])[:max_n]
        if opts:
            ends.append({"side": side, "from": je, "col": int(jf), "options": opts})
    return ends


def grow_lattice(seis, pol, il_idx, xl_idx, Zi, Zx, start, fl=None, live=None, tol=3.0, stats=None):
    """Spread picks from already-picked lines to every line of the net, tying at the crossings.

    Zi : (len(il_idx), nx) picks on the inline family;  Zx : (ni, len(xl_idx)) picks on the crossline family
    start : list of ("i", k) / ("x", m) lines whose picks are new and should seed their crossing lines.
    A crossing is accepted only if the new line agrees with what is already there (within tol samples);
    where it does not, the new line is cut at that crossing.
    """
    ni, nx, nt = seis.shape
    stats = stats if stats is not None else {"tied": 0, "mistied": 0, "tracks": 0}
    queue, tried = deque(start), set()
    il_pos = {int(g): k for k, g in enumerate(il_idx)}
    xl_pos = {int(g): m for m, g in enumerate(xl_idx)}
    while queue:
        axis, k = queue.popleft()
        if axis == "i":
            gi = int(il_idx[k])
            for m, gx in enumerate(xl_idx):
                zc = Zi[k, gx]
                if not np.isfinite(zc) or np.isfinite(Zx[gi, m]) or ("x", m, k) in tried:
                    continue
                tried.add(("x", m, k))
                z, _ = track_line(seis[:, gx, :], gi, zc, pol, None if fl is None else fl[:, gx, :],
                                  None if live is None else live[:, gx])
                stats["tracks"] += 1
                for g2, k2 in il_pos.items():
                    if k2 != k and np.isfinite(z[g2]) and np.isfinite(Zi[k2, gx]):
                        if abs(z[g2] - Zi[k2, gx]) <= tol:
                            stats["tied"] += 1
                        else:
                            stats["mistied"] += 1
                            if g2 > gi:
                                z[g2:] = np.nan
                            else:
                                z[:g2 + 1] = np.nan
                new = np.isfinite(z) & ~np.isfinite(Zx[:, m])
                if new.sum() > 1:
                    Zx[new, m] = z[new]
                    queue.append(("x", m))
        else:
            gx = int(xl_idx[k])
            for m, gi in enumerate(il_idx):
                zc = Zx[gi, k]
                if not np.isfinite(zc) or np.isfinite(Zi[m, gx]) or ("i", m, k) in tried:
                    continue
                tried.add(("i", m, k))
                z, _ = track_line(seis[gi], gx, zc, pol, None if fl is None else fl[gi], None if live is None else live[gi])
                stats["tracks"] += 1
                for g2, k2 in xl_pos.items():
                    if k2 != k and np.isfinite(z[g2]) and np.isfinite(Zx[gi, k2]):
                        if abs(z[g2] - Zx[gi, k2]) <= tol:
                            stats["tied"] += 1
                        else:
                            stats["mistied"] += 1
                            if g2 > gx:
                                z[g2:] = np.nan
                            else:
                                z[:g2 + 1] = np.nan
                new = np.isfinite(z) & ~np.isfinite(Zi[m])
                if new.sum() > 1:
                    Zi[m, new] = z[new]
                    queue.append(("i", m))
    return stats


def to_surface(Zi, Zx, il_idx, xl_idx, seis, pol, live=None, snap=2):
    """Fill in between the picked lines. Returns (z (inline, crossline), confidence 0..1, share of the area picked).

    Between lines the surface is interpolated, then nudged onto the nearest reflector of the right polarity.
    Confidence falls with distance from a picked line and where the nudge had to move the surface.
    """
    from scipy.interpolate import griddata
    from .rgt import fill_dead, snap_to_event
    ni, nx, nt = seis.shape
    picks = np.full((ni, nx), np.nan, dtype=np.float32)
    for k, gi in enumerate(il_idx):
        picks[gi] = Zi[k]
    for m, gx in enumerate(xl_idx):
        col = Zx[:, m]
        ok = np.isfinite(col)
        both = ok & np.isfinite(picks[:, gx])
        picks[ok, gx] = col[ok]
        picks[both, gx] = col[both]
    have = np.isfinite(picks)
    if have.sum() < 8:
        return None, None, 0.0
    ii, xx = np.nonzero(have)
    gi_, gx_ = np.mgrid[0:ni, 0:nx]
    z = griddata((ii, xx), picks[have], (gi_, gx_), method="linear")
    if np.isnan(z).any():
        zn = griddata((ii, xx), picks[have], (gi_, gx_), method="nearest")
        z = np.where(np.isnan(z), zn, z)
    base = gaussian_filter(z.astype(np.float32), 0.8, mode="nearest")
    base[have] = picks[have]
    s = snap_to_event(seis, base, float(pol), half=snap)
    s[have] = picks[have]
    moved = np.abs(s - base)
    dist = distance_transform_edt(~have)
    conf = np.exp(-dist / 10.0) * np.where(moved <= 1.5, 1.0, 0.5)
    if live is not None and not live.all():
        s = fill_dead(s, live)
        conf = np.where(live, conf, 0.0)
    return np.clip(s, 0, nt - 1).astype(np.float32), conf.astype(np.float32), float(have[live].mean() if live is not None else have.mean())


def compare(z_a, z_b, live, dt_ms):
    """Agreement of two surfaces (in samples) over the live area: statistics in ms and the difference map."""
    d = (z_a - z_b) * dt_ms
    v = d[live] if live is not None else d.ravel()
    a = np.abs(v)
    return ({"mean_abs_ms": round(float(a.mean()), 1), "median_abs_ms": round(float(np.median(a)), 1),
             "p90_abs_ms": round(float(np.percentile(a, 90)), 1), "bias_ms": round(float(v.mean()), 1),
             "within_8ms": round(float((a <= 8.0).mean()), 2)}, d.astype(np.float32))
