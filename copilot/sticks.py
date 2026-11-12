"""Fault sticks: short lines that follow a fault on one section, the way an interpreter picks them.

The app proposes sticks from the fault-likelihood cube (ridges of low coherence), measures the throw across
each one, and uses measured throw/fit tests to join sticks on neighbouring lines into
named faults. Units inside this module are grid indices: lateral trace index and time-sample index.
"""
import numpy as np
from scipy.ndimage import label, uniform_filter1d


def section_sticks(fl_sec, min_h=20, max_n=10, thresh=0.3):
    """Candidate sticks on one section. fl_sec: (lateral, time) fault likelihood 0..1.

    Each stick: {"pts": [[lateral, time], ...] top to bottom, "score", "height", "fl"}.
    Blobs and bed-parallel streaks are rejected; only tall, narrow features are kept.
    """
    lab, n = label(fl_sec > thresh, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        la, tt = np.nonzero(lab == c)
        t0, t1 = int(tt.min()), int(tt.max())
        h = t1 - t0
        if h < min_h:
            continue
        w = fl_sec[la, tt].astype(np.float64)
        sw = np.bincount(tt - t0, weights=w, minlength=h + 1)
        sl = np.bincount(tt - t0, weights=w * la, minlength=h + 1)
        cnt = np.bincount(tt - t0, minlength=h + 1)
        ok = sw > 0
        if ok.sum() < min_h * 0.6:
            continue
        cen, ts = sl[ok] / sw[ok], (np.arange(h + 1) + t0)[ok]
        if cnt[ok].mean() > max(10.0, 0.5 * h):
            continue                                            # a blob, not a line
        if np.ptp(cen) > 2.0 * h:
            continue                                            # lies along the bedding
        cs = uniform_filter1d(cen, size=int(min(len(cen), 9)), mode="nearest")
        k = np.unique(np.linspace(0, len(ts) - 1, min(6, len(ts))).astype(int))
        out.append({"pts": [[round(float(cs[i]), 1), float(ts[i])] for i in k], "height": int(h),
                    "fl": round(float(w.mean()), 2), "score": round(float(w.mean() * h), 1)})
    # pieces of one fault that are split by a short gap along the same line are joined into one stick
    out.sort(key=lambda s: s["pts"][0][1])
    merged = []
    for s in out:
        m = next((q for q in merged if 0 <= s["pts"][0][1] - q["pts"][-1][1] <= 14 and abs(s["pts"][0][0] - q["pts"][-1][0]) <= 4), None)
        if m is None:
            merged.append(s)
            continue
        h = m["height"] + s["height"]
        m["fl"] = round((m["fl"] * m["height"] + s["fl"] * s["height"]) / h, 2)
        allp = m["pts"] + s["pts"]
        k = np.unique(np.linspace(0, len(allp) - 1, min(7, len(allp))).astype(int))
        m["pts"] = [allp[i] for i in k]
        m["height"] = int(m["pts"][-1][1] - m["pts"][0][1])
        m["score"] = round(m["fl"] * m["height"], 1)
    merged.sort(key=lambda s: -s["score"])
    return merged[:max_n]


def _lag(sec, ja, jb, t, win, maxlag):
    """Best vertical shift (samples) that lines trace jb up with trace ja around time t, and how well it fits."""
    nlat, nt = sec.shape
    if ja < 0 or jb < 0 or ja >= nlat or jb >= nlat or t - win - maxlag < 0 or t + win + maxlag >= nt:
        return None
    a = sec[ja, t - win:t + win + 1].astype(np.float64)
    best, bl = -2.0, 0
    for lag in range(-maxlag, maxlag + 1):
        b = sec[jb, t - win + lag:t + win + 1 + lag].astype(np.float64)
        den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
        c = float((a * b).sum() / den) - 0.004 * abs(lag)
        if c > best:
            best, bl = c, lag
    return bl, best


def stick_throw(sec, stick, offset=5, win=20, maxlag=15):
    """Vertical offset of the reflectors across a stick, in samples. sec: (lateral, time) seismic.

    The ordinary dip of the beds (measured on each side of the stick) is taken out, so what is left is the
    step at the stick itself. Positive = the high-index side is later (deeper), i.e. that side is downthrown.
    Returns (median throw in samples, mean correlation of the match, throws at the three depths).
    """
    p = np.asarray(stick["pts"], dtype=float)
    lags, cs = [], []
    for frac in (0.25, 0.5, 0.75):
        t = int(round(p[:, 1].min() + frac * np.ptp(p[:, 1])))
        lat = int(round(float(np.interp(t, p[:, 1], p[:, 0]))))
        across = _lag(sec, lat - offset, lat + offset, t, win, maxlag)
        if across is None:
            continue
        left = _lag(sec, lat - 2 * offset, lat - offset, t, win, maxlag)
        right = _lag(sec, lat + offset, lat + 2 * offset, t, win, maxlag)
        dips = [q[0] for q in (left, right) if q is not None and q[1] > 0.5]
        dip = 2.0 * float(np.mean(dips)) if dips else 0.0           # dip over the 2 x offset traces that are spanned
        lags.append(across[0] - dip)
        cs.append(across[1])
    if not lags:
        return 0.0, 0.0, []
    return float(np.median(lags)), float(np.mean(cs)), lags


def decoy_sticks(fl_sec, real, n=1, min_h=20, seed=0, live_row=None):
    """Fake sticks drawn where there is NO sign of a fault. They look like the others on the picture; if the
    assistant calls them faults too, its votes are not to be trusted."""
    rng = np.random.default_rng(seed)
    nlat, nt = fl_sec.shape
    quiet = fl_sec.max(axis=1) < 0.12
    if live_row is not None:
        quiet &= live_row
    taken = [np.mean([q[0] for q in s["pts"]]) for s in real]
    ok = [j for j in range(8, nlat - 8) if quiet[max(0, j - 5):j + 6].all() and all(abs(j - t) > 14 for t in taken)]
    out = []
    h = int(np.median([s["height"] for s in real])) if real else int(min_h * 2)
    h = int(np.clip(h, min_h, max(min_h, nt // 2)))
    for _ in range(n):
        if not ok:
            break
        j = int(rng.choice(ok))
        t0 = int(rng.integers(5, max(6, nt - h - 5)))
        tilt = float(rng.uniform(-4, 4))
        out.append({"pts": [[j + tilt * f, t0 + f * h] for f in (0.0, 0.33, 0.66, 1.0)], "height": h, "fl": 0.0,
                    "score": 0.0, "decoy": True})
        ok = [q for q in ok if abs(q - j) > 14]
    return out


def _lat(st, t):
    p = np.asarray(st["pts"], dtype=float)
    return np.interp(t, p[:, 1], p[:, 0], left=np.nan, right=np.nan)


def link_sticks(sticks, step, fl=None):
    """Join accepted sticks on neighbouring lines into faults. Returns {stick index: group number}.

    Two sticks are joined when they overlap in time, sit close to each other, and - if the fault-likelihood
    cube is given - the cube stays raised along the straight path between them (so two different faults that
    happen to be near each other are not merged).
    """
    parent = list(range(len(sticks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for axis in ("i", "x"):
        byg = {}
        for n, s in enumerate(sticks):
            if s["axis"] == axis and s.get("accepted"):
                byg.setdefault(s["g"], []).append(n)
        gs = sorted(byg)
        for a_g, b_g in zip(gs[:-1], gs[1:]):
            if b_g - a_g > 2 * step + 0.5:
                continue
            pairs = []
            for a in byg[a_g]:
                pa = np.asarray(sticks[a]["pts"], float)
                for b in byg[b_g]:
                    pb = np.asarray(sticks[b]["pts"], float)
                    t0, t1 = max(pa[:, 1].min(), pb[:, 1].min()), min(pa[:, 1].max(), pb[:, 1].max())
                    if t1 - t0 < 0.4 * min(np.ptp(pa[:, 1]), np.ptp(pb[:, 1])):
                        continue
                    ts = np.linspace(t0, t1, 7)
                    la, lb = _lat(sticks[a], ts), _lat(sticks[b], ts)
                    d = float(np.nanmean(np.abs(la - lb)))
                    if d > 1.2 * (b_g - a_g) + 3:
                        continue
                    if fl is not None:
                        tm = int(round(0.5 * (t0 + t1)))
                        lam, lbm = float(_lat(sticks[a], tm)), float(_lat(sticks[b], tm))
                        k = np.linspace(0, 1, max(3, int(b_g - a_g) + 1))
                        gg = np.rint(a_g + k * (b_g - a_g)).astype(int)
                        ll = np.rint(lam + k * (lbm - lam)).astype(int)
                        if axis == "i":
                            v = fl[np.clip(gg, 0, fl.shape[0] - 1), np.clip(ll, 0, fl.shape[1] - 1), int(np.clip(tm, 0, fl.shape[2] - 1))]
                        else:
                            v = fl[np.clip(ll, 0, fl.shape[0] - 1), np.clip(gg, 0, fl.shape[1] - 1), int(np.clip(tm, 0, fl.shape[2] - 1))]
                        if float(np.mean(v)) < 0.12:
                            continue
                    pairs.append((d, a, b))
            for d, a, b in sorted(pairs):
                parent[find(a)] = find(b)
    # the same fault is usually seen on both line families: merge groups whose points coincide in 3-D
    from scipy.spatial import cKDTree

    def cloud(n):
        s = sticks[n]
        p = np.asarray(s["pts"], float)
        ts = np.arange(p[:, 1].min(), p[:, 1].max() + 1, 4.0)
        la = np.interp(ts, p[:, 1], p[:, 0])
        gi, gx = (np.full(len(ts), float(s["g"])), la) if s["axis"] == "i" else (la, np.full(len(ts), float(s["g"])))
        return np.stack([gi, gx, ts * (0.75 * step / 12.0)], axis=1)

    groups = {}
    for n, s in enumerate(sticks):
        if s.get("accepted"):
            groups.setdefault((find(n), s["axis"]), []).append(n)
    keys = list(groups)
    clouds = {k: np.concatenate([cloud(n) for n in groups[k]]) for k in keys}
    trees = {k: cKDTree(clouds[k]) for k in keys}
    for a in keys:
        for b in keys:
            if a[1] != "i" or b[1] != "x" or find(a[0]) == find(b[0]):
                continue
            small, big = (a, b) if len(clouds[a]) <= len(clouds[b]) else (b, a)
            d, _ = trees[big].query(clouds[small], k=1)
            if float((d <= 0.75 * step).mean()) >= 0.3:
                parent[find(a[0])] = find(b[0])
    return {n: find(n) for n, s in enumerate(sticks) if s.get("accepted")}


def describe_faults(sticks, groups, grid, min_sticks=2):
    """Name the faults (largest first) and measure each: strike, length, vertical extent, throw, downthrown side."""
    cell = grid.get("cell_m") or [1.0, 1.0]
    u_il, u_xl = grid.get("u_il", (0.0, 1.0)), grid.get("u_xl", (1.0, 0.0))
    t0, dt = float(np.asarray(grid["times"])[0]), float(grid["dt_ms"])
    il_of, xl_of = np.asarray(grid["ilines"]), np.asarray(grid["xlines"])
    by = {}
    for n, gid in groups.items():
        by.setdefault(gid, []).append(n)
    recs = []
    for gid, members in by.items():
        if len(members) < min_sticks:
            continue
        gi, gx, tt, thr, down_e, down_n = [], [], [], [], 0.0, 0.0
        for n in members:
            s = sticks[n]
            p = np.asarray(s["pts"], float)
            for la, t in p:
                gi.append(s["g"] if s["axis"] == "i" else la)
                gx.append(la if s["axis"] == "i" else s["g"])
                tt.append(t)
            if s.get("throw_ms"):
                thr.append(abs(s["throw_ms"]))
                sgn = 1.0 if s["throw_ms"] > 0 else -1.0          # + = high-index side of the section is down
                u = u_xl if s["axis"] == "i" else u_il
                down_e += sgn * u[0]
                down_n += sgn * u[1]
        gi, gx, tt = np.array(gi), np.array(gx), np.array(tt)
        px, py = gx * cell[1], gi * cell[0]
        strike, length = None, None
        if len(px) >= 3 and (np.ptp(px) + np.ptp(py)) > 0:
            cov = np.cov(np.stack([px - px.mean(), py - py.mean()]))
            w, v = np.linalg.eigh(cov)
            major = v[:, 1]
            proj = (px - px.mean()) * major[0] + (py - py.mean()) * major[1]
            length = float(np.ptp(proj))
            east = major[1] * u_il[0] + major[0] * u_xl[0]
            north = major[1] * u_il[1] + major[0] * u_xl[1]
            strike = float(np.degrees(np.arctan2(east, north)) % 180.0)
        names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
        down = names[int(((np.degrees(np.arctan2(down_e, down_n)) % 360) + 22.5) // 45) % 8] if abs(down_e) + abs(down_n) > 0.5 else None
        trend = None
        if strike is not None:
            trend = ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "ESE-WNW", "SE-NW", "SSE-NNW"][int(((strike + 11.25) % 180) // 22.5)]
        rev = []  # Gemma does not accept, reject, draw or move fault sticks.
        ci, cx = float(np.median(gi)), float(np.median(gx))
        recs.append({"_members": members, "_size": len(members) * float(np.ptp(tt) + 1),
                     "strike_deg": None if strike is None else round(strike), "trend": trend,
                     "length_km": None if length is None or not grid.get("cell_m") else round(length / 1000.0, 2),
                     "twt_top_ms": round(t0 + float(tt.min()) * dt), "twt_base_ms": round(t0 + float(tt.max()) * dt),
                     "throw_ms": round(float(np.median(thr)), 1) if thr else None, "downthrown_towards": down,
                     "sticks": len(members), "sticks_reviewed": len(rev),
                     "sticks_confirmed": sum(1 for s in rev if s.get("verdict") == "fault"),
                     "where": {"il": int(il_of[int(np.clip(round(ci), 0, len(il_of) - 1))]),
                               "xl": int(xl_of[int(np.clip(round(cx), 0, len(xl_of) - 1))]),
                               "twt": round(t0 + float(np.median(tt)) * dt, 1)}})
    recs.sort(key=lambda r: -r["_size"])
    for n, r in enumerate(recs):
        r["id"] = f"F{n + 1}"
        for m in r.pop("_members"):
            sticks[m]["fault"] = r["id"]
        r.pop("_size")
    return recs


def build(grid, seis, fl, live, output, log=print, step=8):
    """Extract and link measured fault sticks without any language-model calls."""
    from .config import write_json
    dt = float(grid["dt_ms"])
    min_h = max(16, int(round(80.0 / dt)))
    ni, nx, _ = seis.shape
    lines = {"i": np.arange((ni // 2) % step, ni, step), "x": np.arange((nx // 2) % step, nx, step)}
    allst = []
    for axis, indices in lines.items():
        for gl in indices:
            sec = seis[gl] if axis == "i" else seis[:, gl]
            fsec = fl[gl] if axis == "i" else fl[:, gl]
            for n, s in enumerate(section_sticks(fsec, min_h, max_n=8)):
                lag, corr, lags = stick_throw(sec, s)
                accepted = bool(s["fl"] >= 0.4 and abs(lag) >= 1.5 and corr >= 0.7 and len(lags) >= 2 and
                                (max(lags) - min(lags)) <= max(4.0, 0.5 * abs(lag)))
                s.update({"id": f"{axis}{int(gl)}-{n + 1}", "axis": axis, "g": int(gl),
                          "line": int(grid["ilines" if axis == "i" else "xlines"][gl]),
                          "throw_samples": round(lag, 1), "throw_ms": round(lag * dt, 1), "throw_fit": round(corr, 2),
                          "throw_by_depth_ms": [round(v * dt, 1) for v in lags], "code_ok": accepted,
                          "accepted": accepted, "verdict": "fault" if accepted else "not_fault", "reviewed_by": "algorithm",
                          "why": "Measured likelihood, reflector offset and cross-fault fit tests " + ("passed" if accepted else "failed"),
                          "disputed": False})
                allst.append(s)
    groups = link_sticks(allst, step, fl)
    named = describe_faults(allst, groups, grid)
    write_json(output / "sticks.json", {"step": step, "lines": {k: [int(v) for v in a] for k, a in lines.items()},
               "sticks": allst, "method": "algorithmic likelihood and measured throw/fit tests", "gemma_used": False,
               "reviewed_lines": 0, "decoys": {"shown": 0, "rejected": 0, "votes_used": False}})
    write_json(output / "fault_surfaces.json", named)
    log(f"Fault sticks: {sum(s['accepted'] for s in allst)} of {len(allst)} kept by measured tests, {len(named)} named faults; no Gemma")
    return {"sticks": len(allst), "accepted": sum(s["accepted"] for s in allst), "faults": len(named), "gemma_used": False}
