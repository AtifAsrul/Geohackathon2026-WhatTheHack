"""Tie the wells to the seismic, and check the rock predictions against what the wells found.

Three things happen here, all on the interpretation grid:

  1. Well tie.   Each well's logs are slid up and down (at most 40 ms) against the seismic property trace
                 along its own path. The shift that fits best is reported, and used only when it is clear.
  2. Horizons.   Where a well crosses a horizon: which log boundary is nearest, how many ms apart they are,
                 and what rock the logs show above and below.
  3. Certainty.  The supplied sand-probability volume is tested the honest way: predict each well from all
                 the OTHER wells, then count how often the prediction was right. That gives statements like
                 "sand, 0.64; when the prediction was 0.6-0.8 the wells showed sand 62% of the time".
"""
import numpy as np

from . import config, wells
from .config import read_json, write_json

I = config.INTERP
FEATURES = ("sand", "ai", "si", "vpvs", "far_near")
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def _to_grid(vals, axis_vals):
    """Line number (or ms) -> fractional index on a grid axis; NaN outside. Works for descending axes too."""
    ax = np.asarray(axis_vals, dtype=np.float64)
    idx = np.arange(len(ax), dtype=np.float64)
    if len(ax) > 1 and ax[0] > ax[-1]:
        ax, idx = ax[::-1], idx[::-1]
    return np.interp(np.asarray(vals, dtype=np.float64), ax, idx, left=np.nan, right=np.nan)


def _path(a, g, shift_ms=0.0):
    """Fractional grid position (gi, gx, gt) of every log sample of one well, and which samples are usable."""
    if "il" not in a or "twt" not in a:
        return None
    gi, gx = _to_grid(a["il"], g["ilines"]), _to_grid(a["xl"], g["xlines"])
    gt = (a["twt"] + shift_ms - float(g["times"][0])) / float(g["dt_ms"])
    nt = len(g["times"])
    ok = np.isfinite(gi) & np.isfinite(gx) & np.isfinite(gt) & (gt >= 0) & (gt <= nt - 1)
    return gi, gx, gt, ok


# ------------------------------------------------------------------ 1. well tie
def tie_shift(a, g, vols, max_ms=40.0):
    """Best time shift between a well's logs and the seismic property trace along its path.

    Uses (1 - shale volume) against the sand-probability volume, or failing that the impedance log
    against inverted P-impedance. Positive shift = the logs have to move DOWN (later) to fit the seismic.
    """
    if "sand" in vols and ("c_vsh" in a or "c_sand" in a):
        log_curve = 1.0 - a["c_vsh"] if "c_vsh" in a else a["c_sand"]
        cube, pair = vols["sand"], "log sand content vs sand-probability volume"
    elif "ai" in vols and "c_ai" in a:
        log_curve, cube, pair = a["c_ai"], vols["ai"], "impedance log vs inverted P-impedance"
    else:
        return None
    p = _path(a, g)
    if p is None:
        return None
    gi, gx, gt, ok = p
    ok = ok & np.isfinite(log_curve)
    if ok.sum() < 30:
        return None
    ni, nx, nt = cube.shape
    k = np.rint(gt[ok]).astype(int)
    k0, k1 = int(k.min()), int(k.max())
    if k1 - k0 < 12:
        return None
    n = k1 - k0 + 1
    cnt = np.bincount(k - k0, minlength=n).astype(np.float64)
    lg = np.bincount(k - k0, weights=log_curve[ok], minlength=n) / np.maximum(cnt, 1)
    ci = np.rint(np.bincount(k - k0, weights=gi[ok], minlength=n) / np.maximum(cnt, 1)).astype(int)
    cx = np.rint(np.bincount(k - k0, weights=gx[ok], minlength=n) / np.maximum(cnt, 1)).astype(int)
    have = cnt > 0
    w = max(1, int(round(16.0 / float(g["dt_ms"]))))            # logs smoothed to roughly seismic resolution
    if w > 1:
        ker = np.ones(w) / w
        lg = np.convolve(np.where(have, lg, 0), ker, mode="same") / np.maximum(np.convolve(have.astype(float), ker, mode="same"), 1e-9)
    L = max(1, int(round(max_ms / float(g["dt_ms"]))))
    res = {}
    for lag in range(-L, L + 1):
        kk = np.arange(k0, k1 + 1) + lag
        m = have & (kk >= 0) & (kk < nt)
        if m.sum() < 12:
            continue
        sv = np.asarray(cube[np.clip(ci[m], 0, ni - 1), np.clip(cx[m], 0, nx - 1), kk[m]], dtype=np.float64)
        if sv.std() < 1e-9 or lg[m].std() < 1e-9:
            continue
        res[lag] = float(np.corrcoef(lg[m], sv)[0, 1])
    if 0 not in res or len(res) < 3:
        return None
    best = max(res, key=res.get)
    clear = bool(res[best] >= 0.4 and (best == 0 or res[best] - res[0] >= 0.05))
    return {"shift_ms": round(best * float(g["dt_ms"]), 1), "r_shifted": round(res[best], 2), "r_raw": round(res[0], 2),
            "clear": clear, "pair": pair, "points": int(have.sum())}


# ------------------------------------------------------------------ logs averaged onto the seismic grid
def well_rows(a, g, vols, facies=None, shift_ms=0.0, bin_ms=20.0, live=None):
    """One row per ~20 ms interval of one well: what the logs show, and what every seismic volume shows there."""
    p = _path(a, g, shift_ms)
    if p is None or "c_sand" not in a:
        return []
    gi, gx, gt, ok = p
    ok = ok & np.isfinite(a["c_sand"])
    if ok.sum() < 10:
        return []
    nt = len(g["times"])
    nb = max(1, int(round(bin_ms / float(g["dt_ms"]))))
    b = np.floor(gt / nb).astype(np.int64)
    rows = []
    for u in np.unique(b[ok]):
        idx = np.flatnonzero(ok & (b == u))
        if idx.size < 5:
            continue
        ci, cx = int(round(float(gi[idx].mean()))), int(round(float(gx[idx].mean())))
        if live is not None and not live[ci, cx]:
            continue
        t0i, t1i = int(u * nb), int(min(nt, u * nb + nb))
        row = {"twt": round(float(g["times"][0]) + (t0i + nb / 2.0) * float(g["dt_ms"]), 1), "gi": ci, "gx": cx,
               "net_sand": round(float(np.nanmean(a["c_sand"][idx])), 3)}
        for c, name in (("vsh", "vsh"), ("hc", "hc"), ("ai", "ai_log"), ("phi", "phi")):
            v = a.get(f"c_{c}")
            if v is not None and np.isfinite(v[idx]).any():
                row[name] = round(float(np.nanmean(v[idx])), 4)
        for role, cube in vols.items():
            if role in ("sand", "ai", "si", "vpvs"):
                row[role] = float(np.mean(cube[ci, cx, t0i:t1i]))
        if "near" in vols and "far" in vols:
            nr = float(np.sqrt(np.mean(np.square(vols["near"][ci, cx, t0i:t1i], dtype=np.float64))))
            fr = float(np.sqrt(np.mean(np.square(vols["far"][ci, cx, t0i:t1i], dtype=np.float64))))
            row["far_near"] = fr - nr
        if facies is not None:
            row["facies"] = int(np.bincount(np.asarray(facies[ci, cx, t0i:t1i]).astype(int)).argmax()) + 1
        rows.append(row)
    return rows


# ------------------------------------------------------------------ 3. certainty
def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _bal_acc(p, y):
    p, y = np.asarray(p) >= 0.5, np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return None
    return round(float(0.5 * ((p & y).sum() / y.sum() + (~p & ~y).sum() / (~y).sum())), 2)


def fit(rows_by_well, log=print):
    """Check the sand prediction against the wells, leaving one well out at a time, and keep the best recipe.

    Recipes compared on the same test: (a) always the average, (b) the sand-probability volume as supplied,
    (c) that volume re-scaled to what the wells show (isotonic), (d) several volumes together (logistic).
    """
    from sklearn.isotonic import IsotonicRegression
    use = {w: [r for r in rows if "sand" in r] for w, rows in rows_by_well.items()}
    use = {w: r for w, r in use.items() if len(r) >= 3}
    out = {"model": "none", "wells_used": sorted(use), "intervals": int(sum(len(r) for r in use.values()))}
    if not use:
        out["why"] = "no well has both usable logs in two-way time and a sand-probability value along its path"
        return out
    wl = sorted(use)
    y = {w: np.array([r["net_sand"] >= 0.5 for r in use[w]], dtype=float) for w in wl}
    p = {w: np.clip(np.array([r["sand"] for r in use[w]], dtype=float), 0, 1) for w in wl}
    ally, allp = np.concatenate([y[w] for w in wl]), np.concatenate([p[w] for w in wl])
    out["sand_share_at_wells"] = round(float(ally.mean()), 3)
    out["sand_prob_range"] = [round(float(allp.min()), 3), round(float(allp.max()), 3)]
    if len(wl) < 3 or ally.min() == ally.max():
        out["why"] = ("fewer than 3 usable wells" if len(wl) < 3 else "the wells show only one rock type") + \
                     ": the sand-probability volume is used as supplied and could not be checked"
        return out
    feats = [k for k in FEATURES if all(k in r for w in wl for r in use[w])]
    X = {w: np.array([[r[k] for k in feats] for r in use[w]], dtype=float) for w in wl}
    pred = {"baseline": [], "raw": [], "isotonic": [], "logistic": []}
    truth = []
    logistic_ok = len(feats) >= 2 and len(ally) >= 40
    for w in wl:
        tr = [q for q in wl if q != w]
        ytr, ptr = np.concatenate([y[q] for q in tr]), np.concatenate([p[q] for q in tr])
        truth.append(y[w])
        pred["baseline"].append(np.full(len(y[w]), ytr.mean()))
        pred["raw"].append(p[w])
        if ytr.min() == ytr.max():
            pred["isotonic"].append(np.full(len(y[w]), ytr.mean()))
            logistic_ok = False
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(ptr, ytr)
        pred["isotonic"].append(iso.predict(p[w]))
        if logistic_ok:
            try:
                from sklearn.linear_model import LogisticRegression
                Xtr = np.concatenate([X[q] for q in tr])
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                lr = LogisticRegression(C=1.0, max_iter=500).fit((Xtr - mu) / sd, ytr)
                pred["logistic"].append(lr.predict_proba((X[w] - mu) / sd)[:, 1])
            except Exception:
                logistic_ok = False
    truth = np.concatenate(truth)
    score = {k: round(_brier(np.concatenate(v), truth), 4) for k, v in pred.items() if v and (k != "logistic" or logistic_ok)}
    best = "isotonic"
    if "logistic" in score and score["logistic"] < 0.97 * score["isotonic"]:
        best = "logistic"
    chosen = np.concatenate(pred[best])
    out.update({"model": best, "features": feats if best == "logistic" else ["sand"],
                "error_score": score,           # Brier score: 0 = perfect, lower is better
                "beats_always_average": bool(score[best] < score["baseline"]),
                "balanced_accuracy": _bal_acc(chosen, truth), "balanced_accuracy_raw": _bal_acc(np.concatenate(pred["raw"]), truth)})
    rel = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (chosen >= lo) & (chosen < hi)
        if m.sum():
            rel.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "predicted": round(float(chosen[m].mean()), 2),
                        "observed": round(float(truth[m].mean()), 2)})
    out["reliability"] = rel
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(allp, ally)
    out["iso"] = {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
    if best == "logistic":
        from sklearn.linear_model import LogisticRegression
        Xa = np.concatenate([X[w] for w in wl])
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=500).fit((Xa - mu) / sd, ally)
        out["logit"] = {"mean": mu.tolist(), "std": sd.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    return out


def p_sand(calib, values):
    """Probability of sand at a place, from its seismic values, as checked against the wells.

    values: {"sand": .., "ai": .., "si": .., "vpvs": .., "far_near": ..} (only "sand" is required).
    Returns {"p", "rock", "certainty", "basis", "inside_tested_range"} or None.
    """
    s = (values or {}).get("sand")
    if s is None or not np.isfinite(s):
        return None
    c = calib or {}
    model, p = c.get("model", "none"), float(np.clip(s, 0, 1))
    if model == "logistic" and all(k in values and values[k] is not None for k in c.get("features", [])):
        L = c["logit"]
        zed = (np.array([values[k] for k in c["features"]], dtype=float) - np.array(L["mean"])) / np.array(L["std"])
        p = float(1.0 / (1.0 + np.exp(-(float(np.dot(zed, L["coef"])) + L["intercept"]))))
    elif model in ("isotonic", "logistic") and c.get("iso"):
        p = float(np.interp(s, c["iso"]["x"], c["iso"]["y"]))
    rel = None
    if model != "none":
        rel = next((r for r in c.get("reliability", []) if r["lo"] <= p < r["hi"] + 1e-9 and r["n"] >= 4), None)
        if rel and rel["n"] >= 10:                  # enough test intervals: quote what the wells actually showed
            p = float(rel["observed"])
    p = float(np.clip(p, 0.03, 0.97))               # a handful of wells can never justify 0 or 1
    out = {"p": round(p, 2), "rock": "sand" if p >= 0.5 else "shale", "certainty": round(max(p, 1 - p), 2)}
    if model == "none":
        out["basis"] = "sand-probability volume as supplied; " + c.get("why", "it could not be checked against wells")
        return out
    basis = f"checked on {c['intervals']} log intervals in {len(c['wells_used'])} wells, each well predicted from the others"
    if rel:
        basis += (f"; when the prediction was {rel['lo']:.1f}-{rel['hi']:.1f} the wells showed sand in "
                  f"{rel['observed']:.0%} of {rel['n']} intervals")
    rng = c.get("sand_prob_range")
    out["inside_tested_range"] = bool(rng is None or rng[0] - 0.02 <= s <= rng[1] + 0.02)
    if not out["inside_tested_range"]:
        basis += "; this value is outside the range the wells sampled, so it is an extrapolation"
    out["basis"] = basis
    return out


def describe(calib, values):
    """One sentence for evidence lists, e.g. 'sand with certainty 0.78 (checked on ...)'. '' if not possible."""
    r = p_sand(calib, values)
    return "" if r is None else f"{r['rock']} with certainty {r['certainty']:.2f} ({r['basis']})"


# ------------------------------------------------------------------ 2. horizons against log boundaries
def horizon_ties(horizons, g, ws, zones, shifts, max_ms=40.0):
    """Where each well crosses each horizon: nearest log boundary, mismatch, rock above and below."""
    t0, dt = float(g["times"][0]), float(g["dt_ms"])
    ties = []
    for w in ws.get("wells", []):
        key = w["key"]
        a = wells.well_arrays(key)
        sh = (shifts.get(w["name"]) or {})
        shift = sh.get("shift_ms", 0.0) if sh.get("clear") else 0.0
        p = _path(a, g, shift)
        if p is None:
            continue
        gi, gx, gt, ok = p
        idx = np.flatnonzero(ok)
        if idx.size < 3:
            continue
        ci = np.rint(gi[idx]).astype(int)
        cx = np.rint(gx[idx]).astype(int)
        tw = a["twt"][idx] + shift
        zb = (zones.get(w["name"]) or {}).get("boundaries", [])
        for hz in horizons:
            z_ms = t0 + hz["z"][ci, cx] * dt
            cross = np.flatnonzero(tw >= z_ms)
            if cross.size == 0 or cross[0] == 0:
                continue
            j = int(cross[0])
            th = float(z_ms[j])
            rec = {"well": w["name"], "horizon": hz["name"], "twt": round(th, 1), "polarity": hz.get("polarity"),
                   "where": {"il": int(g["ilines"][ci[j]]), "xl": int(g["xlines"][cx[j]]), "twt": round(th, 1)}}
            cand = [(abs(b["twt"] + shift - th), b) for b in zb if b.get("twt") is not None]
            if cand:
                d, b = min(cand, key=lambda q: q[0])
                if d <= max_ms:
                    rec.update({"boundary_twt": round(b["twt"] + shift, 1), "boundary": b["kind"],
                                "mismatch_ms": round(b["twt"] + shift - th, 1), "sand_thickness_m": b["thick_m"]})
            up = wells.interval_summary(key, th - 20.0 - shift, th - shift) or {}
            dn = wells.interval_summary(key, th - shift, th + 20.0 - shift) or {}
            for tag, s in (("above", up), ("below", dn)):
                if "net_sand_fraction" in s:
                    rec[tag] = {"net_sand": s["net_sand_fraction"], "rock": "sand" if s["net_sand_fraction"] >= 0.5 else "shale"}
                    if "hydrocarbon_fraction" in s:
                        rec[tag]["hydrocarbon_flag"] = s["hydrocarbon_fraction"]
            if "acoustic_impedance" in up and "acoustic_impedance" in dn:
                step = dn["acoustic_impedance"] - up["acoustic_impedance"]
                if abs(step) > 0.02 * abs(up["acoustic_impedance"]):
                    rec["impedance_step"] = "increase" if step > 0 else "decrease"
            ties.append(rec)
    by = {}
    for hz in horizons:
        t = [r for r in ties if r["horizon"] == hz["name"]]
        mm = [abs(r["mismatch_ms"]) for r in t if "mismatch_ms" in r]
        pairs = [f"{r['above']['rock']} over {r['below']['rock']}" for r in t if "above" in r and "below" in r]
        common = max(set(pairs), key=pairs.count) if pairs else None
        by[hz["name"]] = {"wells": len(t), "with_boundary": len(mm),
                          "median_mismatch_ms": round(float(np.median(mm)), 1) if mm else None,
                          "usual_rocks": common, "usual_rocks_count": pairs.count(common) if common else 0,
                          "rock_pairs": len(pairs)}
    votes = {"increase_is_peak": 0, "increase_is_trough": 0}
    for r in ties:
        if r.get("impedance_step") and r.get("polarity") in ("peak", "trough") and abs(r.get("mismatch_ms", 99)) <= 12:
            same = (r["impedance_step"] == "increase") == (r["polarity"] == "peak")
            votes["increase_is_peak" if same else "increase_is_trough"] += 1
    n = sum(votes.values())
    if n >= 4 and max(votes.values()) >= 0.75 * n:
        votes["reading"] = ("an impedance increase downward shows as a " +
                            ("peak" if votes["increase_is_peak"] > votes["increase_is_trough"] else "trough") +
                            f" in {max(votes['increase_is_peak'], votes['increase_is_trough'])} of {n} well/horizon crossings")
    else:
        votes["reading"] = f"polarity could not be settled from the wells ({n} usable crossings)"
    return {"ties": ties, "by_horizon": by, "polarity": votes}


# ------------------------------------------------------------------ the pipeline step
def run(log=print):
    g = read_json(I / "grid.json")
    if not g:
        raise RuntimeError("the interpretation grid is missing")
    for k in ("ilines", "xlines", "times"):
        g[k] = np.asarray(g[k])
    ws = wells.wells_summary()
    vols = {r: np.load(I / f"vol_{r}.npy", mmap_mode="r") for r in ("sand", "ai", "si", "vpvs", "near", "far")
            if (I / f"vol_{r}.npy").exists()}
    role_ref = next((k for k, v in (g.get("roles") or {}).items() if v == g.get("ref")), None)
    if role_ref in ("near", "far") and role_ref not in vols:
        vols[role_ref] = np.load(I / "seis.npy", mmap_mode="r")
    facies = np.load(I / "facies.npy", mmap_mode="r") if (I / "facies.npy").exists() else None
    live = np.load(I / "live.npy") if (I / "live.npy").exists() else None
    zones, shifts, rows, excluded = {}, {}, {}, []
    for w in ws.get("wells", []):
        name, a = w["name"], wells.well_arrays(w["key"])
        z = wells.log_zones(w["key"])
        if z:
            zones[name] = z
        if not w.get("has", {}).get("logs"):
            excluded.append({"well": name, "why": "no log file"})
            continue
        if "c_sand" not in a:
            excluded.append({"well": name, "why": "no gamma-ray or shale-volume curve, so sand cannot be told from shale"})
            continue
        if "twt" not in a or not np.isfinite(a["twt"]).any():
            excluded.append({"well": name, "why": "no time-depth relation (no checkshot here or in any other well)"})
            continue
        if "il" not in a or not w.get("inside_survey", True):
            excluded.append({"well": name, "why": "outside the seismic survey, or position unknown"})
            continue
        sh = tie_shift(a, g, vols)
        if sh:
            shifts[name] = sh
        borrowed = w.get("td_borrow_km") is not None
        if borrowed and not (sh and sh["clear"]):
            excluded.append({"well": name, "why": "time-depth borrowed from another well and no clear tie to the seismic"})
            r = well_rows(a, g, vols, facies, 0.0, live=live)
            if r:
                rows[name] = {"rows": r, "train": False}
            continue
        r = well_rows(a, g, vols, facies, sh["shift_ms"] if sh and sh["clear"] else 0.0, live=live)
        if len(r) < 3:
            excluded.append({"well": name, "why": "too little log inside the seismic volume"})
            continue
        rows[name] = {"rows": r, "train": True}
    calib = fit({w: v["rows"] for w, v in rows.items() if v["train"]}, log=log)
    calib["wells_excluded"] = excluded
    calib["shifts"] = shifts
    fs = {}
    for w, v in rows.items():
        if not v["train"]:
            continue
        for r in v["rows"]:
            if "facies" in r:
                d = fs.setdefault(str(r["facies"]), {"sum": 0.0, "intervals": 0, "wells": set()})
                d["sum"] += r["net_sand"]
                d["intervals"] += 1
                d["wells"].add(w)
    calib["facies_net_sand"] = {k: {"net_sand": round(d["sum"] / d["intervals"], 3), "intervals": d["intervals"],
                                    "wells": len(d["wells"])} for k, d in fs.items()}
    write_json(I / "calib.json", calib)
    write_json(I / "well_zones.json", zones)
    hz = []
    if (I / "horizons.npz").exists():
        d = np.load(I / "horizons.npz")
        hz = [{"name": str(n), "z": d["z"][k], "polarity": str(d["polarity"][k])} for k, n in enumerate(d["names"])]
    write_json(I / "ties.json", horizon_ties(hz, g, ws, zones, shifts))
    if calib["model"] == "none":
        log(f"Rock certainty: {calib.get('why', 'not available')}")
    else:
        log(f"Rock certainty: {calib['model']} recipe, {calib['intervals']} log intervals in {len(calib['wells_used'])} wells, "
            f"error score {calib['error_score'][calib['model']]} (always-average {calib['error_score']['baseline']})")
    for name, sh in shifts.items():
        log(f"Well tie {name}: best shift {sh['shift_ms']:+.0f} ms, fit {sh['r_shifted']} (unshifted {sh['r_raw']})"
            + ("" if sh["clear"] else " - not clear, not applied"))
    return calib
