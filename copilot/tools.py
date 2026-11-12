"""Look-ups and pictures shared by the web page and by the chat assistant."""
import io

import numpy as np
from scipy.ndimage import binary_erosion, label

from . import config, knowledge, sfm, store, wells
from .config import read_json

I = config.INTERP
HCOLORS = ["#ffd400", "#ff7f0e", "#2ca02c", "#17becf", "#e377c2", "#b39ddb", "#c49c94", "#ff5c5c", "#9edae5", "#dbdb8d"]
FACIES_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]

_c = {}


def _cached(name, path, loader):
    if not path.exists():
        return None
    mt = path.stat().st_mtime
    if _c.get(name, (None,))[0] != mt:
        _c[name] = (mt, loader(path))
    return _c[name][1]


def grid():
    def ld(p):
        g = read_json(p)
        for k in ("ii", "xi", "ti", "ilines", "xlines", "times"):
            g[k] = np.asarray(g[k])
        return g
    return _cached("grid", I / "grid.json", ld)


def horizons(source="rgt"):
    """source 'rgt' = surfaces from the RGT cube; 'gemma' = the assistant's tracked picks."""
    def ld(p):
        d = np.load(p)
        out = {"names": [str(n) for n in d["names"]], "z": d["z"], "polarity": [str(x) for x in d["polarity"]]}
        if "conf" in d.files:
            out["conf"] = d["conf"]
        return out
    if source == "gemma":
        return _cached("hz_gemma", I / "horizons_gemma.npz", ld)
    return _cached("hz", I / "horizons.npz", ld)


def live():
    """(inline, crossline) mask of traces that hold real data, or None if every trace does."""
    m = _cached("live", I / "live.npy", lambda p: np.load(p))
    return None if m is None or m.all() else m


def cube(name):
    """Interpretation-grid cubes: rgt, faults, facies, seis."""
    return _cached(name, I / f"{name}.npy", lambda p: np.load(p, mmap_mode="r"))


def maps():
    """Horizon maps of both interpreters in one dictionary: 'H3__time', 'G2__time', 'G2__diff', ..."""
    a = _cached("maps", I / "maps.npz", lambda p: {k: v for k, v in np.load(p).items()})
    b = _cached("maps_gemma", I / "maps_gemma.npz", lambda p: {k: v for k, v in np.load(p).items()})
    if a is None:
        return None
    return dict(a, **b) if b else a


def lead_map():
    return _cached("lead_map", I / "lead_map.npy", lambda p: np.load(p))


def ready():
    return (I / "interp.json").exists() and knowledge.load() is not None


# ------------------------------------------------------------------ index conversions
def _ref():
    return store.reference_volume()


def to_index(il=None, xl=None, twt=None):
    """Line numbers / ms -> index in the full reference volume."""
    r = _ref()
    out = []
    if il is not None:
        out.append(int(np.abs(r.ilines - il).argmin()))
    if xl is not None:
        out.append(int(np.abs(r.xlines - xl).argmin()))
    if twt is not None:
        out.append(int(np.abs(r.samples - twt).argmin()))
    return out[0] if len(out) == 1 else tuple(out)


def g_index(il=None, xl=None, twt=None):
    """Line numbers / ms -> index on the interpretation grid."""
    g = grid()
    out = []
    if il is not None:
        out.append(int(np.abs(g["ilines"] - il).argmin()))
    if xl is not None:
        out.append(int(np.abs(g["xlines"] - xl).argmin()))
    if twt is not None:
        out.append(int(np.abs(g["times"] - twt).argmin()))
    return out[0] if len(out) == 1 else tuple(out)


# ------------------------------------------------------------------ overlays for the viewer
def horizon_lines(axis, number, source="rgt"):
    """Horizon picks along one section. axis 'i' = inline section, 'x' = crossline section.

    'live' says, per point, whether there is real data there (the line is drawn faint over empty traces).
    """
    h, g = horizons(source), grid()
    if h is None:
        return []
    lv = live()
    out = []
    for k, name in enumerate(h["names"]):
        if axis == "i":
            a = g_index(il=number)
            z, pos, ok = h["z"][k][a, :], g["xlines"], (lv[a, :] if lv is not None else None)
        else:
            b = g_index(xl=number)
            z, pos, ok = h["z"][k][:, b], g["ilines"], (lv[:, b] if lv is not None else None)
        twt = g["times"][0] + z * g["dt_ms"]
        rec = {"name": name, "color": HCOLORS[k % len(HCOLORS)], "pos": [int(p) for p in pos],
               "twt": [round(float(t), 1) for t in twt], "source": source}
        if ok is not None:
            rec["live"] = [bool(v) for v in ok]
        out.append(rec)
    return out


def overlay(layer, axis, number, fault_coherence_max=None):
    """2-D uint8 slice of an interpretation cube + the line-number / ms extent it covers."""
    g = grid()
    if g is None:
        return None
    if layer == "similar":
        c = _c.get("similar")
        if c is None:
            return None
        sim, (ci, cx, ct) = c
        a, b, t = (int(np.abs(ci - g_index(il=number)).argmin()) if axis == "i" else None,
                   int(np.abs(cx - g_index(xl=number)).argmin()) if axis == "x" else None,
                   int(np.abs(ct - g_index(twt=number)).argmin()) if axis == "t" else None)
        s = sim[a] if axis == "i" else (sim[:, b] if axis == "x" else sim[:, :, t])
        ext_i = [int(g["ilines"][min(int(ci[0]), len(g["ilines"]) - 1)]), int(g["ilines"][min(int(ci[-1]), len(g["ilines"]) - 1)])]
        ext_x = [int(g["xlines"][min(int(cx[0]), len(g["xlines"]) - 1)]), int(g["xlines"][min(int(cx[-1]), len(g["xlines"]) - 1)])]
        ext_t = [float(g["times"][min(int(ct[0]), len(g["times"]) - 1)]), float(g["times"][min(int(ct[-1]), len(g["times"]) - 1)])]
        data = (np.clip(s, 0, 1) * 255).astype(np.uint8)
    else:
        preview = layer == "faults" and fault_coherence_max is not None
        cb = cube("faults_base") if preview else None
        if cb is None:
            cb = cube({"faults": "faults", "facies": "facies", "rgt": "rgt"}[layer])
        if cb is None:
            return None
        s = cb[g_index(il=number)] if axis == "i" else (cb[:, g_index(xl=number)] if axis == "x" else cb[:, :, g_index(twt=number)])
        s = np.asarray(s)
        data = (np.clip(s, 0, 1) * 255).astype(np.uint8) if layer == "rgt" else s.astype(np.uint8)
        if preview:
            from .faults import apply_cutoff
            coh = cube("coherence")
            if coh is None:
                coh = cube("semb")
            if coh is None:
                raise ValueError("Coherence is unavailable; refresh the fault stage first")
            cs = coh[g_index(il=number)] if axis == "i" else (coh[:, g_index(xl=number)] if axis == "x" else coh[:, :, g_index(twt=number)])
            if coh.dtype == np.uint8:
                cs = cs.astype(np.float32) / 255.0
            data = (apply_cutoff(data.astype(np.float32) / 255.0, cs, fault_coherence_max) * 255).astype(np.uint8)
        ext_i = [int(g["ilines"][0]), int(g["ilines"][-1])]
        ext_x = [int(g["xlines"][0]), int(g["xlines"][-1])]
        ext_t = [float(g["times"][0]), float(g["times"][-1])]
    # section images are (time rows, lateral columns); maps are (inline rows, crossline columns)
    if axis == "i":
        return np.ascontiguousarray(data.T), {"h": ext_x, "v": ext_t}
    if axis == "x":
        return np.ascontiguousarray(data.T), {"h": ext_i, "v": ext_t}
    return np.ascontiguousarray(data), {"h": ext_x, "v": ext_i}


def lead_outlines(horizon):
    """Outline cells of each lead on one horizon, as line-number pairs for drawing."""
    lm, h, g = lead_map(), horizons(), grid()
    if lm is None or h is None or horizon not in h["names"]:
        return []
    m = lm[h["names"].index(horizon)]
    out = []
    for n in np.unique(m):
        if n == 0:
            continue
        mask = m == n
        edge = mask & ~binary_erosion(mask)
        ii, xx = np.nonzero(edge)
        out.append({"id": f"L{int(n)}", "il": [int(v) for v in g["ilines"][ii]], "xl": [int(v) for v in g["xlines"][xx]]})
    return out


# ------------------------------------------------------------------ look-ups for the assistant
def probe(il, xl, twt):
    """Everything known at one point of the survey."""
    r, g, h = _ref(), grid(), horizons()
    i, x, t = to_index(il, xl, twt)
    out = {"il": int(r.ilines[i]), "xl": int(r.xlines[x]), "twt_ms": float(r.samples[t]), "values": {}}
    xy = r.xy(r.ilines[i], r.xlines[x])
    if xy:
        out["x"], out["y"] = round(float(xy[0]), 1), round(float(xy[1]), 1)
    for v in store.list_volumes():
        try:
            vi = int(np.abs(v.ilines - r.ilines[i]).argmin())
            vx = int(np.abs(v.xlines - r.xlines[x]).argmin())
            vt = int(np.abs(v.samples - r.samples[t]).argmin())
            val = v.value(vi, vx, vt)
            st = v.meta["stats"]
            out["values"][v.meta["label"]] = {"value": round(val, 4), "typical_range": [round(st["lo"], 4), round(st["hi"], 4)]}
        except Exception:
            pass
    if g is not None and h is not None:
        gi, gx, gt = g_index(il, xl, twt)
        times = [float(g["times"][0] + h["z"][k][gi, gx] * g["dt_ms"]) for k in range(len(h["names"]))]
        above = [n for n, tt in zip(h["names"], times) if tt <= twt]
        below = [n for n, tt in zip(h["names"], times) if tt > twt]
        out["between_horizons"] = [above[-1] if above else "above H1", below[0] if below else "below the deepest horizon"]
        out["horizon_times_here_ms"] = {n: round(tt, 1) for n, tt in zip(h["names"], times)}
        fc, fl = cube("facies"), cube("faults")
        lv = live()
        if lv is not None and not lv[gi, gx]:
            out["no_data_here"] = "this trace is empty (outside the surveyed area); values and horizons here are not real"
        if fc is not None:
            out["facies"] = int(fc[gi, gx, gt]) + 1
            out["facies_name"] = facies_name(out["facies"])
        hg = horizons("gemma")
        cmp_ = read_json(I / "hz_compare.json") or {}
        if hg is not None and above:
            pr = next((q for q in cmp_.get("pairs", []) if q.get("rgt") == above[-1]), None)
            if pr and pr["gemma"] in hg["names"]:
                tg = float(g["times"][0] + hg["z"][hg["names"].index(pr["gemma"])][gi, gx] * g["dt_ms"])
                out["horizons_agree"] = {"rgt": above[-1], "assistant": pr["gemma"],
                                         "difference_here_ms": round(tg - times[h["names"].index(above[-1])], 1)}
        if fl is not None:
            out["fault_likelihood"] = round(float(fl[gi, gx, gt]) / 255.0, 2)
        lm = lead_map()
        if lm is not None:
            ids = [int(v) for v in np.unique(lm[:, gi, gx]) if v > 0]
            out["inside_leads"] = [f"L{n}" for n in ids]
    ws = [w for w in wells.wells_summary().get("wells", []) if w.get("il") is not None]
    if ws:
        b = r.bin_size() or (1.0, 1.0)
        di = float(np.median(np.diff(r.ilines))) or 1.0
        dx = float(np.median(np.diff(r.xlines))) or 1.0
        d = [(np.hypot((w["il"] - il) / di * b[0], (w["xl"] - xl) / dx * b[1]), w["name"]) for w in ws]
        dist, name = min(d)
        out["nearest_well"] = {"well": name, "distance_km": round(float(dist) / 1000.0, 2)}
        try:
            lg = wells.interval_summary(wells.norm(name), twt - 12.0, twt + 12.0)
            if lg:
                out["nearest_well_logs_here"] = dict(lg, note=f"logs of {name} within 12 ms of this time, {out['nearest_well']['distance_km']} km away")
        except Exception:
            pass
    try:
        if g is not None:
            vals = {}
            for role in ("sand", "ai", "si", "vpvs"):
                c = gcube(role)
                if c is not None:
                    vals[role] = float(c[gi, gx, gt])
            sc = sand_certainty(vals)
            if sc:
                out["sand_certainty"] = sc
            nf = nearest_fault(il, xl, twt)
            if nf:
                out["nearest_fault"] = nf
    except Exception:
        pass
    return out


def lead(lid):
    k = knowledge.load() or {}
    et = {e["id"]: e for e in k.get("evidence", [])}
    for l in k.get("leads", []):
        if l["id"].lower() == str(lid).lower():
            d = dict(l)
            for key in ("for", "against", "unknown"):
                d[key] = [{"id": e, "text": et[e]["text"]} for e in l[key] if e in et]
            return d
    return {"error": f"no lead called {lid}", "available": [l["id"] for l in k.get("leads", [])]}


def well(name):
    k = knowledge.load() or {}
    key = wells.norm(name)
    for w in k.get("wells", []):
        if wells.norm(w["name"]) == key:
            d = {a: b for a, b in w.items() if a not in ("path", "key")}
            tie = next((t for t in k.get("well_ties", []) if wells.norm(t["well"]) == key), None)
            d["horizons"] = tie["horizons"] if tie else {}
            return d
    return {"error": f"no well called {name}", "available": [w["name"] for w in k.get("wells", [])]}


def horizon(name):
    k = knowledge.load() or {}
    for h in k.get("horizons", []):
        if h["name"].lower() == str(name).lower():
            d = dict(h)
            d["leads_here"] = [{"id": l["id"], "kind": l["kind"], "overall": l["scores"]["overall"], "tested": l["tested"]}
                               for l in k.get("leads", []) if l["horizon"] == h["name"]]
            return d
    return {"error": f"no horizon called {name}", "available": [h["name"] for h in k.get("horizons", [])]}


def lookalike(il, xl, twt):
    """Where else does the seismic look like it does here? (foundation-model features)"""
    g = grid()
    if g is None:
        return {"error": "interpretation has not been run"}
    gi, gx, gt = g_index(il, xl, twt)
    res = sfm.similarity(gi, gx, gt)
    if res is None:
        return {"error": "no foundation-model features available"}
    sim, (ci, cx, ct) = res
    _c["similar"] = (sim, (ci, cx, ct))
    thr = max(0.6, float(np.percentile(sim, 95)))
    lab, n = label(sim > thr)
    regions = []
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() < 2:
            continue
        a, b, t = (int(round(v.mean())) for v in np.nonzero(m))
        pil, pxl = int(ci[a]), int(cx[b])
        regions.append({"il": int(g["ilines"][min(pil, len(g["ilines"]) - 1)]), "xl": int(g["xlines"][min(pxl, len(g["xlines"]) - 1)]),
                        "twt_ms": float(g["times"][min(int(ct[t]), len(g["times"]) - 1)]), "patches": int(m.sum()),
                        "similarity": round(float(sim[m].mean()), 2)})
    regions.sort(key=lambda r: -r["patches"])
    method = (read_json(I / "sfm.json") or {}).get("method")
    return {"reference": {"il": il, "xl": xl, "twt_ms": twt}, "threshold": round(thr, 2), "method": method,
            "share_of_volume_similar": round(float((sim > thr).mean()), 3), "regions": regions[:8]}


# ------------------------------------------------------------------ pictures (for the assistant's eyes)
def _lut(name):
    pts = {
        "gray": [(0, (0, 0, 0)), (1, (255, 255, 255))],
        "seismic": [(0, (20, 40, 160)), (0.5, (250, 250, 250)), (1, (170, 20, 20))],
        "viridis": [(0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1, (253, 231, 37))],
        "terrain": [(0, (180, 30, 30)), (0.25, (240, 160, 60)), (0.5, (245, 240, 150)), (0.75, (90, 180, 120)), (1, (40, 70, 160))],
    }[name]
    xs = np.array([p[0] for p in pts])
    cols = np.array([p[1] for p in pts], dtype=float)
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, cols[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _font(size=14):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_section(axis, number, vol_name=None, twt=None, mark=None, width=900, height=620):
    """PNG of an inline ('i') or crossline ('x') section with horizons drawn on it."""
    from PIL import Image, ImageDraw
    v = store.get_volume(vol_name) if vol_name else _ref()
    v = v or _ref()
    idx = int(np.abs((v.ilines if axis == "i" else v.xlines) - number).argmin())
    s = v.slice(axis, idx)                                   # (lateral, time)
    st = v.meta["stats"]
    t0, t1 = float(v.samples[0]), float(v.samples[-1])
    if twt is not None and (t1 - t0) > 1400:
        a = max(t0, twt - 600)
        b = min(t1, a + 1200)
        k0, k1 = int(np.abs(v.samples - a).argmin()), int(np.abs(v.samples - b).argmin())
        s, t0, t1 = s[:, k0:k1 + 1], float(v.samples[k0]), float(v.samples[k1])
    n = np.clip((s.T - st["lo"]) / (st["hi"] - st["lo"]), 0, 1)
    img = Image.fromarray(_lut("seismic" if st["bipolar"] else "viridis")[(n * 255).astype(np.uint8)])
    img = img.resize((width, height), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    lat = v.xlines if axis == "i" else v.ilines
    l0, l1 = float(lat[0]), float(lat[-1])

    def px(pos, t):
        return ((pos - l0) / (l1 - l0 + 1e-9) * (width - 1), (t - t0) / (t1 - t0 + 1e-9) * (height - 1))

    font = _font(14)
    for hz in horizon_lines(axis, number):
        pts = [px(p, t) for p, t in zip(hz["pos"], hz["twt"])]
        d.line(pts, fill=_hex(hz["color"]), width=2)
        d.text((6, min(max(pts[0][1] - 16, 20), height - 18)), hz["name"], fill=_hex(hz["color"]), font=font)
    if mark:
        mx, my = px(mark[0], mark[1])
        d.line([(mx, 0), (mx, height)], fill=(0, 255, 255), width=1)
        d.ellipse([mx - 7, my - 7, mx + 7, my + 7], outline=(0, 255, 255), width=2)
    title = (f"{'Inline' if axis == 'i' else 'Crossline'} {int(number)} | {v.meta['label']} | "
             f"{'crossline' if axis == 'i' else 'inline'} {int(l0)} (left) to {int(l1)} (right) | TWT {t0:.0f} (top) to {t1:.0f} ms (bottom)")
    d.rectangle([0, 0, width, 18], fill=(0, 0, 0))
    d.text((4, 2), title, fill=(255, 255, 255), font=_font(12))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


MAP_LABEL = {"time": "two-way time (ms)", "amp": "amplitude on the horizon", "rms_below": "RMS amplitude below",
             "sand_below": "sand probability below", "sand_above": "sand probability above", "ai_below": "P-impedance below",
             "vpvs_below": "Vp/Vs below", "si_below": "S-impedance below", "far_over_near": "far / near amplitude",
             "fault": "fault likelihood", "facies": "seismic facies", "closure": "closure height (ms)",
             "diff": "difference from the RGT horizon (ms)", "conf": "confidence of the pick (0-1)"}


def render_map(horizon_name, attr="time", mark=None, size=760):
    """PNG map of a horizon attribute with lead outlines and wells."""
    from PIL import Image, ImageDraw
    mp, g = maps(), grid()
    key = f"{horizon_name}__{attr}"
    if mp is None or key not in mp:
        return None
    a = mp[key]
    lo, hi = np.percentile(a, [2, 98]) if attr not in ("facies",) else (0, max(1, a.max()))
    n = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    img = Image.fromarray(_lut("terrain" if attr == "time" else "viridis")[(n * 255).astype(np.uint8)])
    ni, nx = a.shape
    sc = size / max(ni, nx)
    W, H = int(nx * sc), int(ni * sc)
    img = img.resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    font = _font(14)

    def px(il, xl):
        return ((xl - g["xlines"][0]) / (g["xlines"][-1] - g["xlines"][0] + 1e-9) * (W - 1),
                (il - g["ilines"][0]) / (g["ilines"][-1] - g["ilines"][0] + 1e-9) * (H - 1))

    for o in lead_outlines(horizon_name):
        for il, xl in zip(o["il"], o["xl"]):
            x, y = px(il, xl)
            d.point((x, y), fill=(255, 255, 255))
        x, y = px(float(np.mean(o["il"])), float(np.mean(o["xl"])))
        d.text((x - 8, y - 8), o["id"], fill=(255, 255, 255), font=font)
    for w in wells.wells_summary().get("wells", []):
        if w.get("il") is None:
            continue
        x, y = px(w["il"], w["xl"])
        if 0 <= x < W and 0 <= y < H:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 0, 0), outline=(255, 255, 0))
            d.text((x + 6, y - 7), w["name"], fill=(255, 255, 0), font=font)
    if mark:
        x, y = px(mark[0], mark[1])
        d.line([(x, 0), (x, H)], fill=(0, 255, 255))
        d.line([(0, y), (W, y)], fill=(0, 255, 255))
    d.rectangle([0, 0, W, 18], fill=(0, 0, 0))
    d.text((4, 2), f"Map of {horizon_name}: {MAP_LABEL.get(attr, attr)} | low {lo:.3g} to high {hi:.3g} | "
                   f"inline {int(g['ilines'][0])} (top) to {int(g['ilines'][-1])} (bottom), "
                   f"crossline {int(g['xlines'][0])} (left) to {int(g['xlines'][-1])} (right)", fill=(255, 255, 255), font=_font(12))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ====================================================================== pictures with numbered marks
# Everything below draws from the cubes on the interpretation grid (small and fast), so it can run inside
# the pipeline as well as in the chat. Pictures for the assistant are square, with the labels OUTSIDE the
# data, and anything it has to choose between is drawn as a numbered line or mark.
MARK_COLORS = [(255, 221, 0), (0, 229, 255), (255, 64, 255), (124, 255, 0), (255, 140, 0), (255, 255, 255),
               (255, 90, 90), (90, 160, 255)]


def gcube(role):
    """Cube on the interpretation grid by role: full (reference seismic), coh, fault, sand, ai, vpvs, near, far..."""
    g = grid()
    ref_role = next((k for k, v in ((g or {}).get("roles") or {}).items() if v == (g or {}).get("ref")), "full")
    if role in ("full", ref_role):
        return cube("seis")
    if role == "coh":
        return cube("semb")
    if role == "fault":
        return cube("faults")
    return _cached(f"vol_{role}", I / f"vol_{role}.npy", lambda p: np.load(p, mmap_mode="r"))


def gscale(role):
    """(low, high, colour table) used to draw a cube of that role."""
    key = f"scale_{role}"
    c = gcube(role)
    if c is None:
        return 0.0, 1.0, "gray"
    hit = _c.get(key)
    if hit and hit[0] is c:
        return hit[1]
    if role in ("coh", "fault"):
        res = ((70.0, 255.0, "gray") if role == "coh" else (0.0, 255.0, "viridis"))
    else:
        s = np.asarray(c[::4, ::4, ::4], dtype=np.float32)
        s = s[s != 0]
        if s.size == 0:
            res = (0.0, 1.0, "gray")
        elif s.min() < 0 and abs(s.mean()) < 0.3 * (s.std() + 1e-12):
            a = float(np.percentile(np.abs(s), 99))
            res = (-a, a, "gray")
        else:
            res = (float(np.percentile(s, 1)), float(np.percentile(s, 99)), "viridis")
    _c[key] = (c, res)
    return res


def to_rgb(arr, lo, hi, lut="gray"):
    n = np.clip((np.nan_to_num(np.asarray(arr, dtype=np.float32)) - lo) / (hi - lo + 1e-12), 0, 1)
    return _lut(lut)[(n * 255).astype(np.uint8)]


def grid_section(axis, gidx, role="full"):
    """(picture rows = time, columns = lateral position) of one line of the interpretation grid, plus its extent."""
    g, c = grid(), gcube(role)
    s = np.asarray(c[int(gidx)] if axis == "i" else c[:, int(gidx)], dtype=np.float32).T
    lat = g["xlines"] if axis == "i" else g["ilines"]
    return s, {"h": [float(lat[0]), float(lat[-1])], "v": [float(g["times"][0]), float(g["times"][-1])]}


def grid_slice(gt, role="full"):
    """Time slice of the interpretation grid (rows = inline, columns = crossline), plus its extent."""
    g, c = grid(), gcube(role)
    s = np.asarray(c[:, :, int(gt)], dtype=np.float32)
    return s, {"h": [float(g["xlines"][0]), float(g["xlines"][-1])], "v": [float(g["ilines"][0]), float(g["ilines"][-1])]}


def _ticks(a, b, n=6):
    span = abs(b - a) or 1.0
    raw = span / n
    p = 10 ** np.floor(np.log10(raw))
    step = next((q * p for q in (1, 2, 5, 10) if q * p >= raw), raw)
    t = np.ceil(min(a, b) / step) * step
    out = []
    while t <= max(a, b) + 1e-9:
        out.append(float(t))
        t += step
    return out


def cell_name(col, row):
    """Grid square name used in pictures: column letter + row number, e.g. 'C4'."""
    return f"{chr(ord('A') + int(col))}{int(row) + 1}"


def render_panel(rgb, ext, title="", hlabel="", vlabel="", lines=(), marks=(), cells=None, size=896, shade=None):
    """Draw one picture for the assistant. Returns a PIL image (use png() to get the bytes).

    rgb    : (rows, cols, 3) uint8; row 0 is at ext['v'][0] (top), column 0 at ext['h'][0] (left)
    lines  : [{"pts": [(h, v), ...], "label": "3", "color": (r,g,b), "width": 2, "at": 0.5}]  numbered lines
    marks  : [{"h":, "v":, "label": "2", "color": (r,g,b)}]                                     numbered dots
    cells  : (columns, rows) -> a lettered / numbered grid is drawn instead of axis ticks
    shade  : boolean (rows, cols) array; True cells are dimmed (e.g. no data)
    """
    from PIL import Image, ImageDraw
    L, T, R, B = 64, 48, 14, 30
    W = H = int(size)
    pw, ph = W - L - R, H - T - B
    im = Image.fromarray(np.ascontiguousarray(rgb))
    if shade is not None and shade.any():
        a = np.array(im, dtype=np.float32)
        a[shade] = a[shade] * 0.25 + 12
        im = Image.fromarray(a.astype(np.uint8))
    canvas = Image.new("RGB", (W, H), (16, 20, 24))
    canvas.paste(im.resize((pw, ph), Image.BILINEAR), (L, T))
    d = ImageDraw.Draw(canvas)
    h0, h1 = ext["h"]
    v0, v1 = ext["v"]

    def X(h):
        return L + (h - h0) / (h1 - h0 + 1e-9) * (pw - 1)

    def Y(v):
        return T + (v - v0) / (v1 - v0 + 1e-9) * (ph - 1)

    f_s, f_m, f_b = _font(13), _font(15), _font(18)
    d.text((6, 4), title[:150], fill=(235, 240, 245), font=f_m)
    if cells:
        nc, nr = cells
        for c in range(nc + 1):
            x = L + c * (pw - 1) / nc
            d.line([(x, T), (x, T + ph)], fill=(255, 255, 255), width=1)
            if c < nc:
                d.text((x + (pw / nc) / 2 - 5, T - 21), chr(ord("A") + c), fill=(255, 221, 0), font=f_b)
        for r in range(nr + 1):
            y = T + r * (ph - 1) / nr
            d.line([(L, y), (L + pw, y)], fill=(255, 255, 255), width=1)
            if r < nr:
                d.text((L - 24, y + (ph / nr) / 2 - 10), str(r + 1), fill=(255, 221, 0), font=f_b)
    else:
        for t in _ticks(h0, h1):
            x = X(t)
            d.line([(x, T - 4), (x, T)], fill=(170, 180, 190))
            d.text((x - 14, T - 19), f"{t:.0f}", fill=(170, 180, 190), font=f_s)
        for t in _ticks(v0, v1):
            y = Y(t)
            d.line([(L - 4, y), (L, y)], fill=(170, 180, 190))
            d.text((6, y - 7), f"{t:.0f}", fill=(170, 180, 190), font=f_s)
    d.text((L, H - 22), (hlabel + ("     |     " + vlabel if vlabel else ""))[:160], fill=(170, 180, 190), font=f_s)

    def tag(x, y, text, col):
        text = str(text)
        w = 9 * len(text) + 10
        x = float(np.clip(x, L + 2, L + pw - w - 2))
        y = float(np.clip(y, T + 2, T + ph - 24))
        d.rectangle([x, y, x + w, y + 22], fill=(0, 0, 0), outline=col, width=2)
        d.text((x + 5, y + 2), text, fill=(255, 255, 255), font=f_m)

    for n, ln in enumerate(lines):
        pts = [(X(h), Y(v)) for h, v in ln["pts"] if h is not None and v is not None and np.isfinite(h) and np.isfinite(v)]
        if len(pts) < 2:
            continue
        col = tuple(ln.get("color") or MARK_COLORS[n % len(MARK_COLORS)])
        d.line(pts, fill=col, width=int(ln.get("width", 2)))
        if ln.get("label") is not None:
            p = pts[int(np.clip(round(float(ln.get("at", 0.5)) * (len(pts) - 1)), 0, len(pts) - 1))]
            tag(p[0] - 8, p[1] - 26, ln["label"], col)
    for n, mk in enumerate(marks):
        col = tuple(mk.get("color") or MARK_COLORS[n % len(MARK_COLORS)])
        x, y = X(mk["h"]), Y(mk["v"])
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=3)
        if mk.get("label") is not None:
            sx = float(mk.get("shift", 0))              # labels of marks that sit close together are staggered sideways
            if sx:
                d.line([(x + 6, y), (x + 9 + sx, y)], fill=col, width=1)
            tag(x + 9 + sx, y - 11, mk["label"], col)
    return canvas


def png(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def side_by_side(images, gap=8):
    from PIL import Image
    h = max(im.height for im in images)
    out = Image.new("RGB", (sum(im.width for im in images) + gap * (len(images) - 1), h), (16, 20, 24))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def _dead_rows(axis, gidx, shape):
    lv = live()
    if lv is None:
        return None
    row = ~(lv[int(gidx), :] if axis == "i" else lv[:, int(gidx)])
    return np.broadcast_to(row[None, :], shape)


def _hz_lines(axis, gidx, source="rgt", labels=True):
    h, g = horizons(source), grid()
    if h is None:
        return []
    lat = g["xlines"] if axis == "i" else g["ilines"]
    out = []
    for k, name in enumerate(h["names"]):
        z = h["z"][k][int(gidx), :] if axis == "i" else h["z"][k][:, int(gidx)]
        out.append({"pts": list(zip(lat.tolist(), (g["times"][0] + z * g["dt_ms"]).tolist())),
                    "label": name if labels else None, "color": _hex(HCOLORS[k % len(HCOLORS)]), "at": 0.04, "width": 2})
    return out


def _well_marks(twt=None):
    out = []
    for w in wells.wells_summary().get("wells", []):
        if w.get("il") is None or w.get("inside_survey") is False:
            continue
        q = [w["il"], w["xl"]]
        if twt is not None and w.get("path"):
            best = min((p for p in w["path"] if p[2] is not None and p[2] == p[2]), key=lambda p: abs(p[2] - twt), default=None)
            if best:
                q = best
        out.append({"h": q[1], "v": q[0], "label": w["name"], "color": (255, 225, 77)})
    return out


def render_slice(twt, role="full", mark=None, cells=None, with_coherence=True, size=640):
    """Time slice picture: the chosen volume on the left and coherence on the right, wells and faults marked."""
    g = grid()
    gt = g_index(twt=twt)
    t = float(g["times"][gt])
    lv = live()
    shade = None if lv is None else ~lv
    marks = _well_marks(t) if not cells else []
    if mark:
        marks = marks + [{"h": mark[1], "v": mark[0], "label": "you", "color": (0, 255, 255)}]
    lines = [{"pts": list(zip(tr["xl"], tr["il"])), "label": tr["fault"], "color": (255, 70, 70), "at": 0.5, "width": 2}
             for tr in fault_traces(twt=t)] if not cells else []
    out = []
    roles = [role] + (["coh"] if with_coherence and role != "coh" and gcube("coh") is not None else [])
    for r in roles:
        if gcube(r) is None:
            continue
        s, ext = grid_slice(gt, r)
        lo, hi, lut = gscale(r)
        name = {"full": "seismic amplitude", "coh": "coherence (dark = neighbouring traces differ: faults, channel edges)",
                "sand": "sand probability (bright = sandy)", "ai": "P-impedance", "vpvs": "Vp/Vs"}.get(r, r)
        out.append(render_panel(to_rgb(s, lo, hi, lut), ext, title=f"Time slice {t:.0f} ms - {name}",
                                hlabel="crossline number left to right", vlabel="inline number top to bottom",
                                lines=lines, marks=marks, cells=cells, size=size, shade=shade))
    return png(side_by_side(out) if len(out) > 1 else out[0])


def render_grid_section(axis, number, role="full", twt=None, mark=None, source="both", size=704, extra_lines=(), with_sticks=True):
    """Section picture from the interpretation grid with horizons, fault sticks and (optionally) the crosshair."""
    g = grid()
    gidx = g_index(il=number) if axis == "i" else g_index(xl=number)
    s, ext = grid_section(axis, gidx, role)
    lo, hi, lut = gscale(role)
    lines = []
    if source in ("rgt", "both"):
        lines += _hz_lines(axis, gidx, "rgt")
    if source in ("gemma", "both") and horizons("gemma") is not None:
        for ln in _hz_lines(axis, gidx, "gemma"):
            ln.update({"at": 0.93, "width": 1})
            lines.append(ln)
    for st in (sticks_on(axis, number) if with_sticks else []):
        lines.append({"pts": st["pts"], "label": st.get("fault"), "color": (255, 70, 70), "at": 0.1, "width": 3})
    lines += list(extra_lines)
    marks = [{"h": mark[0], "v": mark[1], "label": "you", "color": (0, 255, 255)}] if mark else []
    num = float((g["ilines"] if axis == "i" else g["xlines"])[gidx])
    return render_panel(to_rgb(s, lo, hi, lut), ext,
                        title=f"{'Inline' if axis == 'i' else 'Crossline'} {num:.0f} - {'seismic' if role == 'full' else role}",
                        hlabel=f"{'crossline' if axis == 'i' else 'inline'} number left to right", vlabel="two-way time in ms, top to bottom",
                        lines=lines, marks=marks, size=size, shade=_dead_rows(axis, gidx, s.shape))


def render_view(view):
    """One picture of what the user is looking at: the section on the left, the time slice on the right."""
    ax = view.get("axis", "i")
    num = float(view["il"] if ax == "i" else view["xl"])
    twt = float(view["twt"])
    sec = render_grid_section(ax, num, "full", twt, (float(view["xl"] if ax == "i" else view["il"]), twt), size=704)
    g = grid()
    gt = g_index(twt=twt)
    s, ext = grid_slice(gt, "full")
    lo, hi, lut = gscale("full")
    lv = live()
    sl = render_panel(to_rgb(s, lo, hi, lut), ext, title=f"Time slice {float(g['times'][gt]):.0f} ms - seismic amplitude",
                      hlabel="crossline number left to right", vlabel="inline number top to bottom",
                      marks=_well_marks(twt) + [{"h": float(view["xl"]), "v": float(view["il"]), "label": "you", "color": (0, 255, 255)}],
                      lines=[{"pts": list(zip(tr["xl"], tr["il"])), "label": tr["fault"], "color": (255, 70, 70)} for tr in fault_traces(twt=twt)],
                      size=704, shade=None if lv is None else ~lv)
    return png(side_by_side([sec, sl]))


def render_well(name, size=896):
    """Log panel of one well for the assistant: shale volume, resistivity, porosity, impedance, rock column."""
    from PIL import Image, ImageDraw
    key = wells.norm(name)
    a = wells.well_arrays(key)
    if "md" not in a:
        return None
    in_time = "twt" in a and np.isfinite(a["twt"]).any()
    depth = a["twt"] if in_time else a["md"]
    tracks = [("vsh", "Shale volume 0-1", False, (120, 200, 120)), ("rt", "Resistivity (log scale)", True, (255, 120, 120)),
              ("phi", "Porosity", False, (120, 180, 255)), ("ai", "Impedance", False, (230, 200, 120))]
    tracks = [t for t in tracks if f"c_{t[0]}" in a]
    if not tracks:
        return None
    ok = np.isfinite(depth) & np.any([np.isfinite(a[f"c_{t[0]}"]) for t in tracks], axis=0)
    if ok.sum() < 10:
        return None
    v0, v1 = float(np.nanmin(depth[ok])), float(np.nanmax(depth[ok]))
    W = H = int(size)
    L, T, B = 64, 70, 16
    ph = H - T - B
    n = len(tracks) + 1
    tw = (W - L - 10) / n
    im = Image.new("RGB", (W, H), (16, 20, 24))
    d = ImageDraw.Draw(im)
    f_s, f_m = _font(13), _font(15)

    def Y(v):
        return T + (v - v0) / (v1 - v0 + 1e-9) * (ph - 1)

    d.text((6, 4), f"Well {name} - logs against {'two-way time (ms)' if in_time else 'measured depth (m)'}", fill=(235, 240, 245), font=f_m)
    for t in _ticks(v0, v1, 10):
        y = Y(t)
        d.line([(L - 4, y), (W - 10, y)], fill=(40, 50, 58))
        d.text((6, y - 7), f"{t:.0f}", fill=(170, 180, 190), font=f_s)
    step = max(1, int(ok.sum() // 1500))
    for k, (c, heading, logscale, col) in enumerate(tracks):
        x0 = L + k * tw
        d.rectangle([x0, T, x0 + tw - 6, T + ph], outline=(60, 72, 82))
        d.text((x0 + 3, 28), heading, fill=col, font=f_s)
        v = a[f"c_{c}"]
        m = ok & np.isfinite(v) & ((v > 0) if logscale else True)
        if m.sum() < 5:
            continue
        vv = np.log10(v[m]) if logscale else v[m]
        lo, hi = (0.0, 1.0) if c == "vsh" else (float(np.percentile(vv, 1)), float(np.percentile(vv, 99)))
        d.text((x0 + 3, 46), f"{(10 ** lo if logscale else lo):.3g} to {(10 ** hi if logscale else hi):.3g}", fill=(150, 160, 170), font=f_s)
        xs = x0 + 2 + np.clip((vv - lo) / (hi - lo + 1e-9), 0, 1) * (tw - 10)
        pts = list(zip(xs[::step].tolist(), [Y(q) for q in depth[m][::step]]))
        if len(pts) > 1:
            d.line(pts, fill=col, width=1)
    x0 = L + len(tracks) * tw
    d.rectangle([x0, T, x0 + tw - 6, T + ph], outline=(60, 72, 82), fill=(70, 84, 78))
    d.text((x0 + 3, 28), "Rock: yellow sand,", fill=(255, 221, 0), font=f_s)
    d.text((x0 + 3, 46), "red = hydrocarbon flag", fill=(255, 90, 90), font=f_s)
    z = wells.log_zones(key) or {"sands": []}
    for s in z["sands"]:
        a0, a1 = (s["top_twt"], s["base_twt"]) if in_time else (s["top_md"], s["base_md"])
        if a0 is None or a1 is None:
            continue
        d.rectangle([x0 + 1, Y(a0), x0 + tw - 7, max(Y(a1), Y(a0) + 1)], fill=(230, 60, 60) if s["pay"] else (240, 205, 40))
        if s["thick_m"] >= 5:
            d.text((x0 + tw - 40, Y(a0) - 2), s["id"], fill=(0, 0, 0), font=f_s)
    k = knowledge.load() or {}
    tie = next((t for t in k.get("well_ties", []) if wells.norm(t["well"]) == key), None)
    if tie and in_time:
        hn = (horizons() or {}).get("names", [])
        for n_, r in tie["horizons"].items():
            y = Y(r["twt"])
            if T <= y <= T + ph:
                col = _hex(HCOLORS[hn.index(n_) % len(HCOLORS)]) if n_ in hn else (255, 255, 255)
                d.line([(L, y), (W - 10, y)], fill=col, width=2)
                d.text((L + 2, y - 15), n_, fill=col, font=f_s)
    return png(im)


# ====================================================================== faults, sticks, comparison, dossier
def sticks_file():
    return _cached("sticks", I / "sticks.json", read_json) or {}


def faults_named():
    return _cached("fault_surfaces", I / "fault_surfaces.json", read_json) or []


def _stick_lat(st, t):
    """Lateral index of a stick at time index t (NaN outside its vertical extent)."""
    p = np.asarray(st["pts"], dtype=float)
    return np.interp(t, p[:, 1], p[:, 0], left=np.nan, right=np.nan)


def sticks_on(axis, number, only_accepted=True):
    """Fault sticks to draw on one section, in line numbers and ms. Between the lines that were actually
    interpreted, a fault's stick is interpolated from its two neighbours."""
    sf, g = sticks_file(), grid()
    if not sf or g is None:
        return []
    lat_axis = g["xlines"] if axis == "i" else g["ilines"]
    gidx = g_index(il=number) if axis == "i" else g_index(xl=number)
    lines = sorted({s["g"] for s in sf.get("sticks", []) if s["axis"] == axis})
    if not lines:
        return []
    step = float(sf.get("step", 8))
    lo = max([q for q in lines if q <= gidx], default=None)
    hi = min([q for q in lines if q >= gidx], default=None)
    use = [s for s in sf["sticks"] if s["axis"] == axis and (s.get("accepted") or not only_accepted)]
    out = []

    def emit(pts, s):
        out.append({"fault": s.get("fault"), "verdict": s.get("verdict"), "why": s.get("why", ""), "throw_ms": s.get("throw_ms"),
                    "pts": [(float(np.interp(a, np.arange(len(lat_axis)), lat_axis)), float(g["times"][0] + b * g["dt_ms"])) for a, b in pts]})

    done = set()
    if lo is not None and hi is not None and lo != hi:
        w = (gidx - lo) / float(hi - lo)
        for a in [s for s in use if s["g"] == lo and s.get("fault")]:
            b = next((s for s in use if s["g"] == hi and s.get("fault") == a["fault"]), None)
            if b is None:
                continue
            pa, pb = np.asarray(a["pts"], float), np.asarray(b["pts"], float)
            t0, t1 = max(pa[:, 1].min(), pb[:, 1].min()), min(pa[:, 1].max(), pb[:, 1].max())
            if t1 - t0 < 3:
                continue
            ts = np.linspace(t0, t1, 6)
            emit([((1 - w) * _stick_lat(a, t) + w * _stick_lat(b, t), t) for t in ts], a)
            done.add(a["fault"])
    for s in use:
        if s.get("fault") in done:
            continue
        if abs(s["g"] - gidx) <= step / 2.0:
            emit(s["pts"], s)
    return out


def fault_traces(twt=None, horizon=None):
    """Where the named faults cut a time slice (or a horizon), as map lines in inline / crossline numbers."""
    sf, g = sticks_file(), grid()
    if not sf or g is None:
        return []
    zmap = None
    if horizon:
        for src in ("rgt", "gemma"):
            h = horizons(src)
            if h is not None and horizon in h["names"]:
                zmap = h["z"][h["names"].index(horizon)]
    gt = None if twt is None else (float(twt) - float(g["times"][0])) / float(g["dt_ms"])
    step = float(sf.get("step", 8))
    out = []
    by = {}
    for s in sf.get("sticks", []):
        if s.get("accepted") and s.get("fault"):
            by.setdefault((s["fault"], s["axis"]), []).append(s)
    ni, nx = len(g["ilines"]), len(g["xlines"])
    for (fid, axis), group in by.items():
        seg, last = [], None
        for s in sorted(group, key=lambda q: q["g"]):
            if zmap is not None:
                p = np.asarray(s["pts"], float)
                ts = np.linspace(p[:, 1].min(), p[:, 1].max(), 24)
                la = np.interp(ts, p[:, 1], p[:, 0])
                zz = (zmap[int(s["g"]), np.clip(np.rint(la).astype(int), 0, nx - 1)] if axis == "i"
                      else zmap[np.clip(np.rint(la).astype(int), 0, ni - 1), int(s["g"])])
                k = np.flatnonzero(np.diff(np.sign(ts - zz)) != 0)
                lat = float(la[k[0]]) if k.size else np.nan
            else:
                lat = float(_stick_lat(s, gt))
            if not np.isfinite(lat):
                continue
            pt = (float(s["g"]), lat) if axis == "i" else (lat, float(s["g"]))
            if last is not None and s["g"] - last > 2 * step + 0.5 and len(seg) > 1:
                out.append({"fault": fid, "pts": seg})
                seg = []
            seg.append(pt)
            last = s["g"]
        if len(seg) > 1:
            out.append({"fault": fid, "pts": seg})
    res = []
    for tr in out:
        gi = np.array([p[0] for p in tr["pts"]])
        gx = np.array([p[1] for p in tr["pts"]])
        res.append({"fault": tr["fault"], "il": np.interp(gi, np.arange(ni), g["ilines"]).round(1).tolist(),
                    "xl": np.interp(gx, np.arange(nx), g["xlines"]).round(1).tolist()})
    return res


def nearest_fault(il, xl, twt):
    """The named fault closest to a point at that time, with the distance to it."""
    g, best = grid(), None
    r = _ref()
    b = (r.bin_size() if r is not None else None) or (1.0, 1.0)
    di = float(np.median(np.diff(g["ilines"]))) or 1.0
    dx = float(np.median(np.diff(g["xlines"]))) or 1.0
    for tr in fault_traces(twt=twt):
        d = np.hypot((np.array(tr["il"]) - il) / di * b[0] * (di / (float(np.median(np.diff(r.ilines))) or 1.0) if r is not None else 1),
                     (np.array(tr["xl"]) - xl) / dx * b[1] * (dx / (float(np.median(np.diff(r.xlines))) or 1.0) if r is not None else 1))
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (float(d[k]), tr["fault"])
    if best is None:
        return None
    return {"fault": best[1], "distance_km": round(best[0] / 1000.0, 2)}


def fault(fid):
    fs = faults_named()
    for f in fs:
        if f["id"].lower() == str(fid).lower():
            d = dict(f)
            sf = sticks_file()
            d["stick_method"] = sf.get("method", "legacy cached picks")
            d["stick_checks"] = [{"line": f"{'inline' if s['axis'] == 'i' else 'crossline'} {s['line']}", "verdict": s.get("verdict"),
                                     "why": s.get("why", ""), "throw_ms": s.get("throw_ms"), "throw_fit": s.get("throw_fit"),
                                     "reviewed_by": s.get("reviewed_by", "legacy")}
                                    for s in sf.get("sticks", []) if s.get("fault") == f["id"] and s.get("verdict") not in (None, "unreviewed")][:8]
            return d
    return {"error": f"no fault called {fid}", "available": [f["id"] for f in fs][:30]}


def compare_horizons(name=None):
    """How the assistant's tracked horizons agree with the RGT horizons."""
    c = read_json(I / "hz_compare.json")
    if not c:
        return {"error": "the assistant's horizons have not been made yet (the AI pass has not run)"}
    if name:
        for p in c.get("pairs", []):
            if name.lower() in (p["gemma"].lower(), str(p.get("rgt", "")).lower()):
                return p
        return {"error": f"no horizon called {name}", "available": [p["gemma"] for p in c.get("pairs", [])]}
    return c


def sand_certainty(values):
    from . import calibrate
    return calibrate.p_sand(read_json(I / "calib.json"), values)


def facies_name(number):
    k = knowledge.load() or {}
    for f in k.get("facies", []):
        if f["facies"] == number:
            return f.get("gemma_name") or f.get("name") or f.get("description")
    return None


def dossier(il=None, xl=None, twt=None, lead_id=None):
    """Everything tied together for one place (or one lead): position, rock certainty, wells, faults, both
    interpreters' horizons, and what the assistant saw there in its own pass. Compact on purpose."""
    k = knowledge.load() or {}
    out = {}
    if lead_id:
        ld = lead(lead_id)
        if "error" in ld:
            return ld
        il, xl, twt = ld["where"]["il"], ld["where"]["xl"], ld.get("zone_twt_ms", ld["crest_twt_ms"])
        out["lead"] = {q: ld.get(q) for q in ("id", "kind", "horizon", "area_km2", "relief_ms", "crest_twt_ms", "reservoir_zone",
                                              "scores", "tested", "wells_inside", "nearest_well")}
        out["lead"]["for"] = [e["text"] for e in ld["for"]][:8]
        out["lead"]["against"] = [e["text"] for e in ld["against"]][:6]
        out["lead"]["unknown"] = [e["text"] for e in ld["unknown"]][:5]
        if ld.get("assistant_saw"):
            out["lead"]["assistant_saw"] = ld["assistant_saw"]
    p = probe(float(il), float(xl), float(twt))
    out["place"] = {q: p.get(q) for q in ("il", "xl", "twt_ms", "between_horizons", "facies", "facies_name", "fault_likelihood",
                                          "inside_leads", "nearest_well", "nearest_fault", "sand_certainty", "nearest_well_logs_here",
                                          "horizons_agree")}
    out["place"]["values"] = {a: b["value"] for a, b in p.get("values", {}).items()}
    near = []
    for e in k.get("evidence", []):
        w = e.get("where") or {}
        if e.get("source") == "assistant-visual" and w.get("il") is not None and \
                abs(w["il"] - il) <= 0.12 * (k["survey"]["inlines"][1] - k["survey"]["inlines"][0]) and \
                abs(w["xl"] - xl) <= 0.12 * (k["survey"]["crosslines"][1] - k["survey"]["crosslines"][0]) and \
                (w.get("twt") is None or abs(w["twt"] - twt) <= 150):
            near.append(f"[{e['id']}] {e['text']}")
    out["assistant_visual_readings_nearby"] = near[:6]
    return out
