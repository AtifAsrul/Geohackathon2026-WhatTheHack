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


def horizons():
    def ld(p):
        d = np.load(p)
        return {"names": [str(n) for n in d["names"]], "z": d["z"], "polarity": [str(x) for x in d["polarity"]]}
    return _cached("hz", I / "horizons.npz", ld)


def cube(name):
    """Interpretation-grid cubes: rgt, faults, facies, seis."""
    return _cached(name, I / f"{name}.npy", lambda p: np.load(p, mmap_mode="r"))


def maps():
    return _cached("maps", I / "maps.npz", lambda p: {k: v for k, v in np.load(p).items()})


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
def horizon_lines(axis, number):
    """Horizon picks along one section. axis 'i' = inline section, 'x' = crossline section."""
    h, g = horizons(), grid()
    if h is None:
        return []
    out = []
    for k, name in enumerate(h["names"]):
        if axis == "i":
            z = h["z"][k][g_index(il=number), :]
            pos = g["xlines"]
        else:
            z = h["z"][k][:, g_index(xl=number)]
            pos = g["ilines"]
        twt = g["times"][0] + z * g["dt_ms"]
        out.append({"name": name, "color": HCOLORS[k % len(HCOLORS)], "pos": [int(p) for p in pos],
                    "twt": [round(float(t), 1) for t in twt]})
    return out


def overlay(layer, axis, number):
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
        cb = cube({"faults": "faults", "facies": "facies", "rgt": "rgt"}[layer])
        if cb is None:
            return None
        s = cb[g_index(il=number)] if axis == "i" else (cb[:, g_index(xl=number)] if axis == "x" else cb[:, :, g_index(twt=number)])
        s = np.asarray(s)
        data = (np.clip(s, 0, 1) * 255).astype(np.uint8) if layer == "rgt" else s.astype(np.uint8)
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
        if fc is not None:
            out["facies"] = int(fc[gi, gx, gt]) + 1
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
             "fault": "fault likelihood", "facies": "seismic facies", "closure": "closure height (ms)"}


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
