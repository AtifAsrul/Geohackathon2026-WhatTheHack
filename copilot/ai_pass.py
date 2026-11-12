"""The assistant's own interpretation pass.

After the measured interpretation is finished, Gemma is shown pictures and asked small, checkable questions:

  facies    does the rule-based facies name fit what the examples look like?
  wells     what do the logs of each well show?
  leads     looking at the two sections through each lead: is the trap visible?
  slices    what features are visible on time slices (channels, fault lines, fold bands)?
  horizons  which numbered reflectors are the key horizons, and how do they continue across faults?

Gemma never gives coordinates. It picks between numbered marks or lettered grid squares that the app drew,
and the app does the measuring and the drawing. Every picture also carries a decoy (a mark that is wrong on
purpose); how often Gemma rejects the decoys is reported, and if it accepts them its votes are not used.
If Gemma cannot be reached, every step still produces its result from the app's own best choice, labelled
"not reviewed by the assistant".
"""
import time

import numpy as np

from . import config, llm, rgt, sticks as sticks_mod, tools, tracker, wells
from .config import read_json, write_json

I = config.INTERP
STEP = 8                                # every 8th line of the interpretation grid is picked
RULES = ("You are an exploration geoscientist looking at a picture made from a 3-D seismic survey. "
         "Reply with JSON only, no other text. Refer only to the numbers and letters drawn on the picture. "
         "In every \"why\" say in under 20 words what you SEE (shape, brightness, offsets), not what you assume.\n\n")
CONF = ("low", "medium", "high")


class Ctx:
    pass


def _ctx(log):
    c = Ctx()
    c.log = log
    c.g = tools.grid()
    c.dt, c.t0 = float(c.g["dt_ms"]), float(c.g["times"][0])
    c.live = np.load(I / "live.npy") if (I / "live.npy").exists() else None
    raw = np.load(I / "seis.npy")
    c.seis = rgt.fill_dead(raw, c.live) if c.live is not None else raw
    c.ni, c.nx, c.nt = c.seis.shape
    if c.live is None:
        c.live = np.ones((c.ni, c.nx), dtype=bool)
    c.fl = np.load(I / "faults.npy").astype(np.float32) / 255.0
    c.k = tools.knowledge.load() or {}
    c.il_idx = np.arange((c.ni // 2) % STEP, c.ni, STEP)
    c.xl_idx = np.arange((c.nx // 2) % STEP, c.nx, STEP)
    c.reviewed = llm.check()[0]
    return c


def _num(c, axis, idx):
    """Line number of a fractional lateral index on a section of the given family."""
    lat = c.g["xlines"] if axis == "i" else c.g["ilines"]
    return np.interp(idx, np.arange(len(lat)), lat)


def _ms(c, t):
    return c.t0 + np.asarray(t, dtype=float) * c.dt


def _line_no(c, axis, gidx):
    return int((c.g["ilines"] if axis == "i" else c.g["xlines"])[int(gidx)])


def _sec(c, axis, gidx, cube=None):
    cube = c.seis if cube is None else cube
    return cube[int(gidx)] if axis == "i" else cube[:, int(gidx)]


def _runs(z):
    """Split a pick array with gaps into its unbroken stretches: [(start, stop), ...]."""
    ok = np.isfinite(z)
    if not ok.any():
        return []
    e = np.flatnonzero(np.diff(ok.astype(np.int8)) != 0) + 1
    b = np.r_[0, e, len(ok)]
    return [(int(p), int(q)) for p, q in zip(b[:-1], b[1:]) if ok[p] and q - p >= 2]


def _pick_lines(c, axis, z, label, color, at=0.5, width=2):
    """One tracked pick array -> picture lines (one per unbroken stretch; only the longest carries the label)."""
    runs = _runs(z)
    if not runs:
        return []
    longest = max(runs, key=lambda r: r[1] - r[0])
    out = []
    for p, q in runs:
        idx = np.arange(p, q)
        out.append({"pts": list(zip(_num(c, axis, idx).tolist(), _ms(c, z[p:q]).tolist())), "color": color, "width": width,
                    "label": label if (p, q) == longest else None, "at": at})
    return out


def _picture(c, axis, gidx, title, lines=(), marks=(), size=896):
    s, ext = tools.grid_section(axis, gidx, "full")
    lo, hi, lut = tools.gscale("full")
    lv = c.live[int(gidx), :] if axis == "i" else c.live[:, int(gidx)]
    shade = np.broadcast_to(~lv[None, :], s.shape) if not lv.all() else None
    im = tools.render_panel(tools.to_rgb(s, lo, hi, "gray"), ext, title=title,
                            hlabel=f"{'crossline' if axis == 'i' else 'inline'} number, left to right",
                            vlabel="two-way time in ms, downwards", lines=lines, marks=marks, size=size, shade=shade)
    return tools.png(im)


def _conf(v):
    return v if v in CONF else "low"


def _text(v, n=160):
    return str(v or "").strip().replace("\n", " ")[:n]


def _trusted(shown, rejected):
    """Do the assistant's votes count? Not if it accepted most of the deliberately wrong marks."""
    return True if shown < 2 else bool(rejected / shown >= 0.6)


# ====================================================================== facies
def review_facies(c):
    fac = [f for f in c.k.get("facies", []) if not f.get("no_data")]
    cube = np.load(I / "facies.npy", mmap_mode="r") if (I / "facies.npy").exists() else None
    if cube is None or not fac:
        return {"reviewed": 0}
    from PIL import Image
    w = int(min(96, c.nx // 2, c.nt // 2))
    jobs, rows = [], []
    lo, hi, _ = tools.gscale("full")
    for f in fac:
        cl = f["facies"] - 1
        best = []
        for gi in np.unique(np.linspace(0, c.ni - 1, 9).astype(int)):
            m = (np.asarray(cube[gi]) == cl).astype(np.float32)
            cs = np.cumsum(np.cumsum(np.pad(m, ((1, 0), (1, 0))), axis=0), axis=1)
            share = (cs[w:, w:] - cs[:-w, w:] - cs[w:, :-w] + cs[:-w, :-w]) / float(w * w)
            x0, t0 = np.unravel_index(int(np.argmax(share)), share.shape)
            best.append((float(share[x0, t0]), int(gi), int(x0), int(t0)))
        best.sort(reverse=True)
        tiles = []
        for n, (sh, gi, x0, t0) in enumerate(best[:4]):
            crop = c.seis[gi, x0:x0 + w, t0:t0 + w].T
            ext = {"h": [float(c.g["xlines"][x0]), float(c.g["xlines"][min(x0 + w, c.nx) - 1])],
                   "v": [float(_ms(c, t0)), float(_ms(c, min(t0 + w, c.nt) - 1))]}
            tiles.append(tools.render_panel(tools.to_rgb(crop, lo, hi, "gray"), ext, size=448,
                                            title=f"Example {n + 1}: inline {_line_no(c, 'i', gi)} ({sh:.0%} this facies)"))
        while len(tiles) < 4:
            tiles.append(Image.new("RGB", (448, 448), (16, 20, 24)))
        top, bot = tools.side_by_side(tiles[:2], 0), tools.side_by_side(tiles[2:4], 0)
        im = Image.new("RGB", (top.width, top.height + bot.height), (16, 20, 24))
        im.paste(top, (0, 0))
        im.paste(bot, (0, top.height))
        facts = [f"amplitude {f.get('amplitude_vs_avg')} times the survey average", f"continuity {f.get('continuity')}",
                 f"frequency {f.get('frequency_vs_avg')} times average", f"{f['share']:.0%} of the volume",
                 f"typically near {f['typical_twt_ms']:.0f} ms"]
        if "sand_mean" in f:
            facts.append(f"mean sand probability {f['sand_mean']}")
        if "well_net_sand" in f:
            facts.append(f"wells: net sand {f['well_net_sand']:.0%} in {f['well_intervals']} log intervals")
        prompt = (RULES + "The picture shows four examples of ONE seismic facies: a group of places whose reflection pattern "
                  "is alike, found automatically. Each example is a small vertical section.\n"
                  f"Measured for this facies: {'; '.join(facts)}.\n"
                  f"The app's rule-based name is \"{f['name']}\" ({f['lithology']}).\n"
                  "Does that name fit what you see? Give the name you would use, in standard seismic-facies terms "
                  "(amplitude, continuity, geometry), and the most likely depositional setting or cause.\n"
                  'JSON: {"agree": true or false, "name": "...", "setting": "...", "why": "...", "confidence": "low|medium|high"}')
        jobs.append({"task": "facies", "prompt": prompt, "image": tools.png(im),
                     "mock": {"agree": True, "name": f["name"], "setting": f["reading"].split(":")[0][:90],
                              "why": "stand-in model: the picture was not examined", "confidence": "low"}})
        rows.append(f)
    out = {}
    for f, a in zip(rows, llm.ask_many(jobs)):
        if isinstance(a, dict) and a.get("name"):
            out[str(f["facies"])] = {"agree": bool(a.get("agree", True)), "name": _text(a["name"], 80), "setting": _text(a.get("setting"), 140),
                                     "why": _text(a.get("why")), "confidence": _conf(a.get("confidence")), "reviewed": True}
    write_json(I / "ai_facies.json", out)
    return {"reviewed": len(out), "of": len(rows)}


# ====================================================================== wells
def read_wells(c):
    zones = read_json(I / "well_zones.json", {}) or {}
    jobs, names = [], []
    for w in wells.wells_summary().get("wells", []):
        z = zones.get(w["name"])
        if not z:
            continue
        img = tools.render_well(w["name"])
        if img is None:
            continue
        s = z["summary"]
        big = sorted(z["sands"], key=lambda q: -q["thick_m"])[:6]
        lines = [f"{q['id']}: {q['top_md']}-{q['base_md']} m" + (f" ({q['top_twt']:.0f} ms)" if q.get("top_twt") else "") +
                 f", {q['thick_m']} m thick" + (f", porosity {q['phi']:.0%}" if q.get("phi") is not None else "") +
                 (f", resistivity {q['rt']:.1f} ohm-m" if q.get("rt") is not None else "") + (", hydrocarbon flag" if q["pay"] else "")
                 for q in sorted(big, key=lambda q: q["top_md"])]
        ids = [q["id"] for q in z["sands"]]
        pay = [q for q in big if q["pay"]]
        prompt = (RULES + f"The picture is the log panel of well {w['name']}: shale volume (low = sand), resistivity, porosity, "
                  "impedance, and a rock column (yellow = sand, red = sand with a hydrocarbon flag, grey = shale). "
                  "Coloured horizontal lines are seismic horizons.\n"
                  f"Measured: {s['net_sand_m']} m of sand in {s['gross_m']} m logged (net-to-gross {s['net_to_gross']}), "
                  f"{s['net_pay_m']} m with a hydrocarbon flag.\nThickest sands:\n" + "\n".join(lines) + "\n"
                  "Read the logs like a petrophysicist: where is the best reservoir, and does anything indicate hydrocarbons "
                  "(high resistivity in a clean, porous sand) or water (low resistivity in a clean sand)?\n"
                  'JSON: {"reading": "<two short sentences>", "best_reservoir": "<a sand id from the list, or null>", '
                  '"fluid": "hydrocarbon|water|unclear", "why": "...", "confidence": "low|medium|high"}')
        mock = {"reading": f"{s['net_sand_m']} m of sand in {s['gross_m']} m logged; " +
                           (f"{s['net_pay_m']} m carries the hydrocarbon flag." if s["net_pay_m"] > 0 else "no interval carries the hydrocarbon flag."),
                "best_reservoir": (pay[0]["id"] if pay else (big[0]["id"] if big else None)),
                "fluid": "hydrocarbon" if s["net_pay_m"] >= 2 else ("water" if s.get("has_fluid_flag") else "unclear"),
                "why": "stand-in model: taken from the measured log zones", "confidence": "low"}
        jobs.append({"task": "well", "prompt": prompt, "image": img, "mock": mock})
        names.append((w["name"], ids))
    out = {}
    for (name, ids), a in zip(names, llm.ask_many(jobs)):
        if isinstance(a, dict) and a.get("reading"):
            br = a.get("best_reservoir")
            out[name] = {"reading": _text(a["reading"], 320), "best_reservoir": br if br in ids else None,
                         "fluid": a.get("fluid") if a.get("fluid") in ("hydrocarbon", "water", "unclear") else "unclear",
                         "why": _text(a.get("why")), "confidence": _conf(a.get("confidence")), "reviewed": True}
    write_json(I / "ai_wells.json", out)
    return {"reviewed": len(out), "of": len(names)}


# ====================================================================== leads
def look_leads(c, top=6):
    jobs, ids = [], []
    yn = ("yes", "no", "unclear")
    for l in c.k.get("leads", [])[:top]:
        il, xl, twt = l["where"]["il"], l["where"]["xl"], l.get("zone_twt_ms", l["crest_twt_ms"])
        # horizons only: the picture must not depend on what a later step of this pass produces
        a = tools.render_grid_section("i", il, "full", twt, (xl, l["crest_twt_ms"]), source="rgt", size=704, with_sticks=False)
        b = tools.render_grid_section("x", xl, "full", twt, (il, l["crest_twt_ms"]), source="rgt", size=704, with_sticks=False)
        prompt = (RULES + f"Two vertical sections cross at the crest of lead {l['id']} (the circle marked 'you'): the inline on the "
                  f"left, the crossline on the right. Coloured lines are mapped horizons, named at the left edge.\n"
                  f"The app mapped this lead as a {l['kind']} on horizon {l['horizon']}: crest {l['crest_twt_ms']:.0f} ms, "
                  f"relief {l['relief_ms']:.0f} ms, reservoir zone {l.get('reservoir_zone', 'n/a')}.\n"
                  "Look at the picture and answer: does horizon " + l["horizon"] + " arch over the mark in the inline section? in the "
                  "crossline section? Is there a noticeably brighter reflection under the crest than beside it? Is there a flat, "
                  "horizontal reflection cutting across dipping ones (a possible fluid contact)? What worries you most?\n"
                  'JSON: {"closes_inline": "yes|no|unclear", "closes_crossline": "yes|no|unclear", "bright_under_crest": "yes|no|unclear", '
                  '"flat_spot": "yes|no|unclear", "concern": "...", "why": "...", "confidence": "low|medium|high"}')
        four = l["kind"] == "four-way dip closure"
        mock = {"closes_inline": "yes" if four else "unclear", "closes_crossline": "yes" if four else "unclear",
                "bright_under_crest": "unclear", "flat_spot": "unclear", "concern": "none stated by the stand-in model",
                "why": "stand-in model: the picture was not examined", "confidence": "low"}
        jobs.append({"task": "lead", "prompt": prompt, "image": tools.png(tools.side_by_side([a, b])), "mock": mock})
        ids.append(l["id"])
    out = {}
    for lid, a in zip(ids, llm.ask_many(jobs)):
        if isinstance(a, dict) and any(a.get(q) in yn for q in ("closes_inline", "closes_crossline")):
            out[lid] = {q: (a.get(q) if a.get(q) in yn else "unclear") for q in ("closes_inline", "closes_crossline", "bright_under_crest", "flat_spot")}
            out[lid].update({"concern": _text(a.get("concern")), "why": _text(a.get("why")), "confidence": _conf(a.get("confidence")), "reviewed": True})
    write_json(I / "ai_leads.json", out)
    return {"reviewed": len(out), "of": len(ids)}


# ====================================================================== time slices
TYPES = {"channel": "channel-like band", "fault": "fault line", "fold_bands": "bands outlining a fold or dipping beds",
         "bright_patch": "isolated bright patch"}
TRENDS = {"up-down": (1, 0), "left-right": (0, 1), "diagonal-down-right": (1, 1), "diagonal-up-right": (-1, 1)}


def _compass(c, d_il, d_xl):
    """Compass trend of a direction given in (inline, crossline) index steps."""
    u_il, u_xl = c.g.get("u_il"), c.g.get("u_xl")
    if not u_il or not u_xl:
        return None
    cell = c.g.get("cell_m") or [1.0, 1.0]
    e = d_il * cell[0] * u_il[0] + d_xl * cell[1] * u_xl[0]
    n = d_il * cell[0] * u_il[1] + d_xl * cell[1] * u_xl[1]
    az = np.degrees(np.arctan2(e, n)) % 180.0
    return ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "ESE-WNW", "SE-NW", "SSE-NNW"][int(((az + 11.25) % 180) // 22.5)]


def read_slices(c, n_cells=8, max_slices=14):
    times = []
    for l in c.k.get("leads", [])[:6]:
        times.append(float(l.get("zone_twt_ms", l["crest_twt_ms"])))
    lo_t, hi_t = float(c.g["times"][0]), float(c.g["times"][-1])
    times += list(np.linspace(lo_t + 0.1 * (hi_t - lo_t), hi_t - 0.1 * (hi_t - lo_t), 8))
    picked = []
    for t in times:
        if all(abs(t - q) >= 40.0 for q in picked) and len(picked) < max_slices:
            picked.append(float(c.g["times"][tools.g_index(twt=t)]))
    picked.sort()
    sand = tools.gcube("sand")
    jobs, meta = [], []
    ei = np.linspace(0, c.ni, n_cells + 1).astype(int)
    ex = np.linspace(0, c.nx, n_cells + 1).astype(int)
    for t in picked:
        gt = tools.g_index(twt=t)
        img = tools.render_slice(t, "full", cells=(n_cells, n_cells))
        fmap = c.fl[:, :, max(0, gt - 3):gt + 4].max(axis=2)
        smap = np.asarray(sand[:, :, max(0, gt - 3):gt + 4], dtype=np.float32).mean(axis=2) if sand is not None else None
        # the stand-in model's answer: squares with a strong fault response, and squares that are clearly sandier
        feats = []
        fc = [tools.cell_name(b, a) for a in range(n_cells) for b in range(n_cells)
              if fmap[ei[a]:ei[a + 1], ex[b]:ex[b + 1]].mean() > 0.12]
        if fc:
            feats.append({"type": "fault", "cells": fc[:10], "trend": "up-down", "why": "stand-in model: squares with low coherence", "confidence": "low"})
        if smap is not None:
            bg = float(smap[c.live].mean())
            sc = [tools.cell_name(b, a) for a in range(n_cells) for b in range(n_cells)
                  if smap[ei[a]:ei[a + 1], ex[b]:ex[b + 1]].mean() > bg + 0.12]
            if sc:
                feats.append({"type": "bright_patch", "cells": sc[:10], "trend": "round", "why": "stand-in model: squares with raised sand probability", "confidence": "low"})
        prompt = (RULES + f"Two map views of the same time slice at {t:.0f} ms. LEFT: seismic amplitude. RIGHT: coherence, where dark "
                  "lines mark places where neighbouring traces differ (faults, channel edges). Both carry the same grid: columns A-H, "
                  "rows 1-8. Dimmed areas have no data.\n"
                  "List up to 4 geological features you can actually see, with the grid squares each one covers:\n"
                  '- "channel": a sinuous or ribbon-like band, usually with dark edges on the coherence view\n'
                  '- "fault": a straight or gently curved dark line on the coherence view that offsets the amplitude pattern\n'
                  '- "fold_bands": concentric or parallel amplitude bands (a time slice cuts dipping layers, so a dome shows as rings)\n'
                  '- "bright_patch": an isolated area of strong amplitude\n'
                  "If nothing is clear, return an empty list.\n"
                  'JSON: {"features": [{"type": "channel|fault|fold_bands|bright_patch", "cells": ["B3", "C3"], '
                  '"trend": "up-down|left-right|diagonal-down-right|diagonal-up-right|round", "why": "...", "confidence": "low|medium|high"}]}')
        jobs.append({"task": "slice", "prompt": prompt, "image": img, "mock": {"features": feats}, "max_tokens": 700})
        meta.append((t, gt, fmap, smap))
    out = []
    for (t, gt, fmap, smap), a in zip(meta, llm.ask_many(jobs)):
        rec = {"twt": round(t, 1), "features": [], "reviewed": isinstance(a, dict)}
        for ft in (a or {}).get("features", [])[:4] if isinstance(a, dict) else []:
            if not isinstance(ft, dict) or ft.get("type") not in TYPES:
                continue
            cells = [s.upper() for s in ft.get("cells", []) if isinstance(s, str) and len(s) == 2 and
                     "A" <= s[0].upper() <= chr(ord("A") + n_cells - 1) and s[1].isdigit() and 1 <= int(s[1]) <= n_cells]
            if not cells:
                continue
            mask = np.zeros((c.ni, c.nx), dtype=bool)
            for s in cells:
                b, a_ = ord(s[0]) - 65, int(s[1]) - 1
                mask[ei[a_]:ei[a_ + 1], ex[b]:ex[b + 1]] = True
            mask &= c.live
            rest = c.live & ~mask
            if mask.sum() < 4 or rest.sum() < 4:
                continue
            ii, xx = np.nonzero(mask)
            meas = {"fault_likelihood_inside": round(float(fmap[mask].mean()), 3), "fault_likelihood_elsewhere": round(float(fmap[rest].mean()), 3)}
            if smap is not None:
                meas.update({"sand_prob_inside": round(float(smap[mask].mean()), 2), "sand_prob_elsewhere": round(float(smap[rest].mean()), 2)})
            if ft["type"] == "fault":
                check = "supported" if meas["fault_likelihood_inside"] > max(0.05, 2.0 * meas["fault_likelihood_elsewhere"]) else "not supported"
            elif ft["type"] in ("channel", "bright_patch") and smap is not None:
                d = meas["sand_prob_inside"] - meas["sand_prob_elsewhere"]
                check = "supported (sandier than the surroundings)" if d > 0.08 else ("not sandier than the surroundings" if d > -0.08 else "less sandy than the surroundings")
            else:
                check = "no measurement applies"
            tr = ft.get("trend") if ft.get("trend") in TRENDS or ft.get("trend") == "round" else None
            rec["features"].append({
                "type": ft["type"], "label": TYPES[ft["type"]], "cells": cells, "trend": tr,
                "compass": _compass(c, *TRENDS[tr]) if tr in TRENDS else None,
                "il_range": [int(c.g["ilines"][ii.min()]), int(c.g["ilines"][ii.max()])],
                "xl_range": [int(c.g["xlines"][xx.min()]), int(c.g["xlines"][xx.max()])],
                "where": {"il": int(c.g["ilines"][int(np.median(ii))]), "xl": int(c.g["xlines"][int(np.median(xx))]), "twt": round(t, 1)},
                "why": _text(ft.get("why")), "confidence": _conf(ft.get("confidence")), "measured": meas, "check": check})
        out.append(rec)
    write_json(I / "ai_slices.json", out)
    return {"slices": len(out), "reviewed": sum(1 for r in out if r["reviewed"]), "features": sum(len(r["features"]) for r in out)}


# ====================================================================== fault sticks
def do_sticks(c, max_review=10):
    """Compatibility entry point: deterministic sticks, never a Gemma request."""
    return sticks_mod.build(c.g, c.seis, c.fl, c.live, I, log=c.log)


# ====================================================================== horizons
def _jump(c, axis, gl, z, pol, name, jumps, stats, max_rounds=3):
    """Carry a tracked line across breaks (faults): offer numbered continuations, let the assistant choose."""
    sec, fsec = _sec(c, axis, gl), _sec(c, axis, gl, c.fl)
    lv = c.live[gl, :] if axis == "i" else c.live[:, gl]
    for _ in range(max_rounds):
        ends = tracker.continuations(sec, z, pol, fsec, lv)
        if not ends:
            break
        progressed = False
        for e in ends:
            opts = sorted(e["options"], key=lambda o: o["z"])                 # numbered top to bottom on the picture
            code_best = int(np.argmax([o["corr"] for o in opts])) + 1
            marks = [{"h": float(_num(c, axis, e["col"])), "v": float(_ms(c, o["z"])), "label": str(n + 1), "shift": 34 * (n % 3),
                      "color": tools.MARK_COLORS[(n + 1) % len(tools.MARK_COLORS)]} for n, o in enumerate(opts)]
            title = f"{'Inline' if axis == 'i' else 'Crossline'} {_line_no(c, axis, gl)} - where does the yellow horizon continue?"
            img = _picture(c, axis, gl, title, _pick_lines(c, axis, z, None, (255, 221, 0), width=3), marks)
            side = "right" if e["side"] == 1 else "left"
            prompt = (RULES + f"The yellow line follows one reflector (horizon {name}) and stops, most likely at a fault. The numbered "
                      f"circles on the {side} of the gap are places where it might continue.\n"
                      "Which number continues the SAME reflector? Compare the pattern of bright and dim layers just above and below "
                      "the yellow line with the pattern around each circle; across a fault the whole pattern is shifted up or down "
                      "together. Answer 0 if none of them fits.\n"
                      'JSON: {"id": 0, "why": "..."}')
            a = llm.ask_json("jump", prompt, img, mock={"id": code_best, "why": "stand-in model: best match of the layer pattern"})
            choice, why, rev = code_best, "", False
            if isinstance(a, dict):
                try:
                    n = int(a.get("id"))
                    if 0 <= n <= len(opts):
                        choice, why, rev = n, _text(a.get("why")), True
                except Exception:
                    pass
            stats["jumps_asked"] += 1
            stats["jumps_reviewed"] += rev
            rec = {"line": f"{'inline' if axis == 'i' else 'crossline'} {_line_no(c, axis, gl)}",
                   "at": int(_num(c, axis, e["col"])), "options": len(opts), "chosen": choice, "app_best": code_best,
                   "agree": bool(choice == code_best), "why": why, "reviewed": rev}
            if choice == 0:
                rec["result"] = "not continued (the assistant saw no match)"
                jumps.append(rec)
                continue
            o = opts[choice - 1]
            rec["step_ms"] = round(float((o["z"] - z[e["from"]]) * c.dt), 1)
            z2, _ = tracker.track_line(sec, e["col"], o["z"], pol, fsec, lv)
            new = np.isfinite(z2) & ~np.isfinite(z)
            if new.sum() < 3:
                rec["result"] = "the continuation could not be followed"
                jumps.append(rec)
                continue
            z[new] = z2[new]
            rec["result"] = f"continued for {int(new.sum())} traces"
            jumps.append(rec)
            progressed = True
        if not progressed:
            break
    return z


def do_horizons(c, n_candidates=14, max_review=10):
    k0 = int(np.argmin(np.abs(c.il_idx - c.ni // 2)))
    live_n = [int(c.live[g].sum()) for g in c.il_idx]
    if live_n[k0] < 0.6 * max(live_n):
        k0 = int(np.argmax(live_n))
    gi0 = int(c.il_idx[k0])
    sec = c.seis[gi0]
    cands = tracker.candidates_on_line(sec, c.live[gi0], c.fl[gi0], n=n_candidates)
    if len(cands) < 2:
        raise RuntimeError("no continuous reflectors found on the seed line")
    rng = np.random.default_rng(gi0)
    decs = [tracker.decoy_line(cands[int(q)], c.nx, c.nt, seed=int(q)) for q in rng.choice(len(cands), size=min(2, len(cands)), replace=False)]
    items = sorted(cands + decs, key=lambda q: q["z0"])
    lines = []
    for n, q in enumerate(items):
        lines += _pick_lines(c, "i", q["z"], str(n + 1), tools.MARK_COLORS[n % len(tools.MARK_COLORS)], at=0.06 + 0.88 * n / max(1, len(items) - 1))
    title = f"Inline {_line_no(c, 'i', gi0)} - numbered candidate reflectors"
    prompt = (RULES + f"{title}. A vertical seismic section (time increases downwards). The app drew the numbered coloured lines. "
              "Most of them follow one reflector across the section; a few do NOT (they drift across several reflectors).\n"
              "Choose the 6 to 8 lines an interpreter would map as key horizons. Each must follow a single reflector all the way, "
              "be strong and continuous, and together they should cover the section from shallow to deep and separate packages "
              "that look different (bright from dim, parallel from chaotic).\n"
              'JSON: {"picks": [{"id": 3, "name": "<short descriptive name>", "why": "..."}], "reject": [{"id": 5, "why": "..."}]}')
    ranked = sorted([n for n, q in enumerate(items) if not q.get("decoy")], key=lambda n: -items[n]["score"])
    code_pick = []
    for n in ranked:
        if len(code_pick) < 8 and all(abs(items[n]["z0"] - items[m]["z0"]) >= c.nt / 24.0 for m in code_pick):
            code_pick.append(n)
    mock = {"picks": [{"id": n + 1, "name": f"{'peak' if items[n]['polarity'] > 0 else 'trough'} near {_ms(c, items[n]['z0']):.0f} ms",
                       "why": "stand-in model: strong and continuous by measurement"} for n in sorted(code_pick)],
            "reject": [{"id": n + 1, "why": "stand-in model: drifts across reflectors"} for n, q in enumerate(items) if q.get("decoy")]}
    a = llm.ask_json("horizon_select", prompt, _picture(c, "i", gi0, title, lines), mock=mock, max_tokens=900)
    chosen, why, sel_rev, dec_shown, dec_rej = [], {}, False, 0, 0
    if isinstance(a, dict) and isinstance(a.get("picks"), list):
        for p in a["picks"]:
            try:
                n = int(p.get("id")) - 1
            except Exception:
                continue
            if 0 <= n < len(items) and n not in chosen:
                chosen.append(n)
                why[n] = (_text(p.get("name"), 60), _text(p.get("why")))
        if chosen:
            sel_rev = True
            dec_shown = sum(1 for q in items if q.get("decoy"))
            dec_rej = dec_shown - sum(1 for n in chosen if items[n].get("decoy"))
    trust = _trusted(dec_shown, dec_rej) if sel_rev else False
    chosen = [n for n in chosen if not items[n].get("decoy")] if trust else []
    for n in ranked:                                                      # top up from the app's own ranking
        if len(chosen) >= (6 if trust else 8):
            break
        if n not in chosen and all(abs(items[n]["z0"] - items[m]["z0"]) >= c.nt / 30.0 for m in chosen):
            chosen.append(n)
    chosen = sorted(chosen, key=lambda n: items[n]["z0"])[:8]
    stats = {"jumps_asked": 0, "jumps_reviewed": 0}
    hz = []
    for hn, n in enumerate(chosen):
        q = items[n]
        name = f"G{hn + 1}"
        jumps, tie = [], {"tied": 0, "mistied": 0, "tracks": 0}
        z = _jump(c, "i", gi0, np.array(q["z"], dtype=np.float32), q["polarity"], name, jumps, stats)
        Zi = np.full((len(c.il_idx), c.nx), np.nan, dtype=np.float32)
        Zx = np.full((c.ni, len(c.xl_idx)), np.nan, dtype=np.float32)
        Zi[k0] = z
        tracker.grow_lattice(c.seis, q["polarity"], c.il_idx, c.xl_idx, Zi, Zx, [("i", k0)], c.fl, c.live, stats=tie)
        nodes = np.array([[c.live[g, x] for x in c.xl_idx] for g in c.il_idx])
        got = np.isfinite(Zi[:, c.xl_idx]) | np.isfinite(Zx[c.il_idx, :])
        if nodes.sum() and (got & nodes).sum() / nodes.sum() < 0.85:      # second chance through the middle crossline
            m0 = int(np.argmin(np.abs(c.xl_idx - c.nx // 2)))
            if np.isfinite(Zx[:, m0]).sum() > 5:
                Zx[:, m0] = _jump(c, "x", int(c.xl_idx[m0]), Zx[:, m0].copy(), q["polarity"], name, jumps, stats)
                tracker.grow_lattice(c.seis, q["polarity"], c.il_idx, c.xl_idx, Zi, Zx, [("x", m0)], c.fl, c.live, stats=tie)
                got = np.isfinite(Zi[:, c.xl_idx]) | np.isfinite(Zx[c.il_idx, :])
        hz.append({"name": name, "label": why.get(n, ("", ""))[0] or f"{'peak' if q['polarity'] > 0 else 'trough'} near {_ms(c, q['z0']):.0f} ms",
                   "why": why.get(n, ("", ""))[1], "chosen_by": "assistant" if n in why and trust else "app (strongest continuous reflectors)",
                   "polarity": int(q["polarity"]), "strength": float(q["score"]), "twt_at_seed_ms": round(float(_ms(c, q["z0"])), 1),
                   "Zi": Zi, "Zx": Zx, "jumps": jumps, "ties": tie, "flags": [],
                   "net_coverage": round(float((got & nodes).sum() / max(1, nodes.sum())), 2)})
    # ---- the assistant checks the picked lines
    rev_shown = rev_rej = rev_lines = 0
    jobs, meta = [], []
    for axis, idxs in (("i", c.il_idx), ("x", c.xl_idx)):
        pick = [int(idxs[int(round(q))]) for q in np.linspace(0, len(idxs) - 1, min(max_review, len(idxs)))]
        for gl in sorted(set(pick)):
            pos = int(np.flatnonzero(idxs == gl)[0])
            rows = []
            for h in hz:
                z = h["Zi"][pos] if axis == "i" else h["Zx"][:, pos]
                if np.isfinite(z).sum() >= 5:
                    rows.append({"h": h, "z": z})
            if not rows:
                continue
            base = rows[int(np.random.default_rng(gl).integers(len(rows)))]
            dz = np.where(np.isfinite(base["z"]), base["z"] + np.linspace(-1, 1, len(base["z"])) * 22.0 + 9.0, np.nan)
            rows.append({"h": None, "z": np.clip(dz, 2, c.nt - 3)})
            rows.sort(key=lambda r: float(np.nanmedian(r["z"])))
            lines = []
            for n, r in enumerate(rows):
                lines += _pick_lines(c, axis, r["z"], chr(65 + n), tools.MARK_COLORS[n % len(tools.MARK_COLORS)], at=0.08 + 0.84 * n / max(1, len(rows) - 1))
            name = f"{'Inline' if axis == 'i' else 'Crossline'} {_line_no(c, axis, gl)}"
            prompt = (RULES + f"{name}: a vertical seismic section with horizon picks drawn as lettered coloured lines. Check each one "
                      "like a senior interpreter reviewing a junior's picks: does the line stay on ONE reflector, or does it jump or "
                      "drift to a different reflector somewhere? A gap in a line is acceptable (a fault or poor data); a line that "
                      "crosses reflectors is not.\n"
                      'JSON: {"lines": [{"id": "A", "verdict": "ok|jumps|unsure", "where": "left|middle|right|all", "why": "..."}]}\n'
                      '"where" is the part of the picture where the problem is (use "all" if it is wrong throughout).')
            mock = {"lines": [{"id": chr(65 + n), "verdict": "jumps" if r["h"] is None else "ok", "where": "all",
                               "why": "stand-in model: drawn across reflectors on purpose" if r["h"] is None else "stand-in model: followed by wiggle match"}
                              for n, r in enumerate(rows)]}
            jobs.append({"task": "horizon_review", "prompt": prompt, "image": _picture(c, axis, gl, name + " - horizon picks to check", lines),
                         "mock": mock, "max_tokens": 700})
            meta.append((axis, gl, pos, rows))
    answers = llm.ask_many(jobs)
    verdicts = []
    for (axis, gl, pos, rows), a in zip(meta, answers):
        got = {}
        for r in ((a or {}).get("lines") or []) if isinstance(a, dict) else []:
            if isinstance(r, dict) and isinstance(r.get("id"), str) and r.get("verdict") in ("ok", "jumps", "unsure"):
                n = ord(r["id"].strip().upper()[:1] or "?") - 65
                if 0 <= n < len(rows):
                    got[n] = r
        if not got:
            continue
        rev_lines += 1
        for n, r in enumerate(rows):
            v = got.get(n)
            if v is None:
                continue
            if r["h"] is None:
                rev_shown += 1
                rev_rej += v["verdict"] == "jumps"
            else:
                verdicts.append((axis, gl, pos, r["h"], v))
    rev_trust = _trusted(rev_shown, rev_rej)
    for axis, gl, pos, h, v in verdicts:
        h.setdefault("checked", 0)
        h["checked"] += 1
        if v["verdict"] != "jumps":
            continue
        where = v.get("where") if v.get("where") in ("left", "middle", "right", "all") else "all"
        line = f"{'inline' if axis == 'i' else 'crossline'} {_line_no(c, axis, gl)}"
        h["flags"].append({"line": line, "where": where, "why": _text(v.get("why")), "removed": bool(rev_trust)})
        if rev_trust:
            n = c.nx if axis == "i" else c.ni
            a_, b_ = {"left": (0, n // 3), "middle": (n // 3, 2 * n // 3), "right": (2 * n // 3, n), "all": (0, n)}[where]
            if axis == "i":
                h["Zi"][pos, a_:b_] = np.nan
            else:
                h["Zx"][a_:b_, pos] = np.nan
    # ---- surfaces
    out_z, out_c, keep = [], [], []
    for h in hz:
        z, conf, _ = tracker.to_surface(h["Zi"], h["Zx"], c.il_idx, c.xl_idx, c.seis, h["polarity"], c.live)
        if z is None:
            continue
        if out_z:
            clash = z < out_z[-1] + 1.0
            h["crossing_cells"] = int((clash & c.live).sum())
            z = np.maximum(z, out_z[-1] + 1.0)
        out_z.append(z)
        out_c.append(conf)
        keep.append(h)
    if not keep:
        raise RuntimeError("no horizon could be turned into a surface")
    np.savez_compressed(I / "horizons_gemma.npz", names=np.array([h["name"] for h in keep]), z=np.stack(out_z).astype(np.float32),
                        rgt=np.full(len(keep), np.nan), polarity=np.array(["peak" if h["polarity"] > 0 else "trough" for h in keep]),
                        strength=np.array([h["strength"] for h in keep]), conf=np.stack(out_c).astype(np.float32))
    # ---- compare with the RGT horizons
    maps, pairs = {}, []
    rh = tools.horizons("rgt")
    lead_map = tools.lead_map()
    for h, z, conf in zip(keep, out_z, out_c):
        maps[f"{h['name']}__time"] = _ms(c, z).astype(np.float32)
        maps[f"{h['name']}__conf"] = conf
        p = {"gemma": h["name"], "label": h["label"], "rgt": None, "same_reflector": False}
        if rh is not None:
            med = [float(np.median(np.abs(z - rz)[c.live])) for rz in rh["z"]]
            j = int(np.argmin(med))
            st, d = tracker.compare(z, rh["z"][j], c.live, c.dt)
            p.update({"rgt": rh["names"][j], "same_reflector": bool(med[j] <= 4.0), **st})
            if p["same_reflector"]:
                maps[f"{h['name']}__diff"] = d
                dd = np.where(c.live, np.abs(d), 0)
                wi, wx = np.unravel_index(int(np.argmax(dd)), dd.shape)
                p["largest_difference"] = {"il": int(c.g["ilines"][wi]), "xl": int(c.g["xlines"][wx]), "twt": round(float(_ms(c, z[wi, wx])), 1),
                                           "ms": round(float(d[wi, wx]), 1)}
                if lead_map is not None:
                    hi_ = rh["names"].index(p["rgt"])
                    for l in c.k.get("leads", []):
                        if l["horizon"] == p["rgt"]:
                            m = lead_map[hi_] == int(l["id"][1:])
                            if m.any():
                                p.setdefault("leads", {})[l["id"]] = round(float(np.abs(d)[m].mean()), 1)
            else:
                p["note"] = (f"no RGT horizon follows this reflector; the nearest is {p['rgt']}, typically "
                             f"{abs(p['bias_ms']):.0f} ms {'shallower' if p['bias_ms'] > 0 else 'deeper'}")
        pairs.append(p)
    np.savez_compressed(I / "maps_gemma.npz", **maps)
    write_json(I / "hz_compare.json", {"pairs": pairs})
    write_json(I / "picks_gemma.json", {
        "seed_line": f"inline {_line_no(c, 'i', gi0)}", "line_spacing": STEP, "selection_reviewed": sel_rev, "selection_votes_used": trust,
        "selection_decoys": {"shown": dec_shown, "rejected": dec_rej}, "review_lines": rev_lines,
        "review_decoys": {"shown": rev_shown, "rejected": rev_rej, "votes_used": rev_trust}, **stats,
        "horizons": [{q: h.get(q) for q in ("name", "label", "why", "chosen_by", "twt_at_seed_ms", "jumps", "ties", "flags",
                                             "net_coverage", "checked", "crossing_cells")} |
                     {"polarity": "peak" if h["polarity"] > 0 else "trough"} for h in keep]})
    same = [p for p in pairs if p["same_reflector"]]
    c.log(f"Assistant's horizons: {len(keep)} surfaces from picks on every {STEP}th line; {len(same)} follow the same reflector as an "
          f"RGT horizon" + (f" (typical difference {np.median([p['median_abs_ms'] for p in same]):.0f} ms)" if same else ""))
    return {"horizons": len(keep), "selection_reviewed": sel_rev, "paired_with_rgt": len(same), "review_lines": rev_lines,
            "decoys_shown": dec_shown + rev_shown, "decoys_rejected": dec_rej + rev_rej, **stats}


# ====================================================================== the pipeline step
def run(log=print):
    t0 = time.time()
    for k in llm.STATS:
        llm.STATS[k] = 0
    llm.set_budget(config.AI_MINUTES)
    c = _ctx(log)
    if not c.reviewed:
        log(f"The assistant ({config.LLM_URL}) cannot be reached: the app's own choices are used and marked as not reviewed")
        llm.set_budget(1e-9)
    summary = {}
    for name, fn, label in (("facies", review_facies, "facies names"), ("wells", read_wells, "well logs"),
                            ("leads", look_leads, "leads"), ("slices", read_slices, "time slices"),
                            ("horizons", do_horizons, "horizons")):
        log(f"Assistant's pass: {label}")
        try:
            summary[name] = fn(c)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            log(f"!! {label} could not be finished ({type(e).__name__}: {str(e)[:160]})")
    done = {"finished": time.strftime("%Y-%m-%d %H:%M"), "model": config.LLM_MODEL, "reachable": bool(c.reviewed),
            "fault_source": (read_json(I / "faults.json", {}) or {}).get("source"),
            "fault_coherence_max": (read_json(I / "faults.json", {}) or {}).get("coherence_max", 1.0),
            "minutes": round((time.time() - t0) / 60.0, 1), "budget_minutes": config.AI_MINUTES,
            "budget_used_up": bool(c.reviewed and not llm.budget_left()), "requests": dict(llm.STATS), "steps": summary}
    llm.set_budget(None)
    write_json(I / "ai_done.json", done)
    log(f"Assistant's pass finished in {done['minutes']} min: {llm.STATS['asked']} questions asked, "
        f"{llm.STATS['cached']} answered from the cache, {llm.STATS['failed']} without a usable answer")
    return done
