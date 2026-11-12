"""Assemble everything the pipeline found into one knowledge file + a short briefing.

The chat assistant never sees the raw cubes. It reasons over this file, over
pictures of the data, and over the answers of a few look-up tools. Every fact has
an id (E1, E2, ...) and a place in the survey, so each claim can be checked.
"""
import time

import numpy as np

from . import config, store, wells
from .config import read_json, write_json

KFILE = config.WORKSPACE / "knowledge.json"
_cache = {}


def load():
    if not KFILE.exists():
        return None
    mt = KFILE.stat().st_mtime
    if _cache.get("mt") != mt:
        _cache.update({"mt": mt, "k": read_json(KFILE)})
    return _cache["k"]


def build():
    I = config.INTERP
    grid = read_json(I / "grid.json") or {}
    parts = read_json(I / "interp.json") or {}
    ws = wells.wells_summary()
    ref = store.reference_volume()
    vols = store.list_volumes()

    survey = {}
    if ref is not None:
        survey = {"inlines": [int(ref.ilines.min()), int(ref.ilines.max())],
                  "crosslines": [int(ref.xlines.min()), int(ref.xlines.max())],
                  "twt_ms": [float(ref.samples.min()), float(ref.samples.max())],
                  "sample_ms": float(np.median(np.diff(ref.samples))) if len(ref.samples) > 1 else None,
                  "size": list(ref.shape)}
        b = ref.bin_size()
        if b:
            survey["bin_m"] = [round(b[0], 1), round(b[1], 1)]
            survey["area_km2"] = round(b[0] * ref.shape[0] * b[1] * ref.shape[1] / 1e6, 1)
    methods = {
        "conversion": sorted({v.meta.get("converter", "?") for v in vols}),
        "rgt": (read_json(I / "rgt.json") or {}).get("method", "not run"),
        "facies": (read_json(I / "sfm.json") or {}).get("method", "not run"),
        "faults": "trace-to-trace similarity (semblance), vertically smoothed",
        "horizons": "iso-surfaces of the RGT cube, snapped to the nearest peak/trough",
        "traps": "spill-point analysis of each horizon time map, with and without sealing faults",
    }
    evidence = list(parts.get("evidence", []))

    def ev(text, where=None, kind="observation"):
        eid = f"E{len(evidence) + 1}"
        evidence.append({"id": eid, "text": text, "where": where, "kind": kind})
        return eid

    for h in parts.get("horizons", []):
        f = h["faults"]
        trend = ", ".join(f"{t['trend']} ({t['share']:.0%})" for t in f["trends"]) or "no clear trend"
        h["evidence"] = [
            ev(f"{h['name']} ({h['polarity']}) lies between {h['twt_min_ms']} and {h['twt_max_ms']} ms and {h['dip_direction']}; "
               f"shallowest point IL {h['shallowest_at']['il']} / XL {h['shallowest_at']['xl']}.", h["shallowest_at"], "structure"),
            ev(f"{h['name']}: {f['count']} fault segments cut this level; dominant strike {trend}."
               + (f" Largest: {f['largest'][0]['length_km']} km long, about {f['largest'][0]['throw_ms']} ms throw."
                  if f["largest"] else ""), f["largest"][0]["where"] if f["largest"] else None, "faults"),
        ]
    for c in parts.get("calibration", []):
        c["evidence"] = ev(f"Well check - {c['check']}: {c['verdict']}"
                           + (f" (correlation {c['correlation']}, {c['points']} points from {c.get('wells', '?')} wells)."
                              if c.get("correlation") is not None else "."), None, "calibration")
    quality = list(ws.get("notes", []))
    for w in ws.get("wells", []):
        for n in w.get("notes", []):
            quality.append(f"{w['name']}: {n}")
        if w.get("td_borrow_km") is not None:
            quality.append(f"{w['name']}: time-depth from another well ({w['td_source']}); expect tens of ms uncertainty")
    if not methods["rgt"].startswith("RGT-Est"):
        quality.append("RGT-Est did not run; horizons come from the simpler classical method and are less reliable across faults")
    elif "reduced" in methods["rgt"]:
        quality.append("RGT-Est had to run on a smaller grid than its authors recommend (GPU memory); fine detail of the horizons may suffer")
    if "NCS" not in methods["facies"]:
        quality.append(f"Seismic facies come from: {methods['facies']}")
    if grid.get("stride") and max(grid["stride"]) > 1:
        s = grid["stride"]
        quality.append(f"Interpretation was done on a thinned grid (every {s[0]} inline, {s[1]} crossline, {s[2]} sample)")

    k = {"built": time.strftime("%Y-%m-%d %H:%M"), "survey": survey,
         "volumes": [{"name": v.name, "label": v.meta["label"], "role": v.meta["role"], "converter": v.meta.get("converter"),
                      "range": [v.meta["stats"]["lo"], v.meta["stats"]["hi"]]} for v in vols],
         "methods": methods, "horizons": parts.get("horizons", []), "leads": parts.get("leads", []),
         "facies": parts.get("facies", []), "calibration": parts.get("calibration", []),
         "well_ties": parts.get("well_ties", []), "wells": ws.get("wells", []), "evidence": evidence,
         "data_quality": quality, "window_ms": parts.get("window_ms", 40)}
    k["briefing"] = briefing(k)
    write_json(KFILE, k)
    return k


def _f(v, nd=2):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def briefing(k, max_leads=10, compact=False):
    """Plain-text summary given to the language model with every question."""
    s, L = k.get("survey", {}), []
    L.append("SURVEY")
    if s:
        L.append(f"- inlines {s['inlines'][0]}-{s['inlines'][1]}, crosslines {s['crosslines'][0]}-{s['crosslines'][1]}, "
                 f"two-way time {s['twt_ms'][0]:.0f}-{s['twt_ms'][1]:.0f} ms, sample {_f(s.get('sample_ms'), 0)} ms"
                 + (f", bins {s['bin_m'][0]} x {s['bin_m'][1]} m, area {s['area_km2']} km2" if s.get("bin_m") else ""))
    L.append("- volumes: " + ", ".join(f"{v['label']} [{v['name']}]" for v in k.get("volumes", [])))
    m = k.get("methods", {})
    L.append(f"- how it was interpreted: horizons from {m.get('rgt')}; facies from {m.get('facies')}; "
             f"faults from {m.get('faults')}; traps from {m.get('traps')}.")
    L.append("\nHORIZONS (shallow to deep)")
    for h in k.get("horizons", []):
        f = h["faults"]
        trend = "/".join(t["trend"] for t in f["trends"][:2]) or "none clear"
        L.append(f"- {h['name']} ({h['polarity']}): {h['twt_min_ms']:.0f}-{h['twt_max_ms']:.0f} ms, {h['dip_direction']}, "
                 f"{f['count']} faults (strike {trend}), {h['closures']} structural closures"
                 + (f", layer below it averages sand prob {h['sand_prob_layer_avg']:.2f}" if h.get("sand_prob_layer_avg") is not None else "")
                 + f" [{', '.join(h.get('evidence', []))}]")
    if k.get("facies"):
        L.append("\nSEISMIC FACIES (from the foundation model features)")
        for f in k["facies"]:
            extra = f", sand prob {f['sand_mean']:.2f}" if "sand_mean" in f else ""
            L.append(f"- Facies {f['facies']}: {f['description']}; {f['share']:.0%} of the volume, around {f['typical_twt_ms']:.0f} ms{extra}")
    L.append("\nWELLS")
    ties = {t["well"]: t for t in k.get("well_ties", [])}
    for w in k.get("wells", []):
        bits = [f"IL {w['il']:.0f} / XL {w['xl']:.0f}" if w.get("il") is not None else "position unknown"]
        if w.get("inside_survey") is False:
            bits.append("outside the seismic survey")
        bits.append("logs: " + (",".join(w.get("curves", [])) or "none"))
        bits.append(w.get("td_source", "no time-depth"))
        info = "; ".join(f"{a}={b}" for a, b in list(w.get("info", {}).items())[:(4 if compact else 8)])
        L.append(f"- {w['name']}: " + "; ".join(bits) + (f" | well list: {info}" if info else ""))
        if not compact and w["name"] in ties:
            cells = []
            for hn, r in ties[w["name"]]["horizons"].items():
                lg = r.get("logs_layer") or {}
                parts = []
                if "net_sand_fraction" in lg:
                    parts.append(f"net sand {lg['net_sand_fraction']:.0%}")
                if "sand_porosity" in lg:
                    parts.append(f"porosity {lg['sand_porosity']:.0%}")
                if "hydrocarbon_fraction" in lg:
                    parts.append(f"hydrocarbon flag {lg['hydrocarbon_fraction']:.0%}")
                ss = (r.get("seismic_layer") or {}).get("sand")
                if ss is not None:
                    parts.append(f"seismic sand prob {ss:.2f}")
                if parts:
                    cells.append(f"{hn} at {r['twt']:.0f} ms: " + ", ".join(parts))
            if cells:
                L.append("    logs in the layer below each horizon -> " + " | ".join(cells))
    if k.get("calibration"):
        L.append("\nWELL CHECKS OF THE SEISMIC PROPERTIES")
        for c in k["calibration"]:
            L.append(f"- {c['check']}: {c['verdict']}" + (f" (r={c['correlation']}, n={c['points']})" if c.get("correlation") is not None else "")
                     + f" [{c.get('evidence', '')}]")
    L.append("\nLEADS (ranked by a transparent screening score 0-1; a structure S# can hold several leads at different levels)")
    etext = {e["id"]: e["text"] for e in k.get("evidence", [])}
    for n, l in enumerate(k.get("leads", [])[:max_leads]):
        sc = l["scores"]
        same = f", same structure as {','.join(l['same_structure_as'])}" if l.get("same_structure_as") else ""
        L.append(f"- {l['id']} (rank {l['rank']}, structure {l.get('structure', '?')}{same}): {l['kind']} at {l['horizon']}, "
                 f"crest IL {l['where']['il']} / XL {l['where']['xl']} "
                 f"at {l['crest_twt_ms']:.0f} ms, area {_f(l.get('area_km2'))} km2, relief {l['relief_ms']:.0f} ms, "
                 f"reservoir zone {l.get('reservoir_zone', 'n/a')} (about {l.get('zone_twt_ms', 0):.0f} ms at the crest), "
                 f"{'ALREADY DRILLED by ' + ','.join(l.get('wells_inside', [])) if l.get('tested') else 'undrilled'}. "
                 f"Scores: trap {sc['trap']}, reservoir {sc['reservoir']}, seal {sc['seal']}, "
                 f"hydrocarbon indicators {sc['hydrocarbon_indicators']}, overall {sc['overall']}.")
        if not compact and n < 8:
            for tag, key in (("for", "for"), ("against", "against"), ("unknown", "unknown")):
                for e in l.get(key, []):
                    L.append(f"    {tag} [{e}] {etext.get(e, '').split(': ', 1)[-1]}")
    if len(k.get("leads", [])) > max_leads:
        L.append(f"- ... and {len(k['leads']) - max_leads} lower-ranked leads (use the lead tool)")
    if k.get("data_quality"):
        L.append("\nDATA QUALITY AND LIMITS")
        for q in k["data_quality"][:(8 if compact else 25)]:
            L.append(f"- {q}")
    return "\n".join(L)
