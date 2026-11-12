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


SRC = {"measured": "measured from the data", "well-log": "from well logs", "well-check": "seismic checked against wells",
       "assistant-visual": "the assistant's own reading of a picture", "comparison": "two interpretations compared"}


def build():
    """Collect every result into knowledge.json. Called twice per run: once when the measured interpretation
    is finished (the app is usable from then), and again after the assistant's own pass. Evidence ids from
    the first call never change in the second: new evidence is only ever appended."""
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
    live = np.load(I / "live.npy") if (I / "live.npy").exists() else None
    if live is not None and not live.all():
        survey["empty_share"] = round(float(1 - live.mean()), 3)
    methods = {
        "conversion": sorted({v.meta.get("converter", "?") for v in vols if not v.meta.get("derived")}),
        "rgt": (read_json(I / "rgt.json") or {}).get("method", "not run"),
        "facies": (read_json(I / "sfm.json") or {}).get("method", "not run"),
        "faults": "coherence (trace-to-trace similarity), vertically smoothed; fault sticks picked on every 8th line",
        "horizons": "iso-surfaces of the RGT cube, snapped to the nearest peak/trough",
        "traps": "spill-point analysis of each horizon time map, with and without sealing faults",
    }
    evidence = list(parts.get("evidence", []))
    for e in evidence:
        e.setdefault("source", "well-log" if e.get("kind") == "well" else ("well-check" if e.get("kind") == "rock" else "measured"))

    def ev(text, where=None, kind="observation", source="measured"):
        eid = f"E{len(evidence) + 1}"
        evidence.append({"id": eid, "text": text, "where": where, "kind": kind, "source": source})
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
                              if c.get("correlation") is not None else "."), None, "calibration", "well-check")
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
    if survey.get("empty_share"):
        quality.append(f"{survey['empty_share']:.0%} of the survey rectangle has no traces (empty edge). It is left out of every "
                       f"measurement; horizons are only carried flat across it for display")
    for sfail in read_json(config.WORKSPACE / "soft_fail.json", []) or []:
        quality.append(f"The step '{sfail['label']}' did not finish ({sfail['error']}); results that depend on it are missing")

    # ---------------- wells: what the logs show (measured), ties, rock certainty
    zones = read_json(I / "well_zones.json", {}) or {}
    wlist = ws.get("wells", [])
    wloc = {w["name"]: ({"il": int(round(w["il"])), "xl": int(round(w["xl"]))} if w.get("il") is not None else None) for w in wlist}
    for w in wlist:
        z = zones.get(w["name"])
        if not z:
            continue
        s = z["summary"]
        big = sorted(z["sands"], key=lambda q: -q["thick_m"])[:3]
        w["zones"] = s
        w["evidence"] = [ev(
            f"Well {w['name']} logs: {s['net_sand_m']} m of sand in {s['gross_m']} m logged (net-to-gross {s['net_to_gross']}); "
            + (f"{s['net_pay_m']} m carries the hydrocarbon flag; " if s.get("has_fluid_flag") else "no resistivity or saturation curve, so no fluid flag; ")
            + "thickest sands: " + ", ".join(f"{q['id']} {q['thick_m']} m at {q['top_md']} m"
                                             + (f" ({q['top_twt']:.0f} ms)" if q.get("top_twt") else "")
                                             + (" with hydrocarbon flag" if q["pay"] else "") for q in big) + ".",
            wloc.get(w["name"]), "well", "well-log")]
    calib = read_json(I / "calib.json")
    ties = read_json(I / "ties.json") or {}
    rock = None
    if calib:
        rock = {q: calib.get(q) for q in ("model", "wells_used", "intervals", "sand_share_at_wells", "error_score", "balanced_accuracy",
                                           "balanced_accuracy_raw", "beats_always_average", "reliability", "why", "features",
                                           "wells_excluded", "shifts")}
        if calib.get("model", "none") != "none":
            es = calib["error_score"]
            rel = "; ".join(f"predicted {r['lo']:.1f}-{r['hi']:.1f}: sand in {r['observed']:.0%} of {r['n']} intervals" for r in calib.get("reliability", []))
            rock["evidence"] = ev(
                f"Rock certainty check: the sand prediction was tested on {calib['intervals']} log intervals in {len(calib['wells_used'])} "
                f"wells, each well predicted from the others. Recipe kept: "
                f"{'the sand-probability volume re-scaled to the wells' if calib['model'] == 'isotonic' else 'several volumes combined (' + ', '.join(calib['features']) + ')'}. "
                f"Error score {es[calib['model']]} (0 is perfect; always guessing the average scores {es['baseline']}, the volume as supplied "
                f"{es['raw']}). Reliability - {rel}.", None, "calibration", "well-check")
            if not calib.get("beats_always_average"):
                quality.append("The sand prediction did no better than always guessing the average at the wells: treat rock certainty with caution")
        else:
            rock["evidence"] = ev(f"Rock certainty could not be checked against the wells: {calib.get('why', 'not enough well data')}.",
                                  None, "calibration", "well-check")
        for name, sh in (calib.get("shifts") or {}).items():
            sh["evidence"] = ev(
                f"Well tie {name} ({sh['pair']}): best fit when the logs are moved {sh['shift_ms']:+.0f} ms (fit {sh['r_shifted']}, "
                f"unshifted {sh['r_raw']}); " + ("the shift is clear and is applied in the well comparisons." if sh["clear"] and sh["shift_ms"]
                                                  else "no shift is applied." if sh["clear"] else "the fit is not clear enough to apply a shift."),
                wloc.get(name), "well-tie", "well-check")
        for x in calib.get("wells_excluded", []):
            quality.append(f"{x['well']} is left out of the rock-certainty check: {x['why']}")
    for h in parts.get("horizons", []):
        t = (ties.get("by_horizon") or {}).get(h["name"])
        if t and t["wells"]:
            h["well_tie"] = t
            txt = f"{h['name']} at the wells: crossed by {t['wells']} wells"
            if t["median_mismatch_ms"] is not None:
                txt += f"; the nearest log boundary is typically {t['median_mismatch_ms']} ms away ({t['with_boundary']} wells)"
            if t["usual_rocks"]:
                txt += f"; the logs show {t['usual_rocks']} across it in {t['usual_rocks_count']} of {t['rock_pairs']} wells"
            h["evidence"].append(ev(txt + ".", None, "well-tie", "well-check"))
    pol = ties.get("polarity")
    if pol and pol.get("reading"):
        pol["evidence"] = ev(f"Polarity from the wells: {pol['reading']}.", None, "well-tie", "well-check")

    # Fault geometry is produced by the measured fault stage, independently of Gemma.
    fm = read_json(I / "faults.json", {}) or {}
    cutoff = fm.get("coherence_max", 1.0)
    methods["faults"] = fm.get("method", "legacy coherence discontinuity") + (
        f"; coherence < {cutoff:.2f}" if cutoff < 1 else "; coherence filter off") + "; algorithmic sticks, no Gemma"
    faults = []
    for fa in read_json(I / "fault_surfaces.json", []) or []:
        txt = f"Fault {fa['id']}: " + (f"strikes {fa['trend']}, " if fa.get("trend") else "") \
            + (f"{fa['length_km']} km long, " if fa.get("length_km") else "") + f"cuts {fa['twt_top_ms']}-{fa['twt_base_ms']} ms" \
            + (f", measured throw about {fa['throw_ms']:.0f} ms" if fa.get("throw_ms") else "") \
            + f"; {fa['sticks']} algorithmic sticks; accepted using likelihood and measured reflector offset/fit, without Gemma"
        fa["evidence"] = ev(txt + ".", fa["where"], "faults", "measured")
        faults.append(fa)

    # ---------------- the assistant's own pass (only present after the 'ai' step)
    done = read_json(I / "ai_done.json")
    facies = parts.get("facies", [])
    ghz, readings = [], []
    leads = parts.get("leads", [])
    if done:
        if cutoff != done.get("fault_coherence_max", 1.0) or fm.get("source") != done.get("fault_source"):
            quality.append("Fault settings changed after the cached assistant pass. Its visual readings are retained from the earlier run; "
                           "fault sticks and lead fault scores use the current settings")
        st = done.get("steps", {})
        methods["assistant_pass"] = (f"{done['model']} looked at pictures and reviewed facies names, well logs, leads, time slices, "
                                     f"and horizons ({done['requests']['asked']} questions, {done['minutes']} min)"
                                     if done.get("reachable") else "the model could not be reached; the app's own choices were used")
        if not done.get("reachable"):
            quality.append("The assistant's own pass could not reach the model: the second set of horizons is the "
                           "app's own choice and was not reviewed by the assistant")
        rq = done.get("requests") or {}
        if done.get("reachable") and rq.get("failed"):
            quality.append(f"{rq['failed']} of {rq['asked']} questions to the assistant in its own pass got no usable answer; "
                           f"for those items the app's own choice was kept and is marked as not reviewed")
        if done.get("budget_used_up"):
            quality.append(f"The assistant's own pass ran out of its {done['budget_minutes']:.0f}-minute budget; later items were not reviewed")
        for name, s in st.items():
            if isinstance(s, dict) and s.get("error"):
                quality.append(f"Assistant's pass, {name}: not finished ({s['error']})")
        if done.get("reachable"):
            if (st.get("horizons") or {}).get("horizons") and not (st.get("horizons") or {}).get("selection_reviewed"):
                quality.append("The assistant gave no usable choice of horizons: the second set of horizons (G#) follows the app's "
                               "strongest continuous reflectors, not reviewed")
        af = read_json(I / "ai_facies.json", {}) or {}
        for f in facies:
            a = af.get(str(f["facies"]))
            if a:
                f.update({"gemma_name": a["name"], "gemma_setting": a["setting"], "gemma_agrees": a["agree"], "gemma_why": a["why"],
                          "gemma_confidence": a["confidence"]})
        aw = read_json(I / "ai_wells.json", {}) or {}
        for w in wlist:
            a = aw.get(w["name"])
            if a:
                w["assistant_reading"] = a
                w.setdefault("evidence", []).append(ev(
                    f"Assistant's reading of well {w['name']}'s log panel: {a['reading']} Best reservoir: {a['best_reservoir'] or 'none named'}; "
                    f"fluid: {a['fluid']} (confidence {a['confidence']}).", wloc.get(w["name"]), "well", "assistant-visual"))
        for r in read_json(I / "ai_slices.json", []) or []:
            for ft in r["features"]:
                m = ft["measured"]
                chk = ft["check"] + (f"; sand probability {m['sand_prob_inside']} inside vs {m['sand_prob_elsewhere']} elsewhere" if "sand_prob_inside" in m else "") \
                    + f"; fault likelihood {m['fault_likelihood_inside']} inside vs {m['fault_likelihood_elsewhere']} elsewhere"
                ft["evidence"] = ev(
                    f"Assistant's reading of the {r['twt']:.0f} ms time slice: {ft['label']} across IL {ft['il_range'][0]}-{ft['il_range'][1]} / "
                    f"XL {ft['xl_range'][0]}-{ft['xl_range'][1]}" + (f", trending {ft['compass']}" if ft.get("compass") else "")
                    + f" - \"{ft['why']}\" (confidence {ft['confidence']}). Measured check: {chk}.", ft["where"], "slice", "assistant-visual")
                readings.append(dict(ft, twt=r["twt"]))
        pk = read_json(I / "picks_gemma.json") or {}
        cmp_ = read_json(I / "hz_compare.json") or {}
        pinfo = {h["name"]: h for h in pk.get("horizons", [])}
        for p in cmp_.get("pairs", []):
            hi = pinfo.get(p["gemma"], {})
            if p.get("same_reflector"):
                txt = (f"Second-opinion horizon {p['gemma']} ({p['label']}; {hi.get('chosen_by', 'chosen by the app')}) follows the same reflector as "
                       f"RGT horizon {p['rgt']}: they differ by {p['median_abs_ms']} ms typically, {p['p90_abs_ms']} ms at the worst tenth of the area")
                ld = p.get("largest_difference")
                if ld:
                    txt += f"; largest difference {ld['ms']:+.0f} ms at IL {ld['il']} / XL {ld['xl']}"
                where = ld
            else:
                txt, where = f"Second-opinion horizon {p['gemma']} ({p['label']}): {p.get('note', 'no RGT counterpart')}", None
            tie = hi.get("ties") or {}
            if tie.get("tied") or tie.get("mistied"):
                txt += f". Line ties: {tie['tied']} of {tie['tied'] + tie['mistied']} crossings agree"
            for j in (hi.get("jumps") or [])[:2]:
                txt += (f". Across a break on {j['line']} near {j['at']}: " +
                        (f"option {j['chosen']} of {j['options']} chosen ({'the assistant' if j['reviewed'] else 'the app'}"
                         + ("" if j["agree"] or not j["reviewed"] else f", the app's best match was option {j['app_best']}") + ")"
                         + (f", a step of {j['step_ms']:+.0f} ms" if j.get("step_ms") is not None else "") if j["chosen"] else "no continuation accepted"))
            for fl_ in (hi.get("flags") or [])[:2]:
                txt += f". The assistant flagged it on {fl_['line']} ({fl_['where']}): \"{fl_['why']}\""
            p["evidence"] = ev(txt + ".", where, "horizons", "comparison")
            p.update({q: hi.get(q) for q in ("why", "chosen_by", "net_coverage", "jumps", "flags", "ties", "polarity")})
            ghz.append(p)
        rd, sd = pk.get("review_decoys") or {}, pk.get("selection_decoys") or {}
        if rd.get("shown") or sd.get("shown"):
            ev(f"Check on the assistant's horizon judgement: when choosing horizons it rejected {sd.get('rejected', 0)} of {sd.get('shown', 0)} "
               f"lines drawn across the reflectors on purpose; when reviewing picks, {rd.get('rejected', 0)} of {rd.get('shown', 0)}.",
               None, "check", "comparison")
        al = read_json(I / "ai_leads.json", {}) or {}
        agree = {}
        for p in cmp_.get("pairs", []):
            for lid, d in (p.get("leads") or {}).items():
                agree[lid] = (p["gemma"], p["rgt"], d)
        for l in leads:
            a = al.get(l["id"])
            if a:
                l["assistant_saw"] = a
                eid = ev(f"{l['id']}: the assistant, looking at the two sections through the crest - horizon closes over it on the inline: "
                         f"{a['closes_inline']}; on the crossline: {a['closes_crossline']}; brighter under the crest: {a['bright_under_crest']}; "
                         f"flat spot: {a['flat_spot']}. Concern: {a['concern'] or 'none'}. (confidence {a['confidence']})", l["where"], "trap", "assistant-visual")
                yes = [a["closes_inline"], a["closes_crossline"]]
                l["for" if yes.count("yes") == 2 else ("against" if "no" in yes else "unknown")].append(eid)
            if l["id"] in agree:
                gname, rname, d = agree[l["id"]]
                eid = ev(f"{l['id']}: the two independent horizon interpretations ({rname} from RGT, {gname} tracked on a net of lines) "
                         f"differ by {d} ms on average inside the lead" + (", so its shape does not depend on the method." if d <= 8 else
                                                                         ", so the mapped shape of the trap depends on the method."),
                         l["where"], "trap", "comparison")
                l["for" if d <= 8 else "against"].append(eid)
                l["horizon_agreement_ms"] = d

    k = {"built": time.strftime("%Y-%m-%d %H:%M"), "survey": survey,
         "volumes": [{"name": v.name, "label": v.meta["label"], "role": v.meta["role"], "converter": v.meta.get("converter"),
                      "range": [v.meta["stats"]["lo"], v.meta["stats"]["hi"]]} for v in vols],
         "methods": methods, "horizons": parts.get("horizons", []), "leads": leads,
         "facies": facies, "calibration": parts.get("calibration", []),
         "well_ties": parts.get("well_ties", []), "wells": wlist, "evidence": evidence,
         "rock_certainty": rock, "horizon_ties": ties.get("ties", []), "polarity": pol,
         "faults": faults, "second_horizons": ghz, "slice_readings": readings, "assistant_pass": done,
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
                 + (f", at wells: {h['well_tie']['usual_rocks']}" + (f", log boundary {h['well_tie']['median_mismatch_ms']} ms off"
                                                                   if h['well_tie'].get('median_mismatch_ms') is not None else "")
                    if (h.get("well_tie") or {}).get("usual_rocks") else "")
                 + f" [{', '.join(h.get('evidence', []))}]")
    if k.get("facies"):
        L.append("\nSEISMIC FACIES (from the foundation model features)")
        for f in k["facies"]:
            if f.get("no_data"):
                L.append(f"- Facies {f['facies']}: no data (empty survey edge)")
                continue
            extra = f", sand prob {f['sand_mean']:.2f}" if "sand_mean" in f else ""
            if "well_net_sand" in f:
                extra += f", wells show net sand {f['well_net_sand']:.0%} ({f['well_intervals']} intervals)"
            L.append(f"- Facies {f['facies']} = {f.get('name', f['description'])} ({f.get('lithology', '')}); "
                     f"{f['share']:.0%} of the volume, around {f['typical_twt_ms']:.0f} ms{extra}."
                     + ("" if compact else f" Usually (general principle): {f.get('reading', '')}.")
                     + (f" Assistant's own name from pictures: {f['gemma_name']} - {f['gemma_setting']}"
                        + ("" if f.get("gemma_agrees", True) else " (it disagreed with the rule-based name)") if f.get("gemma_name") else ""))
    L.append("\nWELLS")
    ties = {t["well"]: t for t in k.get("well_ties", [])}
    for w in k.get("wells", []):
        bits = [f"IL {w['il']:.0f} / XL {w['xl']:.0f}" if w.get("il") is not None else "position unknown"]
        if w.get("inside_survey") is False:
            bits.append("outside the seismic survey")
        bits.append("logs: " + (",".join(w.get("curves", [])) or "none"))
        bits.append(w.get("td_source", "no time-depth"))
        info = "; ".join(f"{a}={b}" for a, b in list(w.get("info", {}).items())[:(4 if compact else 8)])
        L.append(f"- {w['name']}: " + "; ".join(bits) + (f" | well list: {info}" if info else "")
                 + (f" [{', '.join(w['evidence'])}]" if w.get("evidence") else ""))
        if w.get("assistant_reading") and not compact:
            L.append(f"    assistant's log reading: {w['assistant_reading']['reading']} (fluid: {w['assistant_reading']['fluid']})")
        if not compact and w["name"] in ties:
            cells = []
            for hn, r in ties[w["name"]]["horizons"].items():
                lg0 = r.get("logs_layer") or {}
                if not (lg0.get("hydrocarbon_fraction", 0) > 0.02 or lg0.get("net_sand_fraction", 0) >= 0.4):
                    continue                    # only the layers worth a line; the rest is available through the well tool
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
                L.append("    notable layers (below the named horizon) -> " + " | ".join(cells[:4]))
    if k.get("calibration"):
        L.append("\nWELL CHECKS OF THE SEISMIC PROPERTIES")
        for c in k["calibration"]:
            L.append(f"- {c['check']}: {c['verdict']}" + (f" (r={c['correlation']}, n={c['points']})" if c.get("correlation") is not None else "")
                     + f" [{c.get('evidence', '')}]")
    rc = k.get("rock_certainty")
    if rc:
        L.append("\nROCK CERTAINTY (sand or shale, checked against the wells)")
        if rc.get("model", "none") == "none":
            L.append(f"- not checked: {rc.get('why')} [{rc.get('evidence', '')}]")
        else:
            rel = "; ".join(f"{r['lo']:.1f}-{r['hi']:.1f} -> sand in {r['observed']:.0%} (n={r['n']})" for r in rc.get("reliability", []))
            L.append(f"- tested on {rc['intervals']} log intervals in {len(rc['wells_used'])} wells, each well predicted from the others; "
                     f"when the prediction was: {rel} [{rc.get('evidence', '')}]")
        for name, sh in list((rc.get("shifts") or {}).items())[:(6 if compact else 20)]:
            if sh.get("shift_ms") and sh.get("clear"):
                L.append(f"- well tie {name}: logs moved {sh['shift_ms']:+.0f} ms to fit the seismic (fit {sh['r_shifted']}) [{sh.get('evidence', '')}]")
        if (k.get("polarity") or {}).get("evidence"):
            L.append(f"- polarity: {k['polarity']['reading']} [{k['polarity']['evidence']}]")
    if k.get("faults"):
        L.append("\nNAMED FAULTS (algorithmic sticks picked on every 8th line, measured offset/fit checks, no Gemma)")
        for fa in k["faults"][:(6 if compact else 14)]:
            L.append(f"- {fa['id']}: " + (f"strike {fa['trend']}, " if fa.get("trend") else "") + (f"{fa['length_km']} km, " if fa.get("length_km") else "")
                     + f"{fa['twt_top_ms']}-{fa['twt_base_ms']} ms" + (f", throw ~{fa['throw_ms']:.0f} ms" if fa.get("throw_ms") else "")
                     + (f" down to the {fa['downthrown_towards']}" if fa.get("downthrown_towards") else "")
                     + f", near IL {fa['where']['il']} / XL {fa['where']['xl']}; algorithmic sticks, measured throw/fit checks [{fa['evidence']}]")
    if k.get("second_horizons"):
        L.append("\nSECOND-OPINION HORIZONS (G#: reflectors chosen from a picture, tracked along a net of lines, independent of RGT)")
        for p in k["second_horizons"]:
            L.append(f"- {p['gemma']} ({p['label']}): " + (f"same reflector as {p['rgt']}, typical difference {p['median_abs_ms']} ms, worst tenth {p['p90_abs_ms']} ms"
                                                          if p.get("same_reflector") else p.get("note", "no RGT counterpart")) + f" [{p['evidence']}]")
    if k.get("slice_readings"):
        L.append("\nASSISTANT'S OWN READINGS OF TIME SLICES (what a vision model saw in pictures - weaker than measurements; each has a measured check)")
        for ft in k["slice_readings"][:(6 if compact else 16)]:
            L.append(f"- {ft['twt']:.0f} ms: {ft['label']} at IL {ft['il_range'][0]}-{ft['il_range'][1]} / XL {ft['xl_range'][0]}-{ft['xl_range'][1]}"
                     + (f", {ft['compass']}" if ft.get("compass") else "") + f"; check: {ft['check']} [{ft['evidence']}]")
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
