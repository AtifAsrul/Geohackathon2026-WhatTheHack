"""From horizons + volumes + wells to maps, traps and ranked leads.

Everything here is deliberately simple and transparent: each number that feeds a
lead's score is kept, so the assistant can show its working.
"""
import heapq

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

from . import faults as faults_mod
from . import wells as wells_mod

SEISMIC_ROLES = ("full", "near", "mid", "far", "ufar")
ROLE_LABEL = {"full": "full stack", "near": "near stack", "mid": "mid stack", "far": "far stack",
              "ufar": "ultra-far stack", "ai": "P-impedance", "si": "S-impedance", "vpvs": "Vp/Vs",
              "sand": "sand probability"}


# ------------------------------------------------------------------ helpers
def take_at(vol, z):
    k = np.clip(np.rint(z).astype(np.int64), 0, vol.shape[2] - 1)
    return np.take_along_axis(vol, k[..., None], axis=2)[..., 0]


class Windowed:
    """Mean (or RMS) of a cube between two surfaces, via one cumulative sum."""

    def __init__(self, vol, power=1):
        self.power = power
        v = np.asarray(vol, dtype=np.float64)
        self.cs = np.cumsum(v * v if power == 2 else v, axis=2)
        self.nt = vol.shape[2]

    def mean(self, z0, z1):
        k0 = np.clip(np.rint(z0).astype(np.int64), 0, self.nt - 2)
        k1 = np.clip(np.rint(z1).astype(np.int64), 1, self.nt - 1)
        k1 = np.maximum(k1, k0 + 1)
        top = np.take_along_axis(self.cs, k0[..., None], axis=2)[..., 0]
        bot = np.take_along_axis(self.cs, k1[..., None], axis=2)[..., 0]
        m = (bot - top) / (k1 - k0)
        return (np.sqrt(np.maximum(m, 0)) if self.power == 2 else m).astype(np.float32)


def priority_flood(z, wall=None):
    """Fill 'pits' of the surface z (here: structural highs, because z is time, small = shallow).

    Returns the spill level for every cell. filled - z > 0 marks a closure.
    Cells in `wall` (sealing faults) cannot be crossed.
    """
    ni, nx = z.shape
    BIG = 1e12
    zz = z.astype(np.float64).copy()
    if wall is not None:
        zz[wall] = BIG
    filled = np.full((ni, nx), np.inf)
    done = np.zeros((ni, nx), dtype=bool)
    heap = []
    for i in range(ni):
        for x in ((0, nx - 1) if 0 < i < ni - 1 else range(nx)):
            if not done[i, x]:
                done[i, x] = True
                filled[i, x] = zz[i, x]
                heap.append((zz[i, x], i, x))
    heapq.heapify(heap)
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        e, i, x = pop(heap)
        for di, dx in nbrs:
            a, b = i + di, x + dx
            if 0 <= a < ni and 0 <= b < nx and not done[a, b]:
                done[a, b] = True
                f = zz[a, b] if zz[a, b] > e else e
                filled[a, b] = f
                push(heap, (f, a, b))
    # compartments completely boxed in by faults: fill to their own deepest point
    boxed = (filled >= BIG / 2)
    if wall is not None:
        boxed &= ~wall
    if boxed.any():
        lab, n = label(boxed)
        for c in range(1, n + 1):
            m = lab == c
            filled[m] = z[m].max()
    if wall is not None:
        filled[wall] = z[wall]
    return filled


def closures_on(z_ms, wall, min_relief, min_cells):
    f = priority_flood(z_ms, wall)
    depth = f - z_ms
    if wall is not None:
        depth[wall] = 0
    lab, n = label(depth > min_relief, structure=np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        m = lab == c
        cells = int(m.sum())
        if cells < min_cells:
            continue
        zi = np.where(m, z_ms, np.inf)
        ci, cx = np.unravel_index(np.argmin(zi), z_ms.shape)
        out.append({"mask": m, "cells": cells, "relief_ms": float(depth[m].max()),
                    "crest": (int(ci), int(cx)), "crest_ms": float(z_ms[ci, cx]), "spill_ms": float(f[ci, cx])})
    return out, depth


def _zscore_in(prop, mask):
    if prop is None:
        return None
    sd = float(prop.std())
    if sd < 1e-12:
        return 0.0
    return float((prop[mask].mean() - prop.mean()) / sd)


def _ring(mask, width):
    return binary_dilation(mask, iterations=max(2, int(width))) & ~mask


def _lvl(score):
    return "favourable" if score >= 0.6 else ("moderate" if score >= 0.4 else "weak")


# ------------------------------------------------------------------ main builder
def build(grid, seis, vols, horizons, fl, semb, facies, wells, log=print, window_ms=40.0):
    """Returns (knowledge parts dict, maps dict of 2-D arrays, lead_map int16 (nh, ni, nx))."""
    ni, nx, nt = seis.shape
    dt = grid["dt_ms"]
    t0 = grid["times"][0]
    w = max(3, int(round(window_ms / dt)))
    cell = grid.get("cell_m")
    cell_km2 = (cell[0] * cell[1] / 1e6) if cell else None
    il_of = np.asarray(grid["ilines"])
    xl_of = np.asarray(grid["xlines"])
    nh = len(horizons)
    eps = 1e-9

    def ms(z):
        return t0 + np.asarray(z) * dt

    def where(i, x, t_ms=None, horizon=None, vol=None):
        d = {"il": int(il_of[int(np.clip(i, 0, ni - 1))]), "xl": int(xl_of[int(np.clip(x, 0, nx - 1))])}
        if t_ms is not None:
            d["twt"] = float(round(t_ms, 1))
        if horizon:
            d["horizon"] = horizon
        if vol:
            d["vol"] = vol
        return d

    def area(cells):
        return round(cells * cell_km2, 2) if cell_km2 else None

    log("Attributes: preparing layer averages")
    win = {}
    for role, v in vols.items():
        win[role] = Windowed(v, power=2 if role in SEISMIC_ROLES else 1)
    if "full" not in win:
        win["full"] = Windowed(seis, power=2)

    maps, hz_info, leads, evidence = {}, [], [], []
    lead_map = np.zeros((nh, ni, nx), dtype=np.int16)

    def ev(text, loc=None, kind="observation"):
        eid = f"E{len(evidence) + 1}"
        evidence.append({"id": eid, "text": text, "where": loc, "kind": kind})
        return eid

    # ---------------- wells: positions on the interpretation grid
    well_hits = {}
    for wl in wells.get("wells", []):
        a = wells_mod.well_arrays(wl["key"])
        if "il" not in a or not np.isfinite(a.get("twt", np.array([np.nan]))).any():
            continue
        gi = np.interp(a["il"], il_of, np.arange(ni), left=np.nan, right=np.nan)
        gx = np.interp(a["xl"], xl_of, np.arange(nx), left=np.nan, right=np.nan)
        ok = np.isfinite(gi) & np.isfinite(gx) & np.isfinite(a["twt"])
        if ok.sum() < 3:
            continue
        well_hits[wl["key"]] = {"name": wl["name"], "gi": gi, "gx": gx, "twt": a["twt"], "ok": ok, "hz": {}}

    # ---------------- per horizon (each horizon is the top of the layer down to the next horizon)
    for h, hz in enumerate(horizons):
        name, z = hz["name"], hz["z"]
        nxt = horizons[h + 1]["name"] if h + 1 < nh else None
        log(f"Attributes: {name} maps and traps")
        zn = horizons[h + 1]["z"] if nxt else np.clip(z + 3 * w, 0, nt - 1)
        zn = np.minimum(np.maximum(zn, z + 2), nt - 1)
        z_ms = gaussian_filter(ms(z), 1.0)
        nsub = int(np.clip(round(float(np.mean(zn - z)) / w), 1, 6))
        edges = [z + (zn - z) * (j / nsub) for j in range(nsub + 1)]
        slabs = [{role: wd.mean(edges[j], edges[j + 1]) for role, wd in win.items()} for j in range(nsub)]
        off = [(float(np.mean(edges[j] - z)) * dt, float(np.mean(edges[j + 1] - z)) * dt) for j in range(nsub)]
        za = np.clip(z - w, 0, nt - 1)
        whole = {role: win[role].mean(z, zn) for role in ("sand", "ai", "near", "far") if role in win}

        m = {"time": ms(z).astype(np.float32), "amp": take_at(seis, z).astype(np.float32),
             "rms_below": np.max([s["full"] for s in slabs], axis=0)}
        if "sand" in win:
            m["sand_below"] = np.max([s["sand"] for s in slabs], axis=0)
            m["sand_above"] = win["sand"].mean(za, z)
        for role in ("ai", "vpvs", "si"):
            if role in win:
                m[f"{role}_below"] = np.min([s[role] for s in slabs], axis=0)
        if "near" in whole and "far" in whole:
            m["far_over_near"] = (whole["far"] / (whole["near"] + eps)).astype(np.float32)
        segs, fmap = faults_mod.fault_segments(fl, z, cell or (1.0, 1.0), grid.get("u_il", (0, 1)), grid.get("u_xl", (1, 0)))
        m["fault"] = fmap.astype(np.float32)
        if facies is not None:
            m["facies"] = take_at(facies, np.clip(z + w // 2, 0, nt - 1)).astype(np.float32)
        for s in segs:
            s["throw_ms"] = round(s.pop("throw_samples") * dt, 1)
            s["where"] = where(s.pop("ci"), s.pop("cx"), None, name)
            s.pop("cells", None)

        # traps: with and without sealing faults
        min_relief = max(2.0 * dt, 4.0)
        min_cells = max(30, int(0.25 / cell_km2)) if cell_km2 else 60
        four_way, depth4 = closures_on(z_ms, None, min_relief, min_cells)
        wall = binary_dilation(fmap > 0.35, iterations=1)
        fault_cl, depthf = (closures_on(z_ms, wall, min_relief, min_cells) if wall.any() else ([], depth4))
        m["closure"] = np.maximum(depth4, depthf).astype(np.float32)
        cands = [dict(c, kind="four-way dip closure") for c in four_way]
        mask4 = np.zeros((ni, nx), dtype=bool)
        for c in four_way:
            mask4 |= c["mask"]
        for c in fault_cl:
            if (c["mask"] & mask4).sum() / c["cells"] < 0.5:
                cands.append(dict(c, kind="fault-dependent closure"))

        # stratigraphic candidates: sand-rich (or bright) bodies that structure does not explain
        prop_name = "sand_below" if "sand_below" in m else "rms_below"
        prop = gaussian_filter(m[prop_name], 1.5)
        thr = 0.5 * float(np.median(prop) + np.percentile(prop, 98))      # halfway between background and the best
        if prop_name == "sand_below":
            thr = max(thr, 0.4)
        struct_mask = np.zeros((ni, nx), dtype=bool)
        for c in cands:
            struct_mask |= c["mask"]
        # only the part of a body that lies OUTSIDE the structural closures counts here
        outside = (prop > thr) & ~(binary_dilation(struct_mask, iterations=2) if struct_mask.any() else struct_mask)
        lab, n = label(outside, structure=np.ones((3, 3)))
        for c in range(1, n + 1):
            bm = lab == c
            cells = int(bm.sum())
            if cells < min_cells:
                continue
            zi = np.where(bm, z_ms, np.inf)
            ci, cx = np.unravel_index(np.argmin(zi), z_ms.shape)
            ring = _ring(bm, 3)
            shallow_edge = ring & (z_ms < np.percentile(z_ms[bm], 15))
            pinch = bool(shallow_edge.sum() >= 5 and prop[shallow_edge].mean() < 0.75 * prop[bm].mean())
            cands.append({"mask": bm, "cells": cells, "relief_ms": float(z_ms[bm].max() - z_ms[bm].min()),
                          "crest": (int(ci), int(cx)), "crest_ms": float(z_ms[ci, cx]), "spill_ms": None,
                          "kind": "stratigraphic body", "pinch": pinch, "body_prop": prop_name})

        # wells crossing this horizon, and what their logs say down to the next horizon
        hits_here = []
        for key, wh in well_hits.items():
            gi, gx, tw = wh["gi"], wh["gx"], wh["twt"]
            idx = np.nonzero(wh["ok"])[0]
            ci_ = np.clip(np.rint(gi[idx]).astype(int), 0, ni - 1)
            cx_ = np.clip(np.rint(gx[idx]).astype(int), 0, nx - 1)
            zi = z_ms[ci_, cx_]
            cross = np.nonzero(tw[idx] >= zi)[0]
            if cross.size == 0 or (cross[0] == 0 and tw[idx][0] > zi[0] + 4 * dt):
                continue
            j0 = cross[0]
            hi_, hx_, ht = int(ci_[j0]), int(cx_[j0]), float(tw[idx][j0])
            t_base = float(ms(zn[hi_, hx_]))
            rec = {"well": wh["name"], "key": key, "gi": hi_, "gx": hx_, "twt": round(ht, 1), "base_twt": round(t_base, 1),
                   "logs": wells_mod.interval_summary(key, ht, max(t_base, ht + 2 * dt)),
                   "seismic": {r: round(float(whole[r][hi_, hx_]), 4) for r in ("sand", "ai") if r in whole}}
            wh["hz"][name] = rec
            hits_here.append(rec)

        # ---- score every candidate
        for c in cands:
            mask, (ci, cx) = c["mask"], c["crest"]
            ring = _ring(mask, max(3, np.sqrt(c["cells"]) / 4))
            lid = f"L{len(leads) + 1}"
            loc = where(ci, cx, c["crest_ms"], name)
            # best reservoir zone inside this layer
            if "sand" in win:
                sj = [float(s["sand"][mask].mean()) for s in slabs]
            else:
                sj = [float(s["full"][mask].mean() / (s["full"].mean() + eps)) for s in slabs]
            j = int(np.argmax(sj))
            sl = slabs[j]
            zone = f"{off[j][0]:.0f}-{off[j][1]:.0f} ms below {name}"
            zone_twt = c["crest_ms"] + 0.5 * (off[j][0] + off[j][1])
            zloc = dict(loc, twt=round(zone_twt, 1))
            lead = {"id": lid, "kind": c["kind"], "horizon": name, "where": loc, "area_km2": area(c["cells"]),
                    "cells": c["cells"], "relief_ms": round(c["relief_ms"], 1), "crest_twt_ms": round(c["crest_ms"], 1),
                    "spill_twt_ms": round(c["spill_ms"], 1) if c["spill_ms"] is not None else None,
                    "reservoir_zone": zone, "zone_twt_ms": round(zone_twt, 1),
                    "metrics": {}, "for": [], "against": [], "unknown": []}
            mt = lead["metrics"]
            size = f"{lead['area_km2']} km2" if lead["area_km2"] is not None else f"{c['cells']} grid cells"
            # --- trap
            if c["kind"] == "four-way dip closure":
                trap = 0.2 + 0.5 * min(c["relief_ms"] / 30.0, 1) + 0.3 * min((lead["area_km2"] or c["cells"] / 400) / 4.0, 1)
                lead["for"].append(ev(f"{lid}: {name} time map closes in all directions around IL {loc['il']} / XL {loc['xl']}: "
                                      f"crest {lead['crest_twt_ms']} ms, spill {lead['spill_twt_ms']} ms, relief "
                                      f"{lead['relief_ms']} ms, area {size}.", loc, "trap"))
            elif c["kind"] == "fault-dependent closure":
                trap = 0.12 + 0.4 * min(c["relief_ms"] / 30.0, 1) + 0.25 * min((lead["area_km2"] or c["cells"] / 400) / 4.0, 1)
                lead["for"].append(ev(f"{lid}: {name} closes only if the bounding fault seals (relief {lead['relief_ms']} ms, "
                                      f"area {size}, crest {lead['crest_twt_ms']} ms at IL {loc['il']} / XL {loc['xl']}).", loc, "trap"))
                lead["unknown"].append(ev(f"{lid}: fault seal is not tested by any data here; the trap fails if the fault leaks.",
                                          loc, "risk"))
            else:
                trap = 0.45 if c.get("pinch") else 0.25
                what = "sand probability" if c["body_prop"] == "sand_below" else "amplitude"
                lead["for"].append(ev(f"{lid}: a {size} body of high {what} below {name} that does not follow the "
                                      f"structure, shallowest at IL {loc['il']} / XL {loc['xl']} ({lead['crest_twt_ms']} ms).",
                                      zloc, "trap"))
                if c.get("pinch"):
                    lead["for"].append(ev(f"{lid}: the body dies out in the up-dip direction, which is what an up-dip "
                                          f"pinch-out (stratigraphic trap) looks like.", zloc, "trap"))
                else:
                    lead["against"].append(ev(f"{lid}: the body continues up-dip, so there is no obvious up-dip seal.", zloc, "risk"))
            # --- reservoir
            if "sand" in win:
                s_in, s_bg = sj[j], float(sl["sand"].mean())
                mt["sand_prob_in"], mt["sand_prob_survey_avg"] = round(s_in, 3), round(s_bg, 3)
                res = float(np.clip((s_in - 0.25) / 0.45, 0, 1))
                (lead["for"] if s_in >= 0.45 else lead["against"]).append(ev(
                    f"{lid}: the sandiest zone under the lead is {zone}: sand probability {s_in:.2f} inside the lead "
                    f"(same zone across the survey {s_bg:.2f}).", dict(zloc, vol="sand"), "reservoir"))
            else:
                res = 0.4
                lead["unknown"].append(ev(f"{lid}: no sand-probability volume loaded, reservoir presence is not assessed.", loc, "risk"))
            # --- seal
            if "sand" in win:
                if j == 0:
                    s_ab, sdesc = float(m["sand_above"][mask].mean()), f"the {int(window_ms)} ms above {name}"
                else:
                    s_ab = float(slabs[j - 1]["sand"][mask].mean())
                    sdesc = f"the zone directly above it ({off[j - 1][0]:.0f}-{off[j - 1][1]:.0f} ms below {name})"
                mt["sand_prob_above_zone"] = round(s_ab, 3)
                seal = float(1 - np.clip((s_ab - 0.2) / 0.4, 0, 1))
                (lead["for"] if s_ab < 0.35 else lead["against"]).append(ev(
                    f"{lid}: {sdesc} has sand probability {s_ab:.2f} "
                    f"({'shale-prone, a likely top seal' if s_ab < 0.35 else 'sandy, so top seal is a risk'}).",
                    dict(loc, vol="sand"), "seal"))
            else:
                seal = 0.5
            # --- direct hydrocarbon indicators, all measured in the chosen zone
            dhi = []
            for role, word in (("ai", "P-impedance"), ("vpvs", "Vp/Vs")):
                if role in sl:
                    zval = _zscore_in(sl[role], mask)
                    mt[f"{role}_zscore"] = round(zval, 2)
                    dhi.append(float(np.clip(-zval / 1.5, 0, 1)))
                    if zval < -0.5:
                        lead["for"].append(ev(f"{lid}: {word} in that zone is low inside the lead ({zval:+.1f} standard "
                                              f"deviations vs the rest of the survey), as expected for porous or gas-bearing sand.",
                                              dict(zloc, vol=role), "dhi"))
                    elif zval > 0.5:
                        lead["against"].append(ev(f"{lid}: {word} in that zone is higher than the survey average ({zval:+.1f} sd), "
                                                  f"not what hydrocarbon sand usually shows.", dict(zloc, vol=role), "dhi"))
            if "near" in sl and "far" in sl:
                fn = sl["far"] / (sl["near"] + eps)
                r_in, r_bg = float(np.median(fn[mask])), float(np.median(fn))
                mt["far_over_near_in"], mt["far_over_near_survey"] = round(r_in, 2), round(r_bg, 2)
                rel = r_in / (r_bg + eps)
                dhi.append(float(np.clip((rel - 0.9) / 0.6, 0, 1)))
                if rel > 1.15:
                    lead["for"].append(ev(f"{lid}: amplitude grows from near to far offsets more than elsewhere "
                                          f"(far/near {r_in:.2f} vs {r_bg:.2f} survey median) - a rising AVO response.",
                                          dict(zloc, vol="far"), "dhi"))
            if ring.sum() > 10:
                a_in, a_out = float(sl["full"][mask].mean()), float(sl["full"][ring].mean())
                ratio = a_in / (a_out + eps)
                mt["amp_inside_over_outside"] = round(ratio, 2)
                dhi.append(float(np.clip((ratio - 0.95) / 0.5, 0, 1)))
                if ratio > 1.2 and c["kind"] != "stratigraphic body":
                    lead["for"].append(ev(f"{lid}: amplitude in that zone is {ratio:.1f}x brighter inside the closure than "
                                          f"just outside it, i.e. the bright area fits the structure.", dict(zloc, vol="full"), "dhi"))
            dhi_s = float(np.mean(dhi)) if dhi else 0.3
            # --- wells
            inside = [r for r in hits_here if mask[r["gi"], r["gx"]]]
            if hits_here:
                dkm = [(float(np.hypot((r["gi"] - ci) * (cell[0] if cell else 1), (r["gx"] - cx) * (cell[1] if cell else 1))), r)
                       for r in hits_here]
                dmin, near = min(dkm, key=lambda q: q[0])
                lead["nearest_well"] = {"well": near["well"],
                                        "distance_km" if cell else "distance_cells": round(dmin / (1000.0 if cell else 1.0), 1)}
            if inside:
                lead["wells_inside"] = [r["well"] for r in inside]
                for r in inside:
                    e0 = float(ms(edges[j][r["gi"], r["gx"]]))
                    e1 = float(ms(edges[j + 1][r["gi"], r["gx"]]))
                    lg = wells_mod.interval_summary(r["key"], e0, max(e1, e0 + 2 * dt)) or {}
                    bits = [f"net sand {lg['net_sand_fraction']:.0%}" if "net_sand_fraction" in lg else None,
                            f"porosity {lg['sand_porosity']:.0%} in the sand" if "sand_porosity" in lg else None,
                            f"hydrocarbon flag over {lg['hydrocarbon_fraction']:.0%} of the zone" if "hydrocarbon_fraction" in lg else None]
                    bits = ", ".join(b for b in bits if b) or "no usable logs in this zone"
                    hc = lg.get("hydrocarbon_fraction", 0) or 0
                    lead["for" if hc > 0.05 else "unknown"].append(ev(
                        f"{lid}: already drilled by well {r['well']}; in the zone ({e0:.0f}-{e1:.0f} ms at the well) its logs show {bits}.",
                        where(r["gi"], r["gx"], 0.5 * (e0 + e1), name), "well"))
                    mt.setdefault("well_results", {})[r["well"]] = lg
            else:
                lead["unknown"].append(ev(f"{lid}: no well has tested this lead"
                                          + (f"; the nearest is {lead['nearest_well']['well']}" if lead.get("nearest_well") else "") + ".",
                                          loc, "well"))
            lead["unknown"].append(ev(f"{lid}: hydrocarbon charge and timing cannot be judged from seismic and logs alone.", loc, "risk"))
            if c["kind"] != "stratigraphic body" and j > 0:
                lead["unknown"].append(ev(f"{lid}: the closure is mapped on {name}; the reservoir zone is deeper and is assumed "
                                          f"to be parallel to it.", zloc, "risk"))
            if facies is not None:
                mt["dominant_facies"] = int(np.bincount(m["facies"][mask].astype(int)).argmax()) + 1
            overall = float((max(trap, 0.02) * max(res, 0.02) * max(seal, 0.02)) ** (1 / 3) * 0.7 + dhi_s * 0.3)
            lead["scores"] = {"trap": round(trap, 2), "reservoir": round(res, 2), "seal": round(seal, 2),
                              "hydrocarbon_indicators": round(dhi_s, 2), "overall": round(overall, 2)}
            lead["rating"] = {k: _lvl(v) for k, v in lead["scores"].items()}
            lead["tested"] = bool(inside)
            lead["_h"], lead["_mask"] = h, mask
            leads.append(lead)

        hz_info.append({
            "name": name, "polarity": hz["polarity"],
            "layer_below": f"{name} to {nxt}" if nxt else f"{name} and {3 * int(window_ms)} ms below",
            "twt_min_ms": round(float(m["time"].min()), 1),
            "twt_max_ms": round(float(m["time"].max()), 1), "twt_mean_ms": round(float(m["time"].mean()), 1),
            "shallowest_at": where(*np.unravel_index(np.argmin(z_ms), z_ms.shape), float(z_ms.min()), name),
            "deepest_at": where(*np.unravel_index(np.argmax(z_ms), z_ms.shape), float(z_ms.max()), name),
            "dip_direction": _dip_dir(z_ms, grid),
            "faults": {"count": len(segs), "trends": faults_mod.strike_summary(segs), "largest": segs[:5]},
            "closures": sum(1 for c in cands if c["kind"] != "stratigraphic body"),
            "sand_prob_layer_avg": round(float(whole["sand"].mean()), 3) if "sand" in whole else None,
            "wells": [{"well": r["well"], "twt": r["twt"], "logs_layer": r["logs"], "seismic_layer": r["seismic"]} for r in hits_here],
        })
        for k, v in m.items():
            maps[f"{name}__{k}"] = v

    # ---------------- group stacked closures into structures, rank, keep the best, renumber
    parent = list(range(len(leads)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(leads)):
        for b in range(a + 1, len(leads)):
            ma, mb = leads[a]["_mask"], leads[b]["_mask"]
            inter = (ma & mb).sum()
            if inter and inter / min(ma.sum(), mb.sum()) > 0.5:
                parent[find(a)] = find(b)
    order = sorted(range(len(leads)), key=lambda q: -leads[q]["scores"]["overall"])
    per_group, total_group, keep = {}, {}, []
    for q in order:
        g = find(q)
        total_group[g] = total_group.get(g, 0) + 1
        if per_group.get(g, 0) < 3 and len(keep) < 14:
            per_group[g] = per_group.get(g, 0) + 1
            keep.append(q)
    gname = {}
    for q in keep:
        gname.setdefault(find(q), f"S{len(gname) + 1}")
    out_leads, remap = [], {}
    for n, q in enumerate(keep):
        l = leads[q]
        remap[l["id"]] = f"L{n + 1}"
        lead_map[l.pop("_h")][l.pop("_mask")] = n + 1
        l["structure"] = gname[find(q)]
        l["levels_closed_on_structure"] = total_group[find(q)]
        l["rank"] = n + 1
        out_leads.append(l)
    for l in out_leads:
        l["id"] = remap[l["id"]]
    for l in out_leads:
        l["same_structure_as"] = [o["id"] for o in out_leads if o is not l and o["structure"] == l["structure"]]
    kept = set()
    for l in out_leads:
        for k in ("for", "against", "unknown"):
            kept.update(l[k])
    ev_out = []
    for e in evidence:
        if e["id"] not in kept:
            continue
        lid = e["text"].split(":")[0]
        e["text"] = remap.get(lid, lid) + e["text"][len(lid):]
        ev_out.append(e)
    idmap = {e["id"]: f"E{n + 1}" for n, e in enumerate(ev_out)}
    for e in ev_out:
        e["id"] = idmap[e["id"]]
    for l in out_leads:
        for k in ("for", "against", "unknown"):
            l[k] = [idmap[x] for x in l[k]]

    # ---------------- well calibration of the seismic properties
    calib = _calibration(well_hits)
    fac = _facies_table(facies, seis, semb, vols, grid) if facies is not None else []
    wells_out = []
    for key, wh in well_hits.items():
        wells_out.append({"well": wh["name"], "horizons": {hn: {"twt": r["twt"], "base_twt": r["base_twt"], "logs_layer": r["logs"],
                                                                 "seismic_layer": r["seismic"]} for hn, r in wh["hz"].items()}})
    return ({"horizons": hz_info, "leads": out_leads, "evidence": ev_out, "calibration": calib, "facies": fac,
             "well_ties": wells_out, "window_ms": window_ms}, maps, lead_map)


def _dip_dir(z_ms, grid):
    """Overall down-dip compass direction of a time surface."""
    gi, gx = np.gradient(z_ms)
    u_il, u_xl = grid.get("u_il", (0, 1)), grid.get("u_xl", (1, 0))
    e = float(gi.mean() * u_il[0] + gx.mean() * u_xl[0])
    n = float(gi.mean() * u_il[1] + gx.mean() * u_xl[1])
    if abs(e) + abs(n) < 1e-6:
        return "flat"
    az = np.degrees(np.arctan2(e, n)) % 360
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return f"deepens towards the {names[int((az + 22.5) // 45) % 8]}"


def _calibration(well_hits):
    """How well do the seismic property volumes agree with the logs at the wells?"""
    out = []
    pairs = {"sand": ("net_sand_fraction", "sand probability vs log net-sand fraction"),
             "ai": ("acoustic_impedance", "inverted P-impedance vs log impedance")}
    for role, (lg, label_) in pairs.items():
        xs, ys, names = [], [], set()
        for wh in well_hits.values():
            for r in wh["hz"].values():
                if r["logs"] and lg in r["logs"] and role in r["seismic"]:
                    xs.append(r["logs"][lg])
                    ys.append(r["seismic"][role])
                    names.add(wh["name"])
        if len(xs) >= 4 and np.std(xs) > 0 and np.std(ys) > 0:
            r = float(np.corrcoef(xs, ys)[0, 1])
            verdict = "agrees well" if r > 0.6 else ("agrees loosely" if r > 0.3 else "does not agree")
            out.append({"check": label_, "correlation": round(r, 2), "points": len(xs), "wells": len(names),
                        "verdict": f"seismic {verdict} with the wells"})
        else:
            out.append({"check": label_, "correlation": None, "points": len(xs),
                        "verdict": "not enough well/horizon intersections with usable logs to check"})
    return out


def _facies_table(facies, seis, semb, vols, grid):
    k = int(facies.max()) + 1
    sub = (slice(None, None, 2),) * 3
    f = np.asarray(facies[sub])
    amp = np.abs(np.asarray(seis[sub]))
    sm = np.asarray(semb[sub]) if semb is not None else None
    tt = np.broadcast_to(np.asarray(grid["times"])[::2][None, None, :], f.shape)
    g_amp = float(amp.mean()) + 1e-9
    rows = []
    for c in range(k):
        m = f == c
        if m.sum() == 0:
            continue
        row = {"facies": c + 1, "share": round(float(m.mean()), 3), "amplitude_vs_avg": round(float(amp[m].mean() / g_amp), 2),
               "typical_twt_ms": round(float(np.median(tt[m])), 0)}
        if sm is not None:
            row["continuity"] = round(float(sm[m].mean()), 2)
        for role in ("sand", "ai", "vpvs"):
            if role in vols:
                v = np.asarray(vols[role][sub])
                row[f"{role}_mean"] = round(float(v[m].mean()), 3)
                row[f"{role}_vs_avg_sd"] = round(float((v[m].mean() - v.mean()) / (v.std() + 1e-9)), 2)
        words = ["bright" if row["amplitude_vs_avg"] > 1.25 else ("dim" if row["amplitude_vs_avg"] < 0.75 else "moderate-amplitude")]
        if sm is not None:
            words.append("continuous" if row["continuity"] > float(sm.mean()) + 0.03 else
                         ("broken / chaotic" if row["continuity"] < float(sm.mean()) - 0.03 else "semi-continuous"))
        if "sand_mean" in row:
            words.append("sand-prone" if row["sand_vs_avg_sd"] > 0.3 else ("shale-prone" if row["sand_vs_avg_sd"] < -0.3 else "mixed lithology"))
        row["description"] = ", ".join(words)
        rows.append(row)
    return rows
