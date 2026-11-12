"""Well data: well list (xlsx), deviation surveys, checkshots and LAS logs.

The exact file layouts are not known in advance, so every reader is tolerant:
columns are recognised from their header names, with sensible guesses when a
file has no header. Anything that could not be used is reported, never hidden.
"""
import re
from pathlib import Path

import numpy as np

from . import config
from .config import read_json, write_json

WELLS_JSON = config.WORKSPACE / "wells.json"
WELLS_NPZ = config.WORKSPACE / "wells_logs.npz"


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _tok(s):
    return re.sub(r"[\(\[].*?[\)\]]", "", str(s)).strip().lower()


def _isnum(s):
    try:
        float(s)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ generic table
def read_table(path):
    """Return (header tokens or None, 2-D float array, key/value pairs found above the table)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    rows, kv = [], {}
    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        toks = [t for t in re.split(r"[,\t; ]+", raw.lstrip("#!% ")) if t != ""]
        numeric = len(toks) >= 2 and all(_isnum(t) for t in toks)
        rows.append((numeric, toks, raw))
        if not numeric:
            for m in re.finditer(r"([A-Za-z][A-Za-z _\.]{0,24}?)\s*[:=]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", raw):
                kv[_tok(m.group(1))] = float(m.group(2))
    counts = {}
    for numeric, toks, _ in rows:
        if numeric:
            counts[len(toks)] = counts.get(len(toks), 0) + 1
    if not counts:
        return None, np.zeros((0, 0)), kv
    ncol = max(counts, key=counts.get)
    data = np.array([[float(t) for t in toks] for numeric, toks, _ in rows if numeric and len(toks) == ncol])
    # the header is the non-numeric line above the table that names the most known columns;
    # it may be split by tabs, commas, or runs of spaces ("Measured Depth   Inclination ...")
    unit = r"[\(\[]?(m|ft|ms|s|sec|deg|degrees|us/ft|us/m|m/s)[\)\]]?"
    header, best = None, 0
    for numeric, toks, raw in rows:
        if numeric:
            break
        line = raw.lstrip("#!% ")
        for parts in (line.split("	"), line.split(","), line.split(";"), re.split(r"\s{2,}", line), line.split()):
            parts = [q.strip() for q in parts if q.strip() != ""]
            if len(parts) > ncol:
                parts = [q for q in parts if not re.fullmatch(unit, q.lower())]
            if len(parts) != ncol:
                continue
            cand = [_tok(q).replace(" ", "_") for q in parts]
            score = sum(1 for c in cand if any(re.match(pat, c) for pat in _COLS.values()))
            if score > best:
                header, best = cand, score
            break
    return header, data, kv


_COLS = {
    "md": r"^(md|meas.*|mdepth|depth|dept|md_?kb|along.*)$",
    "tvdss": r"^(tvdss|tvd_?ss|subsea.*|z|elev.*|tvdmsl|ss)$",
    "tvd": r"^(tvd|tvd_?kb|tvdkb|vertical.*|true_?vert.*)$",
    "x": r"^(x|east.*|utm_?x|x_?coord.*|xcoord|e)$",
    "y": r"^(y|north.*|utm_?y|y_?coord.*|ycoord|n)$",
    "dx": r"^(dx|x_?off.*|ew|e_?w|deltax|delta_?e.*|dex)$",
    "dy": r"^(dy|y_?off.*|ns|n_?s|deltay|delta_?n.*|dny)$",
    "inc": r"^(inc.*|dev.*|drift|angle)$",
    "az": r"^(az.*|bearing|dir.*)$",
    "twt": r"^(twt.*|time.*|t|two.*|owt.*|one.*|tt)$",
}


def _find(header, key):
    if not header:
        return None
    for i, h in enumerate(header):
        if re.match(_COLS[key], h):
            return i
    return None


# ------------------------------------------------------------------ deviation
def min_curvature(md, inc_deg, az_deg):
    inc, az = np.radians(inc_deg), np.radians(az_deg)
    dx, dy, dz = np.zeros_like(md), np.zeros_like(md), np.zeros_like(md)
    for k in range(1, len(md)):
        dmd = md[k] - md[k - 1]
        cosd = np.cos(inc[k] - inc[k - 1]) - np.sin(inc[k - 1]) * np.sin(inc[k]) * (1 - np.cos(az[k] - az[k - 1]))
        dl = np.arccos(np.clip(cosd, -1, 1))
        rf = 1.0 if dl < 1e-6 else 2 / dl * np.tan(dl / 2)
        dy[k] = dy[k - 1] + dmd / 2 * (np.sin(inc[k - 1]) * np.cos(az[k - 1]) + np.sin(inc[k]) * np.cos(az[k])) * rf
        dx[k] = dx[k - 1] + dmd / 2 * (np.sin(inc[k - 1]) * np.sin(az[k - 1]) + np.sin(inc[k]) * np.sin(az[k])) * rf
        dz[k] = dz[k - 1] + dmd / 2 * (np.cos(inc[k - 1]) + np.cos(inc[k])) * rf
    return dx, dy, dz


def read_deviation(path, x0=None, y0=None, kb=0.0, notes=None):
    """Return dict(md, x, y, tvdss) or None."""
    notes = notes if notes is not None else []
    header, d, kv = read_table(path)
    if d.size == 0:
        notes.append(f"{Path(path).name}: no numeric table found")
        return None
    n = d.shape[1]
    col = {k: _find(header, k) for k in ("md", "tvd", "tvdss", "x", "y", "dx", "dy", "inc", "az")}
    if header is None:
        big = [i for i in range(n) if np.nanmedian(np.abs(d[:, i])) > 2e4]
        mono = [i for i in range(n) if i not in big and np.all(np.diff(d[:, i]) >= -1e-6) and np.ptp(d[:, i]) > 0]
        if len(big) >= 2:
            col["x"], col["y"] = big[0], big[1]
        if mono:
            col["md"] = mono[0]
            if len(mono) > 1:      # a depth column that starts below zero is measured from sea level
                col["tvdss" if d[0, mono[1]] < 0 else "tvd"] = mono[1]
        rest = [i for i in range(n) if i not in big and i not in mono]
        if col["x"] is None and len(rest) >= 2 and np.nanmax(np.abs(d[:, rest[0]])) <= 180:
            col["inc"], col["az"] = rest[0], rest[1]
        notes.append(f"{Path(path).name}: no column names, guessed the layout")
    if x0 is None:
        x0 = kv.get("x", kv.get("easting", kv.get("surface x")))
    if y0 is None:
        y0 = kv.get("y", kv.get("northing", kv.get("surface y")))

    md = d[:, col["md"]] if col["md"] is not None else None
    z = None
    if col["tvdss"] is not None:
        z = d[:, col["tvdss"]].copy()
        if np.nanmedian(z) < 0:
            z = -z
    elif col["tvd"] is not None:
        z = d[:, col["tvd"]] - (kb or 0.0)
    if col["x"] is not None and col["y"] is not None:
        x, y = d[:, col["x"]], d[:, col["y"]]
    elif col["dx"] is not None and col["dy"] is not None and x0 is not None:
        x, y = x0 + d[:, col["dx"]], y0 + d[:, col["dy"]]
    elif col["inc"] is not None and col["az"] is not None and md is not None and x0 is not None:
        dx, dy, dz = min_curvature(md, d[:, col["inc"]], d[:, col["az"]])
        x, y = x0 + dx, y0 + dy
        if z is None:
            z = dz - (kb or 0.0)
    elif x0 is not None:
        x, y = np.full(len(d), x0), np.full(len(d), y0)
        notes.append(f"{Path(path).name}: no positions in the survey, well treated as vertical")
    else:
        notes.append(f"{Path(path).name}: could not work out the well position")
        return None
    if md is None:
        md = (z + (kb or 0.0)) if z is not None else None
    if z is None:
        z = md - (kb or 0.0) if md is not None else None
    if md is None or z is None:
        return None
    order = np.argsort(md)
    return {"md": md[order], "x": x[order], "y": y[order], "tvdss": z[order]}


# ------------------------------------------------------------------ checkshot
def read_checkshot(path, notes=None):
    """Return dict(depth, twt_ms, depth_kind in {md,tvd,tvdss}) or None."""
    notes = notes if notes is not None else []
    header, d, _ = read_table(path)
    if d.size == 0 or d.shape[1] < 2:
        notes.append(f"{Path(path).name}: no depth/time table found")
        return None
    ct = _find(header, "twt")
    kind, cd = None, None
    for k in ("tvdss", "tvd", "md"):
        c = _find(header, k)
        if c is not None and c != ct:
            kind, cd = k, c
            break
    if ct is None or cd is None:
        ncol = d.shape[1]
        small = [i for i in range(ncol) if np.nanmax(np.abs(d[:, i])) < 20]
        ct = small[0] if small else ncol - 1
        cd = [i for i in range(ncol) if i != ct][0]
        kind = kind or "tvdss"
        notes.append(f"{Path(path).name}: no column names, assumed depth then time")
    depth, t = np.abs(d[:, cd]), np.abs(d[:, ct])
    if np.nanmax(t) < 20:
        t = t * 1000.0
    if header and re.match(r"^(owt|one)", header[ct] or ""):
        t = t * 2.0
    order = np.argsort(depth)
    depth, t = depth[order], t[order]
    keep = np.concatenate([[True], np.diff(depth) > 0])
    return {"depth": depth[keep], "twt": t[keep], "kind": kind}


# ------------------------------------------------------------------ well list
def read_well_list(path, notes=None):
    """Return {normalised well name: {name, x, y, kb, td, info{...}}} from the spreadsheet."""
    notes = notes if notes is not None else []
    out = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as e:
        notes.append(f"well list could not be opened: {e}")
        return out
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(c is not None and str(c).strip() != "" for c in r)]
        if len(rows) < 2:
            continue
        hi = 0
        for i, r in enumerate(rows[:15]):
            if sum(isinstance(c, str) and c.strip() != "" for c in r) >= 3:
                hi = i
                break
        hdr = [(_tok(c) if c is not None else "") for c in rows[hi]]

        def col(pat):
            for i, h in enumerate(hdr):
                if re.search(pat, h):
                    return i
            return None

        cn = col(r"well|name|uwi|borehole") or 0
        cx = col(r"^x$|east|x[ _]?coord|surface[ _]?x|^utm[ _]?x|\bx\b")
        cy = col(r"^y$|north|y[ _]?coord|surface[ _]?y|^utm[ _]?y|\by\b")
        ck = col(r"\bkb\b|kelly|\brt\b|rotary|datum|elev")
        ctd = col(r"\btd\b|total[ _]?depth")
        for r in rows[hi + 1:]:
            if cn >= len(r) or r[cn] is None:
                continue
            name = str(r[cn]).strip()
            rec = {"name": name, "x": None, "y": None, "kb": None, "td": None, "info": {}}
            for key, c in (("x", cx), ("y", cy), ("kb", ck), ("td", ctd)):
                if c is not None and c < len(r) and _isnum(r[c]):
                    rec[key] = float(r[c])
            for i, h in enumerate(rows[hi]):
                if i < len(r) and r[i] is not None and i != cn and h is not None:
                    rec["info"][str(h).strip()] = r[i] if isinstance(r[i], (int, float)) else str(r[i])
            out[norm(name)] = rec
    if not out:
        notes.append("well list: no wells recognised in the spreadsheet")
    return out


# ------------------------------------------------------------------ LAS logs
_CURVES = {
    "gr": r"^(gr|gam|sgr|cgr|hgr|ecgr|hsgr|gr_.*)$",
    "rhob": r"^(rhob|rhoz|den|zden|dens|rho|rhom|density)$",
    "nphi": r"^(nphi|tnph|neu|cnc|nphz|npor|cn|neutron)$",
    "dt": r"^(dt|dtc|dtco|ac|dt4p|sonic|dtp|dtcomp|dt24)$",
    "dts": r"^(dts|dtsm|dt4s|dtshear)$",
    "rt": r"^(rt|rdeep|res_?d.*|ild|lld|rd|at90|ah90|rlld|hdrs|rdep|lls?d|rt_.*|resistivity|res)$",
    "vsh": r"^(vsh|vcl|vshale|vclay|vsh_.*|vcl_.*)$",
    "phi": r"^(phie|phit|por|phi|porosity|phie_.*|phit_.*)$",
    "sw": r"^(sw|swe|swt|sw_.*)$",
    "vp": r"^(vp|pvel|velp)$",
    "vs": r"^(vs|svel|vels)$",
    "ai": r"^(ai|pimp|zp|ip|p_?imp.*)$",
}


def read_las(path, notes=None):
    """Return dict(md, curves{canonical: array}, raw_names[...]) or None."""
    notes = notes if notes is not None else []
    try:
        import lasio
        las = lasio.read(str(path), ignore_header_errors=True)
    except Exception as e:
        notes.append(f"{Path(path).name}: LAS could not be read ({e})")
        return None
    md = np.asarray(las.index, dtype=np.float64)
    unit = str(getattr(las.curves[0], "unit", "") or "").lower()
    if unit.startswith("f"):
        md = md * 0.3048
    curves, raw, units = {}, [], {}
    for c in las.curves[1:]:
        mn = str(c.mnemonic).split(":")[0].strip().lower()
        raw.append(str(c.mnemonic))
        for key, pat in _CURVES.items():
            if key not in curves and re.match(pat, mn):
                v = np.asarray(c.data, dtype=np.float64)
                v[np.abs(v + 999.25) < 0.01] = np.nan
                if np.isfinite(v).sum() > 10:
                    curves[key] = v
                    units[key] = str(c.unit or "").lower()
                break
    if md[0] > md[-1]:
        md = md[::-1]
        curves = {k: v[::-1] for k, v in curves.items()}
    return {"md": md, "curves": curves, "raw_names": raw, "units": units}


def derive_logs(log):
    """Add shale volume, porosity, sand / hydrocarbon flags and acoustic impedance where possible."""
    c = log["curves"]
    made = []
    if "vsh" in c:
        v = c["vsh"].copy()
        if np.nanmax(v) > 1.5:
            v = v / 100.0
        c["vsh"] = np.clip(v, 0, 1)
    elif "gr" in c:
        lo, hi = np.nanpercentile(c["gr"], [5, 95])
        c["vsh"] = np.clip((c["gr"] - lo) / max(hi - lo, 1e-6), 0, 1)
        made.append("shale volume from gamma ray")
    if "phi" in c:
        p = c["phi"].copy()
        if np.nanmax(p) > 1.5:
            p = p / 100.0
        c["phi"] = np.clip(p, 0, 0.6)
    elif "rhob" in c:
        rho = c["rhob"] / (1000.0 if np.nanmedian(c["rhob"]) > 100 else 1.0)
        c["phi"] = np.clip((2.65 - rho) / 1.65, 0, 0.6)
        made.append("porosity from density")
    if "sw" in c and np.nanmax(c["sw"]) > 1.5:
        c["sw"] = c["sw"] / 100.0
    if "vsh" in c:
        c["sand"] = (c["vsh"] < 0.4).astype(float)
        c["sand"][~np.isfinite(c["vsh"])] = np.nan
        if "sw" in c:
            c["hc"] = ((c["sand"] > 0.5) & (c["sw"] < 0.6)).astype(float)
            made.append("hydrocarbon flag from water saturation")
        elif "rt" in c:
            shale_rt = np.nanmedian(c["rt"][c["vsh"] > 0.6]) if np.any(c["vsh"] > 0.6) else np.nanmedian(c["rt"])
            c["hc"] = ((c["sand"] > 0.5) & (c["rt"] > 3.0 * shale_rt)).astype(float)
            made.append("hydrocarbon flag from resistivity (3x the shale baseline)")
    if "vp" not in c and "dt" in c:
        dt = c["dt"]
        per_m = "/m" in log.get("units", {}).get("dt", "") or np.nanmedian(dt) > 250
        c["vp"] = (1e6 / dt) if per_m else (304800.0 / dt)
    if "ai" not in c and "vp" in c and "rhob" in c:
        rho = c["rhob"] / (1000.0 if np.nanmedian(c["rhob"]) > 100 else 1.0)
        c["ai"] = c["vp"] * rho
    log["derived"] = made
    return log


# ------------------------------------------------------------------ putting it together
def find_files(data_dir):
    """Sort the files under a folder into well list / LAS logs / deviation surveys / checkshots.

    A file is recognised by its own name or, failing that, by the name of the folder it sits in
    (so 'deviations/m-1.txt' works as well as 'm-1_Deviation.txt').
    """
    out = {"list": [], "las": [], "dev": [], "chk": []}
    root = Path(data_dir)
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith((".", "~")):
            continue
        low, par, ext = p.name.lower(), p.parent.name.lower(), p.suffix.lower()
        if ext in (".xlsx", ".xlsm"):
            out["list"].append(p)
        elif ext == ".las":
            out["las"].append(p)
        elif ext in (".txt", ".csv", ".dev", ".dat", ".asc"):
            if re.search(r"deviat|survey|traj", low) or re.search(r"deviat|survey|traj", par):
                out["dev"].append(p)
            elif re.search(r"check|\btd\b|timedepth|time_depth", low) or re.search(r"check|time_?depth", par):
                out["chk"].append(p)
    return out


def has_files(data_dir):
    """Whether a folder contains recognised well files, without reading their contents."""
    return any(find_files(data_dir).values())


def guess_folder(data_dir):
    """Find well data inside the selection or in a named neighbouring folder.

    Seismic is often selected as challenge/stack while the wells live in
    challenge/well_data. Never suggest a seismic-only folder as the well input.
    """
    if not data_dir:
        return ""
    p = Path(data_dir).expanduser()
    if not p.is_dir():
        return ""
    names = ("well_data", "wells", "well")
    candidates = [p / name for name in names] + [p] + [p.parent / name for name in names]
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir() and has_files(candidate):
            return str(candidate)
    return ""


def _file_key(p):
    s = re.sub(r"(?i)[_\- ]?(deviation|checkshot|survey|traj\w*|composite\w*|interp\w*)", "", Path(p).stem)
    return norm(s)


def scan(data_dir):
    """Quick look at a folder for the import dialog: how many well files of each kind, and which wells."""
    f = find_files(data_dir)
    names = {}
    for p in f["las"] + f["dev"] + f["chk"]:
        names.setdefault(_file_key(p), Path(p).stem.split("_")[0])
    return {"path": str(data_dir), "las": len(f["las"]), "deviation": len(f["dev"]), "checkshot": len(f["chk"]),
            "well_list": len(f["list"]), "wells": sorted(names.values())}


def td_time(z, depth, twt):
    """Depth below sea level -> two-way time through a checkshot table.

    Outside the table the last (or first) interval velocity is carried on, instead of freezing the time.
    Returns (times, number of samples that had to be extrapolated below the table).
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.interp(z, depth, twt)
    below = z > depth[-1]
    if len(depth) >= 2:
        v1 = (twt[-1] - twt[-2]) / max(depth[-1] - depth[-2], 1e-6)
        out[below] = twt[-1] + (z[below] - depth[-1]) * v1
        above = z < depth[0]
        v0 = (twt[1] - twt[0]) / max(depth[1] - depth[0], 1e-6)
        out[above] = np.maximum(twt[0] + (z[above] - depth[0]) * v0, 0.0)
    return out, int(below.sum())


def load_wells(data_dir, ref=None, log=print):
    """Read every well file under data_dir. Saves wells.json + wells_logs.npz and returns the summary."""
    data_dir = Path(data_dir)
    notes = []
    found = find_files(data_dir)
    xlsx, las, dev, chk = found["list"], found["las"], found["dev"], found["chk"]
    key = _file_key

    wl = {}
    for p in xlsx:
        wl.update(read_well_list(p, notes))
    names = {}
    for p in las + dev + chk:
        names.setdefault(key(p), p.stem.split("_")[0])
    for k, rec in wl.items():
        names.setdefault(k, rec["name"])
    if not names and any(found.values()):
        raise ValueError("Well files were found, but no well names could be read. "
                         + "; ".join(notes))

    # 1. positions + paths
    wells = {}
    for k, disp in sorted(names.items()):
        rec = wl.get(k, {})
        w = {"name": rec.get("name", disp), "key": k, "x": rec.get("x"), "y": rec.get("y"),
             "kb": rec.get("kb") or 0.0, "td": rec.get("td"), "info": rec.get("info", {}),
             "has": {"deviation": False, "checkshot": False, "logs": False}, "notes": []}
        dv = next((p for p in dev if key(p) == k), None)
        path = read_deviation(dv, w["x"], w["y"], w["kb"], w["notes"]) if dv else None
        if path is None and w["x"] is not None:
            depth = np.array([0.0, w["td"] or 4000.0])
            path = {"md": depth, "x": np.full(2, w["x"]), "y": np.full(2, w["y"]), "tvdss": depth - w["kb"]}
            w["notes"].append("no deviation survey: treated as a vertical well")
        elif path is not None:
            w["has"]["deviation"] = True
            if w["x"] is None:
                w["x"], w["y"] = float(path["x"][0]), float(path["y"][0])
        w["_path"] = path
        wells[k] = w

    # 2. time-depth functions (depth below sea level -> two-way time)
    td = {}
    for p in chk:
        k = key(p)
        cs = read_checkshot(p, notes)
        w = wells.get(k)
        if cs is None or w is None:
            continue
        depth = cs["depth"]
        if cs["kind"] == "md" and w["_path"] is not None:
            depth = np.interp(depth, w["_path"]["md"], w["_path"]["tvdss"])
        elif cs["kind"] == "tvd":
            depth = depth - w["kb"]
        order = np.argsort(depth)
        td[k] = (depth[order], cs["twt"][order])
        w["has"]["checkshot"] = True
    if not td:
        notes.append("no usable checkshot: wells cannot be placed in two-way time, "
                     "so well-to-seismic comparisons are skipped")

    # 3. logs + conversion to the seismic grid
    arrays = {}
    for k, w in wells.items():
        path = w["_path"]
        lp = next((p for p in las if key(p) == k), None)
        lg = read_las(lp, w["notes"]) if lp else None
        if lg:
            lg = derive_logs(lg)
            w["has"]["logs"] = True
            w["curves"] = sorted(lg["curves"].keys())
            w["raw_curves"] = lg["raw_names"]
            w["derived"] = lg["derived"]
        if path is None:
            w["notes"].append("no position known for this well")
            continue
        # which checkshot to use
        if k in td:
            src, w["td_source"] = k, "own checkshot"
        elif td:
            src = min(td, key=lambda q: (wells[q]["x"] - w["x"]) ** 2 + (wells[q]["y"] - w["y"]) ** 2
                      if wells[q]["x"] is not None and w["x"] is not None else 1e30)
            dist = np.hypot(wells[src]["x"] - w["x"], wells[src]["y"] - w["y"]) if w["x"] is not None else float("nan")
            w["td_source"] = f"checkshot borrowed from {wells[src]['name']} ({dist / 1000:.1f} km away)"
            w["td_borrow_km"] = float(dist / 1000)
        else:
            src = None
        md = lg["md"] if lg else path["md"]
        tvdss = np.interp(md, path["md"], path["tvdss"])
        x = np.interp(md, path["md"], path["x"])
        y = np.interp(md, path["md"], path["y"])
        if src:
            twt, n_below = td_time(tvdss, td[src][0], td[src][1])
            if n_below:
                w["notes"].append(f"the logs go {float(np.nanmax(tvdss) - td[src][0][-1]):.0f} m below the deepest "
                                  f"checkshot point; times there are extrapolated with the last interval velocity")
        else:
            twt = np.full(len(md), np.nan)
        arrays[f"{k}__md"], arrays[f"{k}__tvdss"], arrays[f"{k}__twt"] = md, tvdss, twt
        arrays[f"{k}__x"], arrays[f"{k}__y"] = x, y
        if ref is not None and ref.meta.get("affine"):
            il, xl = ref.ilxl(x, y)
            arrays[f"{k}__il"], arrays[f"{k}__xl"] = il, xl
            w["il"], w["xl"] = float(il[0]), float(xl[0])
            inside = (il >= ref.ilines.min()) & (il <= ref.ilines.max()) & (xl >= ref.xlines.min()) & (xl <= ref.xlines.max())
            w["inside_survey"] = bool(inside.any())
            # light path for drawing
            step = max(1, len(md) // 300)
            w["path"] = [[float(a), float(b), float(c)] for a, b, c in zip(il[::step], xl[::step], twt[::step])]
        if lg:
            for cn, v in lg["curves"].items():
                arrays[f"{k}__c_{cn}"] = v
        w["twt_range"] = [float(np.nanmin(twt)), float(np.nanmax(twt))] if np.isfinite(twt).any() else None
        w["md_range"] = [float(md[0]), float(md[-1])]

    for w in wells.values():
        w.pop("_path", None)
    np.savez_compressed(WELLS_NPZ, **arrays)
    out = {"wells": list(wells.values()), "notes": notes, "data_dir": str(data_dir),
           "counts": {"wells": len(wells), "las": len(las), "deviation": len(dev), "checkshot": len(chk)}}
    write_json(WELLS_JSON, out)
    log(f"Wells: {len(wells)} found, {sum(w['has']['logs'] for w in wells.values())} with logs, "
        f"{sum(w['has']['checkshot'] for w in wells.values())} with checkshots")
    return out


def wells_summary():
    return read_json(WELLS_JSON, {"wells": [], "notes": []})


_npz_cache = {}


def well_arrays(key):
    """All stored arrays of one well as a dict (md, twt, il, xl, c_<curve>...)."""
    if not WELLS_NPZ.exists():
        return {}
    mt = WELLS_NPZ.stat().st_mtime
    if _npz_cache.get("mt") != mt:
        z = np.load(WELLS_NPZ)
        _npz_cache.clear()
        _npz_cache.update({"mt": mt, "data": {n: z[n] for n in z.files}})
    pre = f"{key}__"
    return {n[len(pre):]: v for n, v in _npz_cache["data"].items() if n.startswith(pre)}


def log_zones(key, min_thick=2.0):
    """Blocky reading of one well's logs: sand and shale intervals, pay, and the main boundaries.

    Sand = shale volume below 0.4 (see derive_logs); pay = sand with the hydrocarbon flag over half of it.
    Returns None if the well has no usable shale-volume / gamma-ray curve.
    """
    from scipy.ndimage import median_filter
    a = well_arrays(key)
    if "c_sand" not in a or "md" not in a:
        return None
    md, sand = a["md"], a["c_sand"]
    ok = np.isfinite(sand)
    if ok.sum() < 20:
        return None
    first, last = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
    dz = float(np.nanmedian(np.diff(md))) or 0.15
    win = max(3, int(round(1.0 / max(dz, 1e-3))) | 1)
    s = median_filter(np.where(ok, sand, 0.0), size=win) > 0.5

    def runs(flag):
        e = np.flatnonzero(np.diff(flag.astype(np.int8)) != 0) + 1
        st, en = np.r_[first, e[(e > first) & (e < last)]], np.r_[e[(e > first) & (e < last)], last]
        return [(int(p), int(q)) for p, q in zip(st, en) if q > p]

    for _ in range(3):                                  # absorb beds thinner than min_thick into their neighbours
        changed = False
        for p, q in runs(s):
            if md[q - 1] - md[p] < min_thick and p > first and q < last:
                s[p:q] = ~s[p]
                changed = True
        if not changed:
            break
    twt = a.get("twt", np.full(len(md), np.nan))
    tv = a.get("tvdss", np.full(len(md), np.nan))

    def avg(c, p, q):
        v = a.get(f"c_{c}")
        return None if v is None or not np.isfinite(v[p:q]).any() else round(float(np.nanmean(v[p:q])), 4)

    def t(v):
        return None if not np.isfinite(v) else round(float(v), 1)

    sands, bounds, net, pay = [], [], 0.0, 0.0
    for p, q in runs(s):
        if not s[p]:
            continue
        th = float(md[q - 1] - md[p])
        hc = avg("hc", p, q)
        z = {"top_md": round(float(md[p]), 1), "base_md": round(float(md[q - 1]), 1), "thick_m": round(th, 1),
             "top_tvdss": t(tv[p]), "base_tvdss": t(tv[q - 1]), "top_twt": t(twt[p]), "base_twt": t(twt[q - 1]),
             "vsh": avg("vsh", p, q), "phi": avg("phi", p, q), "rt": avg("rt", p, q), "hc_frac": hc,
             "pay": bool(hc is not None and hc > 0.5)}
        sands.append(z)
        net += th
        pay += th if z["pay"] else 0.0
        if th >= 5.0:
            bounds.append({"md": z["top_md"], "twt": z["top_twt"], "kind": "top of sand (shale over sand)", "thick_m": z["thick_m"]})
            bounds.append({"md": z["base_md"], "twt": z["base_twt"], "kind": "base of sand (sand over shale)", "thick_m": z["thick_m"]})
    gross = float(md[last - 1] - md[first])
    for n, z in enumerate(sorted(sands, key=lambda q: q["top_md"])):
        z["id"] = f"S{n + 1}"
    big = sorted(sands, key=lambda q: -q["thick_m"])
    return {"summary": {"logged_from_md": round(float(md[first]), 1), "logged_to_md": round(float(md[last - 1]), 1),
                        "gross_m": round(gross, 1), "net_sand_m": round(net, 1),
                        "net_to_gross": round(net / gross, 3) if gross > 0 else None, "net_pay_m": round(pay, 1),
                        "sand_count": len(sands), "thickest_sand": big[0]["id"] if big else None,
                        "has_fluid_flag": "c_hc" in a},
            "sands": sorted(big[:40], key=lambda q: q["top_md"]), "boundaries": bounds}


def interval_summary(key, t_top, t_base):
    """Log averages of one well between two two-way times (ms)."""
    a = well_arrays(key)
    if "twt" not in a or not np.isfinite(a["twt"]).any():
        return None
    m = (a["twt"] >= t_top) & (a["twt"] < t_base)
    if m.sum() < 3:
        return None
    out = {"samples": int(m.sum()), "thickness_m": float(np.nanmax(a["tvdss"][m]) - np.nanmin(a["tvdss"][m]))}

    def mean(c):
        v = a.get(f"c_{c}")
        if v is None or not np.isfinite(v[m]).any():
            return None
        return float(np.nanmean(v[m]))

    for c, label in (("sand", "net_sand_fraction"), ("vsh", "shale_volume"), ("phi", "porosity"),
                     ("hc", "hydrocarbon_fraction"), ("rt", "resistivity_ohmm"), ("ai", "acoustic_impedance"),
                     ("gr", "gamma_ray")):
        v = mean(c)
        if v is not None:
            out[label] = round(v, 4)
    if "c_phi" in a and "c_sand" in a:
        s = m & (a["c_sand"] > 0.5)
        if s.sum() > 2 and np.isfinite(a["c_phi"][s]).any():
            out["sand_porosity"] = round(float(np.nanmean(a["c_phi"][s])), 4)
    return out
