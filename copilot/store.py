"""SEG-Y -> AI-ready store, and fast access to the converted volumes.

Conversion order:
  1. SDIP (github.com/zahidaramai/sdip): lossless SEG-Y -> MDIO / Zarr v3.
  2. If SDIP is not installed or refuses the file: read with segyio into a plain
     NumPy store. The rest of the app works the same with either.
"""
import math
import os
import re
import shutil
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import config
from .config import read_json, write_json

# ------------------------------------------------------------------ naming / roles
_ROLES = [  # (regex on lower-case file name, role, human label)
    (r"ultra[_\- ]?far", "ufar", "Ultra-far stack"),
    (r"near", "near", "Near stack"),
    (r"mid", "mid", "Mid stack"),
    (r"far", "far", "Far stack"),
    (r"full", "full", "Full stack"),
    (r"p[_\- ]?imp|acoustic[_\- ]?imp|\bai\b", "ai", "P-impedance"),
    (r"s[_\- ]?imp|shear[_\- ]?imp", "si", "S-impedance"),
    (r"vp[_\- ]?vs", "vpvs", "Vp/Vs"),
    (r"sand[_\- ]?prob|sandprob|prob", "sand", "Sand probability"),
]
ROLE_ORDER = ["full", "near", "mid", "far", "ufar", "ai", "si", "vpvs", "sand", "other"]


def guess_role(filename):
    low = Path(filename).name.lower()
    for pat, role, label in _ROLES:
        if re.search(pat, low):
            return role, label
    stem = Path(filename).stem
    return "other", stem.replace("_", " ")


def clean_name(filename):
    stem = Path(filename).stem
    stem = re.sub(r"(?i)challenge\d+_?", "", stem)
    stem = re.sub(r"(?i)_?cropped|_?output|inversion_?", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem or Path(filename).stem


# ------------------------------------------------------------------ SEG-Y helpers
_BYTE_PAIRS = [(189, 193), (9, 21), (5, 21), (17, 13), (9, 13), (21, 9), (181, 185)]


def _open_segy(path):
    """Open with regular geometry, trying the common inline/crossline header positions."""
    import segyio
    last = None
    for il, xl in _BYTE_PAIRS:
        try:
            f = segyio.open(str(path), "r", iline=il, xline=xl, strict=True)
            if len(f.ilines) > 1 and len(f.xlines) > 1 and len(f.ilines) * len(f.xlines) == f.tracecount:
                return f, il, xl
            f.close()
        except Exception as e:  # noqa: PERF203
            last = e
    raise RuntimeError(f"no regular inline/crossline geometry found in {path}: {last}")


def _scaled(vals, scal):
    vals = np.asarray(vals, dtype=np.float64)
    scal = np.asarray(scal, dtype=np.float64)
    out = vals.copy()
    neg = scal < 0
    pos = scal > 0
    out[neg] = vals[neg] / np.abs(scal[neg])
    out[pos] = vals[pos] * scal[pos]
    return out


def segy_affine(path, il_byte=189, xl_byte=193):
    """Least-squares map (inline number, crossline number) -> (X, Y) from trace headers."""
    import segyio
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n = f.tracecount
        step = max(1, n // 5000)
        while step > 1 and any(step % p == 0 for p in (2, 3, 5, 7)):
            step += 1      # avoid sampling the same crossline over and over
        il = np.asarray(f.attributes(il_byte)[::step], dtype=np.float64)
        xl = np.asarray(f.attributes(xl_byte)[::step], dtype=np.float64)
        scal = np.asarray(f.attributes(71)[::step], dtype=np.float64)
        best = None
        for bx, by in ((181, 185), (73, 77), (81, 85)):
            x = _scaled(f.attributes(bx)[::step], scal)
            y = _scaled(f.attributes(by)[::step], scal)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                best = (x, y)
                break
        if best is None:
            return None
        x, y = best
    A = np.stack([np.ones_like(il), il, xl], axis=1)
    if np.linalg.matrix_rank(A) < 3:
        return None
    cx, *_ = np.linalg.lstsq(A, x, rcond=None)
    cy, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(np.mean((A @ cx - x) ** 2 + (A @ cy - y) ** 2)))
    return {"x0": cx[0], "xi": cx[1], "xx": cx[2], "y0": cy[0], "yi": cy[1], "yx": cy[2], "rms_m": resid}


def _sdip_bin():
    if os.environ.get("COPILOT_NO_SDIP") == "1":
        return None
    cands = [os.environ.get("SDIP_BIN"), str(config.ROOT / ".venv-sdip" / "bin" / "sdip"), shutil.which("sdip")]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _convert_sdip(src, out, log):
    """SEG-Y -> MDIO/Zarr v3 through the SDIP command line. Returns True on success."""
    exe = _sdip_bin()
    if not exe:
        log("SDIP is not installed - using the built-in SEG-Y reader instead")
        return False
    cmd = [exe, "ingest", str(src), str(out), "--overwrite", "--envelope-gib", "0"]
    logfile = config.LOGS / f"sdip_{Path(out).stem}.log"
    log(f"SDIP conversion active; live output: {logfile}")
    try:
        with logfile.open("w", encoding="utf-8") as stream:
            r = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=8 * 3600)
    except Exception as e:
        log(f"SDIP could not run ({e}) - using the built-in SEG-Y reader")
        return False
    if r.returncode != 0:
        with logfile.open("rb") as stream:
            stream.seek(max(0, logfile.stat().st_size - 4096))
            tail = (stream.read().decode("utf-8", errors="replace").strip().splitlines() or ["no message"])[-1]
        log(f"SDIP refused this file ({tail[:160]}) - using the built-in SEG-Y reader")
        shutil.rmtree(out, ignore_errors=True)
        return False
    try:
        a = _zarr_array(out, "amplitude")
        assert a.ndim == 3
        return True
    except Exception as e:
        log(f"SDIP store could not be opened ({e}) - using the built-in SEG-Y reader")
        shutil.rmtree(out, ignore_errors=True)
        return False


def _zarr_array(store, name):
    import zarr
    return zarr.open_array(store=str(store), path=name, mode="r")


def _convert_segyio(src, out_dir, log):
    """Plain NumPy store: amplitude.npy with shape (inline, crossline, sample)."""
    import segyio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        f, ilb, xlb = _open_segy(src)
        with f:
            ilines = np.asarray(f.ilines)
            xlines = np.asarray(f.xlines)
            samples = np.asarray(f.samples, dtype=np.float64)
            mm = np.lib.format.open_memmap(out_dir / "amplitude.npy", mode="w+", dtype=np.float32,
                                           shape=(len(ilines), len(xlines), len(samples)))
            for i, il in enumerate(ilines):
                mm[i] = f.iline[int(il)]
                if i % 50 == 0:
                    log(f"reading inline {i + 1}/{len(ilines)}")
            mm.flush()
            del mm
    except RuntimeError:
        # Irregular survey: place every trace by its own inline/crossline numbers.
        ilb, xlb = 189, 193
        with segyio.open(str(src), "r", ignore_geometry=True) as f:
            n = f.tracecount
            il_all = np.asarray(f.attributes(ilb)[:])
            xl_all = np.asarray(f.attributes(xlb)[:])
            ilines = np.unique(il_all)
            xlines = np.unique(xl_all)
            samples = np.asarray(f.samples, dtype=np.float64)
            mm = np.lib.format.open_memmap(out_dir / "amplitude.npy", mode="w+", dtype=np.float32,
                                           shape=(len(ilines), len(xlines), len(samples)))
            mm[:] = 0
            ii = np.searchsorted(ilines, il_all)
            xi = np.searchsorted(xlines, xl_all)
            for s in range(0, n, 20000):
                e = min(n, s + 20000)
                mm[ii[s:e], xi[s:e]] = f.trace.raw[s:e]
                log(f"reading traces {e}/{n}")
            mm.flush()
            del mm
    return ilines, xlines, samples, ilb, xlb


def ingest(source, log=print, force=False):
    """Convert one SEG-Y file. Returns the metadata dictionary (also saved to disk)."""
    source = Path(source)
    name = clean_name(source.name)
    meta_path = config.STORES / f"{name}.json"
    old = read_json(meta_path)
    if old and not force and Path(old["path"]).exists() and old.get("source") == str(source):
        return old
    role, label = guess_role(source.name)
    log(f"Converting {source.name} to an AI-ready store")

    mdio_path = config.STORES / f"{name}.mdio"
    npy_path = config.STORES / f"{name}.npstore"
    ilb, xlb = 189, 193
    if _convert_sdip(source, mdio_path, log):
        kind, path, converter = "mdio", mdio_path, "SDIP (MDIO / Zarr v3, lossless)"
        ilines = np.asarray(_zarr_array(mdio_path, "inline")[:])
        xlines = np.asarray(_zarr_array(mdio_path, "crossline")[:])
        # the vertical axis is called "time" (or "depth") by the MDIO template
        samples = None
        for nm in ("time", "depth", "sample"):
            try:
                samples = np.asarray(_zarr_array(mdio_path, nm)[:], dtype=np.float64)
                break
            except Exception:
                continue
        if samples is None:
            samples = np.arange(_zarr_array(mdio_path, "amplitude").shape[2], dtype=np.float64)
    else:
        ilines, xlines, samples, ilb, xlb = _convert_segyio(source, npy_path, log)
        kind, path, converter = "npy", npy_path, "segyio (NumPy store)"

    affine = None
    try:
        affine = segy_affine(source, ilb, xlb)
    except Exception as e:
        log(f"could not read X/Y coordinates from trace headers: {e}")
    if affine is None and kind == "mdio":
        try:
            cx = np.asarray(_zarr_array(mdio_path, "cdp_x")[:], dtype=np.float64)
            cy = np.asarray(_zarr_array(mdio_path, "cdp_y")[:], dtype=np.float64)
            I, X = np.meshgrid(ilines, xlines, indexing="ij")
            ok = (cx != 0) & np.isfinite(cx)
            A = np.stack([np.ones(ok.sum()), I[ok], X[ok]], axis=1)
            c1, *_ = np.linalg.lstsq(A, cx[ok], rcond=None)
            c2, *_ = np.linalg.lstsq(A, cy[ok], rcond=None)
            affine = {"x0": c1[0], "xi": c1[1], "xx": c1[2], "y0": c2[0], "yi": c2[1], "yx": c2[2], "rms_m": 0.0}
        except Exception:
            affine = None

    meta = {
        "name": name, "label": label, "role": role, "source": str(source), "kind": kind, "path": str(path),
        "converter": converter, "shape": [len(ilines), len(xlines), len(samples)],
        "ilines": [int(v) for v in ilines], "xlines": [int(v) for v in xlines],
        "samples": [float(v) for v in samples], "affine": affine,
    }
    vol = Volume(meta)
    meta["stats"] = vol.compute_stats()
    write_json(meta_path, meta)
    log(f"{label}: {meta['shape'][0]} inlines x {meta['shape'][1]} crosslines x {meta['shape'][2]} samples "
        f"({converter})")
    return meta


# ------------------------------------------------------------------ volume access
_ram = OrderedDict()          # name -> ndarray kept in memory
_ram_lock = threading.Lock()
_loading = set()


class Volume:
    def __init__(self, meta):
        self.meta = meta
        self.name = meta["name"]
        self.shape = tuple(meta["shape"])
        self.ilines = np.asarray(meta["ilines"])
        self.xlines = np.asarray(meta["xlines"])
        self.samples = np.asarray(meta["samples"], dtype=np.float64)
        self._lazy = None

    # -- raw array -------------------------------------------------------
    def lazy(self):
        if self._lazy is None:
            if self.meta["kind"] == "mdio":
                self._lazy = _zarr_array(self.meta["path"], "amplitude")
            else:
                self._lazy = np.load(Path(self.meta["path"]) / "amplitude.npy", mmap_mode="r")
        return self._lazy

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * 4

    def array(self):
        """In-memory copy if available (instant slicing), else the on-disk array."""
        with _ram_lock:
            if self.name in _ram:
                _ram.move_to_end(self.name)
                return _ram[self.name]
        self.load_async()
        return self.lazy()

    def load_async(self):
        if self.nbytes > config.RAM_BUDGET_GB * 1e9 * 0.6:
            return
        with _ram_lock:
            if self.name in _ram or self.name in _loading:
                return
            _loading.add(self.name)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            a = np.array(self.lazy()[:], dtype=np.float32)
            np.nan_to_num(a, copy=False)
            with _ram_lock:
                _ram[self.name] = a
                total = sum(v.nbytes for v in _ram.values())
                while total > config.RAM_BUDGET_GB * 1e9 and len(_ram) > 1:
                    _, old = _ram.popitem(last=False)
                    total -= old.nbytes
        except Exception as e:
            print(f"background load of {self.name} failed: {e}")
        finally:
            with _ram_lock:
                _loading.discard(self.name)

    # -- slices ----------------------------------------------------------
    def slice(self, axis, idx):
        a = self.array()
        idx = int(np.clip(idx, 0, self.shape["ixt".index(axis)] - 1))
        if axis == "i":
            s = a[idx, :, :]
        elif axis == "x":
            s = a[:, idx, :]
        else:
            s = a[:, :, idx]
        return np.nan_to_num(np.asarray(s, dtype=np.float32))

    def value(self, i, x, t):
        try:
            return float(np.nan_to_num(self.array()[int(i), int(x), int(t)]))
        except Exception:
            return None

    def decimated(self, ii, xi, ti, log=None):
        """Sub-sampled copy on the interpretation grid (index arrays per axis).

        Reads whole blocks of inlines at a time, which is what chunked stores are fast at.
        """
        a = self.lazy()
        ii, xi, ti = np.asarray(ii), np.asarray(xi), np.asarray(ti)
        out = np.empty((len(ii), len(xi), len(ti)), dtype=np.float32)
        block = 64
        for b0 in range(0, self.shape[0], block):
            rows = np.nonzero((ii >= b0) & (ii < b0 + block))[0]
            if rows.size == 0:
                continue
            blk = np.asarray(a[b0:min(b0 + block, self.shape[0])], dtype=np.float32)
            out[rows] = blk[ii[rows] - b0][:, xi][:, :, ti]
            if log:
                log(f"{self.meta['label']}: {min(b0 + block, self.shape[0])}/{self.shape[0]} inlines")
        return np.nan_to_num(out)

    def compute_stats(self):
        a = self.lazy()
        ni = self.shape[0]
        picks = np.unique(np.linspace(0, ni - 1, min(ni, 12)).astype(int))
        vals = np.concatenate([np.asarray(a[int(i), :, :], dtype=np.float32).ravel() for i in picks])
        vals = vals[np.isfinite(vals)]
        live = vals[vals != 0]
        if live.size == 0:
            live = np.zeros(1, dtype=np.float32)
        p = np.percentile(live, [0.5, 1, 50, 99, 99.5])
        mean, std = float(live.mean()), float(live.std())
        bipolar = bool(p[0] < 0 and abs(mean) < 0.3 * (std + 1e-12))
        if bipolar:
            c = float(max(abs(p[1]), abs(p[3])))
            lo, hi = -c, c
        else:
            lo, hi = float(p[1]), float(p[3])
        if hi <= lo:
            hi = lo + 1.0
        return {"mean": mean, "std": std, "p01": float(p[1]), "p50": float(p[2]), "p99": float(p[3]),
                "min": float(live.min()), "max": float(live.max()), "bipolar": bipolar, "lo": lo, "hi": hi}

    # -- geometry --------------------------------------------------------
    def xy(self, il, xl):
        a = self.meta.get("affine")
        if not a:
            return None
        return (a["x0"] + a["xi"] * il + a["xx"] * xl, a["y0"] + a["yi"] * il + a["yx"] * xl)

    def ilxl(self, x, y):
        """Inverse of xy(): world coordinates -> (inline number, crossline number)."""
        a = self.meta.get("affine")
        if not a:
            return None
        det = a["xi"] * a["yx"] - a["xx"] * a["yi"]
        if abs(det) < 1e-12:
            return None
        dx, dy = np.asarray(x) - a["x0"], np.asarray(y) - a["y0"]
        return ((dx * a["yx"] - dy * a["xx"]) / det, (dy * a["xi"] - dx * a["yi"]) / det)

    def bin_size(self):
        """(metres per inline step, metres per crossline step) or None."""
        a = self.meta.get("affine")
        if not a or len(self.ilines) < 2 or len(self.xlines) < 2:
            return None
        di = float(np.median(np.diff(self.ilines)))
        dx = float(np.median(np.diff(self.xlines)))
        return (math.hypot(a["xi"], a["yi"]) * abs(di), math.hypot(a["xx"], a["yx"]) * abs(dx))


_vcache = {}


def list_volumes():
    """All converted volumes, ordered full/near/mid/far/... for the user interface."""
    out = []
    for p in sorted(config.STORES.glob("*.json")):
        mt = p.stat().st_mtime
        hit = _vcache.get(str(p))
        if hit and hit[0] == mt:
            out.append(hit[1])
            continue
        m = read_json(p)
        if m and "stats" in m and Path(m.get("path", "")).exists():
            v = Volume(m)
            _vcache[str(p)] = (mt, v)
            out.append(v)
    out.sort(key=lambda v: (ROLE_ORDER.index(v.meta["role"]) if v.meta["role"] in ROLE_ORDER else 99, v.name))
    return out


def get_volume(name):
    for v in list_volumes():
        if v.name == name:
            return v
    return None


def by_role():
    d = {}
    for v in list_volumes():
        d.setdefault(v.meta["role"], v)
    return d


def reference_volume():
    """The volume that defines the survey grid (full stack if present)."""
    vols = list_volumes()
    if not vols:
        return None
    for role in ("full", "mid", "near", "far", "ufar"):
        for v in vols:
            if v.meta["role"] == role:
                return v
    for v in vols:
        if v.meta["stats"].get("bipolar"):
            return v
    return vols[0]
