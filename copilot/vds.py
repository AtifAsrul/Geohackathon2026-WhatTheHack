"""Local, lossless OpenVDS import and bounded reads in (inline, crossline, time) order.

API reference: https://osdu.pages.opengroup.org/platform/domain-data-mgmt-services/seismic/open-vds/
No cloud connections are used. Existing source files are opened read-only by SEGYImport.
"""
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np


def importer_bin():
    """Find the tool shipped by the wheel, including wheels inherited by a venv."""
    candidates = [os.environ.get("OPENVDS_SEGYIMPORT"),
                  str(Path(sys.prefix) / "bin" / "SEGYImport"),
                  str(Path(sys.prefix) / "bin" / "SEGYImport.exe"),
                  shutil.which("SEGYImport")]
    try:
        dist = importlib.metadata.distribution("openvds")
        candidates.extend(str(dist.locate_file(p)) for p in dist.files or []
                          if p.name in ("SEGYImport", "SEGYImport.exe"))
    except importlib.metadata.PackageNotFoundError:
        pass
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()), None)


class VDSArray:
    """Small NumPy-like adapter; a slice requests only its bounding subset at LOD 0."""
    def __init__(self, path):
        import openvds
        self.path = str(Path(path).resolve())
        self.dtype = np.dtype("float32")
        self.ndim = 3
        with openvds.open(self.path) as handle:
            layout = openvds.getLayout(handle)
            if layout.getDimensionality() != 3:
                raise ValueError("Only 3-D poststack VDS volumes are supported")
            names = [layout.getDimensionName(d).lower().replace(" ", "").replace("_", "") for d in range(3)]
            def find_axis(aliases):
                matches = [d for d, name in enumerate(names) if name in aliases]
                if len(matches) != 1:
                    raise ValueError(f"Unsupported VDS axes: {names}")
                return matches[0]
            self.dims = (find_axis({"inline", "inlinenumber"}),
                         find_axis({"crossline", "crosslinenumber"}),
                         find_axis({"sample", "samples", "time", "depth"}))
            self.shape = tuple(layout.getDimensionNumSamples(d) for d in self.dims)
            self.axes = tuple(np.linspace(layout.getDimensionMin(d), layout.getDimensionMax(d), n)
                              for d, n in zip(self.dims, self.shape))
            if layout.getChannelName(0).lower() != "amplitude":
                raise ValueError("VDS primary channel is not Amplitude")

    def __getitem__(self, key):
        import openvds
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > 3:
            raise IndexError("VDS has three axes")
        key = key + (slice(None),) * (3 - len(key))
        bounds, selection, counts = [], [], []
        for item, n in zip(key, self.shape):
            if isinstance(item, (int, np.integer)):
                index = int(item)
                index = index + n if index < 0 else index
                if not 0 <= index < n:
                    raise IndexError("VDS index out of range")
                bounds.append((index, index + 1))
                selection.append(0)
            elif isinstance(item, slice):
                indices = range(*item.indices(n))
                counts.append(len(indices))
                bounds.append((min(indices[0], indices[-1]), max(indices[0], indices[-1]) + 1) if indices else (0, 0))
                selection.append(slice(None, None, indices.step))
            else:
                raise IndexError("VDS supports integer and slice indexing")
        if any(lo == hi for lo, hi in bounds):
            return np.empty(tuple(counts), dtype=np.float32)
        lo, hi = [0] * 3, [0] * 3
        for d, (start, stop) in zip(self.dims, bounds):
            lo[d], hi[d] = start, stop
        with openvds.open(self.path) as handle:
            request = openvds.getAccessManager(handle).requestVolumeSubset(
                min=tuple(lo), max=tuple(hi), lod=0, channel=0,
                format=openvds.VolumeDataFormat.Format_R32, replacementNoValue=0.0)
            # OpenVDS dimension 0 is fastest; NumPy's last dimension is fastest.
            data = request.data.reshape(tuple(hi[d] - lo[d] for d in (2, 1, 0)))
            logical = data.transpose(tuple(2 - d for d in self.dims))
            return np.array(logical[tuple(selection)], dtype=np.float32, copy=True)


def import_segy(source, output, geometry, log_path, log=print):
    """Import to a temporary local file, validate geometry, then publish atomically."""
    import openvds  # Fail early if the Python reader is unavailable.
    exe = importer_bin()
    if not exe:
        raise RuntimeError("OpenVDS SEGYImport executable was not found")
    ilines, xlines, samples, il_byte, xl_byte, crossline_sorted = geometry
    for values in (ilines, xlines, samples):
        if len(values) < 2 or not np.allclose(np.diff(values), np.diff(values)[0], rtol=0, atol=1e-5):
            raise ValueError("OpenVDS import requires regularly spaced 3-D axes")
    source, output, log_path = Path(source), Path(output), Path(log_path)
    temporary = output.with_name(f"{output.stem}.partial-{uuid.uuid4().hex}.vds")
    primary, secondary = ("CrosslineNumber", "InlineNumber") if crossline_sorted else ("InlineNumber", "CrosslineNumber")
    command = [exe, "--vdsfile", str(temporary), "--compression-method", "Zip", "--lod-levels", "0",
               "--primary-key", primary, "--secondary-key", secondary, "--keep-original-order",
               "--header-field", f"InlineNumber={il_byte}:4", "--header-field", f"CrosslineNumber={xl_byte}:4",
               "--sample-start", str(float(samples[0])), "--sample-unit", "ms",
               "--disable-print-text-header", str(source)]
    started = time.monotonic()
    log(f"OpenVDS: {source.stat().st_size / 1e9:.3f} GB source; lossless Zip; live output: {log_path}")
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=8 * 3600)
        if result.returncode:
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 4096))
                tail = stream.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SEGYImport exit {result.returncode}: {tail[-500:]}")
        volume = VDSArray(temporary)
        for expected, actual in zip((ilines, xlines, samples), volume.axes):
            if len(expected) != len(actual) or not np.allclose(expected, actual, rtol=0, atol=1e-4):
                raise ValueError("OpenVDS output axes do not match the SEG-Y geometry")
        # Exercise actual decoding before making this volume visible to the app.
        volume[0, 0, :]
        duration = max(time.monotonic() - started, 0.001)
        metrics = {"input_bytes": source.stat().st_size, "output_bytes": temporary.stat().st_size,
                   "elapsed_seconds": duration, "compression": "Zip (lossless)",
                   "source_MB_s": source.stat().st_size / 1e6 / duration}
        os.replace(temporary, output)
        log(f"OpenVDS complete: {metrics['input_bytes'] / 1e9:.3f} GB SEG-Y -> "
            f"{metrics['output_bytes'] / 1e9:.3f} GB VDS | {duration:.1f}s | "
            f"{metrics['source_MB_s']:.1f} MB/s source-average (import + validation)")
        return metrics
    finally:
        temporary.unlink(missing_ok=True)
