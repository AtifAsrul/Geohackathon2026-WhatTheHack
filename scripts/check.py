"""Self-check: what is installed, what is reachable, and how the data files are laid out.

    .venv/bin/python scripts/check.py            (environment only)
    .venv/bin/python scripts/check.py --data     (also describe the data files)

The data report lists file names, column names and curve names only - never data values -
so it is safe to copy out when asking for help.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def line(ok, what, detail=""):
    print(f"  [{'OK' if ok else '--'}]  {what}" + (f"   ({detail})" if detail else ""))
    return ok


def environment():
    from copilot import config
    print("\nENVIRONMENT")
    line(True, "Python", sys.version.split()[0])
    for mod in ("numpy", "scipy", "fastapi", "uvicorn", "segyio", "lasio", "openpyxl", "PIL", "sklearn", "zarr", "xarray"):
        try:
            m = __import__(mod)
            line(True, mod, getattr(m, "__version__", ""))
        except Exception as e:
            line(False, mod, f"missing: {e}")
    try:
        import torch
        gpu = torch.cuda.is_available()
        line(True, "PyTorch", f"{torch.__version__}, GPU visible here: {'yes' if gpu else 'no (normal on the login node)'}")
    except Exception as e:
        line(False, "PyTorch", f"missing: {e}")
    try:
        import transformers
        from NCS.inference import run_inference  # noqa: F401
        line(True, "NCS foundation model code", f"transformers {transformers.__version__}")
    except Exception as e:
        line(False, "NCS foundation model code", f"{type(e).__name__}: {str(e)[:120]}")
    try:
        import openvds
        from copilot import vds
        exe = vds.importer_bin()
        line(bool(exe), "OpenVDS reader and SEGYImport", exe or "importer not found - built-in reader will be used")
    except Exception as e:
        line(False, "OpenVDS", f"{e} - built-in SEG-Y reader will be used")
    line(config.RGT_WEIGHTS.exists(), "RGT-Est weights", "" if config.RGT_WEIGHTS.exists() else "missing - classical horizons will be used")
    for d in config.NCS_MODEL_DIRS:
        ok = (d / "model.safetensors").exists()
        line(ok, f"NCS model {d.name}", "" if ok else "missing")
    line(bool(shutil.which("sbatch")), "Slurm (GPU jobs)", "" if shutil.which("sbatch") else "not found - everything will run on this machine")
    from copilot import llm
    ok, msg = llm.check()
    line(ok, f"Gemma assistant at {config.LLM_URL}", msg)
    d = Path(config.DEFAULT_DATA_DIR)
    line(d.exists(), f"Challenge data folder {d}", "" if d.exists() else "not found")


def data_report(folder):
    from copilot import wells
    folder = Path(folder)
    print(f"\nDATA LAYOUT under {folder}  (names only, no values)")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    for p in files:
        if p.suffix.lower() in (".sgy", ".segy"):
            info = f"{p.stat().st_size / 1e9:.2f} GB"
            try:
                import segyio
                with segyio.open(str(p), "r", ignore_geometry=True) as f:
                    info += f", {f.tracecount} traces, {len(f.samples)} samples, sample {f.samples[1] - f.samples[0]:g}, format {int(f.format)}"
            except Exception as e:
                info += f", could not open: {e}"
            print(f"  SEG-Y  {p.relative_to(folder)}   {info}")
    seen = set()
    for p in files:
        low, kind = p.name.lower(), None
        if p.suffix.lower() == ".las":
            kind = "las"
        elif "deviat" in low:
            kind = "deviation"
        elif "check" in low:
            kind = "checkshot"
        elif p.suffix.lower() in (".xlsx", ".xlsm"):
            kind = "well list"
        elif "marker" in low or "top" in low:
            kind = "markers"
        if not kind or kind in seen:
            continue
        seen.add(kind)                                   # one example of each kind is enough
        print(f"  {kind.upper()} example: {p.relative_to(folder)}")
        try:
            if kind == "las":
                lg = wells.read_las(p)
                print(f"      curves in file: {lg['raw_names']}")
                print(f"      understood as : {sorted(lg['curves'])}")
            elif kind == "well list":
                import openpyxl
                wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
                for ws in wb.worksheets:
                    rows = [r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]
                    hdr = next((r for r in rows[:15] if sum(isinstance(c, str) for c in r) >= 3), rows[0] if rows else ())
                    print(f"      sheet '{ws.title}': {len(rows)} rows, columns: {[c for c in hdr if c is not None]}")
                print(f"      wells recognised: {len(wells.read_well_list(p))}")
            else:
                header, d, kv = wells.read_table(p)
                print(f"      column names: {header}   numeric columns: {d.shape[1] if d.size else 0}   rows: {len(d)}")
                print(f"      other labelled values above the table: {sorted(kv)}")
        except Exception as e:
            print(f"      could not be read: {type(e).__name__}: {e}")


if __name__ == "__main__":
    environment()
    if "--data" in sys.argv:
        from copilot import config
        i = sys.argv.index("--data")
        data_report(sys.argv[i + 1] if len(sys.argv) > i + 1 else config.DEFAULT_DATA_DIR)
    print()
