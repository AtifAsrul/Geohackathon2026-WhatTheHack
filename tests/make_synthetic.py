"""Make a small fake survey (seismic + inversion volumes + wells) to try the app without real data.

    python tests/make_synthetic.py  OUTPUT_FOLDER  [--dead] [--wells 6]

--dead   leaves an empty wedge and an empty strip at the survey edge (zero traces), like a cropped survey.
--wells  3 (default) or 6 wells. With 6, one well has a checkshot that is 16 ms too slow, to test the well tie.
The true model is saved next to the data as truth.npz so tests can compare against it.

The folder mimics the hackathon layout: stack/, inversion/, well_data/{las,deviations,checkshots}.
Geology: layered rocks folded into a dome, cut by one fault, with a sandy layer that is
gas-bright on the dome and a separate channel sand off-structure.
"""
import sys
from pathlib import Path

import numpy as np
import segyio


def ricker(f=28.0, dt=0.004, n=41):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main(out, dead=False, n_wells=3):
    out = Path(out)
    rng = np.random.default_rng(7)
    ni, nx, nt, dt = 120, 150, 320, 4.0
    t0 = 1000.0
    il = np.arange(2000, 2000 + ni)
    xl = np.arange(5000, 5000 + nx * 2, 2)
    I, X = np.meshgrid(np.arange(ni), np.arange(nx), indexing="ij")

    # structure: dome + regional dip + a fault with 9 samples of throw
    dome = -22.0 * np.exp(-(((I - 55) / 26.0) ** 2 + ((X - 60) / 30.0) ** 2))
    shift = dome + 0.10 * I + 0.05 * X
    fault = X > (105 + 0.15 * (I - 60))
    shift = shift + np.where(fault, 9.0, 0.0)

    # rock column in "stratigraphic time"
    ntau = nt + 120
    imp = np.zeros(ntau)
    k = 0
    while k < ntau:
        th = rng.integers(4, 16)
        imp[k:k + th] = 6500 + 900 * rng.standard_normal()
        k += th
    sand_tau = (170, 184)                       # the sandy layer
    imp[sand_tau[0]:sand_tau[1]] = 7200
    imp[sand_tau[0] - 14:sand_tau[0]] = 7600    # shale above

    tau = np.arange(nt)[None, None, :] - shift[:, :, None] + 60.0      # stratigraphic coordinate of each sample
    k0 = np.clip(np.floor(tau).astype(int), 0, ntau - 2)
    f = tau - k0
    ai = imp[k0] * (1 - f) + imp[k0 + 1] * f

    in_layer = (tau >= sand_tau[0]) & (tau < sand_tau[1])
    channel = np.exp(-(((X - 25) - 0.25 * (I - 20)) / 7.0) ** 2) > 0.5          # a sinuous-ish channel belt
    on_dome = dome < -9.0
    sand_map = np.clip(0.25 + 0.55 * channel + 0.6 * on_dome, 0, 0.95)
    sand = np.where(in_layer, sand_map[:, :, None], 0.12) + 0.04 * rng.standard_normal((ni, nx, nt))
    sand = np.clip(sand, 0, 1).astype(np.float32)
    gas = in_layer & on_dome[:, :, None]
    ai = np.where(in_layer, ai - 900 * sand_map[:, :, None], ai)
    ai = np.where(gas, ai - 700, ai).astype(np.float32)
    vpvs = (2.05 - 0.35 * sand - np.where(gas, 0.2, 0.0) + 0.02 * rng.standard_normal((ni, nx, nt))).astype(np.float32)

    sand_true, ai_true = sand.copy(), ai.copy()
    refl = np.zeros_like(ai)
    refl[:, :, 1:] = (ai[:, :, 1:] - ai[:, :, :-1]) / (ai[:, :, 1:] + ai[:, :, :-1])
    w = ricker()
    full = np.apply_along_axis(lambda tr: np.convolve(tr, w, mode="same"), 2, refl).astype(np.float32)
    full += 0.004 * rng.standard_normal(full.shape).astype(np.float32)
    boost = 1.0 + 0.9 * np.apply_along_axis(lambda tr: np.convolve(tr, np.ones(9) / 9, mode="same"), 2, gas.astype(np.float32))
    near = (0.9 * full + 0.003 * rng.standard_normal(full.shape)).astype(np.float32)
    far = (full * boost + 0.005 * rng.standard_normal(full.shape)).astype(np.float32)
    si = (ai / vpvs).astype(np.float32)

    dead_mask = np.zeros((ni, nx), dtype=bool)
    if dead:
        dead_mask = (I < 12 - 0.09 * X) | (I >= ni - 6)
        for v in (full, near, far, ai, si, vpvs, sand):
            v[dead_mask] = 0.0

    # coordinates: 25 m bins, rotated 20 degrees
    th = np.radians(20.0)
    def xy(a, b):
        return (400000 + 25.0 * (a * np.sin(th) + b * np.cos(th)), 6200000 + 25.0 * (a * np.cos(th) - b * np.sin(th)))

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = segyio.spec()
        spec.ilines, spec.xlines = il, xl
        spec.samples = t0 + np.arange(nt) * dt
        spec.format, spec.sorting = 5, 2
        with segyio.create(str(path), spec) as fh:
            fh.bin.update({segyio.BinField.Interval: int(dt * 1000), segyio.BinField.Samples: nt})
            tr = 0
            for a in range(ni):
                for b in range(nx):
                    x, y = xy(a, b)
                    fh.header[tr] = {segyio.su.iline: int(il[a]), segyio.su.xline: int(xl[b]), segyio.su.cdpx: int(round(x * 100)),
                                     segyio.su.cdpy: int(round(y * 100)), segyio.su.scalco: -100, segyio.su.delrt: int(t0),
                                     segyio.su.ns: nt, segyio.su.dt: int(dt * 1000)}
                    fh.trace[tr] = data[a, b]
                    tr += 1
        print("wrote", path.name)

    P = "Challenge1_Challenge2_"
    write(out / "stack" / f"{P}Full_Stack_cropped.sgy", full)
    write(out / "stack" / f"{P}Near_Stack_cropped.sgy", near)
    write(out / "stack" / f"{P}Far_Stack_cropped.sgy", far)
    write(out / "inversion" / f"{P}Inversion_P_Impedance_cropped.sgy", ai)
    write(out / "inversion" / f"{P}Inversion_S_Impedance_cropped.sgy", si)
    write(out / "inversion" / f"{P}Inversion_VpVs_cropped.sgy", vpvs)
    write(out / "inversion" / f"{P}Inversion_TotalSandProbability_cropped.sgy", sand)

    # ---- wells (constant velocity 2500 m/s: TWT ms = 0.8 * depth below sea level in m)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from copilot.wells import min_curvature

    def ab(x, y):                                   # inverse of xy(): world -> fractional (inline idx, crossline idx)
        dx, dy = (np.asarray(x) - 400000) / 25.0, (np.asarray(y) - 6200000) / 25.0
        return dx * np.sin(th) + dy * np.cos(th), dx * np.cos(th) - dy * np.sin(th)

    wd = out / "well_data"
    for d in ("las", "deviations", "checkshots"):
        (wd / d).mkdir(parents=True, exist_ok=True)
    # name: (inline idx, xline idx, lateral drift m, result, checkshot bias ms or None for no checkshot)
    wells = {"w-1": (55, 60, 0.0, "Gas discovery", 0.0), "w-2": (30, 30, 400.0, "Water-wet sand", 0.0),
             "w-3": (95, 125, 0.0, "Dry", None)}
    if n_wells >= 6:
        wells.update({"w-4": (70, 38, 0.0, "Water-wet sand", 0.0), "w-5": (60, 100, 0.0, "Dry", 16.0),
                      "w-6": (100, 20, 0.0, "Dry", 0.0)})
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Well Name", "Surface X", "Surface Y", "KB (m)", "TD (m)", "Result"])
    kb = 25.0
    truth_wells = {}
    for name, (a, b, drift, result, bias) in wells.items():
        x0, y0 = xy(a, b)
        ws.append([name, round(x0, 1), round(y0, 1), kb, 2900.0, result])
        md = np.arange(0, 2900.0, 25.0)
        if drift:
            inc = np.clip((md - 800) / 2100 * 14, 0, 14)
            azi = np.full_like(md, 60.0)
            lines = ["MD INCL AZIM"] + [f"{m:.2f} {i:.3f} {z:.2f}" for m, i, z in zip(md, inc, azi)]
            dx, dy, dz = min_curvature(md, inc, azi)
            px, py, ptv = x0 + dx, y0 + dy, dz - kb
        else:
            lines = ["MD X Y TVDSS"] + [f"{m:.2f} {x0:.2f} {y0:.2f} {m - kb:.2f}" for m in md]
            px, py, ptv = np.full(len(md), x0), np.full(len(md), y0), md - kb
        (wd / "deviations" / f"{name}_Deviation.txt").write_text("\n".join(lines))
        if bias is not None:
            z = np.arange(0, 2900.0, 100.0)
            (wd / "checkshots" / f"{name}_Checkshot.txt").write_text(
                "TVDSS TWT\n" + "\n".join(f"{d:.1f} {0.8 * d + (bias if d > 0 else 0):.2f}" for d in z))
        # logs sampled from the model along the true well path
        depth = np.arange(1300.0, 2850.0, 0.5)
        tv = np.interp(depth, md, ptv)
        fa, fb = ab(np.interp(depth, md, px), np.interp(depth, md, py))
        ia = np.clip(np.rint(fa).astype(int), 0, ni - 1)
        ib = np.clip(np.rint(fb).astype(int), 0, nx - 1)
        twt = 0.8 * tv
        k = np.clip(((twt - t0) / dt).astype(int), 0, nt - 1)
        s = sand_true[ia, ib, k]
        g = gas[ia, ib, k]
        gr = 110 - 80 * s + 4 * rng.standard_normal(len(depth))
        rhob = 2.55 - 0.3 * s - 0.08 * g
        dtc = 304800.0 / (ai_true[ia, ib, k] / rhob)
        rt = 2 + 3 * s + 60 * g + 0.2 * rng.standard_normal(len(depth))
        las = ["~Version", " VERS. 2.0 :", " WRAP. NO :", "~Well", f" STRT.M {depth[0]:.2f} :", f" STOP.M {depth[-1]:.2f} :",
               " STEP.M 0.5 :", " NULL. -999.25 :", f" WELL. {name} :", "~Curve", " DEPT.M :", " GR.GAPI :", " RHOB.G/C3 :",
               " DT.US/F :", " RT.OHMM :", "~ASCII"]
        las += [f"{d:.2f} {a1:.3f} {a2:.4f} {a3:.3f} {a4:.3f}" for d, a1, a2, a3, a4 in zip(depth, gr, rhob, dtc, rt)]
        (wd / "las" / f"{name}.las").write_text("\n".join(las))
        truth_wells[name] = bias
    wb.save(wd / "Challenge1_Challenge2_WellList_Information.xlsx")
    np.savez_compressed(out / "truth.npz", dead=dead_mask, shift=shift.astype(np.float32), fault=fault,
                        sand_top_ms=(t0 + dt * (sand_tau[0] - 60 + shift)).astype(np.float32), on_dome=on_dome,
                        channel=channel, il=il, xl=xl, t0=t0, dt=dt,
                        well_names=np.array(list(truth_wells)),
                        well_bias_ms=np.array([np.nan if v is None else v for v in truth_wells.values()]))
    print("done ->", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nw = int(sys.argv[sys.argv.index("--wells") + 1]) if "--wells" in sys.argv else 3
    if "--wells" in sys.argv:
        args = [a for a in args if a != str(nw)]
    main(args[0] if args else "synthetic_data", dead="--dead" in sys.argv, n_wells=nw)
