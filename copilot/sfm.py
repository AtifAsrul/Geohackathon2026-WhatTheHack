"""Seismic foundation model (SFM) features -> seismic facies and look-alike search.

Order of preference:
  1. NCS 2.5D model (github.com/NorskRegnesentral/NCS_models, NCS-v1-2.5d-base)
  2. NCS 2D model   (same package, NCS-v1-2d-base)
  3. Sheng et al. SFM (github.com/shenghanlin/SeismicFoundationModel) if its checkpoint
     has been placed in models/sfm/ by hand (its weights are only on Google Drive / Baidu)
  4. Classical seismic attributes (always available)

Whatever produced the features, the rest is identical: reduce them, group them into
facies, and keep them for "find places that look like this one".
"""
import math
import os
import shutil
from pathlib import Path

import numpy as np

from . import config


# ------------------------------------------------------------------ 1/2: NCS
def ncs_features(seis, dt_ms, model_dir, work_dir, device, log=print):
    """Run the NCS inference pipeline on a cube. Returns (features (a,b,c,F), (ci, cx, ct))."""
    import segyio
    import xarray as xr
    from NCS.inference import run_inference

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sgy = work_dir / "ncs_input.sgy"
    out = work_dir / "ncs_features.zarr"
    if out.exists():
        shutil.rmtree(out)
    log("Foundation model: preparing input")
    segyio.tools.from_array3D(str(sgy), np.ascontiguousarray(seis, dtype=np.float32), dt=int(round(dt_ms * 1000)))
    cuda = str(device).startswith("cuda")
    workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    # densify 2 halves the patch spacing (finer facies) at about 8x the work: worth it on a GPU only
    densify = int(os.environ.get("COPILOT_NCS_DENSIFY", "2" if cuda else "1"))
    log(f"Foundation model: running {Path(model_dir).name} on {device}")
    run_inference(
        model_path=str(model_dir), input_path=str(sgy), output_path=str(out), direction="dir0",
        densify=densify, num_overlap_patches=7, overlap_filter="ramp",
        batch_size=16 if cuda else 4, device=device, dtype="float16" if cuda else "float32",
        num_workers=workers,
    )
    ds = xr.open_zarr(str(out), group="dir0")
    feats = np.asarray(ds["features"].values, dtype=np.float32)
    coords = (np.asarray(ds["inline"].values), np.asarray(ds["xline"].values), np.asarray(ds["time_depth"].values))
    try:
        sgy.unlink()
    except Exception:
        pass
    return feats, coords


# ------------------------------------------------------------------ 3: Sheng et al. SFM (MAE ViT encoder)
def sheng_features(seis, ckpt, device, log=print):
    """Encoder tokens of the SFM masked-autoencoder ViT on inline sections (experimental)."""
    import torch
    import torch.nn as nn

    sd = torch.load(str(ckpt), map_location="cpu")
    sd = sd.get("model", sd)
    dim = sd["pos_embed"].shape[-1]
    depth = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    chans = sd["patch_embed.proj.weight"].shape[1]
    grid = int(round(math.sqrt(sd["pos_embed"].shape[1] - 1)))
    heads = 16 if dim >= 1024 else 12

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

        def forward(self, x):
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
            a = (qkv[0] @ qkv[1].transpose(-2, -1)) * (c // heads) ** -0.5
            return self.proj((a.softmax(-1) @ qkv[2]).transpose(1, 2).reshape(b, n, c))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.act = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim), nn.GELU()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attn()
            self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class PE(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(chans, dim, 16, 16)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PE()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, img):
            x = self.patch_embed.proj(img).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]
            cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], 1)
            for b in self.blocks:
                x = b(x)
            return self.norm(x)[:, 1:]

    enc = Enc()
    enc.load_state_dict({k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token"))}, strict=False)
    enc = enc.to(device).eval()
    size = grid * 16
    ni, nx, nt = seis.shape
    ci = np.arange(8, ni, 16)
    px, pt = math.ceil(nx / 16), math.ceil(nt / 16)
    feats = np.zeros((len(ci), px, pt, dim), dtype=np.float32)
    wsum = np.zeros((px, pt, 1), dtype=np.float32)
    pad = np.zeros((max(px * 16, size), max(pt * 16, size)), dtype=np.float32)
    xs = sorted(set(list(range(0, pad.shape[0] - size + 1, size // 2)) + [pad.shape[0] - size]))
    ts = sorted(set(list(range(0, pad.shape[1] - size + 1, size // 2)) + [pad.shape[1] - size]))
    with torch.no_grad():
        for n, i in enumerate(ci):
            if n % 8 == 0:
                log(f"Foundation model (Sheng SFM): inline {n + 1}/{len(ci)}")
            pad[:] = 0
            pad[:nx, :nt] = seis[i]
            acc = np.zeros((pad.shape[0] // 16, pad.shape[1] // 16, dim), dtype=np.float32)
            wsum[:] = 0
            w_acc = np.zeros(acc.shape[:2] + (1,), dtype=np.float32)
            for x0 in xs:
                crops = np.stack([pad[x0:x0 + size, t0:t0 + size].T for t0 in ts])      # (n, time, xline)
                crops = (crops - crops.mean((1, 2), keepdims=True)) / (crops.std((1, 2), keepdims=True) + 1e-6)
                img = torch.from_numpy(crops)[:, None].repeat(1, chans, 1, 1).to(device)
                tok = enc(img).cpu().numpy().reshape(len(ts), grid, grid, dim)             # (time, xline) patches
                for j, t0 in enumerate(ts):
                    acc[x0 // 16:x0 // 16 + grid, t0 // 16:t0 // 16 + grid] += tok[j].transpose(1, 0, 2)
                    w_acc[x0 // 16:x0 // 16 + grid, t0 // 16:t0 // 16 + grid] += 1
            feats[n] = (acc / np.maximum(w_acc, 1))[:px, :pt]
    return feats, (ci, np.arange(px) * 16 + 8, np.arange(pt) * 16 + 8)


# ------------------------------------------------------------------ 4: classical attributes
def classical_features(seis, extra=None, block=8, log=print):
    """Texture attributes on a coarse grid: amplitude level, continuity, frequency, plus rock properties."""
    log("Facies: computing classical texture attributes")
    from .faults import semblance
    s = seis.astype(np.float32)
    semb = semblance(s)
    dsdt = np.gradient(s, axis=2)

    def pool(a):
        ni, nx, nt = (d // block for d in a.shape)
        a = a[:ni * block, :nx * block, :nt * block]
        return a.reshape(ni, block, nx, block, nt, block).mean(axis=(1, 3, 5))

    rms = np.sqrt(pool(s * s)) + 1e-9
    feats = [np.log(rms), pool(semb), np.sqrt(pool(dsdt * dsdt)) / rms]
    for v in (extra or []):
        feats.append(pool(v.astype(np.float32)))
    f = np.stack(feats, axis=-1)
    f = (f - f.mean((0, 1, 2))) / (f.std((0, 1, 2)) + 1e-9)
    coords = tuple(np.arange(n) * block + block // 2 for n in f.shape[:3])
    return f.astype(np.float32), coords


# ------------------------------------------------------------------ facies + similarity
def reduce_and_cluster(feats, k=6, dims=24, seed=0, log=print):
    """PCA then k-means. Returns (unit-length reduced features (a,b,c,D), labels (a,b,c))."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    shp = feats.shape[:3]
    X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(len(X), 150000), replace=False)]
    d = int(min(dims, X.shape[1], max(2, len(sub) - 1)))
    log(f"Facies: reducing {X.shape[1]} features to {d}")
    # sklearn already depends on threadpoolctl. Limit loaded BLAS pools as well
    # as startup environment variables, including libraries imported before us.
    with threadpool_limits(limits=1, user_api="blas"):
        pca = PCA(n_components=d, random_state=seed).fit(sub)
        Z = pca.transform(X).astype(np.float32)
        Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        log(f"Facies: grouping into {k} seismic facies (BLAS threads: 1)")
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096).fit(Z)
    return Z.reshape(shp + (d,)), km.labels_.reshape(shp).astype(np.uint8)


def upsample_labels(labels, coords, shape):
    """Nearest-patch expansion of a coarse label grid onto the interpretation grid."""
    idx = [np.abs(np.asarray(c)[None, :] - np.arange(n)[:, None]).argmin(axis=1) for c, n in zip(coords, shape)]
    return labels[np.ix_(*idx)]


def compute(seis, dt_ms, out_dir, extra=None, device=None, log=print):
    """Run the best available feature extractor and save facies + features. Returns method name."""
    out_dir = Path(out_dir)
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    feats = coords = method = None
    for mdir in config.NCS_MODEL_DIRS:
        if not (Path(mdir) / "config.json").exists():
            log(f"Foundation model: {Path(mdir).name} is not downloaded, skipping")
            continue
        try:
            feats, coords = ncs_features(seis, dt_ms, mdir, out_dir / "ncs_tmp", device, log)
            method = f"NCS foundation model ({Path(mdir).name})"
            break
        except Exception as e:
            log(f"Foundation model {Path(mdir).name} failed: {type(e).__name__}: {str(e)[:300]}")
    if feats is None:
        for ck in sorted(Path(config.SFM_WEIGHTS_DIR).glob("*.pth")) if Path(config.SFM_WEIGHTS_DIR).exists() else []:
            try:
                feats, coords = sheng_features(seis, ck, device, log)
                method = f"Seismic Foundation Model, Sheng et al. ({ck.name})"
                break
            except Exception as e:
                log(f"Sheng SFM failed: {type(e).__name__}: {str(e)[:300]}")
    if feats is None:
        feats, coords = classical_features(seis, extra, log=log)
        method = "classical seismic attributes (no foundation model could run)"
    Z, lab = reduce_and_cluster(feats, log=log)
    np.save(out_dir / "feat.npy", Z.astype(np.float16))
    np.savez(out_dir / "feat_coords.npz", ci=coords[0], cx=coords[1], ct=coords[2])
    np.save(out_dir / "facies.npy", upsample_labels(lab, coords, seis.shape))
    shutil.rmtree(out_dir / "ncs_tmp", ignore_errors=True)
    config.write_json(out_dir / "sfm.json", {"method": method, "k": int(lab.max()) + 1, "dims": int(Z.shape[-1]),
                                             "grid": list(lab.shape)})
    log(f"Facies ready ({method})")
    return method


# -- look-alike search ---------------------------------------------------------
_feat_cache = {}


def _features():
    p = config.INTERP / "feat.npy"
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    if _feat_cache.get("mt") != mt:
        c = np.load(config.INTERP / "feat_coords.npz")
        _feat_cache.update({"mt": mt, "Z": np.load(p).astype(np.float32), "c": (c["ci"], c["cx"], c["ct"])})
    return _feat_cache["Z"], _feat_cache["c"]


def similarity(gi, gx, gt):
    """Cosine similarity (0..1) of every patch to the patch at grid position (gi, gx, gt)."""
    f = _features()
    if f is None:
        return None
    Z, (ci, cx, ct) = f
    a, b, c = (int(np.abs(ci - gi).argmin()), int(np.abs(cx - gx).argmin()), int(np.abs(ct - gt).argmin()))
    sim = Z @ Z[a, b, c]
    return np.clip(sim, 0, 1).astype(np.float32), (ci, cx, ct)
