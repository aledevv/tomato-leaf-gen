"""Two established feature spaces for vein images, next to the statistics of ``veins.metrics``.

The 16 statistics of ``veins.metrics`` are our choice; if generators rank the same in feature
spaces chosen by others, the ranking does not depend on that choice. ``evaluate.py``
measures the same sets (same leaflets, same seed) in:

- ``dinov2``: the CLS token of DINOv2 ViT-L/14 (Oquab et al. 2024), the encoder Stein et al.
  (NeurIPS 2023, "Exposing flaws of generative model evaluation metrics") recommend over
  Inception. Preprocessing as their dgm-eval code (``models/dinov2.py``): PIL bicubic resize to
  224 x 224, ImageNet mean / std, CLS token of the hub model (``x_norm_clstoken``). The image is
  the 256 px vein raster inside the outline mask on a black background, the outline itself not
  drawn, replicated to 3 channels: with ``binary`` the veins (> ``metrics.VEIN``) at 255 (the
  information of table 2), otherwise the raster's values as grey levels (orders, table 1).
  Deviation: the network is timm's ``vit_large_patch14_dinov2.lvd142m`` (Meta's
  ``dinov2_vitl14_pretrain.pth`` weights, converted) instead of ``torch.hub`` (which runs code
  from GitHub at load time); its position embedding for 16 x 16 patches is computed as DINOv2's
  ``interpolate_pos_encoding`` does (bicubic, scale (16 + 0.1) / 37, no antialias), so the
  network is the hub one. Next to 1-NNA, Frechet distance (``fd``, as dgm-eval) and KID (``kid``,
  unbiased MMD^2, cubic polynomial kernel, as dgm-eval with one subset of all the samples).
- ``botany``: the venation characters of ``veins.architecture`` (Ellis et al. 2009) plus vein
  length per area, ``vla`` = total centreline length / lamina area (Sack & Scoffoni 2013), on the
  skeleton of the veins (> ``metrics.VEIN``) inside the mask, an orthogonal step 1 px, a diagonal
  one sqrt(2) (``skeleton_length``), in units of 1 / leaf length (canonical frame, base-tip = 1).
  Per order, from the skeletons of the veins up to that order: ``vla_midrib`` (skeleton of
  > ``metrics.MIDRIB``), ``vla_secondary`` (that of > ``metrics.SECONDARY`` minus the midrib),
  ``vla_minor`` (the rest of ``vla``). The skeleton of the whole network instead would give the
  midrib the stretch of every secondary inside its stroke (+30 % on a chamber leaflet). With
  ``binary`` only ``vla`` (the characters need the orders).

- ``botany_minor`` (table 1 only): ``botany`` with ``vla_minor`` replaced by the minor-vein
  characters of ``minor_characters`` (``MINOR``), measured on the skeleton graph: VLA of the
  tertiary and the fine band, freely ending veinlets and junctions per area (Sack & Scoffoni
  2013), tertiary angle to the secondary at the junction (adapted from the Manual's tertiary
  angle characters, which describe percurrent tertiaries; measured here at the branching).
  Checked in ``utils/veins/minor_checks.py``.

1-NNA as ``metrics.compare`` (standardised by the real set, ``metrics._one_nna``, as many of each);
``botany`` as ``architecture.compare`` (1-NNA on the images where every character is defined).
Tables are cached per content (images, masks, space) in ``CACHE``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy import ndimage as ndi

from leafgen.paths import DATA
from leafgen.veins import architecture, metrics
from leafgen.veins.raster import BOX_Y

CACHE = DATA / "cache"
BOTANY = (*architecture.NAMES, "vla", "vla_midrib", "vla_secondary", "vla_minor")
MINOR = (
    "vla_tertiary",
    "vla_fine",
    "fev",
    "junctions",
    "tert_angle",
    "tert_angle_sd",
)
BOTANY_MINOR = (*BOTANY[:-1], *MINOR)  # vla_minor split into its two bands
EIGHT = np.ones((3, 3), bool)
SPUR, MARGIN, NEAR = 3, 2.5, 6  # px: shortest kept end branch, margin band, junction neighbourhood
_MODEL = None


def _float(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32) / (255.0 if v.dtype == np.uint8 else 1.0)


def skeleton_length(skel: np.ndarray) -> float:
    """Centreline length (px) of a 1 px skeleton: orthogonal neighbours 1, diagonal ones sqrt(2)
    when no orthogonal path of two steps joins them (else the corner would count twice)."""
    s = skel.astype(bool)
    pad = np.pad(s, 1)
    h, w = s.shape

    def at(dy: int, dx: int) -> np.ndarray:
        return pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    total = float((s & at(0, 1)).sum() + (s & at(1, 0)).sum())
    for dy, dx, c in ((1, 1, (0, 1)), (1, -1, (0, -1))):
        total += np.sqrt(2) * (s & at(dy, dx) & ~(at(*c) | at(1, 0))).sum()
    return total


def vla(veins: np.ndarray, mask: np.ndarray, orders: bool = True) -> np.ndarray:
    """[vla, vla_midrib, vla_secondary, vla_minor] (or [vla]) in 1 / leaf length."""
    from skimage.morphology import skeletonize

    m = mask > 0
    v = _float(veins) * m
    per_length = v.shape[0] / (BOX_Y[1] - BOX_Y[0])  # px per leaf length
    area = max(int(m.sum()), 1)
    levels = (metrics.VEIN, metrics.MIDRIB, metrics.SECONDARY) if orders else (metrics.VEIN,)
    total, *up_to = (skeleton_length(skeletonize(v > t)) / area * per_length for t in levels)
    if not orders:
        return np.array([total])
    return np.array([total, up_to[0], up_to[1] - up_to[0], total - up_to[1]])


def _graph(skel: np.ndarray):
    """Junction clusters (pixels with 3+ neighbours) and the segments between them, with spurs
    (end segments shorter than ``SPUR`` px attached to a junction) removed, twice."""
    s = skel.copy()
    for _ in range(3):
        nb = ndi.convolve(s.astype(np.int32), EIGHT.astype(np.int32), mode="constant") - 1
        junc, end = s & (nb >= 3), s & (nb == 1)
        jlab, nj = ndi.label(junc, EIGHT)
        slab, ns = ndi.label(s & ~junc, EIGHT)
        near = ndi.maximum_filter(jlab, size=3) * (slab > 0)
        size = np.bincount(slab.ravel(), minlength=ns + 1)
        ends = np.bincount(slab[end], minlength=ns + 1)
        touch = np.bincount(slab[near > 0], minlength=ns + 1)
        spur = (size < SPUR) & (ends > 0) & (touch > 0)
        spur[0] = False
        if not spur.any() or _ == 2:
            break
        s &= ~spur[slab]
    pairs = np.unique(np.stack([slab[near > 0], near[near > 0]], 1), axis=0)
    return junc, end, jlab, nj, slab, ns, pairs


def minor_characters(veins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``MINOR`` of one image (nan where undefined): skeleton graph of the veins > ``VEIN``.

    - ``vla_tertiary``, ``vla_fine``: VLA of the bands (``SECONDARY``, ``FINE_TOP``] and
      (``VEIN``, ``FINE_TOP``] (``metrics.FINE_TOP``), as ``vla``;
    - ``fev``: freely ending veinlets per area: ends of minor segments (median value <=
      ``SECONDARY``) more than ``MARGIN`` px inside the outline;
    - ``junctions``: junction clusters touching a minor segment, per area;
    - ``tert_angle``, ``tert_angle_sd``: at each junction of a tertiary segment (median value in
      (``FINE_TOP``, ``SECONDARY``]) with a secondary one, the acute angle between the tertiary
      (junction -> mean of its pixels within ``NEAR`` px) and the secondary's principal axis within
      ``NEAR`` px; mean and standard deviation.
    Areas in leaf lengths squared."""
    from skimage.morphology import skeletonize

    m = mask > 0
    v = _float(veins) * m
    size = v.shape[0]
    per_length = size / (BOX_Y[1] - BOX_Y[0])
    area_px = max(int(m.sum()), 1)
    area = area_px / per_length**2
    out = np.full(len(MINOR), np.nan)
    lengths = [
        skeleton_length(skeletonize(v > t)) / area_px * per_length
        for t in (metrics.VEIN, metrics.FINE_TOP, metrics.SECONDARY)
    ]
    out[0], out[1] = lengths[1] - lengths[2], lengths[0] - lengths[1]
    junc, end, jlab, nj, slab, ns, pairs = _graph(skeletonize(v > metrics.VEIN))
    if ns == 0:
        return out
    med = ndi.median(v, slab, np.arange(1, ns + 1))
    med = np.r_[np.inf, med]
    minor = med <= metrics.SECONDARY
    minor[0] = False
    inside = ndi.distance_transform_edt(m) > MARGIN
    out[2] = (end & minor[slab] & inside).sum() / area
    out[3] = len(np.unique(pairs[minor[pairs[:, 0]], 1])) / area
    # the length of the minor segments between two junctions was dropped: its raster value
    # follows the tree's poorly (Spearman 0.3-0.5, utils/veins/minor_checks.py)
    tert = (med > metrics.FINE_TOP) & (med <= metrics.SECONDARY)
    sec = (med > metrics.SECONDARY) & (med <= metrics.MIDRIB)
    by_j = {}
    for sg, jj in pairs:
        by_j.setdefault(jj, []).append(sg)
    centres = ndi.center_of_mass(junc, jlab, np.arange(1, nj + 1))
    angles = []
    for jj, segs in by_j.items():
        major, minors = [x for x in segs if sec[x]], [x for x in segs if tert[x]]
        if not major or not minors:
            continue
        c = np.asarray(centres[jj - 1])
        r0, c0 = np.maximum(np.floor(c - NEAR).astype(int), 0)
        win = slab[r0 : int(c[0] + NEAR) + 1, c0 : int(c[1] + NEAR) + 1]
        rr, cc = np.indices(win.shape)
        pos = np.stack([rr + r0, cc + c0], -1).astype(float) - c
        ok = np.hypot(pos[..., 0], pos[..., 1]) <= NEAR
        p = pos[ok & np.isin(win, major)]
        if len(p) < 3:
            continue
        axis = np.linalg.svd(p - p.mean(0), full_matrices=False)[2][0]
        for t in minors:
            q = pos[ok & (win == t)]
            if len(q) < 3:
                continue
            d = q.mean(0)
            cos = abs(d @ axis) / (np.linalg.norm(d) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos, 0, 1))))
    if len(angles) >= 2:
        out[4], out[5] = np.mean(angles), np.std(angles)
    return out


def _key(images: np.ndarray, masks: np.ndarray, tag: str) -> Path:
    h = hashlib.sha1(tag.encode())
    for a in (images, masks):
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return CACHE / f"{tag}_{h.hexdigest()[:20]}.npy"


def _cached(fn):
    def wrapped(images: np.ndarray, masks: np.ndarray, binary: bool = False) -> np.ndarray:
        path = _key(images, masks, f"{fn.__name__}{'_bin' if binary else ''}")
        if path.exists():
            return np.load(path)
        out = fn(images, masks, binary)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.npy")
        np.save(tmp, out)
        os.replace(tmp, path)  # several evaluations may share the cache
        return out

    return wrapped


@_cached
def botany(images: np.ndarray, masks: np.ndarray, binary: bool = False) -> np.ndarray:
    if binary:
        return np.array([vla(v, m, orders=False) for v, m in zip(images, masks, strict=True)])
    arch = architecture.table(images, masks)
    return np.hstack([arch, [vla(v, m) for v, m in zip(images, masks, strict=True)]])


@_cached
def botany_minor(images: np.ndarray, masks: np.ndarray, binary: bool = False) -> np.ndarray:
    """``botany`` with ``vla_minor`` replaced by the minor-vein characters (orders needed)."""
    assert not binary, "the minor-vein characters need the vein orders"
    base = botany(images, masks)[:, :-1]
    extra = [minor_characters(v, m) for v, m in zip(images, masks, strict=True)]
    return np.hstack([base, extra])


def _model():
    """DINOv2 ViT-L/14 at 224 px, position embedding resampled as DINOv2's hub code does."""
    global _MODEL
    if _MODEL is None:
        import timm
        import torch
        import torch.nn.functional as F

        name = "vit_large_patch14_dinov2.lvd142m"
        full = timm.create_model(name, pretrained=True, num_classes=0)  # trained at 518 px
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
        pe = full.pos_embed.detach()
        cls, patch = pe[:, :1], pe[:, 1:]
        n, dim, w0 = int(np.sqrt(patch.shape[1])), patch.shape[2], 224 // 14
        s = (w0 + 0.1) / n  # interpolate_offset of DINOv2
        patch = F.interpolate(
            patch.reshape(1, n, n, dim).permute(0, 3, 1, 2),
            scale_factor=(s, s),
            mode="bicubic",
            antialias=False,
        )
        assert patch.shape[-2:] == (w0, w0)
        with torch.no_grad():
            model.pos_embed.copy_(
                torch.cat([cls, patch.permute(0, 2, 3, 1).reshape(1, -1, dim)], 1)
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = model.eval().to(device)
    return _MODEL


def to_rgb(veins: np.ndarray, mask: np.ndarray, binary: bool) -> np.ndarray:
    """uint8 (H, W, 3): veins inside the mask, black background, no outline."""
    v = _float(veins) * (mask > 0)
    g = np.where(v > metrics.VEIN, 255, 0) if binary else np.round(v * 255)
    return np.repeat(g.astype(np.uint8)[..., None], 3, -1)


@_cached
def dinov2(images: np.ndarray, masks: np.ndarray, binary: bool = False) -> np.ndarray:
    import torch
    from PIL import Image

    model = _model()
    device = next(model.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), 32):
        batch = [
            np.asarray(Image.fromarray(to_rgb(v, m, binary)).resize((224, 224), Image.BICUBIC))
            for v, m in zip(images[i : i + 32], masks[i : i + 32], strict=True)
        ]
        x = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            out.append(model.forward_features((x - mean) / std)[:, 0].double().cpu().numpy())
    return np.concatenate(out)


def fd(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance of two feature sets (as dgm-eval ``compute_FD_with_reps``)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(len(s1)) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    return float(
        ((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real)
    )


def kid(a: np.ndarray, b: np.ndarray) -> float:
    """Unbiased MMD^2, k(x, y) = (<x, y> / dim + 1)^3, as many of each (dgm-eval ``compute_mmd``,
    whose subsets of min(n, 1000) are all the samples here)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x @ y.T / x.shape[1] + 1) ** 3

    kxx, kyy, kxy = k(a, a), k(b, b), k(a, b)
    return float(
        (kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean()
    )


def table(space: str, images: np.ndarray, masks: np.ndarray, fine: bool, binary: bool):
    if space == "stats":
        return metrics.table(images, masks, fine, binary)
    return {"dinov2": dinov2, "botany": botany, "botany_minor": botany_minor}[space](
        images, masks, binary
    )


def compare(space: str, generated: np.ndarray, real: np.ndarray) -> dict:
    if space == "stats":
        return metrics.compare(generated, real)
    if space == "botany":
        names = BOTANY if real.shape[1] == len(BOTANY) else ("vla",)
        return architecture.compare(generated, real, names)
    if space == "botany_minor":
        return architecture.compare(generated, real, BOTANY_MINOR)
    mu, sd = real.mean(0), real.std(0) + 1e-9
    return {
        "one_nna": metrics._one_nna((generated - mu) / sd, (real - mu) / sd),
        "one_nna_raw": metrics._one_nna(generated, real),
        "fd": fd(generated, real),
        "kid": kid(generated, real),
    }
