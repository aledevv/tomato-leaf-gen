"""Self-contained HTML 3D viewer of generated leaflets (three.js from a CDN), usable on a phone.

The mesh carries the relief smoothed to its step; the full-resolution maps go in as bump maps and
albedo textures. ``#spin=N`` in the URL turns each leaf once in N frames and moves to the next.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from leafgen.leaf import Leaf, sample_maps
from leafgen.veins.relief import albedo

HERE = Path(__file__).resolve().parent
TEX = 1536  # texture height in px


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _img64(img, ext, quality=None):
    _, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else [])
    kind = "jpeg" if ext == ".jpg" else "png"
    return f"data:image/{kind};base64," + base64.b64encode(buf).decode()


def _resize(a):
    h, w = a.shape[:2]
    return cv2.resize(a, (int(w * TEX / h), TEX), interpolation=cv2.INTER_AREA)


def _bump(height_mm):
    """8-bit bump map, 1 level = 1/127 mm."""
    return _img64(_resize(np.clip(height_mm * 127 + 128, 0, 255).astype(np.uint8)), ".jpg", 94)


def _srgb(x):
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def export_leaf(leaf: Leaf, name: str, length_mm: float) -> dict:
    """Viewer record of one leaf; positions in cm."""
    m = leaf.maps
    box_cm = np.array(m.box) * length_mm / 10
    P = leaf.bent / 10
    ps = float(np.abs(P).max()) / 32000
    x, y = leaf.flat_xy.T
    step = np.linalg.norm(
        leaf.flat_xy[leaf.faces[:, 1]] - leaf.flat_xy[leaf.faces[:, 0]], axis=1
    ).mean()
    # the mesh only carries what its step can hold: narrower creases turn into sawtooth walls
    sig = 0.5 * step / (m.box[3] - m.box[2]) * m.mask.shape[0]
    ht = sample_maps(gaussian_filter(m.top_mm, sig), m.box, x, y) * leaf.thickness
    hb = sample_maps(gaussian_filter(m.bottom_mm, sig), m.box, x, y) * leaf.thickness
    return dict(
        name=name,
        length=round(length_mm),
        box=[float(b) for b in box_cm],
        ps=ps,
        n=len(P),
        tk=_b64((leaf.thickness * 255).round().astype(np.uint8)),
        pos=_b64((P / ps).round().astype(np.int16)),
        nrm=_b64((leaf.normals * 127).round().astype(np.int8)),
        uv=_b64((leaf.uv * 65535).round().astype(np.uint16)),
        idx=_b64(leaf.faces.astype(np.uint32)),
        relief=dict(
            lam=m.lamina_mm,
            crease=float(-m.top_mm.min()),
            rib=float(m.bottom_mm.max()),
            pillow=float(m.top_mm.max()),
            htop=_b64((ht * 20000).round().astype(np.int16)),
            hbot=_b64((hb * 20000).round().astype(np.int16)),
            ctop=_img64(_resize(_srgb(albedo(m, "top"))[..., ::-1]), ".jpg", 88),
            cbot=_img64(_resize(_srgb(albedo(m, "bottom"))[..., ::-1]), ".jpg", 88),
            btop=_bump(m.top_mm),
            bbot=_bump(m.bottom_mm),
        ),
    )


def write_viewer(leaves: list[Leaf], path: Path, length_mm: float = 60.0) -> Path:
    data = [export_leaf(leaf, f"seed {leaf.seed}", length_mm) for leaf in leaves]
    page = (HERE / "head.html").read_text()
    page += f'<script type="application/json" id="leafdata">{json.dumps(data)}</script>\n'
    page += (HERE / "body.html").read_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path
