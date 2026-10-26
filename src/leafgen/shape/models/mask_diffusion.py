"""Shape model "mask_diffusion": flow matching on 128 x 128 raster masks with a small 2-D
U-Net (``diffusers.UNet2DModel``), conditioned on the subset through its class embedding.

The standard image pipeline: closure and topology for free, no parameterisation; the known cost is
data (Kadkhodaie et al., ICLR 2024: raster generators need large sets before they generalise, so
the pooled subsets and early stopping on Z_U). Masks live in the canonical box of ``grid`` (x in
[-0.6, 0.6], y in [-0.1, 1.1]), values -1 outside / +1 inside; a draw is thresholded at 0, its
largest loop traced (``grid.contour_from_sdf``) and re-canonicalised on base and tip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from leafgen.contracts.geometry import LeafRole, LeafShape2D
from leafgen.shape.features import (
    ROLES,
    efd_features,
    landmarks_to_contour,
    pseudo_landmarks,
)
from leafgen.shape.models import neural
from leafgen.shape.models.grid import BOX_X, BOX_Y, GRID, SIZE, contour_from_sdf

NAME = "mask_diffusion"
STEPS, CHANNELS, VAL_DRAWS = 50, (32, 64, 128, 128), 512


def rasterise(contour: np.ndarray, n: int = GRID) -> np.ndarray:
    pts = (contour - [BOX_X[0], BOX_Y[0]]) * (n / SIZE)
    mask = np.zeros((n, n), np.uint8)
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, shift=4)
    return mask


def build(resolution: int = GRID, widths: tuple[int, ...] = CHANNELS):
    import torch
    from diffusers import UNet2DModel

    net = UNet2DModel(
        sample_size=resolution,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=tuple(widths),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=len(neural.SUBSETS),
    )

    class Velocity(torch.nn.Module):
        """``neural``'s calling convention v(x, t, subset, vector) around the diffusers U-Net."""

        def __init__(self):
            super().__init__()
            self.unet = net

        def forward(self, x, t, subset, vector=None):  # noqa: ARG002
            return self.unet(x, t * 1000, class_labels=subset).sample

    return Velocity(), torch


def fit(
    bank_path: Path | list[Path],
    output: Path,
    *,
    minutes: float = 300.0,
    resolution: int = GRID,
    widths: tuple[int, ...] = CHANNELS,
    dataset: str = "",
    seed: int = 0,
    schedule: str = "constant",
    patience: int = 0,
    **_: Any,
) -> dict[str, Any]:
    banks = [Path(b) for b in (bank_path if isinstance(bank_path, list) else [bank_path])]
    data = neural.pooled(banks)
    val = neural.split_val(data["photo"], seed=seed)
    train_idx, val_idx = np.flatnonzero(~val), np.flatnonzero(val)
    masks = np.array(
        [rasterise(landmarks_to_contour(v), resolution) for v in data["landmarks"][train_idx]]
    )
    model, torch = build(resolution, widths)
    if resolution > 128:  # recompute activations: the 256 px U-Net does not fit 12 GB otherwise
        model.unet.enable_gradient_checkpointing()
    dev = neural.device()
    model.to(dev)
    x = torch.as_tensor(masks[:, None].astype(np.float16) * 2 - 1, device=dev)
    subset = torch.as_tensor(data["subset"][train_idx], device=dev)

    def loss_fn(net, x1, subset_rows, vector, generator):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            return neural.fm_loss(net, x1.float(), subset_rows, vector, generator)

    val_subsets = [n for n in ("chamber", "field") if n in data["names"]] or data["names"][:1]
    batch_draws, val_draws = (32, VAL_DRAWS) if resolution <= 128 else (16, VAL_DRAWS // 4)

    def validate(average):
        """Per subset (chamber and field when present): VAL_DRAWS conditioned draws against that
        subset's validation leaflets; the record averages the subsets."""
        g = torch.Generator(device=dev).manual_seed(seed + 1)
        records = []
        for name in val_subsets:
            index = neural.SUBSETS.index(name)
            rows = val_idx[data["subset"][val_idx] == index]
            n = min(val_draws, len(rows))
            gen = []
            for i in range(0, n, batch_draws):  # small batches: memory-hungry U-Net
                s = torch.full((min(batch_draws, n - i),), index, dtype=torch.long, device=dev)
                fields = neural.fm_sample(
                    average, (len(s), 1, resolution, resolution), s, None, g, STEPS, dev
                )
                gen += [contour_from_sdf(-f) for f in fields[:, 0].cpu().numpy()]
            gen = [pseudo_landmarks(c) for c in gen if c is not None]
            r = neural.landmarks_metrics(
                np.array(gen), data["landmarks"][rows], data["landmarks"][train_idx]
            )
            r["valid"] = len(gen) / n
            records.append(r)
        return {k: float(np.mean([r[k] for r in records])) for k in records[0]}

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"  # pause with Ctrl-C / SIGTERM, rerun the same command to resume
    log = (output / "train.log").open("a" if last.exists() else "w")
    run = neural.train(
        model,
        x,
        subset,
        loss_fn=loss_fn,
        minutes=minutes,
        seed=seed,
        batch=32 if resolution <= 128 else 8,
        lr=1e-4,
        every=4000 if resolution <= 128 else 8000,
        validate=validate,
        resume=last,
        log=lambda s: (print(s), log.write(s + "\n"), log.flush()),
        schedule=schedule,
        patience=patience,
    )
    log.close()
    torch.save(model.state_dict(), output / "model.pt")
    label = dataset or "pool"
    metadata = dict(
        model=NAME,
        representation=f"mask{resolution}",
        resolution=resolution,
        widths=list(widths),
        schedule=schedule,
        patience=patience,
        subsets=data["names"],
        roles=list(ROLES),
        role_specific=False,
        dataset=label,
        version=f"{label.lower()}_mask_diffusion",
        banks=[str(b) for b in banks],
        training_counts={r: int(len(train_idx)) for r in ROLES},
        validation_leaflets=int(val.sum()),
        training=dict(steps=run["steps"], minutes=run["minutes"], history=run["history"]),
        sha256={"model.pt": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest()},
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    Generator(output, subset=data["names"][0])
    return metadata


class Generator:
    def __init__(
        self,
        model: Path,
        *,
        subset: str | None = None,
        steps: int = STEPS,
        maximum_geometry_attempts: int = 100,
        **_: Any,
    ) -> None:
        model = Path(model)
        self.maximum_geometry_attempts, self.steps = maximum_geometry_attempts, steps
        self.metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
        self.resolution = int(self.metadata.get("resolution", GRID))
        for name, digest in self.metadata.get("sha256", {}).items():
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{NAME} model checksum mismatch: {name}")
        self.subset = subset or self.metadata["subsets"][0]
        self.subset_index = neural.SUBSETS.index(self.subset)
        self.net, self.torch = build(self.resolution, tuple(self.metadata.get("widths", CHANNELS)))
        self.net.load_state_dict(self.torch.load(model / "model.pt", map_location="cpu"))
        self.device = neural.device()
        self.net.to(self.device).eval()

    def sample(self, count: int, seed: int) -> list[np.ndarray]:
        torch = self.torch
        g = torch.Generator(device=self.device).manual_seed(seed)
        out: list[np.ndarray] = []
        attempts, drawn = 0, 0
        while len(out) < count:
            if attempts >= 20 and len(out) < 0.1 * drawn:  # the model cannot draw valid outlines
                raise RuntimeError(f"{NAME}: {len(out)} valid outlines of {drawn} draws")
            n = min(32 if self.resolution <= 128 else 8, max(8, count - len(out)))
            drawn += n
            s = torch.full((n,), self.subset_index, dtype=torch.long, device=self.device)
            fields = neural.fm_sample(
                self.net,
                (n, 1, self.resolution, self.resolution),
                s,
                None,
                g,
                self.steps,
                self.device,
            )
            for f in fields[:, 0].cpu().numpy():
                c = contour_from_sdf(-f)
                if c is not None:
                    out.append(c)
            attempts += 1
        return out[:count]

    def generate(self, *, seed: int, role: LeafRole) -> LeafShape2D:
        if role not in ROLES:
            raise ValueError("role must be left, right or terminal")
        points = self.sample(1, seed)[0]
        return LeafShape2D(
            contour_xy=points,
            biological_base_xy=np.array([0.0, 0.0]),
            biological_tip_xy=np.array([0.0, 1.0]),
            features_h60=efd_features(points),
            provenance={
                "generator": NAME,
                "provider": NAME,
                "model_version": self.metadata.get("version"),
                "model_sha256": dict(self.metadata.get("sha256", {})),
                "dataset": self.metadata.get("dataset"),
                "subset": self.subset,
                "role": role,
                "seed": seed,
                "reflection_applied": False,
            },
        )
