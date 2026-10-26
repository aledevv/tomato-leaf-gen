"""Shape model "rae_flow" (part 3): the decoder of a trained ``rae`` model with a flow-matching
density on its codes instead of the per-subset Gaussian mixture.

Ghosh et al. (ICLR 2020) leave the ex-post density of a regularised autoencoder free; here it is
the velocity MLP of ``landmark_flow`` (conditional flow matching, Heun sampler) trained on the
whitened codes of the pooled development leaflets, with the subset embedding. The RAE itself is
not retrained: ``fit`` takes the banks the RAE was trained on (its ``metadata.json`` lists them)
and ``rae`` = the RAE folder; the codes are recomputed with the saved encoder and the same
validation photos, and the checkpoint is chosen on 1-NNA / Z_U of decoded draws.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from leafgen.contracts.geometry import LeafRole, LeafShape2D
from leafgen.shape.features import ROLES, efd_features
from leafgen.shape.models import neural, rae

NAME = "rae_flow"
STEPS = 50


def _load_rae(folder: Path):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    net, torch = rae.build(metadata["latent"])
    net.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    with np.load(folder / "norm.npz", allow_pickle=False) as d:
        mean, std = d["mean"], float(d["std"][0])
    return metadata, net, torch, mean, std


def fit(
    bank_path: Path | list[Path],
    output: Path,
    *,
    rae_model: Path,
    minutes: float = 10.0,
    dataset: str = "",
    seed: int = 0,
    width: int = 512,
    depth: int = 4,
    jitter: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    """``jitter``: Gaussian noise (whitened units) on the training codes at every step, against
    the pull towards the training set (as in ``landmark_flow``)."""
    banks = [Path(b) for b in (bank_path if isinstance(bank_path, list) else [bank_path])]
    rae_model = Path(rae_model)
    rae_meta, ae, torch, mean, std = _load_rae(rae_model)
    data = neural.pooled(banks)
    val = neural.split_val(data["photo"], seed=seed)
    train_idx, val_idx = np.flatnonzero(~val), np.flatnonzero(val)
    dev = neural.device()
    ae.to(dev).eval()
    x = torch.as_tensor(
        ((data["landmarks"] - mean) / std).reshape(-1, 256, 2).transpose(0, 2, 1),
        dtype=torch.float32,
        device=dev,
    )
    with torch.no_grad():
        codes = np.concatenate(
            [ae.encode(x[i : i + 2048]).cpu().numpy() for i in range(0, len(x), 2048)]
        )
    c_mean, c_std = codes[train_idx].mean(0), codes[train_idx].std(0) + 1e-9
    latent = codes.shape[1]
    z = torch.as_tensor((codes - c_mean) / c_std, dtype=torch.float32, device=dev)
    subset = torch.as_tensor(data["subset"], device=dev)
    net = neural.build_modules()["MLP"](latent, width=width, depth=depth).to(dev)

    def decode(white: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            c = torch.as_tensor(white * c_std + c_mean, dtype=torch.float32, device=dev)
            y = ae.decode(c).cpu().numpy()
        return y.transpose(0, 2, 1).reshape(len(white), -1) * std + mean

    def validate(average):
        g = torch.Generator(device=dev).manual_seed(seed + 1)
        rows = val_idx[: neural.VAL_SAMPLES]
        s = torch.as_tensor(data["subset"][rows], device=dev)
        w = neural.fm_sample(average, (len(rows), latent), s, None, g, STEPS, dev)
        return neural.landmarks_metrics(
            decode(w.cpu().numpy()), data["landmarks"][val_idx], data["landmarks"][train_idx]
        )

    def loss_fn(model, x1, subset_rows, vector, generator):
        if jitter > 0:
            x1 = x1 + jitter * torch.randn(x1.shape, generator=generator, device=x1.device)
        return neural.fm_loss(model, x1, subset_rows, vector, generator)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "train.log").open("w")
    run = neural.train(
        net,
        z[train_idx],
        subset[train_idx],
        loss_fn=loss_fn,
        minutes=minutes,
        seed=seed,
        validate=validate,
        log=lambda s: (print(s), log.write(s + "\n"), log.flush()),
    )
    log.close()
    torch.save(net.state_dict(), output / "model.pt")
    np.savez(output / "codes_norm.npz", mean=c_mean, std=c_std)
    files = ["model.pt", "codes_norm.npz"]
    label = dataset or "pool"
    metadata = dict(
        model=NAME,
        representation="landmarks_128",
        rae_model=str(rae_model),
        rae_sha256=rae_meta.get("sha256", {}),
        latent=int(latent),
        network=dict(kind="MLP", width=width, depth=depth, steps=STEPS),
        jitter=jitter,
        subsets=data["names"],
        roles=list(ROLES),
        role_specific=False,
        dataset=label,
        version=f"{label.lower()}_rae_flow_z{latent}" + (f"_j{jitter:g}" if jitter else ""),
        banks=[str(b) for b in banks],
        training_counts={r: int(len(train_idx)) for r in ROLES},
        validation_leaflets=int(val.sum()),
        training=dict(steps=run["steps"], minutes=run["minutes"], history=run["history"]),
        sha256={f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    Generator(output, subset=data["names"][0])
    return metadata


class Generator:
    """Draw whitened codes with the flow for ``subset`` and decode them with the RAE."""

    def __init__(
        self,
        model: Path,
        *,
        subset: str | None = None,
        maximum_geometry_attempts: int = 100,
        **_: Any,
    ) -> None:
        model = Path(model)
        self.maximum_geometry_attempts = maximum_geometry_attempts
        self.metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
        for name, digest in self.metadata.get("sha256", {}).items():
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{NAME} model checksum mismatch: {name}")
        self.subset = subset or self.metadata["subsets"][0]
        if self.subset not in self.metadata["subsets"]:
            raise ValueError(f"{NAME}: subset {self.subset} not in the training pool")
        self.subset_index = neural.SUBSETS.index(self.subset)
        rae_folder = Path(self.metadata["rae_model"])
        if not rae_folder.is_absolute() and (model / rae_folder).is_dir():
            rae_folder = model / rae_folder  # stored next to this model
        _, self.ae, self.torch, self.mean, self.std = _load_rae(rae_folder)
        self.device = neural.device()
        self.ae.to(self.device).eval()
        net = self.metadata["network"]
        self.net = neural.build_modules()["MLP"](
            self.metadata["latent"], width=net["width"], depth=net["depth"]
        )
        self.net.load_state_dict(self.torch.load(model / "model.pt", map_location="cpu"))
        self.net.to(self.device).eval()
        with np.load(model / "codes_norm.npz", allow_pickle=False) as d:
            self.c_mean, self.c_std = d["mean"], d["std"]

    def sample(self, count: int, seed: int) -> list[np.ndarray]:
        torch = self.torch
        g = torch.Generator(device=self.device).manual_seed(seed)
        out: list[np.ndarray] = []
        attempts = 0
        while len(out) < count and attempts < self.maximum_geometry_attempts:
            n = min(2048, max(64, count - len(out)))
            s = torch.full((n,), self.subset_index, dtype=torch.long, device=self.device)
            w = neural.fm_sample(self.net, (n, self.metadata["latent"]), s, None, g, STEPS, self.device)
            c = w * torch.as_tensor(self.c_std, dtype=torch.float32, device=self.device)
            c = c + torch.as_tensor(self.c_mean, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                y = self.ae.decode(c).cpu().numpy()
            landmarks = y.transpose(0, 2, 1).reshape(n, -1) * self.std + self.mean
            out += [p for p in neural.contours_from_landmarks(landmarks) if p is not None]
            attempts += 1
        if len(out) < count:
            raise RuntimeError(f"{NAME}: only {len(out)} valid outlines of {count}")
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
                "provider": f"{NAME}_z{self.metadata['latent']}",
                "model_version": self.metadata.get("version"),
                "model_sha256": dict(self.metadata.get("sha256", {})),
                "dataset": self.metadata.get("dataset"),
                "subset": self.subset,
                "role": role,
                "seed": seed,
                "reflection_applied": False,
            },
        )
