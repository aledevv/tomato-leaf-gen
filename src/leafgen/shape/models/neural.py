"""Shared pieces of the neural shape models.

Data: ``pooled`` loads one or more banks (development split) into pseudo-landmarks with a subset
index (the conditioning embedding) and the photo of every leaflet; ``split_val`` sets 15 % of the
photos aside for the generative validation. Networks: ``MLP`` (vectors) and ``UNet1D`` (circular
signals), both taking a time embedding, a subset embedding and an optional conditioning vector
(FiLM).
Objective: conditional flow matching with independent coupling (Lipman et al., ICLR 2023; I-CFM in
the terms of Tong et al., TMLR 2024: no minibatch optimal-transport pairing): straight paths
x_t = (1 - t) x0 + t x1 from Gaussian noise, target x1 - x0; sampling integrates the learned velocity with Heun's method. Training: AdamW, cosine
schedule (opt-in, ``schedule="cosine"``; constant after warmup by default), EMA weights, a time
budget in minutes, optional early stopping (``patience`` validations without a new best), and
model selection on generative metrics (1-NNA
in the landmark space against validation leaflets, and Meehan's Z_U against the training set) every
``every`` steps: the kept checkpoint is the best 1-NNA among those with Z_U > -5 (no copying).
Torch is imported inside the functions, so the package works without it.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from leafgen.shape.features import (
    ShapeBank,
    landmarks_to_contour,
    pseudo_landmarks,
    reconstruct_h60,
    valid_contour,
)

SUBSETS = ("chamber", "BILs", "field", "tomato_entire", "tomato_asymmetry")
VAL_FRACTION, VAL_SAMPLES, Z_U_FLOOR = 0.15, 1000, -5.0


# ----------------------------------------------------------------------------- data
def subset_of(bank_path: Path) -> str:
    """``sam3_<subset>_models/...`` -> subset name; otherwise the bank stem."""
    match = re.search(r"sam3_(.+?)_models", str(bank_path))
    return match.group(1) if match else Path(bank_path).stem


def pooled(bank_paths: list[Path]) -> dict:
    """Landmarks (N, 512), subset index (N,), photo (N,) and the subset names of the banks."""
    landmarks, subset, photo, names = [], [], [], []
    for path in bank_paths:
        bank = ShapeBank.load(Path(path))
        selected = bank.split == "development"
        name = subset_of(path)
        names.append(name)
        index = SUBSETS.index(name) if name in SUBSETS else len(SUBSETS) + len(names) - 1
        for f in bank.features[selected]:
            landmarks.append(pseudo_landmarks(reconstruct_h60(f)))
        subset += [index] * int(selected.sum())
        photo += [str(p) for p in bank.source_path[selected]]
    return dict(
        landmarks=np.asarray(landmarks),
        subset=np.asarray(subset, dtype=np.int64),
        photo=np.asarray(photo),
        names=names,
    )


def split_val(photo: np.ndarray, fraction: float = VAL_FRACTION, seed: int = 0) -> np.ndarray:
    """Boolean mask of the validation leaflets: ``fraction`` of the photos, drawn by seed."""
    photos = np.array(sorted(set(photo)))
    held = set(
        np.random.default_rng(seed).choice(
            photos, int(round(fraction * len(photos))), replace=False
        )
    )
    return np.array([p in held for p in photo])


def landmarks_metrics(generated: np.ndarray, val: np.ndarray, train: np.ndarray) -> dict:
    """1-NNA of generated vs validation landmarks and Z_U (generated vs validation, to training)."""
    from leafgen.metrics.distribution import distances_to, one_nna, z_u

    if len(generated) < 10:  # a model that cannot yet draw valid outlines
        return dict(one_nna=1.0, z_u=float("inf"), dcr_ratio=float("inf"))
    n = min(len(generated), len(val))
    g, v = generated[:n], val[:n]
    rr, gg, rg = cdist(v, v), cdist(g, g), cdist(v, g)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)
    dv, dg = distances_to(v, train)[:, 0], distances_to(g, train)[:, 0]
    return dict(
        one_nna=float(one_nna(rr, gg, rg)),
        z_u=float(z_u(dv, dg)),
        dcr_ratio=float(
            np.median(dg) / max(np.median(dv), 1e-12)
        ),  # < 1: closer to training than real
    )


def contours_from_landmarks(vectors: np.ndarray) -> list[np.ndarray | None]:
    """Closed 1024-point contours, None where the landmarks give an invalid outline."""
    out = []
    for v in vectors:
        try:
            pts = landmarks_to_contour(v)
            out.append(pts if valid_contour(pts) else None)
        except ValueError:
            out.append(None)
    return out


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- networks
def build_modules():
    """Network classes, defined lazily: torch is only needed when a model is fitted or sampled."""
    import torch
    from torch import nn

    class TimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t):  # t in [0, 1], shape (B,)
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            angles = t[:, None] * 1000 * freqs[None]
            return self.net(torch.cat([angles.sin(), angles.cos()], 1))

    class Conditioning(nn.Module):
        """time embedding + subset embedding + optional vector -> one context vector."""

        def __init__(self, dim: int, subsets: int, vector_dim: int):
            super().__init__()
            self.time = TimeEmbedding(dim)
            self.subset = nn.Embedding(subsets, dim)
            self.vector = nn.Linear(vector_dim, dim) if vector_dim else None

        def forward(self, t, subset, vector=None):
            h = self.time(t) + self.subset(subset)
            if self.vector is not None:
                h = h + self.vector(vector)
            return h

    class Block(nn.Module):
        def __init__(self, width: int, context: int):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.lin1, self.lin2 = nn.Linear(width, width), nn.Linear(width, width)
            self.film = nn.Linear(context, 2 * width)

        def forward(self, x, c):
            scale, shift = self.film(c).chunk(2, dim=-1)
            h = self.norm(x) * (1 + scale) + shift
            return x + self.lin2(torch.nn.functional.silu(self.lin1(torch.nn.functional.silu(h))))

    class MLP(nn.Module):
        """Velocity field on vectors: v(x, t | subset, vector)."""

        def __init__(self, dim, width=512, depth=4, context=128, subsets=8, vector_dim=0):
            super().__init__()
            self.cond = Conditioning(context, subsets, vector_dim)
            self.inp = nn.Linear(dim, width)
            self.blocks = nn.ModuleList(Block(width, context) for _ in range(depth))
            self.out = nn.Linear(width, dim)

        def forward(self, x, t, subset, vector=None):
            c = self.cond(t, subset, vector)
            h = self.inp(x)
            for block in self.blocks:
                h = block(h, c)
            return self.out(h)

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, context):
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, 5, padding=2, padding_mode="circular")
            self.conv2 = nn.Conv1d(cout, cout, 5, padding=2, padding_mode="circular")
            self.norm1, self.norm2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
            self.film = nn.Linear(context, 2 * cout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, c):
            h = torch.nn.functional.silu(self.norm1(self.conv1(x)))
            scale, shift = self.film(c)[:, :, None].chunk(2, dim=1)
            h = self.norm2(self.conv2(h)) * (1 + scale) + shift
            return self.skip(x) + torch.nn.functional.silu(h)

    class UNet1D(nn.Module):
        """Velocity field on circular signals (B, channels, length), length divisible by 4."""

        def __init__(self, channels, widths=(64, 128, 128), context=128, subsets=8, vector_dim=0):
            super().__init__()
            self.cond = Conditioning(context, subsets, vector_dim)
            self.inp = nn.Conv1d(channels, widths[0], 3, padding=1, padding_mode="circular")
            self.down = nn.ModuleList()
            for i, w in enumerate(widths):
                cin = widths[i - 1] if i else widths[0]
                self.down.append(ConvBlock(cin, w, context))
            self.mid = ConvBlock(widths[-1], widths[-1], context)
            self.up = nn.ModuleList()
            for i in reversed(range(len(widths))):
                cin = widths[i] + (widths[i + 1] if i + 1 < len(widths) else widths[-1])
                self.up.append(ConvBlock(cin, widths[i], context))
            self.out = nn.Conv1d(widths[0], channels, 3, padding=1, padding_mode="circular")

        def forward(self, x, t, subset, vector=None):
            c = self.cond(t, subset, vector)
            h = self.inp(x)
            skips = []
            for i, block in enumerate(self.down):
                h = block(h, c)
                skips.append(h)
                if i + 1 < len(self.down):
                    h = torch.nn.functional.avg_pool1d(h, 2)
            h = self.mid(h, c)
            for i, block in enumerate(self.up):
                skip = skips[-1 - i]
                if h.shape[-1] != skip.shape[-1]:
                    h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="linear")
                h = block(torch.cat([h, skip], 1), c)
            return self.out(h)

    return dict(MLP=MLP, UNet1D=UNet1D)


# ----------------------------------------------------------------------------- flow matching
def fm_loss(model, x1, subset, vector, generator):
    import torch

    x0 = torch.randn(x1.shape, generator=generator, device=x1.device)
    t = torch.rand(len(x1), generator=generator, device=x1.device)
    tb = t.reshape(-1, *([1] * (x1.ndim - 1)))
    xt = (1 - tb) * x0 + tb * x1
    return ((model(xt, t, subset, vector) - (x1 - x0)) ** 2).mean()


def fm_sample(model, shape, subset, vector, generator, steps=50, device="cpu"):
    """Heun integration of dx/dt = v from t = 0 (noise) to 1; returns a tensor on ``device``."""
    import torch

    x = torch.randn(shape, generator=generator, device=device)
    ones = torch.ones(shape[0], device=device)
    with torch.no_grad():
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            v0 = model(x, ones * t0, subset, vector)
            x1 = x + v0 / steps
            v1 = model(x1, ones * t1, subset, vector)
            x = x + 0.5 * (v0 + v1) / steps
    return x


def train(
    model,
    data,
    subset,
    vector=None,
    *,
    loss_fn=None,
    validate=None,
    minutes: float = 10.0,
    max_steps: int = 200_000,
    batch: int = 256,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    ema: float = 0.999,
    every: int = 1000,
    seed: int = 0,
    log=print,
    resume: Path | None = None,
    save_minutes: float = 10.0,
    schedule: str = "constant",
    patience: int = 0,
) -> dict:
    """Train ``model`` on ``data`` (tensor, first axis = leaflets) with the flow-matching loss
    (or ``loss_fn(model, x, subset, vector, generator)``), for ``minutes`` at most.
    ``validate(ema_model)`` returns dict(one_nna, z_u); the EMA weights of the best validation are
    loaded into ``model`` at the end. Returns the history.

    ``resume``: a file (``last.pt``) holding the whole training state (weights, EMA, optimizer, step,
    random stream, history, best checkpoint, minutes already trained), written every
    ``save_minutes`` and when the process gets SIGINT or SIGTERM, which then exits with code 3 (a
    pause). Rerunning the same command continues from it; the budget counts the minutes of all the
    sessions. The file is removed when the training ends.

    ``schedule``: "constant" (after a 500-step warmup) or "cosine", which decays the learning rate
    to zero at the end of the budget (whichever of ``minutes`` and ``max_steps`` is nearer).
    ``patience``: stop after that many validations in a row without a new best (0 = never)."""
    import copy
    import os
    import signal
    import sys

    import torch

    dev = data.device
    torch.manual_seed(seed)
    generator = torch.Generator(device=dev).manual_seed(seed)
    average = copy.deepcopy(model).eval()
    for p in average.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start, history, best = time.time(), [], None
    best_state, step = copy.deepcopy(average.state_dict()), 0
    if resume is not None and Path(resume).exists():
        state = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        average.load_state_dict(state["average"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"].to(generator.get_state().device))
        step, history, best, best_state = (
            state["step"], state["history"], state["best"], state["best_state"]
        )
        start = time.time() - state["minutes"] * 60
        log(f"resumed at step={step} minutes={state['minutes']:.1f}")

    def save() -> None:
        if resume is None:
            return
        state = dict(
            model=model.state_dict(), average=average.state_dict(),
            optimizer=optimizer.state_dict(), generator=generator.get_state(), step=step,
            history=history, best=best, best_state=best_state,
            minutes=(time.time() - start) / 60,
        )
        torch.save(state, f"{resume}.tmp")
        os.replace(f"{resume}.tmp", resume)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    saved = time.time()
    loss_fn = loss_fn or fm_loss
    n = len(data)
    budget = minutes * 60
    while step < max_steps and time.time() - start < budget:
        if stop:
            save()
            log(f"paused at step={step} minutes={(time.time() - start) / 60:.1f}")
            sys.exit(3)
        if resume is not None and time.time() - saved > save_minutes * 60:
            save()
            saved = time.time()
        idx = torch.randint(0, n, (min(batch, n),), generator=generator, device=dev)
        loss = loss_fn(
            model, data[idx], subset[idx], None if vector is None else vector[idx], generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rate = lr * min(1.0, step / 500)
        if schedule == "cosine":
            progress = min(1.0, max(step / max_steps, (time.time() - start) / budget))
            rate *= 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        with torch.no_grad():
            for a, p in zip(average.parameters(), model.parameters(), strict=True):
                a.lerp_(p, 1 - ema)
            for a, p in zip(average.buffers(), model.buffers(), strict=True):
                a.copy_(p)
        step += 1
        if step % every == 0 or time.time() - start >= budget or step == max_steps:
            record = dict(step=step, loss=loss.item(), minutes=(time.time() - start) / 60)
            if validate is not None:
                record.update(validate(average))
                score = record["one_nna"] if record["z_u"] > Z_U_FLOOR else 1.0 + record["one_nna"]
                if best is None or score < best:
                    best, best_state = score, copy.deepcopy(average.state_dict())
                    record["best"] = True
            history.append(record)
            log(
                " ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in record.items()
                )
            )
            since = next(
                (i for i, r in enumerate(reversed(history)) if r.get("best")), len(history)
            )
            if validate is not None and patience > 0 and since >= patience:
                log(f"early stop: {patience} validations without a new best")
                break
    model.load_state_dict(best_state if validate is not None else average.state_dict())
    model.eval()
    if resume is not None and Path(resume).exists():
        Path(resume).unlink()
    return dict(history=history, steps=step, minutes=(time.time() - start) / 60)
