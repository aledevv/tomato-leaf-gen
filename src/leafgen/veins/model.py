"""VeinFlow: veins drawn inside an outline by a conditional flow-matching U-Net.

A U-Net (diffusers ``UNet2DModel``) learns p(veins | outline mask) on the vein rasters of
``veins.raster`` (256 px with the finer veins, the order coded by brightness), with the mask as a
second input channel and the leaf collection as class embedding; flow matching with straight
paths from Gaussian noise (Lipman et al., ICLR 2023), the training loop of ``shape.models.neural``
(EMA weights, time budget, cosine schedule). Validation every ``every`` steps: draws for ``VAL``
development outlines, 1-NNA of their vein statistics against the real ones (``veins.metrics``);
the checkpoint kept is the one with 1-NNA nearest 0.5 (``one_nna`` holds 0.5 + |1-NNA - 0.5|).
``--split photo --dev splits/veins_dev_ids.txt`` keeps whole photos aside for that validation.

    python -m leafgen.veins.model fit --output weights/veinflow --size 256 --fine \
        --widths 32 64 128 256 256 --batch 8 --split photo --dev splits/veins_dev_ids.txt \
        --schedule cosine --minutes 175
    python -m leafgen.veins.model sample --model weights/veinflow --subset chamber \
        --output drawings.npz [--limit 500]

Options kept from the experiments, not used by the released model: ``--cascade`` (second stage of
a 128 -> 256 cascade conditioned on a noisy coarse drawing, Ho et al. 2022), ``--orders`` (one
output channel per vein order), ``--skeleton-weight`` (Skeleton Recall loss), ``--init``
(fine-tuning).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leafgen.paths import DATA
from leafgen.veins import metrics, raster

DERIVED = DATA
SUBSETS = ("chamber", "BILs", "field", "tomato_entire", "tomato_asymmetry")
CHANNELS, STEPS, VAL, DEV = (32, 64, 128, 128), 50, 256, 300
SIGMA_MAX, SIGMA = 0.5, 0.1  # cascade: noise on the coarse image, training range and sampling


def build(
    size: int = 128,
    widths: tuple[int, ...] = CHANNELS,
    layers: int = 1,
    extra: int = 0,
    channels: int = 1,
):
    """``extra``: condition channels after the mask (2 for the cascade: coarse veins, sigma);
    ``channels``: vein channels (4 with ``--orders``)."""
    import torch
    from diffusers import UNet2DModel

    net = UNet2DModel(
        sample_size=size,
        in_channels=channels + 1 + extra,
        out_channels=channels,
        layers_per_block=layers,
        block_out_channels=tuple(widths),
        down_block_types=("DownBlock2D",) * (len(widths) - 1) + ("AttnDownBlock2D",),
        up_block_types=("AttnUpBlock2D",) + ("UpBlock2D",) * (len(widths) - 1),
        num_class_embeds=len(SUBSETS),
    )

    class Velocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = net

        def forward(self, x, t, subset, cond):  # cond: mask [+ coarse veins, sigma]
            return self.unet(torch.cat([x, cond], 1), t * 1000, class_labels=subset).sample

    return Velocity(), torch


def _load(subsets: list[str], bank: str, size: int, coarse: bool = False):
    """``coarse``: also the 128 px three-order raster of each leaflet (``coarse``, uint8 128 px)."""
    veins, masks, subset, ids, photo, small = [], [], [], [], [], []
    for name in subsets:
        r = np.load(DERIVED / f"sam3_{name}_veins" / f"raster{size}_{bank}.npz")
        veins.append(r["veins"])
        masks.append(r["masks"])
        if coarse:
            c = np.load(DERIVED / f"sam3_{name}_veins" / f"raster128_{bank}.npz")
            where = {k: i for i, k in enumerate(c["instance_id"])}
            small.append(c["veins"][[where[k] for k in r["instance_id"]]])
        subset += [SUBSETS.index(name)] * len(r["veins"])
        ids += list(r["instance_id"])
        photo += ["__".join(i.split("__")[:3]) for i in r["instance_id"]]
    return dict(
        veins=np.concatenate(veins),
        masks=np.concatenate(masks),
        subset=np.asarray(subset),
        ids=np.asarray(ids),
        photo=np.asarray(photo),
        row=np.arange(len(ids)),  # position in the subsets as loaded (for ``_Orders``)
    ) | ({"coarse": np.concatenate(small)} if coarse else {})


class _Orders:
    """Per-order rasters of ``veins.orders`` for the leaflets of ``_load`` (same subsets, same
    order), memory-mapped; ``fetch`` reads the rows of a batch."""

    def __init__(self, subsets: list[str], bank: str, size: int):
        from leafgen.veins import orders

        self.parts = [np.load(orders.path(s, bank, size), mmap_mode="r") for s in subsets]
        self.start = np.cumsum([0] + [len(p) for p in self.parts])

    def fetch(self, rows: np.ndarray) -> np.ndarray:
        part = np.searchsorted(self.start, rows, side="right") - 1
        return np.stack([self.parts[k][r - self.start[k]] for k, r in zip(part, rows, strict=True)])


def _condition(torch, mask, coarse, sigma, noise):
    """Condition channels on the device: mask in [-1, 1] (B, 1, H, W), then for the cascade the
    uint8 coarse veins upsampled to H x W, in [-1, 1], plus ``noise`` (same shape) * sigma, and a
    constant sigma channel."""
    mask = mask.float() * 2 - 1
    if coarse is None:
        return mask
    c = torch.nn.functional.interpolate(coarse.float() / 255 * 2 - 1, size=mask.shape[2:])
    level = sigma.reshape(-1, 1, 1, 1)
    return torch.cat([mask, c + noise * level, level.expand_as(mask)], 1)


def _f1(pred: np.ndarray, real: np.ndarray, masks: np.ndarray, tolerance: int = 1) -> float:
    """Pixel F1 of vein pixels (> 0.3) with a ``tolerance`` px dilation, averaged over images."""
    from scipy.ndimage import binary_dilation

    out = []
    for p, r, m in zip(pred, real, masks, strict=True):
        p, r = (p > metrics.VEIN) & (m > 0), (r > metrics.VEIN) & (m > 0)
        precision = (p & binary_dilation(r, iterations=tolerance)).sum() / max(p.sum(), 1)
        recall = (r & binary_dilation(p, iterations=tolerance)).sum() / max(r.sum(), 1)
        out.append(2 * precision * recall / max(precision + recall, 1e-9))
    return float(np.mean(out))


def _skeleton_recall(torch, xt, v, x1, t, threshold: float = 0.4, tau: float = 0.1):
    """Skeleton Recall (Kirchhoff et al., ECCV 2024) on the clean image the velocity predicts,
    x1_hat = x_t + (1 - t) v: 1 - share of the target's coarse-vein skeleton (value >
    ``threshold``, skeletonised, dilated 1 px) covered by the soft vein map
    sigmoid((x1_hat - threshold) / tau), in [0, 1] units; weighted by (2t - 1)+, since at early times
    x1_hat is a blurred mean."""
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize

    target = ((x1[:, 0] + 1) / 2 > threshold).cpu().numpy()
    tube = np.stack([binary_dilation(skeletonize(a), iterations=1) for a in target])
    tube = torch.as_tensor(tube, device=v.device).float()
    x1_hat = (xt + (1 - t.reshape(-1, 1, 1, 1)) * v)[:, 0]
    soft = torch.sigmoid(((x1_hat + 1) / 2 - threshold) / tau)
    recall = (soft * tube).sum((1, 2)) / tube.sum((1, 2)).clamp(min=1)
    w = (2 * t - 1).clamp(min=0)
    return (w * (1 - recall)).sum() / w.sum().clamp(min=1e-6)


def draw(
    net,
    torch,
    masks: np.ndarray,
    subset: np.ndarray,
    seed: int,
    device,
    batch: int = 32,
    coarse: np.ndarray | None = None,
    sigma: float = SIGMA,
    channels: int = 1,
):
    """Vein images in [0, 1], float32, for uint8/bool ``masks`` (and uint8 128 px ``coarse``
    veins for the cascade); per-order channels are composed into one (``raster.compose``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, len(masks), batch):
        m = torch.as_tensor(masks[i : i + batch, None].astype(np.uint8), device=device)
        s = torch.as_tensor(subset[i : i + batch], device=device)
        c = None if coarse is None else torch.as_tensor(coarse[i : i + batch, None], device=device)
        level = torch.full((len(m),), sigma, device=device)
        noise = None
        if c is not None:
            noise = torch.randn((len(m), 1, *m.shape[2:]), generator=g, device=device)
        cond = _condition(torch, m, c, level, noise)
        x = ((fm_sample(net, cond, s, g, channels).clamp(-1, 1) + 1) / 2).cpu().numpy()
        x = raster.compose(x) if channels > 1 else x[:, 0]
        out.append(x * (masks[i : i + batch] > 0))
    return np.concatenate(out).astype(np.float32)


def fm_sample(net, cond, subset, generator, channels: int = 1, steps: int = STEPS):
    from leafgen.shape.models import neural

    shape = (len(cond), channels, *cond.shape[2:])
    return neural.fm_sample(net, shape, subset, cond, generator, steps, cond.device)


def fit(
    subsets: list[str],
    output: Path,
    *,
    size: int = 128,
    widths: tuple[int, ...] = CHANNELS,
    fine: bool = False,
    minutes: float = 60.0,
    overfit: int = 0,
    seed: int = 0,
    schedule: str = "constant",
    patience: int = 0,
    every: int = 2000,
    batch: int = 32,
    layers: int = 1,
    vein_weight: float = 0.0,
    times: str = "uniform",
    split: str = "leaflet",
    cascade: bool = False,
    dev_ids: Path | None = None,
    orders: bool = False,
    init: Path | None = None,
    skeleton_weight: float = 0.0,
    lr: float = 1e-4,
) -> dict:
    from leafgen.shape.models import neural

    # model selection and early stopping on VAL training leaflets left out of the fit, so the
    # held-out split stays untouched until the final evaluation
    train = _load(subsets, "split75", size, cascade)
    rng = np.random.default_rng(seed)
    if overfit:
        keep = np.sort(rng.permutation(len(train["veins"]))[:overfit])
        train = {k: v[keep] for k, v in train.items()}
        held = train
    elif split == "leaflet":
        order = rng.permutation(len(train["veins"]))
        held = {k: v[np.sort(order[:VAL])] for k, v in train.items()}
        train = {k: v[np.sort(order[VAL:])] for k, v in train.items()}
    else:
        # by photo: whole photos go to the validation leaflets (VAL, model selection) and, per
        # subset, to a development set (DEV, to compare variants; written to dev_ids.txt), so
        # no leaflet of a validation or development photo is in the fit
        photos = rng.permutation(np.unique(train["photo"]))
        out, val_photos, dev_photos = np.zeros(len(train["veins"]), int), set(), set()
        count = dict.fromkeys(SUBSETS, 0)
        per_photo = {p: np.flatnonzero(train["photo"] == p) for p in photos}
        given = None
        if dev_ids is not None:
            listed = np.isin(train["ids"], Path(dev_ids).read_text().split())
            given = set(train["photo"][listed])
        for p in photos:
            rows = per_photo[p]
            name = SUBSETS[train["subset"][rows[0]]]
            if given is not None:
                if p in given:
                    dev_photos.add(p)
                elif (out == 1).sum() < VAL:
                    val_photos.add(p)
                    out[rows] = 1
            elif count[name] < DEV:
                dev_photos.add(p)
                count[name] += len(rows)
            elif (out == 1).sum() < VAL:
                val_photos.add(p)
                out[rows] = 1
        dev_rows = np.flatnonzero(np.isin(train["photo"], list(dev_photos)))
        out[dev_rows] = 2
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "dev_ids.txt").write_text("\n".join(train["ids"][dev_rows]) + "\n")
        held = {k: v[out == 1] for k, v in train.items()}
        train = {k: v[out == 0] for k, v in train.items()}
    val = np.sort(rng.permutation(len(held["veins"]))[: min(VAL, len(held["veins"]))])
    real_stats = metrics.table(held["veins"][val], held["masks"][val], fine)
    if orders and (cascade or not fine):
        raise ValueError("--orders needs --fine and no --cascade")
    channels = 4 if orders else 1
    store = _Orders(subsets, "split75", size) if orders else None
    net, torch = build(size, widths, layers, 2 if cascade else 0, channels)
    if init is not None:  # fine-tuning: start from a trained model's weights
        net.load_state_dict(torch.load(Path(init) / "model.pt", map_location="cpu"))
    dev = neural.device()
    net.to(dev)
    # uint8 in host memory, so the GPU keeps room for the display too; a batch goes to the GPU
    # as [-1, 1] float32 inside the loss (noise and times drawn on the CPU, where the training
    # random stream lives)
    # with --orders the batch rows are read from the memory-mapped per-order rasters in the loss
    x = torch.as_tensor(train["row"]) if orders else torch.as_tensor(train["veins"][:, None])
    m = torch.as_tensor(train["masks"][:, None].astype(np.uint8))
    s = torch.as_tensor(train["subset"])
    # the cascade needs the coarse image of each batch row: neural.train then hands over row
    # numbers in the subset slot (same batches, it only indexes them)
    small = torch.as_tensor(train["coarse"][:, None]) if cascade else None
    rows = torch.arange(len(s)) if cascade else s

    def loss_fn(model, x1, subset_rows, mask_rows, generator):
        if orders:
            x1 = torch.as_tensor(store.fetch(x1.numpy()))
        coarse, level, noise = None, None, None
        if cascade:
            coarse, subset_rows = small[subset_rows].to(dev), s[subset_rows]
            level = (torch.rand(len(x1), generator=generator) * SIGMA_MAX).to(dev)
            noise = torch.randn(x1.shape, generator=generator).to(dev)
        cond = _condition(torch, mask_rows.to(dev), coarse, level, noise)
        subset_rows = subset_rows.to(dev)
        x0 = torch.randn(x1.shape, generator=generator).to(dev)
        if times == "logit-normal":  # Esser et al. 2024: more weight on the middle times
            t = torch.sigmoid(torch.randn(len(x1), generator=generator)).to(dev)
        else:
            t = torch.rand(len(x1), generator=generator).to(dev)
        weight = 1 + vein_weight * (x1 > 0).to(dev).float()  # vein pixels count more
        x1 = x1.to(dev).float() / 255 * 2 - 1
        tb = t.reshape(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * x1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            v = model(xt, t, subset_rows, cond)
        loss = (weight * (v.float() - (x1 - x0)) ** 2).sum() / weight.sum()
        if skeleton_weight:
            loss = loss + skeleton_weight * _skeleton_recall(torch, xt, v.float(), x1, t)
        return loss

    def validate(average):
        coarse = held["coarse"][val] if cascade else None  # real coarse veins (training only)
        pred = draw(
            average,
            torch,
            held["masks"][val],
            held["subset"][val],
            seed + 1,
            dev,
            coarse=coarse,
            channels=channels,
        )
        gen = metrics.table(pred, held["masks"][val], fine)
        r = metrics.compare(gen, real_stats)
        from leafgen.veins import connect

        lcc = connect.summary(pred, held["masks"][val])["lcc_coarse_median"]
        f1 = _f1(pred, held["veins"][val] / 255.0, held["masks"][val])
        # neural.train keeps the lowest one_nna; memorising (--overfit) is judged by F1
        score = 1 - f1 if overfit else 0.5 + abs(r["one_nna"] - 0.5)
        return {
            "one_nna": score,
            "nna": r["one_nna"],
            "w1_mean": r["w1_mean"],
            "f1": f1,
            "lcc": lcc,
            "z_u": 0.0,
        }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    log = (output / "train.log").open("a" if last.exists() else "w")
    run = neural.train(
        net,
        x,
        rows,
        m,
        loss_fn=loss_fn,
        validate=validate,
        minutes=minutes,
        seed=seed,
        batch=batch,
        lr=lr,
        every=every,
        resume=last,
        schedule=schedule,
        patience=patience,
        log=lambda line: (print(line), log.write(line + "\n"), log.flush()),
    )
    log.close()
    torch.save(net.state_dict(), output / "model.pt")
    meta = dict(
        size=size,
        widths=list(widths),
        layers=layers,
        fine=fine,
        vein_weight=vein_weight,
        times=times,
        split=split,
        cascade=cascade,
        orders=orders,
        init=None if init is None else str(init),
        skeleton_weight=skeleton_weight,
        lr=lr,
        subsets=subsets,
        overfit=overfit,
        schedule=schedule,
        batch=batch,
        training_leaflets=len(train["veins"]),
        steps=run["steps"],
        minutes=run["minutes"],
        history=run["history"],
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def sample(
    model: Path,
    subset: str,
    output: Path,
    limit: int = 500,
    seed: int = 0,
    ids: Path | None = None,
    coarse: Path | None = None,
    sigma: float = SIGMA,
) -> None:
    """Drawings for held-out outlines, or with ``ids`` for those training outlines (the
    development set of a ``--split photo`` run). A cascade model needs ``coarse``: a ``sample``
    output of a 128 px model on the same outlines, or ``real`` for the real 128 px veins (upper
    bound: what the second stage does with perfect coarse veins)."""
    from leafgen.shape.models import neural

    meta = json.loads((model / "metadata.json").read_text())
    cascade = meta.get("cascade", False)
    channels = 4 if meta.get("orders", False) else 1
    net, torch = build(
        meta["size"], tuple(meta["widths"]), meta.get("layers", 1), 2 if cascade else 0, channels
    )
    net.load_state_dict(torch.load(model / "model.pt", map_location="cpu"))
    dev = neural.device()
    net.to(dev).eval()
    bank = "split75_holdout" if ids is None else "split75"
    held = _load([subset], bank, meta["size"], cascade and str(coarse) == "real")
    if ids is not None:
        keep = np.isin(held["ids"], Path(ids).read_text().split())
        held = {k: v[keep] for k, v in held.items()}
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.permutation(len(held["veins"]))[:limit])
    small = None
    if cascade:
        if str(coarse) == "real":
            small = held["coarse"][pick]
        else:
            g = np.load(coarse)
            where = {k: i for i, k in enumerate(g["instance_id"])}
            keep = np.asarray([k in where for k in held["ids"][pick]])
            pick = pick[keep]  # outlines the 128 px model drew
            small = g["veins"][[where[k] for k in held["ids"][pick]]]
    pred = draw(
        net,
        torch,
        held["masks"][pick],
        held["subset"][pick],
        seed,
        dev,
        coarse=small,
        sigma=sigma,
        channels=channels,
    )
    np.savez_compressed(
        output, instance_id=held["ids"][pick], veins=(pred * 255).round().astype(np.uint8)
    )
    print(f"{len(pick)} drawings -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--size", type=int, default=128)
    f.add_argument("--widths", type=int, nargs="+", default=list(CHANNELS))
    f.add_argument("--fine", action="store_true", help="rasters with the finer veins")
    f.add_argument("--minutes", type=float, default=60.0)
    f.add_argument("--overfit", type=int, default=0)
    f.add_argument("--every", type=int, default=2000)
    f.add_argument("--batch", type=int, default=32)
    f.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    f.add_argument("--patience", type=int, default=0)
    f.add_argument("--layers", type=int, default=1, help="resnet blocks per U-Net level")
    f.add_argument("--vein-weight", type=float, default=0.0, help="extra loss weight on veins")
    f.add_argument("--times", choices=("uniform", "logit-normal"), default="uniform")
    f.add_argument("--split", choices=("leaflet", "photo"), default="leaflet")
    f.add_argument("--cascade", action="store_true", help="also condition on 128 px veins")
    f.add_argument("--dev", type=Path, default=None, help="dev_ids.txt to reuse (--split photo)")
    f.add_argument("--orders", action="store_true", help="one channel per vein order (--fine)")
    f.add_argument("--init", type=Path, default=None, help="model folder to fine-tune")
    f.add_argument("--skeleton-weight", type=float, default=0.0, help="Skeleton Recall loss")
    f.add_argument("--lr", type=float, default=1e-4)
    g = sub.add_parser("sample")
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--subset", required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--limit", type=int, default=500)
    g.add_argument("--ids", type=Path, default=None, help="dev_ids.txt: training outlines")
    g.add_argument("--coarse", default=None, help="cascade: 128 px drawings .npz, or real")
    g.add_argument("--sigma", type=float, default=SIGMA, help="cascade: noise on the coarse")
    g.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "fit":
        meta = fit(
            args.subsets,
            args.output,
            size=args.size,
            widths=tuple(args.widths),
            fine=args.fine,
            minutes=args.minutes,
            overfit=args.overfit,
            schedule=args.schedule,
            patience=args.patience,
            every=args.every,
            batch=args.batch,
            layers=args.layers,
            vein_weight=args.vein_weight,
            times=args.times,
            split=args.split,
            cascade=args.cascade,
            dev_ids=args.dev,
            orders=args.orders,
            init=args.init,
            skeleton_weight=args.skeleton_weight,
            lr=args.lr,
        )
        print(json.dumps({k: meta[k] for k in ("steps", "minutes", "training_leaflets")}))
    else:
        sample(
            args.model,
            args.subset,
            args.output,
            args.limit,
            seed=args.seed,
            ids=args.ids,
            coarse=args.coarse,
            sigma=args.sigma,
        )


if __name__ == "__main__":
    main()
