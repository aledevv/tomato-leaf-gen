"""Stage 6: fit a shape model on a bank.

    python -m leafgen.shape.fit --bank bank.npz --output model_dir \
        --model rae|rae_flow|mask_diffusion [--minutes M] [--dataset NAME]

Writes the model folder that ``generate.py`` and ``evaluate.py`` load. The recipes of the final
models are in docs/reproduce.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leafgen.shape.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bank",
        type=Path,
        nargs="+",
        required=True,
        help="one bank, or several for the pooled models",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset", default="", help="label of the outline collection")
    parser.add_argument(
        "--minutes", type=float, default=None, help="training budget (neural models)"
    )
    parser.add_argument("--rae", type=Path, default=None, help="trained rae folder (rae_flow)")
    parser.add_argument("--latent", type=int, default=None, help="latent size (rae)")
    parser.add_argument("--resolution", type=int, default=None, help="mask pixels (mask_diffusion)")
    parser.add_argument("--widths", default=None, help="U-Net channels, e.g. 64,128,256,256")
    parser.add_argument(
        "--schedule", choices=("constant", "cosine"), default=None, help="learning rate (neural)"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="validations without a new best before stopping"
    )
    args = parser.parse_args()
    options = {"dataset": args.dataset}
    for key in ("minutes", "latent", "resolution", "schedule", "patience"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.rae is not None:
        options["rae_model"] = args.rae
    if args.widths:
        options["widths"] = tuple(int(w) for w in args.widths.split(","))
    bank = args.bank[0] if len(args.bank) == 1 else args.bank
    metadata = MODELS[args.model].fit(bank, args.output, **options)
    keys = ("model", "version", "training_counts", "density")
    print(json.dumps({k: metadata[k] for k in keys if k in metadata}, indent=2))


if __name__ == "__main__":
    main()
