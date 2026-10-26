"""Where the downloaded data and weights live (see docs/data.md); both folders sit at the repository
root unless LEAFGEN_DATA / LEAFGEN_WEIGHTS point elsewhere."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LEAFGEN_DATA", ROOT / "data"))
WEIGHTS = Path(os.environ.get("LEAFGEN_WEIGHTS", ROOT / "weights"))
