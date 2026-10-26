"""Shape models: a name -> module registry.

A model module has ``fit(bank_path, output, **options)`` and a ``Generator`` class with
``generate(*, seed, role) -> LeafShape2D`` and ``sample(count, seed) -> list of contours``.
``rae`` is the autoencoder that ``rae_flow`` decodes with; ``rae_flow`` (default) and
``mask_diffusion`` are the two generators.
"""

from __future__ import annotations

from leafgen.shape.models import mask_diffusion, rae, rae_flow

MODELS = {"rae": rae, "rae_flow": rae_flow, "mask_diffusion": mask_diffusion}
