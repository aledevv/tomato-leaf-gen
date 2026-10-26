"""Veins: VeinFlow drawings inside an outline, their repair into one tree, the pillow relief.

    model     VeinFlow, flow matching on vein rasters conditioned on the outline mask
    connect   repair of a drawing: loops cut, detached pieces joined (bridge_loopcut)
    tree      drawing -> drainage tree rooted at the base, branches ranked by flow
    relief    UV height maps and albedo from a tree (pillow style)
    metrics   the 16 vein statistics; known: DINOv2 and the botanical space (VLA + Ellis)
"""

from leafgen.veins.tree import VeinTree

__all__ = ["VeinTree"]
