"""Convert reviewed pixel masks into dimensionless attachment-tip contours."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from leafgen.contracts.geometry import contour_self_intersects, signed_area
from leafgen.jsonl import read_jsonl


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    keep = lengths > 1e-12
    points = points[keep]
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(lengths.sum())
    if len(points) < 4 or perimeter <= 1e-12:
        raise ValueError("mask contour is degenerate")
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    targets = np.arange(count, dtype=np.float64) * perimeter / count
    edges = np.searchsorted(np.cumsum(lengths), targets, side="right")
    fractions = (targets - starts[edges]) / lengths[edges]
    return points[edges] + fractions[:, None] * (following[edges] - points[edges])


def canonical_contour(points: np.ndarray, base: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Proper rotation and scale (no reflection) taking base to (0, 0) and tip to (0, 1),
    counter-clockwise, 1024 points by arc length, sigma-0.5 smoothing; raises when degenerate."""
    base, tip = np.asarray(base, dtype=np.float64), np.asarray(tip, dtype=np.float64)
    direction = tip - base
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError("base and tip collapse")
    longitudinal = direction / length
    lateral = np.asarray([longitudinal[1], -longitudinal[0]])
    relative = np.asarray(points, dtype=np.float64) - base
    canonical = np.column_stack([relative @ lateral / length, relative @ longitudinal / length])
    if signed_area(canonical) < 0.0:
        canonical = canonical[::-1]
    canonical = resample_closed(canonical, 1024)
    return gaussian_filter1d(canonical, sigma=0.5, axis=0, mode="wrap")


def _contour(workspace: Path, row: dict[str, Any]) -> np.ndarray:
    with np.load(workspace / row["mask"], allow_pickle=False) as data:
        mask = np.asarray(data["mask"], dtype=np.uint8)
        x0, y0, _, _ = map(int, data["bbox_xyxy"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"mask has no external contour: {row['leaflet_id']}")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    contour += np.asarray([x0, y0], dtype=np.float64)
    return contour


def _snap_outside_landmark(point: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, float]:
    """Snap a landmark only when it falls outside the reviewed silhouette."""
    location = (float(point[0]), float(point[1]))
    if cv2.pointPolygonTest(contour.astype(np.float32), location, False) >= 0.0:
        return point.copy(), 0.0
    distances2 = np.sum((contour - point) ** 2, axis=1)
    index = int(np.argmin(distances2))
    return contour[index].copy(), float(np.sqrt(distances2[index]))


def canonicalize_shapes(*, workspace: Path, output: Path | None = None) -> dict[str, Any]:
    """Apply reviewed landmarks without reflection and preserve all contour lobes."""
    workspace = Path(workspace).resolve()
    output = Path(output).resolve() if output else workspace / "canonical"
    if output.exists() and any(output.iterdir()):
        raise ValueError("canonical output must be absent or empty")
    items = {row["leaflet_id"]: row for row in read_jsonl(workspace / "items.jsonl")}
    mask_path = workspace / "final/masks.jsonl"
    shape_path = workspace / "final/shapes.jsonl"
    if not mask_path.is_file() or not shape_path.is_file():
        raise ValueError("finalize mask and shape reviews before canonicalization")
    masks = {row["leaflet_id"]: row for row in read_jsonl(mask_path)}
    shapes = {row["leaflet_id"]: row for row in read_jsonl(shape_path)}
    selected = sorted(
        leaflet_id
        for leaflet_id, decision in shapes.items()
        if decision.get("decision") == "accept"
        and masks.get(leaflet_id, {}).get("decision") == "accept"
    )
    if not selected:
        raise ValueError("no accepted shapes are available")
    contour_root = output / "contours"
    contour_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for leaflet_id in selected:
        row = items[leaflet_id]
        review = shapes[leaflet_id]
        reviewed_base = np.asarray(review["base"], dtype=np.float64)
        reviewed_tip = np.asarray(review["tip"], dtype=np.float64)
        pixel_contour = _contour(workspace, row)
        base, base_snap_distance = _snap_outside_landmark(reviewed_base, pixel_contour)
        tip, tip_snap_distance = _snap_outside_landmark(reviewed_tip, pixel_contour)
        length = float(np.linalg.norm(tip - base))
        if length <= 1e-9:
            raise ValueError(f"base and tip collapse for {leaflet_id}")
        try:
            canonical = canonical_contour(pixel_contour, base, tip)
        except ValueError as exc:
            rejections.append(
                {
                    "leaflet_id": leaflet_id,
                    "reason": str(exc),
                    "stage": "canonical_resampling",
                }
            )
            continue
        if signed_area(canonical) <= 1e-12 or contour_self_intersects(canonical):
            rejections.append(
                {
                    "leaflet_id": leaflet_id,
                    "reason": "canonical contour is degenerate or self-intersecting",
                    "stage": "canonical_validation",
                }
            )
            continue
        path = contour_root / f"{leaflet_id}.npz"
        np.savez_compressed(
            path,
            contour_xy=canonical,
            base_xy=np.asarray([0.0, 0.0]),
            tip_xy=np.asarray([0.0, 1.0]),
        )
        catalog.append(
            {
                "leaflet_id": leaflet_id,
                "source_group": row["source_group"],
                "source_file_name": row["source_file_name"],
                "split": row["split"],
                "role": review["role"],
                "contour": path.relative_to(output).as_posix(),
                "base_tip_length_px": length,
                "reviewed_base_xy": reviewed_base.tolist(),
                "reviewed_tip_xy": reviewed_tip.tolist(),
                "base_snap_distance_px": base_snap_distance,
                "tip_snap_distance_px": tip_snap_distance,
                "contour_sha256": _sha256_array(canonical),
                "reflection_applied": False,
            }
        )
    catalog_path = output / "catalog.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in catalog),
        encoding="utf-8",
        newline="\n",
    )
    (output / "rejections.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rejections
        ),
        encoding="utf-8",
        newline="\n",
    )
    if not catalog:
        raise ValueError("all accepted shapes failed canonical validation")
    manifest = {
        "schema_version": 1,
        "status": "complete_with_rejections" if rejections else "complete",
        "coordinate_frame": "base=(0,0), tip=(0,1), right-handed, +Y tip",
        "resampled_points": 1024,
        "smoothing": "periodic_gaussian_sigma_0.5_samples",
        "accepted_shapes": len(catalog),
        "rejected_shapes": len(rejections),
        "split_counts": {
            split: sum(row["split"] == split for row in catalog)
            for split in ("development", "validation", "holdout")
        },
        "role_counts": {
            role: sum(row["role"] == role for row in catalog)
            for role in sorted({row["role"] for row in catalog})
        },
        "snapped_landmarks": sum(row["base_snap_distance_px"] > 0.0 for row in catalog)
        + sum(row["tip_snap_distance_px"] > 0.0 for row in catalog),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 4: canonical contours (base (0,0), tip (0,1)) of the accepted leaflets"
    )
    parser.add_argument("--workspace", type=Path, required=True, help="prepared workspace")
    parser.add_argument("--output", type=Path, help="default <workspace>/canonical")
    args = parser.parse_args()
    print(json.dumps(canonicalize_shapes(workspace=args.workspace, output=args.output), indent=2))


if __name__ == "__main__":
    main()
