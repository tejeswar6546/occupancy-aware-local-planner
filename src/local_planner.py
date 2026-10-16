#!/usr/bin/env python3
"""Test 40 lookahead-path (LP) candidates on saved occupancy-grid NPY files.

Current behavior:
- Reference direction = straight ahead.
- Generate exactly 40 smooth ~3 m forward candidate LPs.
- Check the COMPLETE candidate against the occupancy grid.
- Occupied (100) or outside-map -> invalid.
- Unknown (-1) is allowed, but penalized.
- Prefer paths close to straight-ahead reference, with clearance and less unknown.
- The endpoint of the winning LP is labeled as the local planning goal.

Grid convention expected from final_occupancy_grid.py:
- row 0 = robot, rows increase forward
- x_min = -5 m, x_max = +5 m
- resolution = 0.10 m/cell
- -1 unknown, 0 free, 100 occupied
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def generate_candidates(
    n: int = 40,
    horizon_m: float = 3.0,
    lateral_max_m: float = 2.0,
    points_per_path: int = 61,
):
    """Generate smooth LPs around a straight reference direction.

    Coordinates per point are [x, z] in meters:
      x: lateral, +right
      z: forward

    Lateral motion grows quadratically with progress so every LP begins
    aligned with the straight reference direction.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if horizon_m <= 0 or lateral_max_m < 0:
        raise ValueError("invalid horizon/lateral range")
    if points_per_path < 2:
        raise ValueError("points_per_path must be >= 2")

    z = np.linspace(0.0, horizon_m, points_per_path, dtype=np.float64)
    t = z / horizon_m
    offsets = np.linspace(-lateral_max_m, lateral_max_m, n, dtype=np.float64)

    candidates = []
    for end_x in offsets:
        # Smooth S-curve: starts aligned with reference direction, moves sideways
        # early enough to avoid a nearby obstacle, and settles at the endpoint.
        blend = 3.0 * (t ** 2) - 2.0 * (t ** 3)
        x = end_x * blend
        candidates.append(np.column_stack((x, z)))
    return candidates


def world_to_grid(
    x_m: float,
    z_m: float,
    resolution: float = 0.10,
    x_min: float = -5.0,
):
    row = int(round(z_m / resolution))
    col = int(round((x_m - x_min) / resolution))
    return row, col


def _disk_offsets(radius_cells: int):
    offsets = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                offsets.append((dr, dc))
    return offsets


def _clearance_map_m(grid: np.ndarray, resolution: float):
    """Distance in meters from each cell center to nearest occupied cell."""
    occupied = (grid == 100).astype(np.uint8)
    if not np.any(occupied):
        return np.full(grid.shape, 999.0, dtype=np.float32)

    # distanceTransform measures distance to zero cells, so occupied -> zero.
    free_for_dt = np.where(occupied > 0, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(free_for_dt, cv2.DIST_L2, 5) * float(resolution)


def evaluate_candidate(
    path_xz: np.ndarray,
    grid: np.ndarray,
    resolution: float = 0.10,
    x_min: float = -5.0,
    footprint_radius_m: float = 0.45,
    clearance_map: np.ndarray | None = None,
):
    """Check the full LP.

    Occupied or outside map => invalid.
    Unknown is allowed but contributes to unknown_fraction.
    """
    if clearance_map is None:
        clearance_map = _clearance_map_m(grid, resolution)

    radius_cells = max(0, int(math.ceil(footprint_radius_m / resolution)))
    offsets = _disk_offsets(radius_cells)

    unknown = 0
    sampled = 0
    min_clearance = 999.0
    collision_at = None
    outside = False

    for i, (x_m, z_m) in enumerate(path_xz):
        row, col = world_to_grid(x_m, z_m, resolution, x_min)

        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            outside = True
            collision_at = i
            break

        min_clearance = min(min_clearance, float(clearance_map[row, col]))

        for dr, dc in offsets:
            rr = row + dr
            cc = col + dc

            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                # Ignore footprint cells behind the robot at z≈0, but otherwise
                # leaving the known map is invalid.
                if rr < 0 and row <= radius_cells:
                    continue
                outside = True
                collision_at = i
                break

            val = int(grid[rr, cc])
            sampled += 1

            if val == 100:
                collision_at = i
                return {
                    "valid": False,
                    "collision": True,
                    "outside": False,
                    "collision_index": collision_at,
                    "unknown_fraction": unknown / max(sampled, 1),
                    "min_clearance_m": min_clearance,
                }

            if val == -1:
                unknown += 1

        if outside:
            break

    return {
        "valid": not outside,
        "collision": False,
        "outside": outside,
        "collision_index": collision_at,
        "unknown_fraction": unknown / max(sampled, 1),
        "min_clearance_m": min_clearance,
    }


def candidate_score(path_xz: np.ndarray, result: dict):
    if not result["valid"]:
        return float("inf")

    end_x = abs(float(path_xz[-1, 0]))

    # Simple smoothness measure from lateral second difference.
    x = path_xz[:, 0]
    smoothness = float(np.mean(np.abs(np.diff(x, n=2)))) if len(x) >= 3 else 0.0

    unknown_penalty = 3.0 * float(result["unknown_fraction"])
    reference_deviation_penalty = 0.75 * end_x
    smoothness_penalty = 8.0 * smoothness

    # Reward useful clearance, capped so it does not dominate.
    clearance_reward = 0.35 * min(float(result["min_clearance_m"]), 2.0)

    return (
        unknown_penalty
        + reference_deviation_penalty
        + smoothness_penalty
        - clearance_reward
    )


def choose_best_candidate(candidates, results):
    scores = [candidate_score(c, r) for c, r in zip(candidates, results)]
    finite = [i for i, s in enumerate(scores) if np.isfinite(s)]
    if not finite:
        return None
    return min(finite, key=lambda i: scores[i])


def _grid_to_bgr(grid: np.ndarray):
    img = np.full(grid.shape, 127, dtype=np.uint8)
    img[grid == 0] = 255
    img[grid == 100] = 0
    img = np.flipud(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _path_pixels(path_xz, grid_shape, resolution, x_min, scale):
    h, w = grid_shape
    pts = []
    for x_m, z_m in path_xz:
        row, col = world_to_grid(x_m, z_m, resolution, x_min)
        if 0 <= row < h and 0 <= col < w:
            disp_row = h - 1 - row
            pts.append((int(col * scale), int(disp_row * scale)))
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def visualize(
    grid,
    candidates,
    results,
    winner_idx,
    out_path: Path,
    resolution=0.10,
    x_min=-5.0,
    scale=6,
):
    base = _grid_to_bgr(grid)
    vis = cv2.resize(
        base,
        (base.shape[1] * scale, base.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    # All candidates: valid blue, rejected red.
    for i, (path, result) in enumerate(zip(candidates, results)):
        pts = _path_pixels(path, grid.shape, resolution, x_min, scale)
        if pts is None:
            continue
        color = (220, 100, 30) if result["valid"] else (40, 40, 220)
        thickness = 1
        cv2.polylines(vis, [pts], False, color, thickness, cv2.LINE_AA)

    if winner_idx is not None:
        winner = candidates[winner_idx]
        pts = _path_pixels(winner, grid.shape, resolution, x_min, scale)
        if pts is not None:
            cv2.polylines(vis, [pts], False, (0, 180, 0), 4, cv2.LINE_AA)

        end_x, end_z = winner[-1]
        row, col = world_to_grid(end_x, end_z, resolution, x_min)
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            px = int(col * scale)
            py = int((grid.shape[0] - 1 - row) * scale)
            cv2.circle(vis, (px, py), 9, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                "LOCAL PLANNING GOAL",
                (min(px + 12, max(vis.shape[1] - 220, 0)), max(py - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    # Robot marker.
    robot_row, robot_col = world_to_grid(0.0, 0.0, resolution, x_min)
    rx = int(robot_col * scale)
    ry = int((grid.shape[0] - 1 - robot_row) * scale)
    cv2.circle(vis, (rx, ry), 8, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        vis,
        "ROBOT",
        (rx + 10, max(ry - 8, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), vis)


def process_one(
    og_path: Path,
    output_dir: Path,
    n_candidates=40,
    horizon_m=3.0,
    lateral_max_m=2.0,
    points_per_path=61,
    resolution=0.10,
    x_min=-5.0,
    footprint_radius_m=0.45,
):
    grid = np.load(og_path)
    if grid.ndim != 2:
        raise ValueError(f"{og_path}: expected 2D occupancy grid")
    if not np.all(np.isin(np.unique(grid), [-1, 0, 100])):
        raise ValueError(f"{og_path}: grid contains values other than -1/0/100")

    candidates = generate_candidates(
        n=n_candidates,
        horizon_m=horizon_m,
        lateral_max_m=lateral_max_m,
        points_per_path=points_per_path,
    )

    clearance_map = _clearance_map_m(grid, resolution)
    results = [
        evaluate_candidate(
            path,
            grid,
            resolution=resolution,
            x_min=x_min,
            footprint_radius_m=footprint_radius_m,
            clearance_map=clearance_map,
        )
        for path in candidates
    ]

    scores = [candidate_score(c, r) for c, r in zip(candidates, results)]
    winner_idx = choose_best_candidate(candidates, results)

    stem = og_path.stem
    image_path = output_dir / f"{stem}_40LP_selection.png"
    json_path = output_dir / f"{stem}_40LP_selection.json"

    visualize(
        grid,
        candidates,
        results,
        winner_idx,
        image_path,
        resolution=resolution,
        x_min=x_min,
    )

    summary = {
        "og_file": str(og_path),
        "candidate_count": len(candidates),
        "horizon_m": horizon_m,
        "lateral_max_m": lateral_max_m,
        "footprint_radius_m": footprint_radius_m,
        "valid_count": int(sum(r["valid"] for r in results)),
        "rejected_count": int(sum(not r["valid"] for r in results)),
        "winner_index": winner_idx,
        "winner_endpoint_m": None,
        "winner_score": None,
        "winner_unknown_fraction": None,
        "winner_min_clearance_m": None,
        "candidates": [],
    }

    for i, (path, result, score) in enumerate(zip(candidates, results, scores)):
        summary["candidates"].append({
            "index": i,
            "endpoint_x_m": float(path[-1, 0]),
            "endpoint_z_m": float(path[-1, 1]),
            "valid": bool(result["valid"]),
            "collision": bool(result["collision"]),
            "outside": bool(result["outside"]),
            "unknown_fraction": float(result["unknown_fraction"]),
            "min_clearance_m": float(result["min_clearance_m"]),
            "score": None if not np.isfinite(score) else float(score),
        })

    if winner_idx is not None:
        winner = candidates[winner_idx]
        wr = results[winner_idx]
        summary["winner_endpoint_m"] = {
            "x": float(winner[-1, 0]),
            "z": float(winner[-1, 1]),
        }
        summary["winner_score"] = float(scores[winner_idx])
        summary["winner_unknown_fraction"] = float(wr["unknown_fraction"])
        summary["winner_min_clearance_m"] = float(wr["min_clearance_m"])

    json_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== {og_path.name} ===")
    print(f"Candidates: {len(candidates)}")
    print(f"Valid:      {summary['valid_count']}")
    print(f"Rejected:   {summary['rejected_count']}")
    if winner_idx is None:
        print("WINNER:     NONE -- no valid LP")
    else:
        ep = summary["winner_endpoint_m"]
        print(f"Winner:     LP #{winner_idx}")
        print(f"Endpoint:   x={ep['x']:+.2f} m, z={ep['z']:.2f} m")
        print(f"Clearance:  {summary['winner_min_clearance_m']:.2f} m")
        print(f"Unknown:    {summary['winner_unknown_fraction']*100:.1f}%")
    print(f"Figure:     {image_path}")
    print(f"JSON:       {json_path}")

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--og", nargs="+", required=True, help="One or more occupancy-grid .npy files")
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--horizon", type=float, default=3.0)
    p.add_argument("--lateral-max", type=float, default=2.0)
    p.add_argument("--points", type=int, default=61)
    p.add_argument("--resolution", type=float, default=0.10)
    p.add_argument("--x-min", type=float, default=-5.0)
    p.add_argument("--footprint-radius", type=float, default=0.45)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for raw in args.og:
        process_one(
            Path(raw).expanduser().resolve(),
            output,
            n_candidates=args.candidates,
            horizon_m=args.horizon,
            lateral_max_m=args.lateral_max,
            points_per_path=args.points,
            resolution=args.resolution,
            x_min=args.x_min,
            footprint_radius_m=args.footprint_radius,
        )


if __name__ == "__main__":
    main()
