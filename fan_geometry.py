"""GeometricNeuronV25: population-fan experiments.

This file tests the restart premise without invoking neuron biology:

    fan out -> local competition -> retain -> use successful fan geometry -> fan again

The key object is a matrix of parallel evolutionary lineages.  Each lineage is
a tiny (1 + lambda) evolution strategy.  The shared state is not a centroid;
it is the set of successful mutation directions observed across lineages.

Gate F0 asks prospectively whether that historical fan predicts the directions
of successful steps in the next generation.

Gate F1 asks whether using the discovered multi-direction fan to orient a fixed
fraction of future mutations improves search under an exactly norm-matched
mutation budget.  Controls are isotropic mutation, one-vector centroid
momentum, current-population covariance, and shuffled fan history.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np


Mode = Literal[
    "isotropic",
    "momentum",
    "current_covariance",
    "history_fan",
    "shuffled_history",
]


@dataclass(frozen=True)
class FanConfig:
    lineages: int = 24
    children_per_lineage: int = 8
    generations: int = 60
    branches: int = 3
    mutation_norm: float = 0.25
    guide_fraction: float = 0.65
    history_steps: int = 300
    min_history_steps: int = 12
    perpendicular_penalty: float = 1.20
    seed: int = 0


def branch_directions(branches: int) -> np.ndarray:
    """Equally spaced unit rays in a 2-D search plane."""
    angles = np.linspace(0.0, 2.0 * np.pi, branches, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def branch_score(
    points: np.ndarray,
    directions: np.ndarray,
    perpendicular_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score progress along the nearest ray, penalizing distance off that ray.

    The branch label is used only for measurement. Search never receives the
    label and is never told which lineage should occupy which branch.
    """
    dots = points @ directions.T
    r2 = np.sum(points * points, axis=1, keepdims=True)
    perpendicular2 = np.maximum(r2 - dots * dots, 0.0)
    values = dots - perpendicular_penalty * perpendicular2
    return values.max(axis=1), values.argmax(axis=1)


def _fixed_norm_noise(
    rng: np.random.Generator,
    count: int,
    dim: int,
    norm: float,
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    x *= norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x


def discover_fan_centers(
    successful_steps: np.ndarray,
    branches: int,
    iterations: int = 12,
) -> np.ndarray | None:
    """Discover several persistent step directions using cosine k-means.

    Initialization is deterministic farthest-first. The true branch directions
    are never supplied.
    """
    x = np.asarray(successful_steps, dtype=float)
    if x.ndim != 2 or len(x) < branches:
        return None

    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    centers = [x[0]]

    while len(centers) < branches:
        similarity = np.stack([x @ c for c in centers], axis=1)
        distance = 1.0 - np.max(similarity, axis=1)
        centers.append(x[int(np.argmax(distance))])

    c = np.stack(centers)
    for _ in range(iterations):
        assignment = np.argmax(x @ c.T, axis=1)
        updated = []
        for j in range(branches):
            pts = x[assignment == j]
            if len(pts) == 0:
                updated.append(c[j])
                continue
            center = pts.mean(axis=0)
            center /= np.linalg.norm(center) + 1e-12
            updated.append(center)
        c = np.stack(updated)
    return c


def _rotate_centers(centers: np.ndarray, angle: float) -> np.ndarray:
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle),  math.cos(angle)]],
        dtype=float,
    )
    return centers @ rot.T


def _covariance_guided_noise(
    rng: np.random.Generator,
    points: np.ndarray,
    count: int,
    mutation_norm: float,
) -> np.ndarray:
    """Sample current lineage covariance, then norm-match every mutation."""
    x = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(x.T) + 1e-8 * np.eye(x.shape[1])
    eigval, eigvec = np.linalg.eigh(covariance)
    root = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 1e-12)))
    noise = rng.normal(size=(count, x.shape[1])) @ root.T
    noise *= mutation_norm / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12)
    return noise


def _apply_direction_guide(
    base_noise: np.ndarray,
    directions: np.ndarray,
    rng: np.random.Generator,
    guide_fraction: float,
    mutation_norm: float,
) -> np.ndarray:
    """Bias a subset toward several directions without increasing step norm."""
    out = base_noise.copy()
    count = len(out)
    guided_n = int(round(count * guide_fraction))
    if guided_n <= 0 or len(directions) == 0:
        return out

    chosen = rng.choice(count, size=min(guided_n, count), replace=False)
    guide = directions[rng.integers(0, len(directions), size=len(chosen))]
    mixed = 0.55 * out[chosen] + 1.00 * mutation_norm * guide
    mixed *= mutation_norm / (np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-12)
    out[chosen] = mixed
    return out


def run_matrix_fan(cfg: FanConfig, mode: Mode = "isotropic") -> dict:
    """Run the parallel-lineage search with equal candidate and norm budgets."""
    rng = np.random.default_rng(cfg.seed)
    metric_rng = np.random.default_rng(cfg.seed + 50_000)
    directions = branch_directions(cfg.branches)

    parents = rng.normal(scale=0.05, size=(cfg.lineages, 2))
    successful_history: list[np.ndarray] = []
    centroid_history: list[np.ndarray] = []

    coverage_curve: list[int] = []
    min_branch_curve: list[int] = []
    mean_score_curve: list[float] = []
    best_score_curve: list[float] = []
    prospective_rows: list[dict] = []

    for generation in range(cfg.generations):
        parent_score, _ = branch_score(
            parents, directions, cfg.perpendicular_penalty
        )
        centroid = parents.mean(axis=0)
        centroid_history.append(centroid.copy())

        fan = None
        recent = successful_history[-cfg.history_steps:]
        if len(recent) >= cfg.min_history_steps:
            fan = discover_fan_centers(np.stack(recent), cfg.branches)

        momentum = None
        if len(centroid_history) >= 2:
            delta = centroid_history[-1] - centroid_history[-2]
            if np.linalg.norm(delta) > 1e-12:
                momentum = delta / np.linalg.norm(delta)

        next_parents = []
        realized_steps = []

        for lineage, parent in enumerate(parents):
            noise = _fixed_norm_noise(
                rng,
                cfg.children_per_lineage,
                dim=2,
                norm=cfg.mutation_norm,
            )

            if mode == "momentum" and momentum is not None:
                noise = _apply_direction_guide(
                    noise,
                    momentum[None, :],
                    rng,
                    cfg.guide_fraction,
                    cfg.mutation_norm,
                )
            elif mode == "current_covariance" and generation > 1:
                guided_n = int(round(cfg.children_per_lineage * cfg.guide_fraction))
                if guided_n > 0:
                    chosen = rng.choice(
                        cfg.children_per_lineage,
                        size=min(guided_n, cfg.children_per_lineage),
                        replace=False,
                    )
                    noise[chosen] = _covariance_guided_noise(
                        rng,
                        parents,
                        len(chosen),
                        cfg.mutation_norm,
                    )
            elif mode in {"history_fan", "shuffled_history"} and fan is not None:
                used_fan = fan
                if mode == "shuffled_history":
                    used_fan = _rotate_centers(
                        fan, metric_rng.uniform(0.0, 2.0 * np.pi)
                    )
                noise = _apply_direction_guide(
                    noise,
                    used_fan,
                    rng,
                    cfg.guide_fraction,
                    cfg.mutation_norm,
                )
            elif mode != "isotropic":
                if mode not in {
                    "momentum",
                    "current_covariance",
                    "history_fan",
                    "shuffled_history",
                }:
                    raise ValueError(f"unknown mode: {mode}")

            children = parent[None, :] + noise
            child_score, _ = branch_score(
                children, directions, cfg.perpendicular_penalty
            )
            winner = int(np.argmax(child_score))

            if child_score[winner] > parent_score[lineage]:
                next_parents.append(children[winner])
                realized_steps.append(noise[winner])
            else:
                next_parents.append(parent)
                realized_steps.append(np.zeros(2, dtype=float))

        realized = np.stack(realized_steps)
        success_mask = np.linalg.norm(realized, axis=1) > 1e-12
        successful = realized[success_mask]

        # Gate F0 uses only history available before this generation.
        if fan is not None and len(successful):
            unit = successful / (
                np.linalg.norm(successful, axis=1, keepdims=True) + 1e-12
            )
            history_alignment = float(np.mean(np.max(unit @ fan.T, axis=1)))

            rotated = _rotate_centers(
                fan, metric_rng.uniform(0.0, 2.0 * np.pi)
            )
            shuffled_alignment = float(np.mean(np.max(unit @ rotated.T, axis=1)))

            if momentum is None:
                momentum_alignment = 0.0
            else:
                momentum_alignment = float(np.mean(unit @ momentum))

            prospective_rows.append(
                {
                    "generation": generation,
                    "history_fan_alignment": history_alignment,
                    "shuffled_history_alignment": shuffled_alignment,
                    "centroid_momentum_alignment": momentum_alignment,
                    "successful_lineages": int(len(successful)),
                }
            )

        parents = np.stack(next_parents)
        successful_history.extend(step.copy() for step in successful)

        score, labels = branch_score(
            parents, directions, cfg.perpendicular_penalty
        )
        counts = np.bincount(labels, minlength=cfg.branches)
        coverage_curve.append(int(np.count_nonzero(counts)))
        min_branch_curve.append(int(counts.min()))
        mean_score_curve.append(float(score.mean()))
        best_score_curve.append(float(score.max()))

    final_score, final_labels = branch_score(
        parents, directions, cfg.perpendicular_penalty
    )
    final_counts = np.bincount(final_labels, minlength=cfg.branches)

    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(final_score.mean()),
        "final_best_score": float(final_score.max()),
        "final_branch_coverage": int(np.count_nonzero(final_counts)),
        "final_branch_counts": final_counts.tolist(),
        "final_min_branch_count": int(final_counts.min()),
        "mean_score_curve": mean_score_curve,
        "best_score_curve": best_score_curve,
        "coverage_curve": coverage_curve,
        "min_branch_curve": min_branch_curve,
        "prospective_rows": prospective_rows,
    }


def gate_f0(seed: int = 0, generations: int = 60) -> dict:
    """Measure fan predictiveness on an unguided trajectory."""
    cfg = FanConfig(seed=seed, generations=generations)
    run = run_matrix_fan(cfg, mode="isotropic")
    rows = run["prospective_rows"]

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")

    return {
        "seed": seed,
        "comparisons": len(rows),
        "history_fan_alignment": mean("history_fan_alignment"),
        "shuffled_history_alignment": mean("shuffled_history_alignment"),
        "centroid_momentum_alignment": mean("centroid_momentum_alignment"),
    }


def gate_f1(seed: int = 0, generations: int = 60) -> dict:
    cfg = FanConfig(seed=seed, generations=generations)
    modes: tuple[Mode, ...] = (
        "isotropic",
        "momentum",
        "current_covariance",
        "history_fan",
        "shuffled_history",
    )
    runs = {mode: run_matrix_fan(cfg, mode=mode) for mode in modes}
    return {
        "seed": seed,
        "runs": {
            mode: {
                "final_mean_score": run["final_mean_score"],
                "final_best_score": run["final_best_score"],
                "final_branch_coverage": run["final_branch_coverage"],
                "final_branch_counts": run["final_branch_counts"],
                "final_min_branch_count": run["final_min_branch_count"],
            }
            for mode, run in runs.items()
        },
    }


def run_battery(seeds: int = 32, generations: int = 60) -> dict:
    f0 = [gate_f0(seed, generations) for seed in range(seeds)]
    f1 = [gate_f1(seed, generations) for seed in range(seeds)]

    f0_summary = {
        "history_fan_alignment": float(
            np.mean([row["history_fan_alignment"] for row in f0])
        ),
        "shuffled_history_alignment": float(
            np.mean([row["shuffled_history_alignment"] for row in f0])
        ),
        "centroid_momentum_alignment": float(
            np.mean([row["centroid_momentum_alignment"] for row in f0])
        ),
        "history_beats_shuffled_seeds": int(
            sum(
                row["history_fan_alignment"] > row["shuffled_history_alignment"]
                for row in f0
            )
        ),
        "history_beats_momentum_seeds": int(
            sum(
                row["history_fan_alignment"] > row["centroid_momentum_alignment"]
                for row in f0
            )
        ),
    }

    modes = list(f1[0]["runs"].keys())
    f1_summary = {}
    for mode in modes:
        f1_summary[mode] = {
            "mean_final_score": float(
                np.mean([row["runs"][mode]["final_mean_score"] for row in f1])
            ),
            "mean_branch_coverage": float(
                np.mean([row["runs"][mode]["final_branch_coverage"] for row in f1])
            ),
            "mean_min_branch_count": float(
                np.mean([row["runs"][mode]["final_min_branch_count"] for row in f1])
            ),
        }

    iso = np.array(
        [row["runs"]["isotropic"]["final_mean_score"] for row in f1], dtype=float
    )
    fan = np.array(
        [row["runs"]["history_fan"]["final_mean_score"] for row in f1], dtype=float
    )
    cov = np.array(
        [row["runs"]["current_covariance"]["final_mean_score"] for row in f1],
        dtype=float,
    )
    shuffled = np.array(
        [row["runs"]["shuffled_history"]["final_mean_score"] for row in f1],
        dtype=float,
    )

    f1_summary["history_fan"]["ratio_to_isotropic"] = float(np.mean(fan / iso))
    f1_summary["history_fan"]["ratio_to_current_covariance"] = float(
        np.mean(fan / cov)
    )
    f1_summary["history_fan"]["ratio_to_shuffled_history"] = float(
        np.mean(fan / shuffled)
    )
    f1_summary["history_fan"]["beats_isotropic_seeds"] = int(np.sum(fan > iso))
    f1_summary["history_fan"]["beats_current_covariance_seeds"] = int(
        np.sum(fan > cov)
    )
    f1_summary["history_fan"]["beats_shuffled_history_seeds"] = int(
        np.sum(fan > shuffled)
    )

    return {
        "seeds": seeds,
        "generations": generations,
        "gate_f0": f0_summary,
        "gate_f1": f1_summary,
        "per_seed_f0": f0,
        "per_seed_f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/fan_receipt.json"))
    args = parser.parse_args()

    result = run_battery(args.seeds, args.generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "seeds": result["seeds"],
            "generations": result["generations"],
            "gate_f0": result["gate_f0"],
            "gate_f1": result["gate_f1"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
