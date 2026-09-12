"""Gate F2: nested fan selection under bounded observation.

The experiment is intentionally synthetic.  It asks whether compartmentalized
search protects several independent routes when only one route is observable at
any instant.

Each of B branches has a hidden preferred direction.  On generation g only one
branch/context is exposed.  A candidate is accepted from the score available in
that context; the all-branch score is measured only for reporting.

Arms use the same local-score evaluation budget per generation:
  * flat: fixed-norm mutation across the whole branch matrix.
  * flat_active_matched: the active branch gets the same step norm as a local
    arm, but inactive branches drift too (a deliberately strong attacker).
  * flat_replay: flat mutation, but spends the same score-probe budget checking
    the active context plus a few remembered contexts.
  * local: only the currently eligible branch changes.
  * history: local, with a fraction of proposals bent toward that branch's own
    recent accepted deltas.
  * shuffled_history: same, but uses another branch's history when available.

No arm is given the hidden target directions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import argparse
import json
import numpy as np

Mode = Literal[
    "flat",
    "flat_active_matched",
    "flat_replay",
    "local",
    "history",
    "shuffled_history",
]


@dataclass(frozen=True)
class NestedConfig:
    branches: int = 12
    dim: int = 6
    cycles: int = 40
    eval_budget: int = 32
    mutation_norm: float = 0.30
    curvature: float = 0.20
    guide_fraction: float = 0.65
    history_depth: int = 8
    replay_contexts: int = 3
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _targets(cfg: NestedConfig, rng: np.random.Generator) -> np.ndarray:
    return _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))


def branch_score(weights: np.ndarray, targets: np.ndarray, curvature: float) -> np.ndarray:
    """Saturating progress along each branch's hidden ray, with off-ray penalty."""
    projection = np.sum(weights * targets, axis=-1)
    norm2 = np.sum(weights * weights, axis=-1)
    perpendicular2 = np.maximum(norm2 - projection * projection, 0.0)
    return np.tanh(projection) - curvature * perpendicular2


def _fixed_norm_rows(rng: np.random.Generator, count: int, dim: int, norm: float) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def _flat_deltas(rng: np.random.Generator, count: int, cfg: NestedConfig) -> np.ndarray:
    d = rng.normal(size=(count, cfg.branches, cfg.dim))
    flat = d.reshape(count, -1)
    flat *= cfg.mutation_norm / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    return flat.reshape(count, cfg.branches, cfg.dim)


def _active_matched_deltas(
    rng: np.random.Generator, count: int, active: int, cfg: NestedConfig
) -> np.ndarray:
    """Strong flat attacker: active step is norm-matched, but inactive routes also move.

    The inactive block gets an additional total norm equal to mutation_norm, so this
    attacker is actually allowed sqrt(2) times the total movement of the local arm.
    """
    d = np.zeros((count, cfg.branches, cfg.dim), dtype=float)
    d[:, active, :] = _fixed_norm_rows(rng, count, cfg.dim, cfg.mutation_norm)
    inactive = [i for i in range(cfg.branches) if i != active]
    if inactive:
        z = rng.normal(size=(count, len(inactive), cfg.dim))
        flat = z.reshape(count, -1)
        flat *= cfg.mutation_norm / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
        d[:, inactive, :] = flat.reshape(count, len(inactive), cfg.dim)
    return d


def _history_vector(history: list[np.ndarray], depth: int) -> np.ndarray | None:
    if not history:
        return None
    h = np.mean(np.stack(history[-depth:]), axis=0)
    n = np.linalg.norm(h)
    return None if n < 1e-12 else h / n


def _local_deltas(
    rng: np.random.Generator,
    count: int,
    active: int,
    cfg: NestedConfig,
    history: list[list[np.ndarray]],
    mode: Mode,
) -> np.ndarray:
    d = np.zeros((count, cfg.branches, cfg.dim), dtype=float)
    v = _fixed_norm_rows(rng, count, cfg.dim, cfg.mutation_norm)

    if mode in {"history", "shuffled_history"}:
        for i in range(count):
            if rng.random() >= cfg.guide_fraction:
                continue
            source = active
            if mode == "shuffled_history":
                candidates = [j for j in range(cfg.branches) if j != active and history[j]]
                if candidates:
                    source = int(rng.choice(candidates))
            h = _history_vector(history[source], cfg.history_depth)
            if h is None:
                continue
            mixed = 0.45 * v[i] + cfg.mutation_norm * h
            v[i] = mixed * (cfg.mutation_norm / (np.linalg.norm(mixed) + 1e-12))

    d[:, active, :] = v
    return d


def _candidate_scores(
    weights: np.ndarray,
    deltas: np.ndarray,
    targets: np.ndarray,
    contexts: list[int],
    curvature: float,
) -> np.ndarray:
    candidate_w = weights[None, :, :] + deltas
    wc = candidate_w[:, contexts, :]
    tc = targets[None, contexts, :]
    projection = np.sum(wc * tc, axis=-1)
    norm2 = np.sum(wc * wc, axis=-1)
    perpendicular2 = np.maximum(norm2 - projection * projection, 0.0)
    local = np.tanh(projection) - curvature * perpendicular2
    return np.mean(local, axis=1)


def run_nested(cfg: NestedConfig, mode: Mode) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _targets(cfg, rng)
    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    history: list[list[np.ndarray]] = [[] for _ in range(cfg.branches)]

    all_score_curve: list[float] = []
    accepted = 0
    total_score_probes = 0

    for generation in range(cfg.cycles * cfg.branches):
        active = generation % cfg.branches

        if mode == "flat_replay":
            others = [i for i in range(cfg.branches) if i != active]
            k = min(cfg.replay_contexts, len(others))
            replay = list(rng.choice(others, size=k, replace=False)) if k else []
            contexts = [active] + replay
            candidate_count = max(1, cfg.eval_budget // len(contexts))
        else:
            contexts = [active]
            candidate_count = cfg.eval_budget

        parent_local = branch_score(weights[contexts], targets[contexts], cfg.curvature)
        parent_score = float(np.mean(parent_local))

        if mode in {"flat", "flat_replay"}:
            deltas = _flat_deltas(rng, candidate_count, cfg)
        elif mode == "flat_active_matched":
            deltas = _active_matched_deltas(rng, candidate_count, active, cfg)
        elif mode in {"local", "history", "shuffled_history"}:
            deltas = _local_deltas(rng, candidate_count, active, cfg, history, mode)
        else:
            raise ValueError(f"unknown mode: {mode}")

        scores = _candidate_scores(weights, deltas, targets, contexts, cfg.curvature)
        total_score_probes += candidate_count * len(contexts)
        winner = int(np.argmax(scores))
        if float(scores[winner]) > parent_score:
            weights += deltas[winner]
            accepted += 1
            if mode in {"local", "history", "shuffled_history"}:
                history[active].append(deltas[winner, active].copy())

        all_score_curve.append(float(np.mean(branch_score(weights, targets, cfg.curvature))))

    final_branch_scores = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(np.mean(final_branch_scores)),
        "final_min_branch_score": float(np.min(final_branch_scores)),
        "final_branch_scores": final_branch_scores.tolist(),
        "acceptance_rate": accepted / float(cfg.cycles * cfg.branches),
        "score_probes": int(total_score_probes),
        "all_score_curve": all_score_curve,
    }


def _reach_cycle(curve: list[float], branches: int, threshold: float) -> float | None:
    for g, value in enumerate(curve):
        if value >= threshold:
            return (g + 1) / float(branches)
    return None


def run_battery(seeds: int = 16, cycles: int = 40) -> dict:
    modes: tuple[Mode, ...] = (
        "flat",
        "flat_active_matched",
        "flat_replay",
        "local",
        "history",
        "shuffled_history",
    )
    rows = []
    for seed in range(seeds):
        seed_runs = {}
        for mode in modes:
            cfg = NestedConfig(seed=seed, cycles=cycles)
            run = run_nested(cfg, mode)
            seed_runs[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "acceptance_rate": run["acceptance_rate"],
                "score_probes": run["score_probes"],
                "reach_095_cycles": _reach_cycle(run["all_score_curve"], cfg.branches, 0.95),
                "reach_099_cycles": _reach_cycle(run["all_score_curve"], cfg.branches, 0.99),
            }
        rows.append({"seed": seed, "runs": seed_runs})

    summary = {}
    for mode in modes:
        vals = np.array([r["runs"][mode]["final_mean_score"] for r in rows], dtype=float)
        minvals = np.array([r["runs"][mode]["final_min_branch_score"] for r in rows], dtype=float)
        r95 = [r["runs"][mode]["reach_095_cycles"] for r in rows if r["runs"][mode]["reach_095_cycles"] is not None]
        r99 = [r["runs"][mode]["reach_099_cycles"] for r in rows if r["runs"][mode]["reach_099_cycles"] is not None]
        summary[mode] = {
            "mean_final_score": float(np.mean(vals)),
            "std_final_score": float(np.std(vals)),
            "mean_min_branch_score": float(np.mean(minvals)),
            "reach_095_seeds": len(r95),
            "mean_reach_095_cycles": float(np.mean(r95)) if r95 else None,
            "reach_099_seeds": len(r99),
            "mean_reach_099_cycles": float(np.mean(r99)) if r99 else None,
            "mean_score_probes": float(np.mean([r["runs"][mode]["score_probes"] for r in rows])),
        }

    history = np.array([r["runs"]["history"]["final_mean_score"] for r in rows])
    comparisons = {}
    for other in ("flat", "flat_active_matched", "flat_replay", "local", "shuffled_history"):
        x = np.array([r["runs"][other]["final_mean_score"] for r in rows])
        comparisons[f"history_beats_{other}_seeds"] = int(np.sum(history > x))
        comparisons[f"history_minus_{other}_mean"] = float(np.mean(history - x))

    return {
        "gate": "F2_nested_fan_bounded_observation",
        "seeds": seeds,
        "cycles": cycles,
        "summary": summary,
        "comparisons": comparisons,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/nested_fan_receipt.json"))
    args = parser.parse_args()
    receipt = run_battery(args.seeds, args.cycles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"gate": receipt["gate"], "seeds": receipt["seeds"], "summary": receipt["summary"], "comparisons": receipt["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
