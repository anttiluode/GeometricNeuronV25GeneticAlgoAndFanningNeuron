"""Gate F3: delayed global consequence times local eligibility.

F2 still had an immediate active-branch score. F3 hides that score from the
learner. A branch proposes a fixed-norm delta, the world generates only a
scalar consequence, and that consequence is delivered 4--20 events later.

The question is whether a persistent local eligibility tag (branch id + delta)
can bridge the delay. Controls apply the same delayed positive consequence to
the wrong branch or to whichever branch happens to be active when it arrives.

The hidden task is intentionally simple. The result is about temporal credit
assignment and route identity, not biological realism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import argparse
import heapq
import json
import numpy as np

from nested_fan import branch_score

Mode = Literal[
    "immediate",
    "tagged",
    "history",
    "wrong_tag",
    "current_branch",
]


@dataclass(frozen=True)
class DelayedConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 2400
    mutation_norm: float = 0.18
    curvature: float = 0.20
    delay_min: int = 4
    delay_max: int = 20
    consequence_noise: float = 0.002
    guide_fraction: float = 0.50
    history_depth: int = 4
    history_mix: float = 0.20
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_delta(rng: np.random.Generator, dim: int, norm: float) -> np.ndarray:
    v = rng.normal(size=dim)
    return v * (norm / (np.linalg.norm(v) + 1e-12))


def _history_direction(history: list[np.ndarray], depth: int) -> np.ndarray | None:
    if not history:
        return None
    h = np.mean(np.stack(history[-depth:]), axis=0)
    n = np.linalg.norm(h)
    return None if n < 1e-12 else h / n


def run_delayed(cfg: DelayedConfig, mode: Mode) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))
    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    histories: list[list[np.ndarray]] = [[] for _ in range(cfg.branches)]

    # (delivery_time, sequence, source_branch, delta, scalar_consequence)
    pending: list[tuple[int, int, int, np.ndarray, float]] = []
    sequence = 0
    accepted = 0
    score_curve: list[float] = []

    for t in range(cfg.steps):
        # Consequences return before the new event at time t.
        while pending and pending[0][0] <= t:
            _, _, source, delta, consequence = heapq.heappop(pending)
            if consequence <= 0.0:
                continue

            if mode in {"tagged", "history"}:
                destination = source
            elif mode == "wrong_tag":
                destination = (source + int(rng.integers(1, cfg.branches))) % cfg.branches
            elif mode == "current_branch":
                destination = t % cfg.branches
            else:
                raise ValueError(f"unexpected delayed mode: {mode}")

            weights[destination] += delta
            accepted += 1
            if mode in {"tagged", "history"}:
                histories[destination].append(delta.copy())

        active = t % cfg.branches
        delta = _fixed_norm_delta(rng, cfg.dim, cfg.mutation_norm)

        if mode == "history" and histories[active] and rng.random() < cfg.guide_fraction:
            h = _history_direction(histories[active], cfg.history_depth)
            if h is not None:
                mixed = (1.0 - cfg.history_mix) * delta + (
                    cfg.history_mix * cfg.mutation_norm * h
                )
                delta = mixed * (
                    cfg.mutation_norm / (np.linalg.norm(mixed) + 1e-12)
                )

        # The world can evaluate the consequence; the learner is not given the
        # target direction or branch score. It receives only this scalar later.
        before = float(
            branch_score(
                weights[active : active + 1],
                targets[active : active + 1],
                cfg.curvature,
            )[0]
        )
        after = float(
            branch_score(
                (weights[active] + delta)[None, :],
                targets[active : active + 1],
                cfg.curvature,
            )[0]
        )
        consequence = (after - before) + float(
            rng.normal(scale=cfg.consequence_noise)
        )

        if mode == "immediate":
            if consequence > 0.0:
                weights[active] += delta
                histories[active].append(delta.copy())
                accepted += 1
        else:
            delay = int(rng.integers(cfg.delay_min, cfg.delay_max + 1))
            heapq.heappush(
                pending,
                (t + delay, sequence, active, delta.copy(), consequence),
            )
            sequence += 1

        if t % cfg.branches == cfg.branches - 1:
            score_curve.append(
                float(np.mean(branch_score(weights, targets, cfg.curvature)))
            )

    final_scores = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(np.mean(final_scores)),
        "final_min_branch_score": float(np.min(final_scores)),
        "acceptance_rate": accepted / float(cfg.steps),
        "pending_at_end": len(pending),
        "score_curve": score_curve,
    }


def _reach_cycle(curve: list[float], threshold: float) -> int | None:
    for cycle, value in enumerate(curve, start=1):
        if value >= threshold:
            return cycle
    return None


def run_battery(seeds: int = 16, steps: int = 2400) -> dict:
    modes: tuple[Mode, ...] = (
        "immediate",
        "tagged",
        "history",
        "wrong_tag",
        "current_branch",
    )
    rows = []
    for seed in range(seeds):
        runs = {}
        for mode in modes:
            cfg = DelayedConfig(seed=seed, steps=steps)
            run = run_delayed(cfg, mode)
            runs[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "acceptance_rate": run["acceptance_rate"],
                "reach_095_cycle": _reach_cycle(run["score_curve"], 0.95),
                "reach_099_cycle": _reach_cycle(run["score_curve"], 0.99),
            }
        rows.append({"seed": seed, "runs": runs})

    summary = {}
    for mode in modes:
        finals = np.array([r["runs"][mode]["final_mean_score"] for r in rows])
        mins = np.array([r["runs"][mode]["final_min_branch_score"] for r in rows])
        r95 = [r["runs"][mode]["reach_095_cycle"] for r in rows if r["runs"][mode]["reach_095_cycle"] is not None]
        r99 = [r["runs"][mode]["reach_099_cycle"] for r in rows if r["runs"][mode]["reach_099_cycle"] is not None]
        summary[mode] = {
            "mean_final_score": float(np.mean(finals)),
            "std_final_score": float(np.std(finals)),
            "mean_min_branch_score": float(np.mean(mins)),
            "reach_095_seeds": len(r95),
            "mean_reach_095_cycle": float(np.mean(r95)) if r95 else None,
            "reach_099_seeds": len(r99),
            "mean_reach_099_cycle": float(np.mean(r99)) if r99 else None,
        }

    history = np.array([r["runs"]["history"]["final_mean_score"] for r in rows])
    tagged = np.array([r["runs"]["tagged"]["final_mean_score"] for r in rows])
    immediate = np.array([r["runs"]["immediate"]["final_mean_score"] for r in rows])

    return {
        "gate": "F3_delayed_consequence_local_eligibility",
        "seeds": seeds,
        "steps": steps,
        "summary": summary,
        "comparisons": {
            "tagged_minus_immediate_mean": float(np.mean(tagged - immediate)),
            "history_minus_tagged_mean": float(np.mean(history - tagged)),
            "history_beats_tagged_seeds": int(np.sum(history > tagged)),
            "tagged_beats_wrong_tag_seeds": int(
                np.sum(tagged > np.array([r["runs"]["wrong_tag"]["final_mean_score"] for r in rows]))
            ),
            "tagged_beats_current_branch_seeds": int(
                np.sum(tagged > np.array([r["runs"]["current_branch"]["final_mean_score"] for r in rows]))
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument(
        "--output", type=Path, default=Path("results/delayed_credit_receipt.json")
    )
    args = parser.parse_args()
    receipt = run_battery(args.seeds, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            {
                "gate": receipt["gate"],
                "seeds": receipt["seeds"],
                "summary": receipt["summary"],
                "comparisons": receipt["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
