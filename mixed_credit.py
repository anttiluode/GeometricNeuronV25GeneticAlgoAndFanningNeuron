"""Gate F4: mixed global consequence times local eligibility traces.

F3 used a delayed packet that still carried the causal branch identity. F4
removes that source label. Several branches perturb at once. Their effects are
mixed by a recurrent downstream scalar state, and the learner receives only
that one scalar modulation. Each branch may retain only its own decaying local
eligibility trace.

The gate asks whether global consequence x local eligibility can recover
route-specific learning without an explicit reward-to-source map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import argparse
import json
import numpy as np

from nested_fan import branch_score

Mode = Literal[
    "local_oracle",
    "immediate_global",
    "eligibility",
    "current_only",
    "shuffled_eligibility",
    "pooled_eligibility",
]


@dataclass(frozen=True)
class MixedCreditConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 6000
    active_branches: int = 3
    mutation_norm: float = 0.18
    curvature: float = 0.20
    downstream_decay: float = 0.70
    eligibility_decay: float = 0.70
    learning_rate: float = 0.15
    consequence_noise: float = 0.001
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_rows(
    rng: np.random.Generator, count: int, dim: int, norm: float
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def run_mixed_credit(cfg: MixedCreditConfig, mode: Mode) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))
    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    eligibility = np.zeros_like(weights)
    downstream_state = 0.0
    score_curve: list[float] = []

    scale = cfg.learning_rate / (2.0 * cfg.mutation_norm * cfg.mutation_norm)

    for t in range(cfg.steps):
        active = rng.choice(
            cfg.branches, size=cfg.active_branches, replace=False
        )
        delta = np.zeros_like(weights)
        local_delta = _fixed_norm_rows(
            rng, cfg.active_branches, cfg.dim, cfg.mutation_norm
        )
        delta[active] = local_delta

        if mode == "local_oracle":
            plus = branch_score(
                weights[active] + local_delta,
                targets[active],
                cfg.curvature,
            )
            minus = branch_score(
                weights[active] - local_delta,
                targets[active],
                cfg.curvature,
            )
            local_consequence = plus - minus
            weights[active] += scale * local_consequence[:, None] * local_delta
        else:
            # One antithetic scalar consequence for the entire simultaneous event.
            # It contains no source label and mixes all active branches.
            plus = float(
                np.mean(
                    branch_score(weights + delta, targets, cfg.curvature)
                )
            )
            minus = float(
                np.mean(
                    branch_score(weights - delta, targets, cfg.curvature)
                )
            )
            event_consequence = (plus - minus) + float(
                rng.normal(scale=cfg.consequence_noise)
            )

            if mode == "immediate_global":
                weights += scale * event_consequence * delta
            else:
                # Recurrent downstream mixing: the observed scalar contains a
                # decaying mixture of several earlier overlapping events.
                downstream_state = (
                    cfg.downstream_decay * downstream_state + event_consequence
                )
                modulation = (1.0 - cfg.downstream_decay) * downstream_state

                eligibility = cfg.eligibility_decay * eligibility + delta

                if mode == "eligibility":
                    used = eligibility
                elif mode == "current_only":
                    used = delta
                elif mode == "shuffled_eligibility":
                    used = eligibility[rng.permutation(cfg.branches)]
                elif mode == "pooled_eligibility":
                    pooled = np.mean(eligibility, axis=0, keepdims=True)
                    used = np.repeat(pooled, cfg.branches, axis=0)
                else:
                    raise ValueError(f"unknown mode: {mode}")

                weights += scale * modulation * used

        if (t + 1) % cfg.branches == 0:
            score_curve.append(
                float(np.mean(branch_score(weights, targets, cfg.curvature)))
            )

    final = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "final_branch_scores": final.tolist(),
        "score_curve": score_curve,
    }


def _reach_cycle(curve: list[float], threshold: float) -> int | None:
    for i, value in enumerate(curve, start=1):
        if value >= threshold:
            return i
    return None


def run_battery(seeds: int = 16, steps: int = 6000) -> dict:
    modes: tuple[Mode, ...] = (
        "local_oracle",
        "immediate_global",
        "eligibility",
        "current_only",
        "shuffled_eligibility",
        "pooled_eligibility",
    )
    rows = []
    for seed in range(seeds):
        runs = {}
        for mode in modes:
            cfg = MixedCreditConfig(seed=seed, steps=steps)
            run = run_mixed_credit(cfg, mode)
            runs[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "reach_050_cycle": _reach_cycle(run["score_curve"], 0.50),
                "reach_075_cycle": _reach_cycle(run["score_curve"], 0.75),
            }
        rows.append({"seed": seed, "runs": runs})

    summary = {}
    for mode in modes:
        vals = np.array(
            [row["runs"][mode]["final_mean_score"] for row in rows], dtype=float
        )
        mins = np.array(
            [row["runs"][mode]["final_min_branch_score"] for row in rows],
            dtype=float,
        )
        r50 = [
            row["runs"][mode]["reach_050_cycle"]
            for row in rows
            if row["runs"][mode]["reach_050_cycle"] is not None
        ]
        r75 = [
            row["runs"][mode]["reach_075_cycle"]
            for row in rows
            if row["runs"][mode]["reach_075_cycle"] is not None
        ]
        summary[mode] = {
            "mean_final_score": float(np.mean(vals)),
            "std_final_score": float(np.std(vals)),
            "mean_min_branch_score": float(np.mean(mins)),
            "reach_050_seeds": len(r50),
            "mean_reach_050_cycle": float(np.mean(r50)) if r50 else None,
            "reach_075_seeds": len(r75),
            "mean_reach_075_cycle": float(np.mean(r75)) if r75 else None,
        }

    elig = np.array(
        [row["runs"]["eligibility"]["final_mean_score"] for row in rows]
    )
    comparisons = {}
    for other in (
        "current_only",
        "shuffled_eligibility",
        "pooled_eligibility",
    ):
        x = np.array([row["runs"][other]["final_mean_score"] for row in rows])
        comparisons[f"eligibility_beats_{other}_seeds"] = int(np.sum(elig > x))
        comparisons[f"eligibility_minus_{other}_mean"] = float(np.mean(elig - x))

    return {
        "gate": "F4_mixed_global_consequence_local_eligibility",
        "seeds": seeds,
        "steps": steps,
        "summary": summary,
        "comparisons": comparisons,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mixed_credit_receipt.json"),
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
                "steps": receipt["steps"],
                "summary": receipt["summary"],
                "comparisons": receipt["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
