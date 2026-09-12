"""Gate F7: frequency-addressed causal eligibility.

F6 let each branch choose among several decay-only eligibility traces. F7 asks
whether the temporal address can include a resonant component as well as a
lifetime.

Each branch in the hidden world is assigned one damped temporal frequency. Its
local perturbation consequence enters a damped complex oscillator. All branch
oscillators are summed into one global scalar modulation.

The learner never receives the hidden branch frequencies. Instead, every branch
maintains a bank of damped complex eligibility traces with candidate frequencies
and a tiny local predictor for each trace. The predictor that best explains the
observed global modulation controls plasticity for that branch.

This is a synthetic source-separation / temporal-credit experiment, not a claim
that biological dendrites explicitly implement complex arithmetic.
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
    "adaptive",
    "oracle",
    "fixed0",
    "fixed1",
    "fixed2",
    "fixed3",
    "shuffled",
]

DEFAULT_FREQUENCIES = (0.0, np.pi / 12.0, np.pi / 6.0, np.pi / 3.0)


@dataclass(frozen=True)
class ResonantConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 5000
    active_branches: int = 3
    mutation_norm: float = 0.15
    learning_rate: float = 0.25
    curvature: float = 0.20
    world_decay: float = 0.95
    eligibility_decay: float = 0.95
    predictor_lr: float = 0.03
    error_ema: float = 0.005
    warmup_steps: int = 300
    frequencies: tuple[float, ...] = DEFAULT_FREQUENCIES
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_rows(
    rng: np.random.Generator, count: int, dim: int, norm: float
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def run_resonant(cfg: ResonantConfig, mode: Mode) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))

    frequencies = np.asarray(cfg.frequencies, dtype=float)
    channels = len(frequencies)
    true_index = np.arange(cfg.branches) % channels
    true_frequency = frequencies[true_index]

    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)

    # Hidden downstream consequence state. Each branch is filtered through a
    # different damped oscillator before all branches are mixed into one scalar.
    world_state = np.zeros(cfg.branches, dtype=np.complex128)
    world_phase = np.exp(1j * true_frequency)

    # Local branch x candidate-frequency eligibility bank.
    eligibility = np.zeros(
        (cfg.branches, channels, cfg.dim), dtype=np.complex128
    )
    candidate_phase = np.exp(1j * frequencies)

    # Tiny local linear predictors. A complex trace is represented by its real
    # and imaginary coordinates. Prediction error provides only a selector; the
    # predictor is not given the target direction or true frequency.
    predictor = np.zeros((cfg.branches, channels, 2 * cfg.dim), dtype=float)
    error = np.ones((cfg.branches, channels), dtype=float)
    selection_counts = np.zeros((cfg.branches, channels), dtype=int)
    score_curve: list[float] = []

    for t in range(cfg.steps):
        active = rng.choice(
            cfg.branches, size=cfg.active_branches, replace=False
        )
        local_delta = _fixed_norm_rows(
            rng, cfg.active_branches, cfg.dim, cfg.mutation_norm
        )
        delta = np.zeros_like(weights)
        delta[active] = local_delta

        # The world sees the local antithetic consequence, but the learner does
        # not. Each local consequence is injected into the branch's hidden
        # resonator; only their mixed scalar projection is exposed.
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
        hidden_input = np.zeros(cfg.branches, dtype=float)
        hidden_input[active] = local_consequence
        world_state = (
            cfg.world_decay * world_phase * world_state + hidden_input
        )
        modulation = float(np.real(world_state).mean())

        eligibility = (
            cfg.eligibility_decay
            * candidate_phase[None, :, None]
            * eligibility
            + delta[:, None, :]
        )

        if mode == "adaptive":
            features = np.concatenate(
                [eligibility.real, eligibility.imag], axis=2
            )
            prediction = np.sum(predictor * features, axis=2)
            residual = modulation - prediction
            feature_norm = np.sum(features * features, axis=2) + 1e-4
            predictor += (
                cfg.predictor_lr
                * (residual / feature_norm)[:, :, None]
                * features
            )
            error = (
                (1.0 - cfg.error_ema) * error
                + cfg.error_ema * residual * residual
            )
            selected = np.argmin(error, axis=1)
            if t < cfg.warmup_steps:
                selected = rng.integers(channels, size=cfg.branches)
        elif mode == "oracle":
            selected = true_index
        elif mode.startswith("fixed"):
            selected = np.full(cfg.branches, int(mode[-1]), dtype=int)
        elif mode == "shuffled":
            selected = true_index[rng.permutation(cfg.branches)]
        else:
            raise ValueError(f"unknown mode: {mode}")

        selection_counts[np.arange(cfg.branches), selected] += 1
        used = eligibility[np.arange(cfg.branches), selected, :].real
        weights += cfg.learning_rate * modulation * used

        if (t + 1) % cfg.branches == 0:
            score_curve.append(
                float(np.mean(branch_score(weights, targets, cfg.curvature)))
            )

    final = branch_score(weights, targets, cfg.curvature)
    dominant = np.argmax(selection_counts, axis=1)
    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "mean_curve_score": float(np.mean(score_curve)),
        "frequency_id_accuracy": float(np.mean(dominant == true_index)),
        "true_frequency_index": true_index.tolist(),
        "selection_counts": selection_counts.tolist(),
        "score_curve": score_curve,
    }


def run_battery(seeds: int = 8, steps: int = 5000) -> dict:
    modes: tuple[Mode, ...] = (
        "adaptive",
        "oracle",
        "fixed0",
        "fixed1",
        "fixed2",
        "fixed3",
        "shuffled",
    )
    rows = []
    for seed in range(seeds):
        runs = {}
        for mode in modes:
            cfg = ResonantConfig(seed=seed, steps=steps)
            run = run_resonant(cfg, mode)
            runs[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "mean_curve_score": run["mean_curve_score"],
                "frequency_id_accuracy": run["frequency_id_accuracy"],
            }
        rows.append({"seed": seed, "runs": runs})

    summary = {}
    for mode in modes:
        summary[mode] = {
            "mean_final_score": float(
                np.mean([r["runs"][mode]["final_mean_score"] for r in rows])
            ),
            "mean_min_branch_score": float(
                np.mean(
                    [r["runs"][mode]["final_min_branch_score"] for r in rows]
                )
            ),
            "mean_curve_score": float(
                np.mean([r["runs"][mode]["mean_curve_score"] for r in rows])
            ),
            "mean_frequency_id_accuracy": float(
                np.mean(
                    [r["runs"][mode]["frequency_id_accuracy"] for r in rows]
                )
            ),
        }

    adaptive = np.array(
        [r["runs"]["adaptive"]["final_mean_score"] for r in rows]
    )
    comparisons = {}
    for other in ("fixed0", "fixed1", "fixed2", "fixed3", "shuffled"):
        x = np.array([r["runs"][other]["final_mean_score"] for r in rows])
        comparisons[f"adaptive_beats_{other}_seeds"] = int(
            np.sum(adaptive > x)
        )
        comparisons[f"adaptive_minus_{other}_mean"] = float(
            np.mean(adaptive - x)
        )

    return {
        "gate": "F7_frequency_addressed_causal_eligibility",
        "seeds": seeds,
        "steps": steps,
        "summary": summary,
        "comparisons": comparisons,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/resonant_credit_receipt.json"),
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
