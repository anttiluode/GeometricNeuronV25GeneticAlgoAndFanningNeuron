"""Gate F8: break the preloaded-frequency privilege from F7.

F7 put each hidden branch frequency exactly on the learner's candidate bank.
F8 moves every hidden frequency between the candidate channels.

The question is deliberately adversarial: is the F7 mechanism a genuine
spectral geometry, or merely a discrete channel selector?

Arms:
  * discrete: the original F7 prediction-error selector.
  * softmix: soft error-weighted mixture of all candidate eligibility traces.
  * nearest: privileged attacker that is handed the nearest candidate channel.
  * oracle: privileged upper bound using the exact hidden off-grid frequency.

If the first three remain far below oracle, the current bank cannot synthesize
an unseen temporal mode. That negative result is the intended scientific test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import argparse
import json
import numpy as np

from nested_fan import branch_score

Mode = Literal["discrete", "softmix", "nearest", "oracle"]

CANDIDATE_FREQUENCIES = (0.0, np.pi / 12.0, np.pi / 6.0, np.pi / 3.0)
OFFGRID_FREQUENCIES = (np.pi / 24.0, np.pi / 8.0, np.pi / 4.0)


@dataclass(frozen=True)
class OffGridConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 4000
    active_branches: int = 3
    mutation_norm: float = 0.15
    learning_rate: float = 0.25
    curvature: float = 0.20
    world_decay: float = 0.95
    eligibility_decay: float = 0.95
    predictor_lr: float = 0.03
    error_ema: float = 0.005
    warmup_steps: int = 300
    soft_beta: float = 100.0
    candidate_frequencies: tuple[float, ...] = CANDIDATE_FREQUENCIES
    hidden_frequencies: tuple[float, ...] = OFFGRID_FREQUENCIES
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_rows(
    rng: np.random.Generator, count: int, dim: int, norm: float
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def run_offgrid(cfg: OffGridConfig, mode: Mode) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))

    candidate = np.asarray(cfg.candidate_frequencies, dtype=float)
    hidden_bank = np.asarray(cfg.hidden_frequencies, dtype=float)
    hidden_index = np.arange(cfg.branches) % len(hidden_bank)
    hidden_frequency = hidden_bank[hidden_index]

    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    world_state = np.zeros(cfg.branches, dtype=np.complex128)
    world_phase = np.exp(1j * hidden_frequency)

    eligibility = np.zeros(
        (cfg.branches, len(candidate), cfg.dim), dtype=np.complex128
    )
    candidate_phase = np.exp(1j * candidate)
    predictor = np.zeros(
        (cfg.branches, len(candidate), 2 * cfg.dim), dtype=float
    )
    error = np.ones((cfg.branches, len(candidate)), dtype=float)

    oracle_trace = np.zeros((cfg.branches, cfg.dim), dtype=np.complex128)
    score_curve: list[float] = []
    selected_counts = np.zeros((cfg.branches, len(candidate)), dtype=int)

    nearest_index = np.argmin(
        np.abs(candidate[None, :] - hidden_frequency[:, None]), axis=1
    )

    for t in range(cfg.steps):
        active = rng.choice(
            cfg.branches, size=cfg.active_branches, replace=False
        )
        local_delta = _fixed_norm_rows(
            rng, cfg.active_branches, cfg.dim, cfg.mutation_norm
        )
        delta = np.zeros_like(weights)
        delta[active] = local_delta

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

        if mode == "discrete":
            selected = np.argmin(error, axis=1)
            if t < cfg.warmup_steps:
                selected = rng.integers(len(candidate), size=cfg.branches)
            selected_counts[np.arange(cfg.branches), selected] += 1
            used = eligibility[np.arange(cfg.branches), selected].real
        elif mode == "softmix":
            relative = error - np.min(error, axis=1, keepdims=True)
            alpha = np.exp(-cfg.soft_beta * relative)
            alpha /= np.sum(alpha, axis=1, keepdims=True)
            if t < cfg.warmup_steps:
                alpha[:] = 1.0 / len(candidate)
            used = np.sum(alpha[:, :, None] * eligibility.real, axis=1)
        elif mode == "nearest":
            used = eligibility[np.arange(cfg.branches), nearest_index].real
        elif mode == "oracle":
            oracle_trace = (
                cfg.eligibility_decay
                * np.exp(1j * hidden_frequency)[:, None]
                * oracle_trace
                + delta
            )
            used = oracle_trace.real
        else:
            raise ValueError(f"unknown mode: {mode}")

        weights += cfg.learning_rate * modulation * used

        if (t + 1) % cfg.branches == 0:
            score_curve.append(
                float(np.mean(branch_score(weights, targets, cfg.curvature)))
            )

    final = branch_score(weights, targets, cfg.curvature)
    dominant = np.argmax(selected_counts, axis=1)
    dominant_frequency = candidate[dominant]
    return {
        "config": asdict(cfg),
        "mode": mode,
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "mean_curve_score": float(np.mean(score_curve)),
        "nearest_frequency_mae": float(
            np.mean(np.abs(candidate[nearest_index] - hidden_frequency))
        ),
        "dominant_frequency_mae": float(
            np.mean(np.abs(dominant_frequency - hidden_frequency))
        ) if mode == "discrete" else None,
        "score_curve": score_curve,
    }


def run_battery(seeds: int = 8, steps: int = 4000) -> dict:
    modes: tuple[Mode, ...] = ("discrete", "softmix", "nearest", "oracle")
    rows = []
    for seed in range(seeds):
        runs = {}
        for mode in modes:
            run = run_offgrid(OffGridConfig(seed=seed, steps=steps), mode)
            runs[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "mean_curve_score": run["mean_curve_score"],
                "dominant_frequency_mae": run["dominant_frequency_mae"],
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
        }

    oracle = np.array(
        [r["runs"]["oracle"]["final_mean_score"] for r in rows]
    )
    return {
        "gate": "F8_offgrid_resonant_credit_attacker",
        "seeds": seeds,
        "steps": steps,
        "candidate_frequencies": list(CANDIDATE_FREQUENCIES),
        "hidden_frequencies": list(OFFGRID_FREQUENCIES),
        "summary": summary,
        "comparisons": {
            "oracle_minus_discrete_mean": float(
                np.mean(
                    oracle
                    - np.array(
                        [r["runs"]["discrete"]["final_mean_score"] for r in rows]
                    )
                )
            ),
            "oracle_minus_nearest_mean": float(
                np.mean(
                    oracle
                    - np.array(
                        [r["runs"]["nearest"]["final_mean_score"] for r in rows]
                    )
                )
            ),
            "discrete_beats_softmix_seeds": int(
                np.sum(
                    np.array(
                        [r["runs"]["discrete"]["final_mean_score"] for r in rows]
                    )
                    > np.array(
                        [r["runs"]["softmix"]["final_mean_score"] for r in rows]
                    )
                )
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/offgrid_resonance_receipt.json"),
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
