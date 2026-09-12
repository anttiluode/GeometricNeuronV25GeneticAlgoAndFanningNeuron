"""Gate F9: move the resonant pole instead of mixing fixed channels.

F8 showed that a coarse fixed bank cannot synthesize hidden off-grid temporal
modes. F9 lets every branch adapt its own continuous resonant frequency.

The mechanism deliberately reuses the original V25 motif at a new level:

    fan nearby frequencies -> compare local predictive error -> retain winner
    -> shrink radius -> fan again

No branch is given its hidden true frequency. Candidate poles are judged only
by how well their local trace predicts the same one global scalar modulation.

This is still an engineered synthetic mechanism, not a biological identity
claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from nested_fan import branch_score
from offgrid_resonance import (
    CANDIDATE_FREQUENCIES,
    OFFGRID_FREQUENCIES,
    OffGridConfig,
    run_offgrid,
)


@dataclass(frozen=True)
class AdaptiveResonatorConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 6000
    active_branches: int = 3
    mutation_norm: float = 0.15
    learning_rate: float = 0.25
    curvature: float = 0.20
    world_decay: float = 0.95
    eligibility_decay: float = 0.95
    predictor_lr: float = 0.03
    epoch_steps: int = 800
    predictor_settle_steps: int = 80
    initial_frequency: float = np.pi / 6.0
    initial_fan_radius: float = np.pi / 12.0
    fan_shrink: float = 0.85
    min_frequency: float = 0.0
    max_frequency: float = np.pi / 3.0
    hidden_frequencies: tuple[float, ...] = OFFGRID_FREQUENCIES
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_rows(
    rng: np.random.Generator, count: int, dim: int, norm: float
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def run_adaptive_resonator(cfg: AdaptiveResonatorConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))

    hidden_bank = np.asarray(cfg.hidden_frequencies, dtype=float)
    hidden_index = np.arange(cfg.branches) % len(hidden_bank)
    hidden_frequency = hidden_bank[hidden_index]

    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    world_state = np.zeros(cfg.branches, dtype=np.complex128)
    world_phase = np.exp(1j * hidden_frequency)

    # Every branch begins from the same temporal hypothesis. Separation and
    # experience must pull them toward different hidden frequencies.
    omega = np.full(cfg.branches, cfg.initial_frequency, dtype=float)
    initial_mae = float(np.mean(np.abs(omega - hidden_frequency)))
    radius = cfg.initial_fan_radius
    offsets = np.array([-1.0, 0.0, 1.0])

    candidate_trace = np.zeros(
        (cfg.branches, len(offsets), cfg.dim), dtype=np.complex128
    )
    predictor = np.zeros(
        (cfg.branches, len(offsets), 2 * cfg.dim), dtype=float
    )
    error_sum = np.zeros((cfg.branches, len(offsets)), dtype=float)
    error_count = 0

    # Plasticity uses the currently retained pole while the nearby fan evaluates
    # alternatives in parallel.
    retained_trace = np.zeros((cfg.branches, cfg.dim), dtype=np.complex128)
    candidate_frequency = np.clip(
        omega[:, None] + offsets[None, :] * radius,
        cfg.min_frequency,
        cfg.max_frequency,
    )

    score_curve: list[float] = []
    omega_history = [omega.tolist()]
    radius_history = [float(radius)]

    for t in range(cfg.steps):
        within_epoch = t % cfg.epoch_steps

        if within_epoch == 0 and t > 0:
            mean_error = error_sum / max(error_count, 1)
            winner = np.argmin(mean_error, axis=1)
            omega = np.clip(
                omega + offsets[winner] * radius,
                cfg.min_frequency,
                cfg.max_frequency,
            )
            radius *= cfg.fan_shrink

            candidate_frequency = np.clip(
                omega[:, None] + offsets[None, :] * radius,
                cfg.min_frequency,
                cfg.max_frequency,
            )
            candidate_trace.fill(0.0)
            predictor.fill(0.0)
            error_sum.fill(0.0)
            error_count = 0
            retained_trace.fill(0.0)
            omega_history.append(omega.tolist())
            radius_history.append(float(radius))

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

        candidate_trace = (
            cfg.eligibility_decay
            * np.exp(1j * candidate_frequency)[:, :, None]
            * candidate_trace
            + delta[:, None, :]
        )
        features = np.concatenate(
            [candidate_trace.real, candidate_trace.imag], axis=2
        )
        prediction = np.sum(predictor * features, axis=2)
        residual = modulation - prediction
        feature_norm = np.sum(features * features, axis=2) + 1e-4
        predictor += (
            cfg.predictor_lr
            * (residual / feature_norm)[:, :, None]
            * features
        )

        if within_epoch >= cfg.predictor_settle_steps:
            error_sum += residual * residual
            error_count += 1

        retained_trace = (
            cfg.eligibility_decay
            * np.exp(1j * omega)[:, None]
            * retained_trace
            + delta
        )
        weights += cfg.learning_rate * modulation * retained_trace.real

        if (t + 1) % cfg.branches == 0:
            score_curve.append(
                float(np.mean(branch_score(weights, targets, cfg.curvature)))
            )

    # Let the last local fan cast its vote as a diagnostic, without applying
    # another plasticity epoch after the final frequency update.
    if error_count:
        winner = np.argmin(error_sum / error_count, axis=1)
        diagnostic_omega = np.clip(
            omega + offsets[winner] * radius,
            cfg.min_frequency,
            cfg.max_frequency,
        )
    else:
        diagnostic_omega = omega.copy()

    final = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "mean_curve_score": float(np.mean(score_curve)),
        "initial_frequency_mae": initial_mae,
        "final_frequency_mae": float(
            np.mean(np.abs(diagnostic_omega - hidden_frequency))
        ),
        "retained_frequency": omega.tolist(),
        "diagnostic_frequency": diagnostic_omega.tolist(),
        "hidden_frequency": hidden_frequency.tolist(),
        "omega_history": omega_history,
        "radius_history": radius_history,
        "score_curve": score_curve,
    }


def run_battery(seeds: int = 8, steps: int = 6000) -> dict:
    rows = []
    for seed in range(seeds):
        adaptive = run_adaptive_resonator(
            AdaptiveResonatorConfig(seed=seed, steps=steps)
        )
        # Matched F8 controls use the same hidden world and event budget.
        controls = {}
        for mode in ("discrete", "nearest", "oracle"):
            run = run_offgrid(OffGridConfig(seed=seed, steps=steps), mode)
            controls[mode] = {
                "final_mean_score": run["final_mean_score"],
                "final_min_branch_score": run["final_min_branch_score"],
                "mean_curve_score": run["mean_curve_score"],
            }
        rows.append(
            {
                "seed": seed,
                "adaptive": {
                    "final_mean_score": adaptive["final_mean_score"],
                    "final_min_branch_score": adaptive["final_min_branch_score"],
                    "mean_curve_score": adaptive["mean_curve_score"],
                    "initial_frequency_mae": adaptive["initial_frequency_mae"],
                    "final_frequency_mae": adaptive["final_frequency_mae"],
                },
                "controls": controls,
            }
        )

    adaptive_final = np.array(
        [r["adaptive"]["final_mean_score"] for r in rows], dtype=float
    )
    summary = {
        "adaptive": {
            "mean_final_score": float(np.mean(adaptive_final)),
            "mean_min_branch_score": float(
                np.mean([r["adaptive"]["final_min_branch_score"] for r in rows])
            ),
            "mean_curve_score": float(
                np.mean([r["adaptive"]["mean_curve_score"] for r in rows])
            ),
            "mean_initial_frequency_mae": float(
                np.mean([r["adaptive"]["initial_frequency_mae"] for r in rows])
            ),
            "mean_final_frequency_mae": float(
                np.mean([r["adaptive"]["final_frequency_mae"] for r in rows])
            ),
        }
    }
    comparisons = {}
    for mode in ("discrete", "nearest", "oracle"):
        vals = np.array(
            [r["controls"][mode]["final_mean_score"] for r in rows],
            dtype=float,
        )
        summary[mode] = {
            "mean_final_score": float(np.mean(vals)),
            "mean_min_branch_score": float(
                np.mean(
                    [r["controls"][mode]["final_min_branch_score"] for r in rows]
                )
            ),
            "mean_curve_score": float(
                np.mean([r["controls"][mode]["mean_curve_score"] for r in rows])
            ),
        }
        comparisons[f"adaptive_beats_{mode}_seeds"] = int(
            np.sum(adaptive_final > vals)
        )
        comparisons[f"adaptive_minus_{mode}_mean"] = float(
            np.mean(adaptive_final - vals)
        )

    return {
        "gate": "F9_adaptive_continuous_resonator_fan",
        "seeds": seeds,
        "steps": steps,
        "hidden_frequencies": list(OFFGRID_FREQUENCIES),
        "coarse_reference_bank": list(CANDIDATE_FREQUENCIES),
        "summary": summary,
        "comparisons": comparisons,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/adaptive_resonator_receipt.json"),
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
