"""Gate F6: each branch learns which eligibility timescale to trust.

F5 mapped which fixed eligibility decay works best for different downstream
consequence timescales. F6 removes the experimenter's choice. Every branch
carries the same bank of local eligibility traces. A tiny local predictor is
trained only from the observed global modulation and that branch's own trace.
The branch uses the trace whose predictor has the lowest recent error.

Halfway through the default run the downstream recurrence changes abruptly
from a fast world to a slow world. The learner is not told the switch time or
the new decay.

This is a synthetic metaplasticity experiment. The predictor bank is not a
claim about a particular molecular implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from nested_fan import branch_score


@dataclass(frozen=True)
class AdaptiveTimescaleConfig:
    branches: int = 12
    dim: int = 6
    steps: int = 6000
    active_branches: int = 3
    mutation_norm: float = 0.18
    curvature: float = 0.20
    learning_rate: float = 0.15
    consequence_noise: float = 0.001
    trace_decays: tuple[float, ...] = (0.20, 0.70, 0.95, 0.98)
    downstream_decay_before: float = 0.0
    downstream_decay_after: float = 0.90
    switch_fraction: float = 0.50
    predictor_learning_rate: float = 0.02
    predictor_error_decay: float = 0.99
    seed: int = 0


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _fixed_norm_rows(
    rng: np.random.Generator, count: int, dim: int, norm: float
) -> np.ndarray:
    x = rng.normal(size=(count, dim))
    return x * (norm / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12))


def _event(
    rng: np.random.Generator,
    weights: np.ndarray,
    targets: np.ndarray,
    cfg: AdaptiveTimescaleConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    active = rng.choice(cfg.branches, size=cfg.active_branches, replace=False)
    delta = np.zeros_like(weights)
    delta[active] = _fixed_norm_rows(
        rng, cfg.active_branches, cfg.dim, cfg.mutation_norm
    )
    plus = float(np.mean(branch_score(weights + delta, targets, cfg.curvature)))
    minus = float(np.mean(branch_score(weights - delta, targets, cfg.curvature)))
    consequence = (plus - minus) + float(rng.normal(scale=cfg.consequence_noise))
    return active, delta, consequence


def _switch_step(cfg: AdaptiveTimescaleConfig) -> int:
    return int(round(cfg.steps * cfg.switch_fraction))


def run_fixed_trace(cfg: AdaptiveTimescaleConfig, trace_decay: float) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))
    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    eligibility = np.zeros_like(weights)
    downstream_state = 0.0
    curve: list[float] = []
    switch = _switch_step(cfg)
    scale = cfg.learning_rate / (2.0 * cfg.mutation_norm * cfg.mutation_norm)

    for t in range(cfg.steps):
        downstream_decay = (
            cfg.downstream_decay_before if t < switch else cfg.downstream_decay_after
        )
        _, delta, consequence = _event(rng, weights, targets, cfg)
        downstream_state = downstream_decay * downstream_state + consequence
        modulation = (1.0 - downstream_decay) * downstream_state
        eligibility = trace_decay * eligibility + delta
        weights += scale * modulation * eligibility
        if (t + 1) % cfg.branches == 0:
            curve.append(float(np.mean(branch_score(weights, targets, cfg.curvature))))

    final = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "trace_decay": trace_decay,
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "score_curve": curve,
    }


def run_adaptive(cfg: AdaptiveTimescaleConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    targets = _unit_rows(rng.normal(size=(cfg.branches, cfg.dim)))
    weights = np.zeros((cfg.branches, cfg.dim), dtype=float)
    k_count = len(cfg.trace_decays)

    # Every branch owns all traces and all predictor states. Nothing here stores
    # a reward-to-source label. The only shared signal is scalar modulation.
    traces = np.zeros((k_count, cfg.branches, cfg.dim), dtype=float)
    predictors = np.zeros_like(traces)
    predictor_error = np.full((cfg.branches, k_count), 1e-3, dtype=float)

    downstream_state = 0.0
    curve: list[float] = []
    switch = _switch_step(cfg)
    scale = cfg.learning_rate / (2.0 * cfg.mutation_norm * cfg.mutation_norm)
    choice_before = np.zeros(k_count, dtype=np.int64)
    choice_after = np.zeros(k_count, dtype=np.int64)
    total_before = 0
    total_after = 0

    for t in range(cfg.steps):
        downstream_decay = (
            cfg.downstream_decay_before if t < switch else cfg.downstream_decay_after
        )
        active, delta, consequence = _event(rng, weights, targets, cfg)
        downstream_state = downstream_decay * downstream_state + consequence
        modulation = (1.0 - downstream_decay) * downstream_state

        for k, decay in enumerate(cfg.trace_decays):
            traces[k] = decay * traces[k] + delta

        # Each active branch asks how well each of its local traces predicts the
        # observed global modulation. Normalized LMS prevents long traces from
        # winning merely because they have larger norm.
        for branch in active:
            local_trace = traces[:, branch, :]
            prediction = np.einsum(
                "ki,ki->k", predictors[:, branch, :], local_trace
            )
            residual = modulation - prediction
            norm2 = np.einsum("ki,ki->k", local_trace, local_trace) + 1e-6
            predictors[:, branch, :] += (
                cfg.predictor_learning_rate
                * (residual / norm2)[:, None]
                * local_trace
            )
            predictor_error[branch] = (
                cfg.predictor_error_decay * predictor_error[branch]
                + (1.0 - cfg.predictor_error_decay) * residual * residual
            )

        # Selection is local: two branches may trust different causal memories.
        chosen = np.argmin(predictor_error, axis=1)
        used = np.zeros_like(weights)
        for branch in range(cfg.branches):
            used[branch] = traces[chosen[branch], branch]
        weights += scale * modulation * used

        counts = np.bincount(chosen, minlength=k_count)
        if t < switch:
            choice_before += counts
            total_before += cfg.branches
        else:
            choice_after += counts
            total_after += cfg.branches

        if (t + 1) % cfg.branches == 0:
            curve.append(float(np.mean(branch_score(weights, targets, cfg.curvature))))

    final = branch_score(weights, targets, cfg.curvature)
    return {
        "config": asdict(cfg),
        "final_mean_score": float(np.mean(final)),
        "final_min_branch_score": float(np.min(final)),
        "score_curve": curve,
        "trace_fraction_before": (choice_before / max(total_before, 1)).tolist(),
        "trace_fraction_after": (choice_after / max(total_after, 1)).tolist(),
        "mean_predictor_error": np.mean(predictor_error, axis=0).tolist(),
    }


def _curve_area(curve: list[float]) -> float:
    return float(np.mean(curve)) if curve else float("nan")


def _score_at_switch(curve: list[float], cfg: AdaptiveTimescaleConfig) -> float:
    if not curve:
        return float("nan")
    cycle = max(1, _switch_step(cfg) // cfg.branches)
    return float(curve[min(cycle - 1, len(curve) - 1)])


def run_battery(seeds: int = 8, steps: int = 6000) -> dict:
    rows = []
    for seed in range(seeds):
        cfg = AdaptiveTimescaleConfig(seed=seed, steps=steps)
        adaptive = run_adaptive(cfg)
        fixed = {}
        for decay in cfg.trace_decays:
            run = run_fixed_trace(cfg, decay)
            fixed[str(decay)] = {
                "final_score": run["final_mean_score"],
                "curve_area": _curve_area(run["score_curve"]),
                "score_at_switch": _score_at_switch(run["score_curve"], cfg),
            }
        rows.append(
            {
                "seed": seed,
                "adaptive": {
                    "final_score": adaptive["final_mean_score"],
                    "curve_area": _curve_area(adaptive["score_curve"]),
                    "score_at_switch": _score_at_switch(adaptive["score_curve"], cfg),
                    "trace_fraction_before": adaptive["trace_fraction_before"],
                    "trace_fraction_after": adaptive["trace_fraction_after"],
                },
                "fixed": fixed,
            }
        )

    cfg0 = AdaptiveTimescaleConfig(steps=steps)
    adaptive_final = np.array([r["adaptive"]["final_score"] for r in rows])
    summary_fixed = {}
    comparisons = {}
    for decay in cfg0.trace_decays:
        key = str(decay)
        vals = np.array([r["fixed"][key]["final_score"] for r in rows])
        summary_fixed[key] = {
            "mean_final_score": float(np.mean(vals)),
            "std_final_score": float(np.std(vals)),
            "mean_curve_area": float(
                np.mean([r["fixed"][key]["curve_area"] for r in rows])
            ),
            "mean_score_at_switch": float(
                np.mean([r["fixed"][key]["score_at_switch"] for r in rows])
            ),
        }
        comparisons[f"adaptive_beats_fixed_{key}_seeds"] = int(
            np.sum(adaptive_final > vals)
        )

    adaptive_summary = {
        "mean_final_score": float(np.mean(adaptive_final)),
        "std_final_score": float(np.std(adaptive_final)),
        "mean_curve_area": float(
            np.mean([r["adaptive"]["curve_area"] for r in rows])
        ),
        "mean_score_at_switch": float(
            np.mean([r["adaptive"]["score_at_switch"] for r in rows])
        ),
        "mean_trace_fraction_before": np.mean(
            [r["adaptive"]["trace_fraction_before"] for r in rows], axis=0
        ).tolist(),
        "mean_trace_fraction_after": np.mean(
            [r["adaptive"]["trace_fraction_after"] for r in rows], axis=0
        ).tolist(),
    }

    return {
        "gate": "F6_adaptive_local_timescale_selection",
        "seeds": seeds,
        "steps": steps,
        "switch_step": _switch_step(cfg0),
        "downstream_decay_before": cfg0.downstream_decay_before,
        "downstream_decay_after": cfg0.downstream_decay_after,
        "trace_decays": list(cfg0.trace_decays),
        "adaptive": adaptive_summary,
        "fixed": summary_fixed,
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
        default=Path("results/adaptive_timescale_receipt.json"),
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
                "adaptive": receipt["adaptive"],
                "fixed": receipt["fixed"],
                "comparisons": receipt["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
