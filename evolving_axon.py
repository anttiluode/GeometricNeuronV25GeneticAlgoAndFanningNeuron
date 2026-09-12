"""GeometricNeuronV25TheEvolvingAxon

Minimal experiments for treating the sequence of GA generations as a signal.

The state we observe is the elite centroid of a population of linear operators.
A low-rank DMD-like model is fit online to that trajectory and asked to predict
its next state.  A second experiment uses the predicted displacement to bias a
fraction of mutations, testing whether generational history can become useful
rather than merely descriptive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import json
import math
import numpy as np


@dataclass
class GAConfig:
    dim: int = 10
    population: int = 96
    elite_fraction: float = 0.20
    generations: int = 100
    mutation_sigma: float = 0.12
    seed: int = 0
    predictor_rank: int = 6
    predictor_window: int = 24
    predictor_min_history: int = 10
    guide_fraction: float = 0.35
    guide_strength: float = 0.70
    path_beta: float = 0.80
    fitness_probe_fraction: float = 1.0


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    elite_fitness: float
    centroid: np.ndarray
    best: np.ndarray


def _target_operator(dim: int, seed: int) -> np.ndarray:
    """Create a fixed, structured linear operator with mixed spatial scales."""
    rng = np.random.default_rng(seed)
    q1, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    q2, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    spectrum = np.linspace(1.25, 0.15, dim)
    return q1 @ np.diag(spectrum) @ q2.T


def _probe_bank(dim: int, seed: int, n: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def _fitness(pop: np.ndarray, target: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Negative mean squared output error on a fixed public probe bank."""
    desired = probes @ target.T
    predicted = np.einsum("pd,ndk->npk", probes, pop)
    return -np.mean((predicted - desired[None, :, :]) ** 2, axis=(1, 2))


def _initial_population(cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(scale=0.25, size=(cfg.population, cfg.dim, cfg.dim))


def _elite_summary(
    pop: np.ndarray,
    selection_fitness: np.ndarray,
    elite_n: int,
    report_fitness: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    idx = np.argsort(selection_fitness)[-elite_n:]
    elites = pop[idx]
    report = selection_fitness if report_fitness is None else report_fitness
    best_idx = int(np.argmax(report))
    return elites.mean(axis=0), pop[best_idx].copy(), float(report[best_idx]), float(report[idx].mean())


def low_rank_next_prediction(history: Iterable[np.ndarray], rank: int = 6, window: int = 24) -> np.ndarray | None:
    """Predict the next flattened operator with a low-rank linear flow."""
    hs = [np.asarray(h, dtype=float).reshape(-1) for h in history]
    if len(hs) < 4:
        return None
    h = np.stack(hs[-(window + 1):], axis=0)
    center = h.mean(axis=0)
    hc = h - center
    _, s, vt = np.linalg.svd(hc, full_matrices=False)
    nonzero = int(np.sum(s > 1e-10))
    r = min(rank, nonzero, h.shape[0] - 2)
    if r < 1:
        return hs[-1].copy()
    basis = vt[:r].T
    z = hc @ basis
    x = np.c_[z[:-1], np.ones(len(z) - 1)]
    y = z[1:]
    ridge = 1e-6
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, x.T @ y)
    z_next = np.r_[z[-1], 1.0] @ coef
    return center + basis @ z_next


def _make_children(
    elites: np.ndarray,
    count: int,
    sigma: float,
    rng: np.random.Generator,
    guide_delta: np.ndarray | None = None,
    guide_fraction: float = 0.0,
    guide_strength: float = 0.0,
) -> np.ndarray:
    parents = elites[rng.integers(0, len(elites), size=count)]
    noise = rng.normal(scale=sigma, size=parents.shape)
    children = parents + noise
    if guide_delta is not None and np.linalg.norm(guide_delta) > 1e-12:
        k = int(round(count * guide_fraction))
        if k > 0:
            chosen = rng.choice(count, size=k, replace=False)
            flat_noise = noise.reshape(count, -1)
            typical = float(np.median(np.linalg.norm(flat_noise, axis=1)))
            d = guide_delta.reshape(-1)
            d = d / (np.linalg.norm(d) + 1e-12) * typical * guide_strength
            children[chosen] += d.reshape(guide_delta.shape)
    return children


def run_ga(cfg: GAConfig, guided: bool = False, guide_mode: str = "trajectory") -> list[GenerationRecord]:
    rng = np.random.default_rng(cfg.seed)
    target = _target_operator(cfg.dim, cfg.seed + 1000)
    probes = _probe_bank(cfg.dim, cfg.seed + 2000)
    pop = _initial_population(cfg, rng)
    elite_n = max(2, int(round(cfg.population * cfg.elite_fraction)))
    records: list[GenerationRecord] = []
    centroids: list[np.ndarray] = []
    evolution_path = np.zeros((cfg.dim, cfg.dim), dtype=float)

    for g in range(cfg.generations):
        true_fit = _fitness(pop, target, probes)
        if cfg.fitness_probe_fraction < 1.0:
            m = max(4, int(round(len(probes) * cfg.fitness_probe_fraction)))
            chosen_probes = probes[rng.choice(len(probes), size=m, replace=False)]
            selection_fit = _fitness(pop, target, chosen_probes)
        else:
            selection_fit = true_fit
        centroid, best, best_fit, elite_fit = _elite_summary(
            pop, selection_fit, elite_n, report_fitness=true_fit
        )
        records.append(GenerationRecord(g, best_fit, elite_fit, centroid.copy(), best))
        centroids.append(centroid.copy())
        elites = pop[np.argsort(selection_fit)[-elite_n:]]

        if len(centroids) >= 2:
            step = centroid - centroids[-2]
            evolution_path = cfg.path_beta * evolution_path + (1.0 - cfg.path_beta) * step

        guide_delta = None
        if guided:
            if guide_mode == "velocity" and len(centroids) >= 2:
                guide_delta = centroid - centroids[-2]
            elif guide_mode == "path" and len(centroids) >= 2:
                guide_delta = evolution_path
            elif guide_mode == "trajectory" and len(centroids) >= cfg.predictor_min_history:
                pred = low_rank_next_prediction(centroids, cfg.predictor_rank, cfg.predictor_window)
                if pred is not None:
                    guide_delta = pred.reshape(cfg.dim, cfg.dim) - centroid
            elif guide_mode not in {"velocity", "path", "trajectory"}:
                raise ValueError(f"unknown guide_mode: {guide_mode}")

        child_count = cfg.population - elite_n
        children = _make_children(
            elites,
            child_count,
            cfg.mutation_sigma,
            rng,
            guide_delta=guide_delta,
            guide_fraction=cfg.guide_fraction if guided else 0.0,
            guide_strength=cfg.guide_strength if guided else 0.0,
        )
        pop = np.concatenate([elites.copy(), children], axis=0)

    return records


def gate0_prediction(cfg: GAConfig) -> dict:
    records = run_ga(cfg, guided=False)
    c = [r.centroid for r in records]
    rows = []
    for t in range(cfg.predictor_min_history, len(c) - 1):
        actual = c[t + 1].reshape(-1)
        current = c[t].reshape(-1)
        persistence = current
        velocity = current + (current - c[t - 1].reshape(-1))
        dmd = low_rank_next_prediction(c[: t + 1], cfg.predictor_rank, cfg.predictor_window)
        if dmd is None:
            continue
        denom = np.linalg.norm(actual - current) + 1e-12
        actual_step = actual - current
        velocity_step = current - c[t - 1].reshape(-1)
        trajectory_step = dmd - current

        def cosine(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        rows.append({
            "generation": t + 1,
            "persistence": float(np.linalg.norm(actual - persistence)),
            "velocity": float(np.linalg.norm(actual - velocity)),
            "trajectory": float(np.linalg.norm(actual - dmd)),
            "normalized_trajectory_error": float(np.linalg.norm(actual - dmd) / denom),
            "velocity_direction_cosine": cosine(velocity_step, actual_step),
            "trajectory_direction_cosine": cosine(trajectory_step, actual_step),
        })

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    wins_persist = sum(r["trajectory"] < r["persistence"] for r in rows)
    wins_velocity = sum(r["trajectory"] < r["velocity"] for r in rows)
    return {
        "seed": cfg.seed,
        "comparisons": len(rows),
        "mean_error": {
            "persistence": mean("persistence"),
            "velocity": mean("velocity"),
            "trajectory": mean("trajectory"),
        },
        "trajectory_win_rate_vs_persistence": wins_persist / max(1, len(rows)),
        "trajectory_win_rate_vs_velocity": wins_velocity / max(1, len(rows)),
        "mean_direction_cosine": {
            "velocity": mean("velocity_direction_cosine"),
            "trajectory": mean("trajectory_direction_cosine"),
        },
        "rows": rows,
    }


def _first_generation_at(records: list[GenerationRecord], threshold: float) -> int | None:
    for r in records:
        if -r.best_fitness <= threshold:
            return r.generation
    return None


def gate1_guidance(cfg: GAConfig, mse_threshold: float = 0.008) -> dict:
    base = run_ga(cfg, guided=False)
    velocity = run_ga(cfg, guided=True, guide_mode="velocity")
    trajectory = run_ga(cfg, guided=True, guide_mode="trajectory")
    return {
        "seed": cfg.seed,
        "threshold_mse": mse_threshold,
        "baseline_final_mse": -base[-1].best_fitness,
        "velocity_final_mse": -velocity[-1].best_fitness,
        "trajectory_final_mse": -trajectory[-1].best_fitness,
        "baseline_reach_generation": _first_generation_at(base, mse_threshold),
        "velocity_reach_generation": _first_generation_at(velocity, mse_threshold),
        "trajectory_reach_generation": _first_generation_at(trajectory, mse_threshold),
        "baseline_curve": [-r.best_fitness for r in base],
        "velocity_curve": [-r.best_fitness for r in velocity],
        "trajectory_curve": [-r.best_fitness for r in trajectory],
    }


def gate2_path_memory(cfg: GAConfig, betas=(0.0, 0.5, 0.8, 0.95)) -> dict:
    base = run_ga(cfg, guided=False)
    out = {"seed": cfg.seed, "baseline_final_mse": -base[-1].best_fitness, "paths": {}}
    for beta in betas:
        local = GAConfig(**{**cfg.__dict__, "path_beta": float(beta)})
        rec = run_ga(local, guided=True, guide_mode="path")
        out["paths"][str(beta)] = {
            "final_mse": -rec[-1].best_fitness,
            "ratio_to_baseline": (-rec[-1].best_fitness) / (-base[-1].best_fitness + 1e-12),
            "curve": [-r.best_fitness for r in rec],
        }
    return out


def gate3_bounded_fitness(
    cfg: GAConfig,
    fractions=(0.25, 0.125),
    betas=(0.0, 0.5, 0.8, 0.95),
) -> dict:
    out = {"seed": cfg.seed, "fractions": {}}
    for fraction in fractions:
        local_base = GAConfig(**{**cfg.__dict__, "fitness_probe_fraction": float(fraction)})
        base = run_ga(local_base, guided=False)
        frow = {"baseline_final_mse": -base[-1].best_fitness, "paths": {}}
        for beta in betas:
            local = GAConfig(**{
                **cfg.__dict__,
                "fitness_probe_fraction": float(fraction),
                "path_beta": float(beta),
            })
            rec = run_ga(local, guided=True, guide_mode="path")
            frow["paths"][str(beta)] = {
                "final_mse": -rec[-1].best_fitness,
                "ratio_to_baseline": (-rec[-1].best_fitness) / (-base[-1].best_fitness + 1e-12),
            }
        out["fractions"][str(fraction)] = frow
    return out


def run_battery(seeds: int = 12, generations: int = 100) -> dict:
    gate0 = []
    gate1 = []
    gate2 = []
    for seed in range(seeds):
        cfg = GAConfig(seed=seed, generations=generations)
        gate0.append(gate0_prediction(cfg))
        gate1.append(gate1_guidance(cfg))
        gate2.append(gate2_path_memory(cfg))

    pred_ratio = [g["mean_error"]["trajectory"] / (g["mean_error"]["persistence"] + 1e-12) for g in gate0]
    vel_ratio = [g["mean_error"]["trajectory"] / (g["mean_error"]["velocity"] + 1e-12) for g in gate0]
    trajectory_ratio = [g["trajectory_final_mse"] / (g["baseline_final_mse"] + 1e-12) for g in gate1]
    velocity_ratio_final = [g["velocity_final_mse"] / (g["baseline_final_mse"] + 1e-12) for g in gate1]
    traj_vs_vel = [g["trajectory_final_mse"] / (g["velocity_final_mse"] + 1e-12) for g in gate1]
    reach_triples = [
        (g["baseline_reach_generation"], g["velocity_reach_generation"], g["trajectory_reach_generation"])
        for g in gate1
        if g["baseline_reach_generation"] is not None
        and g["velocity_reach_generation"] is not None
        and g["trajectory_reach_generation"] is not None
    ]
    beta_keys = list(gate2[0]["paths"].keys()) if gate2 else []
    path_ratios = {beta: [g["paths"][beta]["ratio_to_baseline"] for g in gate2] for beta in beta_keys}
    summary = {
        "seeds": seeds,
        "gate0_trajectory_over_persistence_mean": float(np.mean(pred_ratio)),
        "gate0_trajectory_over_velocity_mean": float(np.mean(vel_ratio)),
        "gate0_better_than_persistence_seeds": int(sum(r < 1.0 for r in pred_ratio)),
        "gate0_better_than_velocity_seeds": int(sum(r < 1.0 for r in vel_ratio)),
        "gate0_velocity_direction_cosine_mean": float(np.mean([g["mean_direction_cosine"]["velocity"] for g in gate0])),
        "gate0_trajectory_direction_cosine_mean": float(np.mean([g["mean_direction_cosine"]["trajectory"] for g in gate0])),
        "gate1_velocity_over_baseline_final_mse_mean": float(np.mean(velocity_ratio_final)),
        "gate1_trajectory_over_baseline_final_mse_mean": float(np.mean(trajectory_ratio)),
        "gate1_trajectory_over_velocity_final_mse_mean": float(np.mean(traj_vs_vel)),
        "gate1_velocity_better_final_seeds": int(sum(r < 1.0 for r in velocity_ratio_final)),
        "gate1_trajectory_better_final_seeds": int(sum(r < 1.0 for r in trajectory_ratio)),
        "gate1_trajectory_beats_velocity_seeds": int(sum(r < 1.0 for r in traj_vs_vel)),
        "gate1_reach_triples": reach_triples,
        "gate2_path_ratio_mean": {beta: float(np.mean(vals)) for beta, vals in path_ratios.items()},
        "gate2_path_better_seeds": {beta: int(sum(v < 1.0 for v in vals)) for beta, vals in path_ratios.items()},
    }
    return {"summary": summary, "gate0": gate0, "gate1": gate1, "gate2": gate2}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--output", default="results.json")
    args = p.parse_args()
    out = run_battery(args.seeds, args.generations)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
