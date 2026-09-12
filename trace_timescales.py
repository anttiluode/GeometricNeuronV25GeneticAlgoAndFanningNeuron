"""Gate F5: eligibility timescale versus downstream consequence timescale.

F4 showed that a local eligibility trace can extract useful credit from one
unlabelled, temporally mixed global consequence. F5 asks whether the trace can
have an arbitrary decay. It cannot: the useful local memory timescale depends
on how slowly the downstream consequence is mixed.

This gate is a sweep, not yet an adaptive rule. It maps the landscape that a
future adaptive trace selector would have to learn.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from mixed_credit import MixedCreditConfig, run_mixed_credit

DEFAULT_DOWNSTREAM = (0.20, 0.50, 0.70, 0.90)
DEFAULT_ELIGIBILITY = (0.00, 0.20, 0.50, 0.70, 0.85, 0.95, 0.98)


def run_sweep(
    seeds: int = 8,
    steps: int = 4000,
    downstream_decays: tuple[float, ...] = DEFAULT_DOWNSTREAM,
    eligibility_decays: tuple[float, ...] = DEFAULT_ELIGIBILITY,
) -> dict:
    rows = []
    for downstream in downstream_decays:
        for eligibility in eligibility_decays:
            vals = []
            mins = []
            for seed in range(seeds):
                cfg = MixedCreditConfig(
                    seed=seed,
                    steps=steps,
                    downstream_decay=downstream,
                    eligibility_decay=eligibility,
                )
                run = run_mixed_credit(cfg, "eligibility")
                vals.append(run["final_mean_score"])
                mins.append(run["final_min_branch_score"])
            rows.append(
                {
                    "downstream_decay": downstream,
                    "eligibility_decay": eligibility,
                    "mean_final_score": float(np.mean(vals)),
                    "std_final_score": float(np.std(vals)),
                    "mean_min_branch_score": float(np.mean(mins)),
                }
            )

    best = []
    for downstream in downstream_decays:
        candidates = [r for r in rows if r["downstream_decay"] == downstream]
        winner = max(candidates, key=lambda r: r["mean_final_score"])
        fixed = min(candidates, key=lambda r: abs(r["eligibility_decay"] - 0.70))
        best.append(
            {
                "downstream_decay": downstream,
                "best_eligibility_decay": winner["eligibility_decay"],
                "best_mean_final_score": winner["mean_final_score"],
                "fixed_070_score": fixed["mean_final_score"],
                "best_minus_fixed_070": winner["mean_final_score"]
                - fixed["mean_final_score"],
            }
        )

    best_decays = [r["best_eligibility_decay"] for r in best]
    monotone_non_decreasing = all(
        b >= a for a, b in zip(best_decays, best_decays[1:])
    )

    return {
        "gate": "F5_eligibility_timescale_map",
        "seeds": seeds,
        "steps": steps,
        "downstream_decays": list(downstream_decays),
        "eligibility_decays": list(eligibility_decays),
        "best_by_downstream": best,
        "best_trace_slows_monotonically": monotone_non_decreasing,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/trace_timescale_receipt.json")
    )
    args = parser.parse_args()
    receipt = run_sweep(args.seeds, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            {
                "gate": receipt["gate"],
                "seeds": receipt["seeds"],
                "steps": receipt["steps"],
                "best_by_downstream": receipt["best_by_downstream"],
                "best_trace_slows_monotonically": receipt[
                    "best_trace_slows_monotonically"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
