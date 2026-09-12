# Gate F2 — Nested fan selection under bounded observation

This gate follows the Aizenbud/Leterrier-inspired interpretation of V25:

```text
semi-independent branch routes
        ↓
local nonlinear consequence
        ↓
selection / eligibility bottleneck
        ↓
retain route-specific change
        ↓
branch-specific delta history shapes the next fan
```

The experiment is synthetic. It does **not** claim that dendrites or the AIS literally run this algorithm.

## Why bounded observation matters

There are 12 independent branches. Each branch has a hidden preferred direction in a 6-D parameter space.

Only one branch/context is observable on a generation. The optimizer can therefore know whether a candidate helped the currently active route, but it cannot see what happened to the other 11 routes unless it explicitly spends measurements on replay.

This creates the credit problem we care about:

> Can a locally separated architecture improve the currently active route without silently damaging routes that are not currently observable?

Every arm receives the same **15,360 local score probes** over 40 cycles.

## Arms

- `flat`: fixed-total-norm mutation across the whole 12×6 matrix; accept using only the active context.
- `flat_active_matched`: a deliberately strong attacker. The active branch gets the same mutation norm as a local arm, *and* the inactive block gets an additional equal total norm. It is allowed more total movement, but unseen routes can drift.
- `flat_replay`: flat mutation, but each candidate is checked on the active context plus three remembered contexts. Because probes are budget-matched, it can evaluate fewer candidates.
- `local`: mutate only the currently active/eligible branch.
- `history`: local mutation, with 65% of proposals bent toward that branch's own recent accepted deltas.
- `shuffled_history`: same as `history`, but use another branch's delta history when available.

No arm receives the hidden target directions.

## 16-seed result

| policy | mean final score | mean worst branch | mean cycles to 0.95 | mean cycles to 0.99 |
|---|---:|---:|---:|---:|
| flat | 0.85970 | 0.63203 | — | — |
| flat, active-step matched | 0.98148 | 0.95866 | 13.35 | — |
| flat + replay | 0.95039 | 0.88974 | 35.27* | — |
| local eligibility | 0.99644 | 0.99435 | 9.24 | 16.36 |
| **local + own delta highway** | **0.99964** | **0.99940** | **7.89** | **10.99** |
| local + wrong branch history | 0.99506 | 0.99147 | 10.08 | 20.23 |

`*` 14/16 replay seeds reached 0.95; the other two did not within the run. All local/history arms reached both thresholds in 16/16 seeds.

The branch-specific history arm beat every attacker in **16/16 matched seeds**, including plain local eligibility and shuffled branch history.

## What is actually learned here

The large gap between `flat` and `local` is **not** evidence that a neuron is a genetic algorithm. It demonstrates a narrower credit-assignment fact:

> Under bounded observation, changing only the currently eligible compartment protects unobserved routes from accidental drift.

That is already relevant to the biological analogy because dendritic branches are semi-independent computational subunits rather than one perfectly mixed parameter vector.

The smaller but consistent gap between `history` and `local` is the V25-specific result:

```text
local final score                0.99644
own-history final score          0.99964
wrong-branch-history final score 0.99506
```

Correct route ancestry reaches a mean all-branch score of 0.99 in **10.99 cycles**, versus **16.36** for local search without history and **20.23** when history is taken from the wrong branch.

So the successful deltas are not merely a global bias. Their **identity matters**.

That is the first executable version of the "delta highway" idea:

```text
branch A: ΔA1 -> ΔA2 -> ΔA3 -> next fan around A-history
branch B: ΔB1 -> ΔB2 -> ΔB3 -> next fan around B-history
...
```

Mixing those ancestries makes search worse.

## Why the strong flat attacker matters

`flat_active_matched` gives the visible branch an equally large local step and additionally perturbs all inactive branches. It therefore has *more*, not less, total movement than the local arm.

It still finishes at `0.98148`, versus `0.99644` for local separation and `0.99964` for local history.

That isolates the important effect: the problem is not only that a global mutation spreads its norm thinly. **Unobserved dimensions drift when they are allowed to change without being evaluated.**

Replay repairs some of this, but spends measurement budget protecting old routes. This connects directly to the older bounded-observation / replay problem: structural separation can provide protection that otherwise has to be purchased with extra observations.

## Claim boundary

This gate still has an artificial privilege: the active branch returns an immediate scalar score. There is no delayed downstream world yet, and there is no explicit biophysical dendrite/AIS/axon model.

So F2 earns only:

> **In a bounded-observation multi-route search, compartment-specific eligibility prevents unobserved route drift, and route-specific successful-delta history accelerates subsequent search more than pooled/wrong-route history under the same score-probe budget.**

## Next gate

F3 should remove the immediate branch score.

A local event should travel through a small recurrent downstream system, produce a delayed consequence, and return only a sparse scalar/modulatory signal plus a local eligibility trace:

```text
branch event
    ↓
AIS event
    ↓
axonal fan / downstream network
    ↓
delayed consequence
    ↓
global modulation × local eligibility
    ↓
retain or reject the branch delta
```

That is the point where the experiment becomes a candidate neuron-shaped learning rule rather than a compartmentalized optimizer.
