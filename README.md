# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** useful learning may live in separated route histories and local causal state, not only in one global weight vector or one global gradient.

V25 began from a genetic-algorithm picture:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE THE NEXT FAN -> repeat
```

A neuron is not literally a GA. The biological question is whether the same organizational motif becomes useful again because a neuron is a branching, recurrent, plastic object with semi-independent dendritic routes, an AIS bottleneck, divergent axonal consequences, and local state that can outlive one event.

The project has now moved through seven gates.

## F0–F1 — keep the fan, not only the winner

The first draft collapsed each generation to one centroid and tried to predict the next centroid. That failed against strong simple baselines and remains in [`RESULTS.md`](RESULTS.md).

The restart keeps the whole route fan. [`fan_geometry.py`](fan_geometry.py) runs 24 parallel local evolutionary lineages and retains successful mutation directions.

```text
historical fan predicts next successful directions: 0.90114 cosine
rotated/shuffled fan:                         0.82760
single centroid momentum:                     0.15014
```

History beats both controls in **32/32 seeds**. When used to orient future mutations under matched budgets, the historical fan reaches `13.61372` versus `13.25967` for current covariance and `13.22529` isotropic. The advantage over current covariance is modest (~2.7%).

See [`FAN_RESULTS.md`](FAN_RESULTS.md).

## F2 — bounded observation makes separation useful

[`nested_fan.py`](nested_fan.py) gives the system 12 separate branches. Only one branch is observable at a time, so global updates can silently damage unseen routes.

| policy | mean final | worst branch | cycles to 0.99 |
|---|---:|---:|---:|
| flat global | 0.85970 | 0.63203 | — |
| flat, active-step matched | 0.98148 | 0.95866 | — |
| local eligibility | 0.99644 | 0.99435 | 16.36 |
| **local + own delta highway** | **0.99964** | **0.99940** | **10.99** |
| local + wrong branch history | 0.99506 | 0.99147 | 20.23 |

Correct branch ancestry beats every attacker in **16/16 seeds**.

F2 earns two ideas:

```text
separation protects unobserved routes
route-specific successful deltas can shape that route's next fan
```

See [`NESTED_FAN_RESULTS.md`](NESTED_FAN_RESULTS.md).

## F3 — delayed consequence with explicit causal tag

[`delayed_credit.py`](delayed_credit.py) removes immediate branch fitness. A noisy scalar consequence returns 4–20 events later while other branches continue firing.

Correct delayed tagging reaches `0.99055`, essentially the same as immediate credit (`0.99058`). Wrong-route credit collapses learning. Adding the route's own delta history reaches `0.99188` and reduces time to 0.99.

See [`DELAYED_CREDIT_RESULTS.md`](DELAYED_CREDIT_RESULTS.md).

## F4 — one unlabeled global consequence × local eligibility

[`mixed_credit.py`](mixed_credit.py) removes the source label. Three branches perturb simultaneously. Their effects are compressed into **one scalar consequence** and recurrently mixed with earlier consequences.

| policy | mean final | worst branch |
|---|---:|---:|
| local immediate oracle | 0.99315 | 0.99268 |
| immediate global scalar | 0.90019 | 0.89077 |
| **mixed scalar × local eligibility** | **0.81811** | **0.79239** |
| current perturbation only | 0.64450 | 0.60952 |
| shuffled eligibility | 0.06668 | -0.21892 |
| pooled eligibility | 0.08556 | -0.16301 |

Local eligibility beats current-only, shuffled, and pooled controls in **16/16 matched seeds**.

The useful state is not merely memory. It is memory that remains attached to the route that caused it.

See [`MIXED_CREDIT_RESULTS.md`](MIXED_CREDIT_RESULTS.md).

## F5 — eligibility itself has a timescale

[`trace_timescales.py`](trace_timescales.py) sweeps local trace decay against downstream consequence mixing.

| downstream decay | best eligibility decay | best score | fixed 0.70 |
|---:|---:|---:|---:|
| 0.20 | 0.85 | 0.83244 | 0.82826 |
| 0.50 | 0.95 | 0.81643 | 0.78961 |
| 0.70 | 0.95 | 0.80460 | 0.72357 |
| 0.90 | 0.98 | 0.75274 | 0.47707 |

> **Slow consequences require longer-lived local causal state.**

See [`TRACE_TIMESCALE_RESULTS.md`](TRACE_TIMESCALE_RESULTS.md).

## F6 — each branch learns how long to remember

[`adaptive_timescale.py`](adaptive_timescale.py) gives every branch a simultaneous bank of fast/medium/slow/very-slow traces. Each branch keeps a tiny predictor for each trace and trusts the trace whose local state best predicts the observed global modulation.

The world switches halfway through from immediate to strongly mixed consequence:

```text
first half:   downstream decay = 0.00
second half:  downstream decay = 0.90
```

Trace use across 8 seeds:

| trace decay | before switch | after switch |
|---:|---:|---:|
| 0.20 | **0.93380** | 0.08941 |
| 0.70 | 0.04733 | 0.01400 |
| 0.95 | 0.00763 | 0.08815 |
| 0.98 | 0.01125 | **0.80844** |

The adaptive arm reaches `0.87925`, beating every fixed-timescale control in **8/8 matched seeds**.

So V25 now separates:

```text
WHERE did the cause occur?        -> branch identity
WHAT change remains creditable?   -> local eligibility state
HOW LONG should it remain alive?  -> locally selected timescale
```

See [`ADAPTIVE_TIMESCALE_RESULTS.md`](ADAPTIVE_TIMESCALE_RESULTS.md).

## F7 — frequency can become part of the causal address

The third-arm spectral-router synthesis suggested a clean biological test without importing GAx itself: replace purely decaying traces with damped resonant traces,

```math
z_{t+1}=\rho e^{i\omega}z_t+u_t.
```

[`resonant_credit.py`](resonant_credit.py) gives each hidden branch one temporal channel from

```text
0, pi/12, pi/6, pi/3
```

and mixes all branch consequences into one global scalar modulation. Every branch receives the same four candidate resonant eligibility channels and must infer which temporal mode belongs to it from the global modulation alone.

8 seeds × 5,000 events:

| policy | mean final | worst branch | mean curve | freq ID |
|---|---:|---:|---:|---:|
| **adaptive resonant selector** | **0.65490** | **0.51437** | **0.37821** | **1.000** |
| oracle true frequency | 0.71906 | 0.65020 | 0.44946 | 1.000 |
| best fixed channel | 0.34120 | 0.08167 | 0.18562 | 0.250 |
| shuffled branch↔frequency | 0.39009 | 0.23339 | 0.20160 | 0.281 |

The adaptive selector beats every fixed channel and the shuffled assignment in **8/8 matched seeds**.

The synthetic task deliberately places each hidden frequency exactly on the candidate bank, so this does **not** prove arbitrary frequency discovery. What it earns is narrower:

> **one unlabeled global scalar can still carry enough correlation structure for separated local traces to recover which temporal channel belongs to which branch.**

That extends the causal address to:

```text
WHERE?        branch identity
WHAT?         eligible local change
HOW LONG?     decay / persistence
WHICH MODE?   local temporal frequency
```

or compactly,

```math
(branch,\rho,\omega).
```

See [`RESONANT_CREDIT_RESULTS.md`](RESONANT_CREDIT_RESULTS.md).

## Biological picture

```text
dendritic fan
  semi-independent nonlinear compartments
        ↓
soma / AIS
  global event gate + compartment boundary
        ↓
axonal fan
  many downstream consequences
        ↓
delay / recurrence / modulation
        ↓
local eligibility across several lifetimes and temporal modes
        ↓
local evidence selects useful causal channel
        ↓
changed local route + changed next fan
```

The central architectural idea remains **separation before recombination**.

A dendritic compartment may therefore be more than a nonlinear feature detector. In the V25 hypothesis it can also be a **credit-preserving compartment**: local causal state survives after global events have mixed information.

F7 does not claim that dendrites literally run complex arithmetic or Fourier transforms. A complex trace is only a compact representation of a real two-state damped rotation. The biological question is whether local processes with distinct persistence / phase dynamics can serve as causal addresses.

## Next attackers

F7 currently gives the learner the correct hidden frequencies in its candidate bank. That is the privilege to break next.

```text
1. hidden frequencies BETWEEN candidate channels
2. branch frequencies that drift over time
3. two simultaneous temporal modes on one branch
4. aperiodic / broadband consequence where no resonant channel should win
5. compare resonant channels with an equally sized generic linear state-space bank
```

If V25 only succeeds when the answer already exists as one discrete channel, F7 is a selector. If the local temporal basis can interpolate, split, or reorganize, the mechanism becomes much closer to a self-organizing temporal router.

## Interactive page

[Fan → Select → Retain → Fan](docs/index.html)

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python fan_geometry.py --seeds 32 --generations 60
python nested_fan.py --seeds 16 --cycles 40
python delayed_credit.py --seeds 16 --steps 2400
python mixed_credit.py --seeds 16 --steps 6000
python trace_timescales.py --seeds 8 --steps 4000
python adaptive_timescale.py --seeds 8 --steps 6000
python resonant_credit.py --seeds 8 --steps 5000
```

## Claim boundary

V25 does **not** claim that neurons literally run genetic algorithms, that the AIS is a fitness function, that biological cells run normalized LMS, or that these toys are superior general-purpose optimizers.

The earned result so far is narrower:

> **Separated local routes can protect unobserved knowledge, preserve route-specific successful-change history, bridge mixed delayed consequences with local causal state, adapt the lifetime of that causal state when consequence dynamics change, and use distinct local temporal modes as an additional address for credit in a controlled resonant toy.**
