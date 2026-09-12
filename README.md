# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** useful learning may live in separated route histories and local causal state, not only in one global weight vector or one global gradient.

V25 began from a genetic-algorithm picture:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE NEXT FAN -> repeat
```

A neuron is not literally a GA. The biological question is whether the same organizational motif becomes useful again because a neuron is a branching, recurrent, plastic object with semi-independent dendritic routes, an AIS bottleneck, divergent axonal consequences, and local state that can outlive one event.

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

F3 still stores the exact causal tag:

```text
eligibility = (route, attempted delta, time)
```

Correct delayed tagging reaches `0.99055`, essentially the same as immediate credit (`0.99058`). Wrong-route credit collapses learning. Adding the route's own delta history reaches `0.99188` and reduces time to 0.99.

See [`DELAYED_CREDIT_RESULTS.md`](DELAYED_CREDIT_RESULTS.md).

## F4 — one unlabeled global consequence × local eligibility

F3's reward packet is too generous. [`mixed_credit.py`](mixed_credit.py) removes the source label.

Three branches perturb simultaneously. Their effects are compressed into **one scalar consequence** and recurrently mixed with earlier consequences. No reward carries a branch ID. Each branch may keep only its own decaying eligibility trace.

| policy | mean final | worst branch | reaches 0.75 |
|---|---:|---:|---:|
| local immediate oracle | 0.99315 | 0.99268 | 16/16 |
| immediate global scalar | 0.90019 | 0.89077 | 16/16 |
| **mixed scalar × local eligibility** | **0.81811** | **0.79239** | **16/16** |
| current perturbation only | 0.64450 | 0.60952 | 0/16 |
| shuffled eligibility | 0.06668 | -0.21892 | 0/16 |
| pooled eligibility | 0.08556 | -0.16301 | 0/16 |

Local eligibility beats current-only, shuffled, and pooled controls in **16/16 matched seeds**.

This is deliberately less perfect than F3. Without an explicit source tag, information is genuinely lost. But a large amount remains recoverable if branch identity and temporal state remain local.

See [`MIXED_CREDIT_RESULTS.md`](MIXED_CREDIT_RESULTS.md).

## F5 — eligibility itself has a timescale

F4 fixed the local memory constant by hand. [`trace_timescales.py`](trace_timescales.py) sweeps eligibility decay against the timescale of downstream consequence mixing.

8 seeds × 4,000 events per cell:

| downstream decay | best eligibility decay | best score | fixed 0.70 trace | gain |
|---:|---:|---:|---:|---:|
| 0.20 | 0.85 | 0.83244 | 0.82826 | +0.00418 |
| 0.50 | 0.95 | 0.81643 | 0.78961 | +0.02681 |
| 0.70 | 0.95 | 0.80460 | 0.72357 | +0.08103 |
| 0.90 | 0.98 | 0.75274 | 0.47707 | +0.27566 |

The best local trace slows monotonically as the downstream consequence becomes slower.

> **Slow consequences require longer-lived local causal state. A fixed medium trace increasingly loses credit as temporal mixing becomes slower.**

See [`TRACE_TIMESCALE_RESULTS.md`](TRACE_TIMESCALE_RESULTS.md).

## F6 — each branch learns how long to remember

F5 was only a map: the experimenter chose the best trace after the run. [`adaptive_timescale.py`](adaptive_timescale.py) gives every branch all four traces simultaneously and lets experience decide which one controls plasticity.

Each branch maintains a tiny predictor for each local trace. The predictor sees only that trace and the observed **global scalar modulation**. The branch trusts the trace with the lowest recent prediction error.

The world switches halfway through the run without announcing it:

```text
first half:   downstream decay = 0.00
second half:  downstream decay = 0.90
```

Mean trace use across 8 seeds:

| trace decay | before switch | after switch |
|---:|---:|---:|
| 0.20 | **0.93380** | 0.08941 |
| 0.70 | 0.04733 | 0.01400 |
| 0.95 | 0.00763 | 0.08815 |
| 0.98 | 0.01125 | **0.80844** |

The same architecture therefore moves from fast causal memory to slow causal memory without receiving the regime label or switch time.

It also improves learning:

| policy | score at switch | final score | curve area |
|---|---:|---:|---:|
| fixed 0.20 | 0.78463 | 0.80984 | 0.65815 |
| fixed 0.70 | 0.78418 | 0.83336 | 0.66398 |
| fixed 0.95 | 0.75893 | 0.86595 | 0.65694 |
| fixed 0.98 | 0.70940 | 0.85432 | 0.62218 |
| **adaptive local selector** | **0.78810** | **0.87925** | **0.68435** |

The adaptive arm beats every fixed-timescale control in **8/8 matched seeds** on final score.

So V25 now separates three pieces of causal learning:

```text
WHERE did the cause occur?        -> branch identity
WHAT change remains creditable?   -> local eligibility state
HOW LONG should it remain alive?  -> locally selected timescale
```

See [`ADAPTIVE_TIMESCALE_RESULTS.md`](ADAPTIVE_TIMESCALE_RESULTS.md).

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
local eligibility at several timescales
        ↓
local evidence selects useful causal lifetime
        ↓
changed local route + changed next fan
```

The central architectural idea is **separation before recombination**.

A dendritic compartment may therefore be more than a nonlinear feature detector. In the V25 hypothesis it can also be a **credit-preserving compartment**: local causal state survives after global events have mixed information.

F6 makes the old fast/medium/slow idea operational rather than decorative. The timescales now have a job in temporal credit assignment, and a small meta-state decides which lifetime currently deserves influence.

This is compatible with the modern picture of dendritic compartmentalization and an adaptive AIS, but none of the synthetic gates proves that a specific dendritic, AIS, axonal, or neuromodulatory mechanism implements this exact algorithm.

## Next gate — fan over the memories themselves

F6 still receives a hand-designed bank of four lifetimes.

The next recursion is natural:

```text
start with a few causal lifetimes
        ↓
persistent prediction mismatch
        ↓
FAN OUT new candidate lifetimes
        ↓
COMPETE on experienced consequence
        ↓
RETAIN useful lifetime / prune redundant lifetime
        ↓
repeat
```

That would bring the original V25 generate–compete–retain motif back **inside the learning rule itself**. The neuron-shaped system would not only select a memory timescale; it would restructure the set of timescales available to it.

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
```

## Claim boundary

V25 does **not** claim that neurons literally run genetic algorithms, that the AIS is a fitness function, that biological cells run normalized LMS, or that these toys are superior general-purpose optimizers.

The earned result so far is narrower:

> **Separated local routes can protect unobserved knowledge, preserve route-specific successful-change history, bridge mixed delayed consequences with local causal state, and adapt the lifetime of that causal state when the consequence dynamics change.**
