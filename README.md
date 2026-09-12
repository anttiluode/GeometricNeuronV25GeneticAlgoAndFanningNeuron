# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** useful learning may live in *separated route histories of successful change*, not only in one global weight vector or one global gradient.

V25 began from a simple genetic-algorithm picture:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE THE NEXT FAN -> repeat
```

A neuron is not literally a genetic algorithm. But it is a branching, recurrent, plastic object with semi-independent dendritic routes, a sharp soma/AIS bottleneck, divergent axonal consequences, and local state that can outlive one event.

The repo asks whether the same organizational motif becomes useful again at that higher computational scale.

## 1. Keep the fan, not only the winner

The first V25 draft collapsed each generation to one centroid and tried to forecast the next centroid with a DMD-like model. That richer predictor did not beat strong simple baselines.

That negative result is preserved in [`RESULTS.md`](RESULTS.md) and [`evolving_axon.py`](evolving_axon.py).

The restart keeps the population / route fan:

```text
P0 -> P1 -> P2 -> ...
```

and asks what useful structure survives across generations.

## 2. Gates F0–F1 — successful history forms a directional fan

[`fan_geometry.py`](fan_geometry.py) runs 24 parallel local evolutionary lineages. Each lineage fans eight fixed-norm mutations and keeps a local improvement. The true task branches are never supplied; the shared object is only the set of successful mutation directions.

### F0 — prospective fan geometry

On an unguided isotropic trajectory:

```text
historical multi-direction fan     0.90114 mean cosine
rotated/shuffled fan               0.82760
single centroid momentum           0.15014

history > shuffled                32 / 32 seeds
history > momentum                32 / 32 seeds
```

### F1 — use the fan to shape future mutations

| policy | mean final score |
|---|---:|
| isotropic | 13.22529 |
| centroid momentum | 11.87747 |
| current-population covariance | 13.25967 |
| **historical multi-direction fan** | **13.61372** |
| rotated/shuffled history | 13.16867 |

The historical fan beat isotropic, current covariance and shuffled history in **32/32 matched seeds**. The gain over current covariance is only about **2.7%**: useful signal, not a new-general-optimizer claim.

See [`FAN_RESULTS.md`](FAN_RESULTS.md) and [`results/fan_receipt.json`](results/fan_receipt.json).

## 3. Gate F2 — bounded observation reveals why separation matters

[`nested_fan.py`](nested_fan.py) gives the system **12 separate branches**. Each branch has its own hidden useful direction, but on any generation only one branch/context is observable.

A global update can therefore improve the visible route while silently damaging the other 11.

All arms receive the same **15,360 local score probes** over 40 cycles.

| policy | mean final | worst branch | cycles to 0.99 |
|---|---:|---:|---:|
| flat global mutation | 0.85970 | 0.63203 | — |
| flat, active-step matched | 0.98148 | 0.95866 | — |
| flat + replay | 0.95039 | 0.88974 | — |
| local eligibility | 0.99644 | 0.99435 | 16.36 |
| **local + own delta highway** | **0.99964** | **0.99940** | **10.99** |
| local + wrong branch history | 0.99506 | 0.99147 | 20.23 |

The own-history arm beat **every attacker in 16/16 matched seeds**.

The strong `flat_active_matched` arm gives the visible branch the same local step norm as the separated arm **and** gives the inactive block extra movement. It still loses because unobserved routes drift.

So F2 earns a credit-isolation result:

> **When observation is local and intermittent, compartment-specific eligibility protects routes that are not currently observable.**

And the smaller V25-specific result is the **delta highway**:

```text
branch A: ΔA1 -> ΔA2 -> ΔA3 -> orient next A fan
branch B: ΔB1 -> ΔB2 -> ΔB3 -> orient next B fan
...
```

Correct route ancestry is useful; borrowing history from another route is worse.

See [`NESTED_FAN_RESULTS.md`](NESTED_FAN_RESULTS.md) and [`results/nested_fan_receipt.json`](results/nested_fan_receipt.json).

## 4. Gate F3 — delayed consequence × local eligibility

F2 still gave the active branch an immediate scalar score. [`delayed_credit.py`](delayed_credit.py) removes that privilege from the learner.

A branch proposes one fixed-norm delta. The hidden world produces only a noisy scalar consequence, returned **4–20 events later** while other branches continue firing.

The key object is now:

```text
eligibility = (route identity, attempted delta, time)
```

When the delayed scalar returns, a persistent local tag can connect it back to the causal route.

| policy | mean final | worst branch | cycles to 0.99 |
|---|---:|---:|---:|
| immediate oracle | 0.99058 | 0.98316 | 185.50* |
| delayed + correct tag | 0.99055 | 0.98406 | 181.36* |
| **delayed tag + own delta history** | **0.99188** | **0.98623** | **168.25** |
| delayed + wrong tag | -0.86513 | -2.12278 | — |
| delayed + current branch | -0.55092 | -1.66520 | — |

`*` Immediate and tagged reached 0.99 in 14/16 seeds; history reached it in 16/16.

Correct delayed tagging was almost indistinguishable from immediate credit in mean final score:

```text
tagged - immediate = -0.0000296
```

The correct tag beat both mis-credit controls in **16/16 seeds**.

That is the first point where the fanning-neuron picture becomes a temporal learning mechanism rather than only a search geometry:

> **A delayed global consequence can still train a local route if causal route identity persists until the consequence returns.**

The weak route-history term then provides a second timescale: consolidated successful deltas shape later exploration on the same route.

See [`DELAYED_CREDIT_RESULTS.md`](DELAYED_CREDIT_RESULTS.md) and [`results/delayed_credit_receipt.json`](results/delayed_credit_receipt.json).

## 5. The biological picture we are testing

The modern neuron picture motivates this architecture without proving it:

```text
dendritic fan
  semi-independent nonlinear compartments
        ↓
soma / AIS
  global event gate + somatodendritic/axonal boundary
        ↓
axonal fan
  one event -> many downstream consequences
        ↓
delay / recurrence / modulation
        ↓
local eligibility + persistent route state
        ↓
changed future fan
```

The important idea is **separation before recombination**.

A flat artificial neuron encourages all parameters to mix. A biological neuron repeatedly preserves distinctions: dendritic compartments remain partly independent, the AIS maintains a sharp compartment boundary and controls spike initiation, and the axon diverges again.

V25 is testing whether that separation can solve a learning problem by preserving *who caused what* across both space and time.

## 6. Next gate — no reward packet may carry a source label

F3 still cheats in one way: the simulator stores an exact `(branch, delta)` packet until its consequence arrives.

F4 should remove that explicit queue mapping.

Several branch events should overlap inside a recurrent downstream state, producing one delayed scalar modulation. Each branch may keep only its own decaying local eligibility trace:

```text
many overlapping local events
        ↓
recurrent downstream state
        ↓
one delayed scalar consequence
        ×
independent local eligibility traces
        ↓
plasticity
```

The test becomes:

> **Can `global consequence × local eligibility` recover the correct separated highways when the reward itself contains no source identity?**

That is the next biological boundary.

## Interactive page

[Fan → Select → Retain → Fan](docs/index.html)

The page is a hypothesis visualizer, not biological evidence.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python fan_geometry.py --seeds 32 --generations 60 --output results/fan_receipt_full.json
python nested_fan.py --seeds 16 --cycles 40 --output results/nested_fan_receipt_full.json
python delayed_credit.py --seeds 16 --steps 2400 --output results/delayed_credit_receipt_full.json
```

## Claim boundary

V25 does **not** currently claim that neurons literally run genetic algorithms, that the AIS is a fitness function, or that these toys are superior general-purpose optimizers.

The current earned statement is:

> **Successful local changes can define separated directional histories. Compartment-specific eligibility protects unobserved routes, and persistent causal route identity can bridge delayed scalar consequences. Reusing a route's own consolidated delta history can then bias its future fan.**
