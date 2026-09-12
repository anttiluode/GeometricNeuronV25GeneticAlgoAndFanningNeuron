# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** useful learning may live in *separated route histories of successful change*, not only in one global weight vector or one global gradient.

V25 began from a simple genetic-algorithm picture:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE THE NEXT FAN -> repeat
```

A neuron is not literally a genetic algorithm. But it is a branching, recurrent, plastic object with semi-independent dendritic routes, a sharp soma/AIS bottleneck, divergent axonal consequences, and local state that can outlive one event.

The repo asks whether the same organizational motif becomes useful again at that higher computational scale.

## 1. Keep the fan, not only the winner

The first V25 draft collapsed each generation to one centroid and tried to forecast the next centroid with a DMD-like model. That richer predictor did not beat strong simple baselines. The negative result is preserved in [`RESULTS.md`](RESULTS.md) and [`evolving_axon.py`](evolving_axon.py).

The restart keeps the population / route fan:

```text
P0 -> P1 -> P2 -> ...
```

and asks what useful structure survives across generations.

## 2. Gates F0–F1 — successful history forms a directional fan

[`fan_geometry.py`](fan_geometry.py) runs 24 parallel local evolutionary lineages. Each lineage fans eight fixed-norm mutations and keeps a local improvement. The true task branches are never supplied; the shared object is only the set of successful mutation directions.

On an unguided isotropic trajectory, the historical multi-direction fan predicts the next successful directions with mean cosine `0.90114`, versus `0.82760` for a rotated/shuffled fan and `0.15014` for one centroid-momentum vector. History beats both controls in **32/32 seeds**.

When used to orient future mutations under the same candidate count and mutation norm, the historical fan reaches `13.61372` mean final score versus `13.22529` isotropic, `13.25967` current covariance, `13.16867` rotated history, and `11.87747` centroid momentum. The gain over current covariance is only about **2.7%**: useful signal, not a new-general-optimizer claim.

See [`FAN_RESULTS.md`](FAN_RESULTS.md) and [`results/fan_receipt.json`](results/fan_receipt.json).

## 3. Gate F2 — bounded observation reveals why separation matters

[`nested_fan.py`](nested_fan.py) gives the system **12 separate branches**. Each branch has its own hidden useful direction, but on any generation only one branch/context is observable.

A global update can therefore improve the visible route while silently damaging the other 11.

| policy | mean final | worst branch | cycles to 0.99 |
|---|---:|---:|---:|
| flat global mutation | 0.85970 | 0.63203 | — |
| flat, active-step matched | 0.98148 | 0.95866 | — |
| flat + replay | 0.95039 | 0.88974 | — |
| local eligibility | 0.99644 | 0.99435 | 16.36 |
| **local + own delta highway** | **0.99964** | **0.99940** | **10.99** |
| local + wrong branch history | 0.99506 | 0.99147 | 20.23 |

The own-history arm beat every attacker in **16/16 matched seeds**.

F2 earns two separate statements:

> **Compartment-specific eligibility protects routes that are not currently observable.**

and

> **A branch's own successful-delta ancestry is more useful for its next fan than another branch's ancestry.**

See [`NESTED_FAN_RESULTS.md`](NESTED_FAN_RESULTS.md) and [`results/nested_fan_receipt.json`](results/nested_fan_receipt.json).

## 4. Gate F3 — delayed consequence × explicit causal tag

[`delayed_credit.py`](delayed_credit.py) removes immediate branch fitness. A branch proposes a fixed-norm delta; the hidden world returns only a noisy scalar consequence **4–20 events later** while other branches continue firing.

F3 still stores an exact causal tag:

```text
eligibility = (route identity, attempted delta, time)
```

| policy | mean final | worst branch | cycles to 0.99 |
|---|---:|---:|---:|
| immediate oracle | 0.99058 | 0.98316 | 185.50* |
| delayed + correct tag | 0.99055 | 0.98406 | 181.36* |
| **delayed tag + own delta history** | **0.99188** | **0.98623** | **168.25** |
| delayed + wrong tag | -0.86513 | -2.12278 | — |
| delayed + current branch | -0.55092 | -1.66520 | — |

Correct tagging is almost indistinguishable from immediate credit in mean final score. Destroying causal route identity destroys learning.

See [`DELAYED_CREDIT_RESULTS.md`](DELAYED_CREDIT_RESULTS.md) and [`results/delayed_credit_receipt.json`](results/delayed_credit_receipt.json).

## 5. Gate F4 — one unlabeled global consequence × local traces

F3's exact reward packet is too generous. [`mixed_credit.py`](mixed_credit.py) removes it.

Three branches perturb simultaneously. Their effects are compressed into **one global scalar consequence**, and a recurrent downstream state mixes that scalar with consequences from earlier events. No reward carries a branch ID.

Each branch may keep only its own decaying eligibility trace.

```text
several local perturbations
        ↓
one global scalar consequence
        ↓
recurrent temporal mixing
        ×
separate local eligibility traces
        ↓
plastic change
```

16 seeds × 6,000 events:

| policy | mean final | worst branch | reaches 0.75 |
|---|---:|---:|---:|
| local immediate oracle | 0.99315 | 0.99268 | 16/16 |
| immediate global scalar | 0.90019 | 0.89077 | 16/16 |
| **mixed scalar × local eligibility** | **0.81811** | **0.79239** | **16/16** |
| mixed scalar × current perturbation only | 0.64450 | 0.60952 | 0/16 |
| mixed scalar × shuffled eligibility | 0.06668 | -0.21892 | 0/16 |
| mixed scalar × pooled eligibility | 0.08556 | -0.16301 | 0/16 |

Local eligibility beats the current-only, shuffled, and pooled controls in **16/16 matched seeds**.

This gate is deliberately less perfect than F3. Without an explicit source tag, some credit information is genuinely lost. But a large amount remains recoverable if the branch preserves its own temporal state.

The earned statement is:

> **A global consequence can remain useful without an explicit source label when local branches preserve their own temporally extended eligibility. Destroying either the temporal trace or the branch identity sharply reduces learning in this toy.**

See [`MIXED_CREDIT_RESULTS.md`](MIXED_CREDIT_RESULTS.md) and [`results/mixed_credit_receipt.json`](results/mixed_credit_receipt.json).

## 6. Biological picture

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

V25 is now testing whether that separation has a learning function:

> **A dendritic compartment may be not only a nonlinear feature detector but also a credit-preserving compartment.**

That is the biological hypothesis earned by F2–F4. It is not yet a claim about a specific molecular pathway.

## 7. Next gate — learn the eligibility timescale

F4 fixes both downstream and eligibility decay by hand.

The next biological question is whether a branch can discover which memory timescale is useful. Give each route several local traces:

```text
fast eligibility     e_f
medium eligibility   e_m
slow eligibility     e_s
```

and let only their predictive success determine which trace controls plasticity.

If the downstream consequence changes its correlation time, the winning trace should change with it.

That would connect the old V24 fast/medium/slow idea to the new causal-route mechanism without importing the mathematics of GAx.

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
python mixed_credit.py --seeds 16 --steps 6000 --output results/mixed_credit_receipt_full.json
```

## Claim boundary

V25 does **not** claim that neurons literally run genetic algorithms, that the AIS is a fitness function, or that these toys are superior general-purpose optimizers.

The current earned statement is:

> **Separated local routes can protect unobserved knowledge, retain route-specific successful-change history, bridge delayed consequences with local causal state, and extract useful learning from a temporally mixed unlabeled global consequence when branch identity remains local.**
