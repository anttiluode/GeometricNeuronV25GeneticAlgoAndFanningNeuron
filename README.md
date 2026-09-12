# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** useful learning may live in separated route histories and local causal state, not only in one global weight vector or one global gradient.

V25 began from a simple motif:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE THE NEXT FAN -> repeat
```

A neuron is not literally a genetic algorithm. This repo asks whether that organizational motif becomes useful again inside a branching, recurrent, plastic system with semi-independent dendritic routes, a soma/AIS bottleneck, divergent axonal consequences, and local state that can outlive one event.

The mathematical GA/operator branch lives elsewhere on purpose. V25 keeps testing the biological/computational question independently.

## The progression

| gate | question | result |
|---|---|---|
| F0–F1 | Does successful population history contain a useful multidirectional fan? | yes; prospective fan cosine `0.901`, modest search gain over covariance |
| F2 | Does branch separation help under bounded observation? | yes; local eligibility protects unseen routes and own delta history accelerates search |
| F3 | Can local route identity bridge delayed consequence? | yes when the explicit causal tag is retained |
| F4 | Can one unlabeled global scalar train separated local routes? | partly; local eligibility `0.818`, pooled/shuffled traces nearly fail |
| F5 | Does eligibility need the right lifetime? | yes; slower consequence requires slower local trace |
| F6 | Can a branch choose its own causal lifetime online? | yes; fast→slow trace use shifts after an unseen environment switch |
| F7 | Can temporal frequency be part of causal address? | yes on a supplied resonant bank; branch frequency recovered exactly in the synthetic on-grid world |
| F8 | Does that bank generalize to unseen off-grid frequencies? | **no**; selection/blending fails badly relative to exact-frequency oracle |
| F9 | Can the branch move its own resonant pole? | **partly yes**; local pole-fan closes much of the off-grid gap and reduces direct frequency error |

The negative gates stay. They are part of the object.

---

## F0–F1 — keep the fan, not only the winner

[`fan_geometry.py`](fan_geometry.py) keeps successful directions from 24 parallel local evolutionary lineages instead of collapsing a generation to one centroid.

```text
historical multi-direction fan     0.90114 prospective cosine
rotated/shuffled fan               0.82760
single centroid momentum           0.15014
```

When used to orient later search, the historical fan reaches `13.61372` versus `13.25967` for current covariance and `13.22529` for isotropic search. See [`FAN_RESULTS.md`](FAN_RESULTS.md).

## F2 — separation protects causal history

[`nested_fan.py`](nested_fan.py) gives the system 12 branches but lets it observe only one at a time.

```text
flat global                    0.85970
flat active-step matched       0.98148
local eligibility              0.99644
local + own delta highway      0.99964
local + wrong branch history   0.99506
```

Own branch history reaches the `0.99` threshold in `10.99` cycles versus `16.36` for local search and `20.23` with the wrong ancestry. See [`NESTED_FAN_RESULTS.md`](NESTED_FAN_RESULTS.md).

The first important architectural statement is:

> **separation can preserve the address of change when observation is bounded.**

## F3–F4 — delayed and mixed credit

F3 stores an exact delayed `(branch, delta)` eligibility tag. Correct delayed credit (`0.99055`) is essentially as good as immediate credit (`0.99058`); assigning the same consequence to the wrong route destroys learning. See [`DELAYED_CREDIT_RESULTS.md`](DELAYED_CREDIT_RESULTS.md).

F4 removes that source label. Three branches perturb simultaneously and their effects are compressed into one recurrently mixed scalar consequence.

| policy | mean final |
|---|---:|
| local oracle | 0.99315 |
| immediate global scalar | 0.90019 |
| **mixed scalar × local eligibility** | **0.81811** |
| current perturbation only | 0.64450 |
| shuffled eligibility | 0.06668 |
| pooled eligibility | 0.08556 |

See [`MIXED_CREDIT_RESULTS.md`](MIXED_CREDIT_RESULTS.md).

So memory is not enough. **Memory must remain attached to the route that produced it.**

## F5–F6 — the branch learns how long a cause survives

F5 sweeps local eligibility decay against downstream consequence decay. The useful local trace gets slower as downstream mixing gets slower. See [`TRACE_TIMESCALE_RESULTS.md`](TRACE_TIMESCALE_RESULTS.md).

F6 gives each branch four traces simultaneously:

```text
0.20   fast
0.70   medium
0.95   slow
0.98   very slow
```

The world secretly switches halfway from immediate consequence to strongly mixed consequence.

| trace | before switch | after switch |
|---:|---:|---:|
| 0.20 | **0.93380** | 0.08941 |
| 0.70 | 0.04733 | 0.01400 |
| 0.95 | 0.00763 | 0.08815 |
| 0.98 | 0.01125 | **0.80844** |

The adaptive arm reaches `0.87925`, beating every fixed-timescale control in `8/8` matched seeds. See [`ADAPTIVE_TIMESCALE_RESULTS.md`](ADAPTIVE_TIMESCALE_RESULTS.md).

At this point the causal address is already:

```text
WHERE?       branch identity
WHAT?        eligible local change
HOW LONG?    local causal lifetime
```

## F7 — add a temporal mode

F7 replaces purely decaying traces with damped resonant traces:

```math
z_{t+1}=\rho e^{i\omega}z_t+u_t.
```

Complex notation is only a compact implementation of a real 2-D damped rotation.

Each hidden branch uses one frequency from

```text
0, pi/12, pi/6, pi/3
```

and all branch consequences are mixed into one scalar modulation. Each branch receives the same four candidate resonant traces and locally selects the one that best predicts that global scalar.

| policy | mean final | frequency ID |
|---|---:|---:|
| **adaptive resonant selector** | **0.65490** | **1.000** |
| exact-frequency oracle | 0.71906 | 1.000 |
| best fixed channel | 0.34120 | 0.250 |
| shuffled branch↔frequency | 0.39009 | 0.281 |

See [`RESONANT_CREDIT_RESULTS.md`](RESONANT_CREDIT_RESULTS.md).

F7 earns only:

> **frequency can function as part of the address of causality when the useful temporal mode already exists in the local bank.**

The address becomes approximately `(branch, rho, omega)`.

## F8 — off-grid attacker: the fixed bank breaks

F7's true frequencies were preloaded. F8 deliberately moves them between the candidate channels:

```text
candidate bank:  0, pi/12, pi/6, pi/3
hidden world:    pi/24, pi/8, pi/4
```

[`offgrid_resonance.py`](offgrid_resonance.py) compares discrete selection, soft mixing, a privileged nearest-grid channel, and an exact off-grid oracle.

| policy | mean final | mean curve |
|---|---:|---:|
| F7 discrete selector | 0.29704 | 0.15452 |
| soft bank mixture | 0.22768 | 0.11764 |
| privileged nearest channel | 0.34488 | 0.18652 |
| **exact off-grid oracle** | **0.61058** | **0.35826** |

See [`OFFGRID_RESONANCE_RESULTS.md`](OFFGRID_RESONANCE_RESULTS.md).

This is an important negative result:

> **the F7 mechanism is a selector, not yet a self-organizing spectrum.**

A weighted mixture of neighboring resonators does not become the missing intermediate resonator; their phases eventually separate.

## F9 — move the pole itself

F9 changes the object instead of hiding the F8 failure.

Each branch starts at the same `omega = pi/6` and opens a three-way local fan:

```text
omega - delta
omega
omega + delta
```

Each candidate trace tries to predict the same global modulation. After an epoch the branch keeps the lowest-error temporal pole, shrinks `delta`, and fans again.

```text
FAN nearby poles
   ↓
COMPETE by local predictive error
   ↓
RETAIN winner
   ↓
SHRINK radius
   ↓
FAN again
```

No branch sees its hidden true frequency.

8 matched seeds × 6,000 events:

| policy | mean final | worst branch | mean curve |
|---|---:|---:|---:|
| F8 discrete selector | 0.41883 | 0.18243 | 0.22357 |
| privileged nearest grid channel | 0.46206 | 0.20583 | 0.25980 |
| **adaptive continuous pole fan** | **0.57950** | **0.28583** | **0.29426** |
| exact-frequency oracle | 0.73561 | 0.68632 | 0.46520 |

The adaptive pole fan beats both the discrete selector and the privileged nearest-grid channel in **8/8 matched seeds**.

Direct frequency error falls from

```text
initial mean |omega - omega*|   0.26180 rad
final mean   |omega - omega*|   0.07657 rad
```

See [`ADAPTIVE_RESONATOR_RESULTS.md`](ADAPTIVE_RESONATOR_RESULTS.md).

This is the first V25 gate where the **temporal dynamics that store causal history are themselves rewritten by consequence**.

And the original motif has recurred at another level:

```text
synaptic / parameter fan
    -> retain useful local deltas

timescale bank
    -> retain useful causal lifetime

temporal pole fan
    -> retain useful resonant dynamics
    -> fan again
```

## Current biological hypothesis

```text
dendritic fan
  semi-independent nonlinear / plastic compartments
        ↓
soma / AIS
  global event gate + compartment boundary
        ↓
axonal fan
  divergent downstream consequences
        ↓
delay / recurrence / modulation
        ↓
branch-local causal state
  route × lifetime × temporal mode
        ↓
consequence selects / moves local causal dynamics
        ↓
future plasticity and future fan change
```

This is compatible with thinking about dendritic compartmentalization as a way to preserve causal identity and about multiple cellular timescales as a substrate for temporal credit. It does **not** establish that a real dendrite, AIS, axon, oscillation, ion channel, or neuromodulator implements these exact synthetic rules.

## Why the spectral-router synthesis is interesting but still separate

The independent mathematical arm is studying context-dependent modal gain and evolving operators. V25 has now independently reached a lower-level object:

```text
local route × local temporal mode × consequence-driven mode adaptation
```

That is enough contact for comparison, not enough for unification.

The useful sentence for V25 is:

> **The reward carries no detailed address; the substrate keeps the address. The substrate can now also move the temporal mode in which that address persists.**

## Next attackers

F9 still has one stationary hidden mode per branch and regular epoch boundaries.

The strongest next tests are:

```text
1. continuously drift each hidden frequency;
2. place TWO temporal modes on one branch and require both to survive;
3. remove explicit pole-selection epochs;
4. compare with an equally sized generic two-state linear dynamical trace;
5. use a broadband/no-mode world and require the system not to invent a fake stable resonance.
```

The two-mode branch is especially important. A real spectral routing mechanism should not always purify to one winner; sometimes **separation must preserve several futures at once**.

## Interactive page

[Static V25 hypothesis page](docs/index.html)

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
python offgrid_resonance.py --seeds 8 --steps 4000
python adaptive_resonator.py --seeds 8 --steps 6000
```

## Claim boundary

V25 does **not** claim that neurons literally run genetic algorithms, that the AIS is a fitness function, that biological cells run normalized LMS, or that these toys are a new general-purpose optimizer.

The current earned statement is narrower:

> **Separated local routes can preserve causal identity under mixed delayed consequences. Local causal state can adapt its useful lifetime, resonant modes can provide an additional temporal address, a coarse fixed mode bank fails off-grid, and a local generate–compete–retain search over the resonant pole can partially recover unseen branch-specific temporal modes without receiving their true frequencies.**
