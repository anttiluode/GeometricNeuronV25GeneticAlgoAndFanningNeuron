# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Restarted premise:** evolution is not only a sequence of winners. Each generation is a *fan* of alternatives that is compressed by selection and then expanded again. A neuron has the same gross geometry — fan-in, bottleneck, fan-out — and its plastic structure can make that cycle persistent across time.

```text
FAN OUT
   ↓
LOCAL CONSEQUENCES / COMPETITION
   ↓
RETAIN
   ↓
SUCCESSFUL ROUTE GEOMETRY SHAPES THE NEXT FAN
   ↓
repeat
```

The hypothesis is **not** that a neuron literally runs a genetic algorithm or mutates DNA during ordinary computation.

The hypothesis is that the useful geometry of evolution — **variation → competition → retention → biased future variation** — may recur at a higher level of neural processing because a neuron is already a recurrent branching system whose route strengths can persistently change.

## The object is a fan, not a winner

At evolutionary scale, keep the whole population matrix:

```text
P_0 -> P_1 -> P_2 -> ...
```

Do not immediately collapse `P_g` to one centroid.

At neuronal scale the structural analogy is:

```text
many dendritic routes
      ↓
local nonlinear competition + soma / AIS bottleneck
      ↓
one or a few output events
      ↓
large axonal fan-out
      ↓
different downstream consequences
      ↓
feedback / plasticity changes which routes matter next time
```

The testable motif is therefore:

```text
FAN-IN -> BOTTLENECK -> FAN-OUT -> WORLD -> FEEDBACK -> CHANGED FAN
```

## The literal matrix-of-GAs experiment

The restarted experiment uses **24 parallel local evolutionary lineages**. Each lineage is a tiny `(1 + 8)` search:

```text
lineage 1: parent -> 8 mutations -> keep local winner
lineage 2: parent -> 8 mutations -> keep local winner
...
lineage 24
```

There is no global winner. This is deliberate: a globally selected population rapidly collapses onto one branch and destroys the object we want to study.

Across the lineages we retain only the mutation directions that actually improved their parent. Those successful directions form a historical fan. The true task branches are never supplied to the algorithm.

The shared object is therefore:

```text
successful local step 1
successful local step 2
successful local step 3
...
        ↓
cluster directional history
        ↓
several persistent fan directions
        ↓
orient part of the next mutations
```

Every mutation is norm-matched across all controls.

## Gate F0 — does the old fan predict the next successful fan?

This gate is measured on an **unguided isotropic run**. The history cannot affect the trajectory being scored.

Before generation `g`, V25 clusters only successful mutation directions from generations `< g`, then measures alignment to the successful steps that actually occur at `g`.

32 seeds, 60 generations:

```text
historical multi-direction fan     0.90114 mean cosine
shuffled/rotated history           0.82760
single centroid momentum           0.15014

history > shuffled                32 / 32 seeds
history > momentum                32 / 32 seeds
```

So the first clean result of the restart is:

> **A generation can leave behind more than a winner. The geometry of successful alternatives can contain prospective information about the next successful alternatives.**

This is a stationary three-ray toy, so persistence of useful directions is expected. The important controls are that the directions are learned only from previous accepted changes, are scored prospectively, and are compared with geometry-preserving shuffled history.

## Gate F1 — can that geometry shape future search?

Five policies receive identical lineage count, children per lineage and mutation norm:

| policy | mean final score | coverage / 3 |
|---|---:|---:|
| isotropic | 13.22529 | 3.000 |
| centroid momentum | 11.87747 | 3.000 |
| current-population covariance | 13.25967 | 3.000 |
| **historical multi-direction fan** | **13.61372** | **3.000** |
| shuffled historical fan | 13.16867 | 3.000 |

The historical fan beat isotropic, current covariance and shuffled history in **32/32 matched seeds**.

```text
history fan / isotropic             1.02940
history fan / current covariance    1.02676
history fan / shuffled history      1.03381
```

The advantage over current covariance is only about **2.7%**. That is deliberately not sold as a new optimizer. The result is narrower:

> **In this branching toy, retaining the directional mixture of successful history is more useful than collapsing it to one momentum vector, using only present population covariance, or destroying its orientation.**

See [`FAN_RESULTS.md`](FAN_RESULTS.md), [`fan_geometry.py`](fan_geometry.py), and [`results/fan_receipt.json`](results/fan_receipt.json).

## Why this is different from the first V25 draft

The old experiment reduced every GA generation to an elite-centroid operator and tried to predict its next position with a DMD-like model. That richer predictor did not beat persistence; simple last-step guidance was more useful.

Those negative results remain in [`RESULTS.md`](RESULTS.md) and [`evolving_axon.py`](evolving_axon.py).

The restart changes the object:

```text
old:     W_0 -> W_1 -> W_2                one point per generation

new:     P_0 -> P_1 -> P_2                whole population / route fan
              \  |  /
               successful directional mixture
```

Collapsing to one point can erase the very branching geometry we care about.

## Why the neuron belongs here

The biological mapping remains a hypothesis, not a result.

- **Dendritic fan-in:** many synapses and compartments provide many routes into the cell.
- **Bottleneck:** local dendritic nonlinearities and soma/AIS reduce that fan to a much smaller output event set.
- **Axonal fan-out:** a spike train reaches many branches and terminals with heterogeneous consequences.
- **Retention:** synaptic efficacy, spine structure, release properties, excitability and slower structural variables can change with activity and feedback.

The interesting possibility is not `dendrite = selection` and `axon = mutation`.

It is that the **whole recurrent cycle** may repeatedly instantiate:

```text
GENERATE ROUTE VARIANTS
        ↓
DIFFERENT CONSEQUENCES
        ↓
DIFFERENTIAL STABILIZATION
        ↓
CHANGED ROUTE POPULATION
        ↓
NEW FAN
```

That is the higher-level repetition of the evolutionary motif we want to test.

## Next gate — remove the external fitness oracle

F0–F1 still use an explicit artificial score. That is the major privilege.

Gate F2 should replace it with a recurrent world:

```text
local route fan
     ↓
downstream system
     ↓
delayed consequence
     ↓
local return signal
     ↓
stabilize / weaken route
```

The hard question becomes whether the matrix can recover useful fan geometry when no component is handed a global fitness value and credit is delayed and mixed.

Only after that survives should V25 claim a neuron-shaped learning mechanism.

## Interactive page

The static page visualizes the repeated motif:

**[Fan → Select → Retain → Fan](docs/index.html)**

It is a hypothesis visualizer, not biological evidence.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python fan_geometry.py --seeds 32 --generations 60 --output results/fan_receipt_full.json
```

## Claim boundary

Established fields already include genetic algorithms, evolution strategies, covariance adaptation, island models, structural synaptic plasticity, activity-dependent stabilization, axonal branching, synaptic competition and recurrent learning.

This repo does **not** currently claim:

- that neurons literally run genetic algorithms;
- that fan-in equals selection or fan-out equals mutation one-to-one;
- that the same mechanism necessarily repeats at every biological scale;
- that the branching toy is a new general optimizer;
- that genes alone explain neural learning.

The current earned statement is smaller:

> **Generate–compete–retain–regenerate can be implemented as a matrix of local evolutionary fans, and in a controlled branching toy the directional geometry of previously successful fans contains prospective information and modestly improves the next search under matched budgets.**
