# Fan geometry restart — Gates F0–F1

The restart keeps the whole generational fan instead of collapsing each generation to one centroid.

The testbed is deliberately small and synthetic:

- 24 parallel `(1 + 8)` evolutionary lineages;
- three equally good rays in a 2-D search plane;
- every lineage independently fans eight fixed-norm mutations and retains only a local improvement;
- no lineage is assigned a branch and no diversity reward is supplied;
- the shared history contains only mutation steps that actually improved their parent.

The historical fan is discovered by cosine clustering of successful steps. The true branch directions are never supplied.

## Gate F0 — does past fan geometry predict the next successful fan?

Gate F0 is measured on the **ordinary isotropic search**. History does not affect the run.

Before generation `g`, the method clusters only successful steps from generations `< g`. It then asks how well those historical directions align with the successful lineage steps that actually occur at `g`.

32 seeds, 60 generations:

| prospective signal | mean alignment |
|---|---:|
| historical multi-direction fan | **0.90114** |
| rotated/shuffled historical fan | 0.82760 |
| one centroid-momentum direction | 0.15014 |

The historical fan beat the shuffled-history control in **32/32 seeds** and the single momentum direction in **32/32 seeds**.

This is the first result in the restarted V25 that directly supports the original intuition:

> **A generation can leave behind more than a winner. The geometry of successful alternatives can contain prospective information about the next successful alternatives.**

The result is not surprising in a stationary three-ray world; the important control is that the fan is learned only from previous accepted changes and scored prospectively.

## Gate F1 — can the fan shape the next search?

The same task was then run under five mutation policies. Every lineage receives the same number of candidate children, and **every child mutation has the same Euclidean norm** in every arm.

| policy | mean final score | branch coverage / 3 | mean least-populated branch |
|---|---:|---:|---:|
| isotropic | 13.22529 | 3.000 | 5.406 |
| centroid momentum | 11.87747 | 3.000 | 5.375 |
| current-population covariance | 13.25967 | 3.000 | 5.406 |
| **historical multi-direction fan** | **13.61372** | **3.000** | **5.406** |
| shuffled historical fan | 13.16867 | 3.000 | 5.406 |

Across the 32 matched seeds, historical fan guidance beat:

- isotropic search: **32/32** seeds;
- current-population covariance: **32/32** seeds;
- shuffled history: **32/32** seeds.

Mean score ratios:

```text
history fan / isotropic             1.02940
history fan / current covariance    1.02676
history fan / shuffled history      1.03381
```

The advantage over current covariance is only about **2.7%**. That modest margin is important: this is not evidence for a radically superior optimizer. It is evidence that, in this branching toy, preserving the *directional mixture of successful history* contains useful information that a one-vector momentum signal and current population covariance do not capture as well.

## Why the matrix matters

A single globally selected population rapidly collapses onto one branch in this task. The restarted experiment therefore uses a literal **matrix of local evolutionary searches**:

```text
lineage 1: parent -> fan -> local winner
lineage 2: parent -> fan -> local winner
...
lineage N: parent -> fan -> local winner
                 |
                 +-> shared history of successful directions
                              |
                              v
                     geometry of next fans
```

This is not a trick hidden after the result. It is the object the new V25 is trying to study: many local fan/retain cycles whose successful route geometry can be shared without averaging the routes into one global winner.

That is much closer to the neuron hypothesis than the first centroid/DMD draft.

## Neuron interpretation boundary

The experiment establishes only an artificial-search result.

It does **not** establish that a neuron performs genetic evolution.

The bridge to test later is organizational:

```text
many local route variants
        ↓
local consequences / competition
        ↓
persistent stabilization
        ↓
changed route population
        ↓
new fan of possible consequences
```

A neuron-shaped Gate F2/F3 would need to remove the explicit external fitness score and replace it with recurrent downstream consequences and local credit signals.

## Run

```bash
python fan_geometry.py --seeds 32 --generations 60 --output results/fan_receipt.json
python -m unittest discover -s tests -v
```

The summary and per-seed final receipts are in `results/fan_receipt.json`.
