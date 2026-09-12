# GeometricNeuronV25 — Genetic Algorithms and the Fanning Neuron

> **Working hypothesis:** learning may depend not only on which variants succeed, but on preserving the *separated route histories* of successful change.

V25 started from a simple observation about genetic algorithms:

```text
FAN OUT -> COMPETE -> RETAIN -> SHAPE THE NEXT FAN -> repeat
```

A neuron is not literally a genetic algorithm. But it is a branching, recurrent, plastic object with semi-independent dendritic routes, a sharp soma/AIS bottleneck, divergent axonal consequences, and persistent structural variables.

The repo asks whether the evolutionary motif becomes useful again at that higher computational scale.

## 1. Keep the fan, not only the winner

The first V25 draft collapsed each generation to one centroid and tried to forecast the next centroid with a DMD-like model. That was the wrong object. The richer predictor did not beat strong simple baselines.

The restart preserves a population / route fan:

```text
P0 -> P1 -> P2 -> ...
```

and asks whether successful directional structure in previous generations helps orient future search.

The old negative result is preserved in [`RESULTS.md`](RESULTS.md) and [`evolving_axon.py`](evolving_axon.py).

## 2. Gates F0–F1 — successful history forms a directional fan

[`fan_geometry.py`](fan_geometry.py) uses 24 parallel local evolutionary lineages. Each lineage fans eight fixed-norm mutations and keeps a local improvement. There is no global winner.

The true task branches are never supplied. The shared state is only the set of mutation steps that actually improved their parent.

### F0 — prospective fan geometry

On an unguided isotropic trajectory, previous successful directions predict the next successful directions:

```text
historical multi-direction fan     0.90114 mean cosine
rotated/shuffled fan               0.82760
single centroid momentum           0.15014

history > shuffled                32 / 32 seeds
history > momentum                32 / 32 seeds
```

### F1 — use the fan to shape future mutations

Under identical lineage count, child count and mutation norm:

| policy | mean final score |
|---|---:|
| isotropic | 13.22529 |
| centroid momentum | 11.87747 |
| current-population covariance | 13.25967 |
| **historical multi-direction fan** | **13.61372** |
| rotated/shuffled history | 13.16867 |

The historical fan beat isotropic, current covariance and shuffled history in **32/32 matched seeds**. The gain over current covariance is only about **2.7%**; this is evidence for useful directional history in this toy, not a claim of a new general optimizer.

See [`FAN_RESULTS.md`](FAN_RESULTS.md) and [`results/fan_receipt.json`](results/fan_receipt.json).

## 3. Gate F2 — the biological matrix under bounded observation

The Aizenbud/Leterrier connection suggested a stricter object:

```text
semi-independent local routes
        ↓
local nonlinear consequence
        ↓
selection / eligibility bottleneck
        ↓
route-specific retention
        ↓
next local fan
```

[`nested_fan.py`](nested_fan.py) therefore gives the system **12 separate branches**. Each branch has its own hidden useful direction, but on any generation only one branch/context is observable.

That creates a credit-assignment problem. A global update can improve the visible route while silently damaging routes that were not measured.

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

The strong `flat_active_matched` attacker matters: its active branch receives the same local step norm as the separated arm, and its inactive block is allowed an additional equal total mutation norm. It therefore gets *more total movement*. It still loses because unobserved routes drift.

So the large F2 result is not “GA history magically learns.” It is a credit-isolation result:

> **When observation is local and intermittent, compartment-specific eligibility prevents unobserved routes from being rewritten.**

The smaller V25-specific result is the delta highway:

```text
branch A: ΔA1 -> ΔA2 -> ΔA3 -> orient next A fan
branch B: ΔB1 -> ΔB2 -> ΔB3 -> orient next B fan
...
```

Correct branch ancestry reaches 0.99 in **10.99 cycles**, versus **16.36** for local isotropic search and **20.23** when history is borrowed from the wrong branch.

So route identity matters; the useful object is not one global bias.

See [`NESTED_FAN_RESULTS.md`](NESTED_FAN_RESULTS.md) and [`results/nested_fan_receipt.json`](results/nested_fan_receipt.json).

## 4. The neuron hypothesis has become more specific

The modern biological picture motivates this organization without proving the algorithm.

```text
dendritic fan
  many semi-independent nonlinear compartments
        ↓
soma / AIS
  adaptive global event gate and compartment boundary
        ↓
axonal fan
  one event produces many downstream consequences
        ↓
recurrent feedback / plasticity
        ↓
changed local route state
```

The important idea is **separation before recombination**.

A flat artificial neuron encourages everything to mix into one parameter vector. A real neuron repeatedly preserves distinctions: dendritic compartments remain partly independent, the AIS sharply separates somatodendritic and axonal identity, and the axon diverges again.

V25 asks whether that physical separation can solve a learning problem:

> protect route-specific history until there is evidence about that route again.

This connects the fanning idea to bounded observation and continual learning much more directly than the original “dendrite = selection / axon = mutation” cartoon.

## 5. Next gate — delayed consequence, not immediate fitness

F2 still has one major privilege: the currently active branch immediately returns a scalar score.

F3 should remove that.

```text
local branch event
      ↓
AIS event
      ↓
axonal fan / recurrent downstream system
      ↓
DELAY
      ↓
scalar consequence / modulation
      ×
local eligibility trace
      ↓
retain or reject the branch delta
```

The key question becomes:

> Can the separated delta highways still form when credit returns late, globally, and ambiguously?

If yes, the mechanism starts to look like a neuron-shaped learning rule rather than a compartmentalized optimizer.

## Interactive page

[Fan → Select → Retain → Fan](docs/index.html)

The page is a hypothesis visualizer, not biological evidence.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python fan_geometry.py --seeds 32 --generations 60 --output results/fan_receipt_full.json
python nested_fan.py --seeds 16 --cycles 40 --output results/nested_fan_receipt_full.json
```

## Claim boundary

V25 does **not** currently claim that:

- neurons literally run genetic algorithms;
- dendritic branches are GA individuals;
- the AIS is literally a fitness function;
- the repeating motif is established at every biological scale;
- these toys are superior general-purpose optimizers.

The current earned statement is:

> **Successful local changes can define separated directional histories. Under bounded observation, keeping change compartment-specific protects unobserved routes, and reusing the correct route's own successful-delta history accelerates later search under a matched measurement budget.**
