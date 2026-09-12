# GeometricNeuronV25 — The Evolving Axon

> **Restarted premise:** evolution is not only a sequence of winners. Each generation is a *fan* of alternatives that is compressed by selection and then expanded again. A neuron has the same gross geometry — fan-in, bottleneck, fan-out — and its plastic structure can make that cycle persistent across time.

This repo now asks a simpler question than the first draft:

```text
GENERATE A FAN
      ↓
TEST / COMPETE
      ↓
RETAIN A SMALLER SET
      ↓
USE THE SURVIVORS TO SHAPE THE NEXT FAN
      ↓
repeat
```

The hypothesis is **not** that a neuron literally runs a genetic algorithm or mutates DNA during ordinary computation.

The hypothesis is that the useful geometry of evolution — **variation → competition → retention → biased future variation** — may recur at a higher level of neural processing because a neuron already has branching populations of routes, a strong bottleneck, activity-dependent stabilization, and repeated interaction with a recurrent world.

## The repeating motif

At the genetic / evolutionary scale, one generation can be written as a population matrix

```text
P_g = [candidate_1
       candidate_2
       ...
       candidate_N]
```

Selection contracts that population. Mutation and recombination expand it again.

```text
population fan-out
      ↓
fitness / selection
      ↓
surviving basin
      ↓
new population fan-out
```

A neuron has a strikingly similar *geometry*:

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

That is an analogy at the level of **algorithmic organization**, not a claim of literal Darwinian evolution inside every spike.

## The core V25 question

Do **not** collapse each generation immediately to one centroid.

Keep the whole fan:

```text
P_0 -> P_1 -> P_2 -> ...
```

and ask whether the geometry of successful past fans tells us how to orient the next one.

A minimal version is

```text
mu_g       = selected center
D_g        = successful directions / subspace
Sigma_g    = surviving spread

candidate_(g+1) = mu_g + D_g a + L_g epsilon
```

where `L_g L_g^T = Sigma_g` and `a` samples directions that were useful in previous generations.

The strongest possible result would be:

> **Past successful population geometry compiles the search geometry of the next generation.**

That would connect directly to the older repo line:

> **structure compiles an operator**

and add:

> **successful operator populations compile the space in which the next operator is searched.**

## Why the neuron belongs here

The biological mapping is intentionally cautious.

- **Dendritic fan-in:** many synapses and dendritic compartments provide a population of routes into the cell.
- **Bottleneck:** local dendritic nonlinearities and the soma/AIS turn that fan into a much smaller set of output events.
- **Axonal fan-out:** one spike train reaches many branches and terminals with heterogeneous downstream effects.
- **Retention:** synaptic strengths, spine structure, release properties, excitability and slower structural variables can be changed by activity and feedback.

So the neuron can be viewed as a recurrent fan system:

```text
FAN-IN -> BOTTLENECK -> FAN-OUT -> WORLD -> FEEDBACK -> CHANGED FAN
```

The important V25 question is whether **learning is partly the changing geometry of that fan**.

## Interactive restart

Open the static experiment page:

**[Fan → Select → Retain → Fan](docs/index.html)**

The page shows the same motif at two scales:

1. a population that fans out, is selected, and uses the survivors to orient the next fan;
2. a neuron-scale abstraction with dendritic fan-in, an AIS bottleneck, axonal fan-out, and feedback-dependent route stabilization.

It is a visual hypothesis generator, not biological evidence.

## What happened to the first V25 experiments?

They stay in the repository as a useful failed detour.

The original version reduced each generation to an elite-centroid operator and tried to forecast the next operator with a DMD-like temporal model. That richer predictor did **not** beat a strong persistence baseline. A much simpler last-generation displacement was useful as mutation guidance in the toy problem.

Those results are retained in [`RESULTS.md`](RESULTS.md), [`evolving_axon.py`](evolving_axon.py), and `results/`.

They now have a narrower interpretation:

> collapsing a whole generation to one moving point may throw away the very population geometry that motivated the idea.

The restart therefore treats **the fan itself as the state**.

## New experimental gates

### Gate F0 — does population shape predict useful search directions?

Evolve a matrix of linear operators. Preserve the full selected population at every generation. Estimate the successful low-dimensional subspace from *between-generation survivor geometry* and test it prospectively on the next generation.

Attackers:

- isotropic mutation;
- simple centroid momentum;
- covariance from the current generation only;
- shuffled generation history.

A pass requires history-conditioned fan geometry to improve future search under the **same candidate and mutation-norm budget**.

### Gate F1 — fan branching, not just covariance

Construct multimodal tasks where two or more distinct successful directions coexist. The mechanism should keep several branches alive instead of collapsing them to one averaged direction.

This is important for the neuron analogy: dendrites and axons are branched, not one principal component.

### Gate F2 — recurrent feedback selects routes

Replace the external static fitness function with downstream consequences that return through a recurrent network. Ask whether route stabilization still emerges when no global optimizer is handed the answer directly.

### Gate F3 — neuron-shaped implementation

Only after F0–F2 survive controls, map the mechanism onto an explicit dendrite → AIS → axon model and ask which biological privileges are actually required.

## Claim boundary

Established fields already include genetic algorithms, evolution strategies, covariance adaptation, structural synaptic plasticity, activity-dependent stabilization, axonal branching, synaptic competition and recurrent learning.

This repo does **not** currently claim:

- that neurons literally run genetic algorithms;
- that fan-in equals selection or fan-out equals mutation in a one-to-one biological sense;
- that the same mechanism repeats fractally at every biological scale;
- that genes explain neural computation by themselves;
- a new learning rule.

The testable hypothesis is narrower:

> **Generate–compete–retain–regenerate is a useful organizational motif at evolutionary scale. Because neurons are recurrent branching systems whose route strengths can persistently change, the same motif may be implementable again at the computational scale.**

That is what V25 now tests.
