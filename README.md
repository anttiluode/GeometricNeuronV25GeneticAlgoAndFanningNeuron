# GeometricNeuronV25 — The Evolving Axon

> What if a sequence of genetic-algorithm generations is not merely optimization history, but **data** from which a system can infer its own next transfer operator?

This repo starts from one narrow claim and refuses the biological metaphor until the claim survives controls.

```text
operator at generation 0
        ↓
operator at generation 1
        ↓
operator at generation 2
        ↓
       ...
        ↓
fit a low-rank temporal operator to the trajectory
        ↓
predict generation g+1
        ↓
ask whether the prediction is useful for learning
```

The motivating neuron picture is deliberately provisional:

```text
dendrite                 AIS / axon history                 terminal
spatial integration  ->  recent transfer trajectory  ->  history-shaped output
```

The hypothesis is **not** that a biological axon runs a genetic algorithm. The hypothesis is that *change of a transfer function through time can itself be state*, and that a system can use that state to anticipate useful future change.

## Gate 0 — does evolutionary history predict the next operator?

A population evolves small linear operators `W`. Fitness is output error on a fixed public probe bank. At every generation we save the elite centroid

```text
W_0, W_1, W_2, ...
```

and fit a low-rank DMD-like one-step model to the recent trajectory.

The prediction is compared with two attackers:

1. **persistence**: `W_(g+1) = W_g`
2. **velocity**: `W_(g+1) = W_g + (W_g - W_(g-1))`

A pass requires the trajectory model to beat both controls across seeds. Merely fitting the already-seen trajectory is not enough; scoring is one-step-ahead.

## Gate 1 — can the prediction become learning?

The same GA is run twice with identical population budget, mutation scale and elitism.

- `BASELINE`: isotropic mutations only.
- `TRAJECTORY_GUIDED`: after enough history exists, a fixed fraction of offspring receives a mutation component along the predicted next-generation displacement.

The guide is norm-matched to an ordinary mutation so it does not receive a hidden larger-step budget.

A useful result would be lower final error and/or earlier arrival at the same error threshold across seeds. A negative result is informative: evolutionary history can be predictable without being a useful control signal.

## Why this belongs after V24

V24 established that the pulse alone is not always the evidence: **address + pulse history** can reveal a hidden system, and persistent writes can change future sensing. It also hit a biological bottleneck: addressed dendritic probes could open full algebraic rank while soma noise still hid several directions.

V25 changes the observable. Instead of demanding that one instantaneous soma amplitude carry everything, it asks whether a *trajectory of changing transfer states* contains recoverable and predictive structure.

Related family:

- `GeometricNeuronV24` — bounded observation, READ/WRITE, active measurement
- `SighImageSuper` — transient history, travelling traces, changed material
- `CausalHorizon` — time-ordered changing operators
- `IttnasNoruen` — preserving responses is not the same as preserving future access
- `BlackBoxLab` — the natural later testbed for actual inherited lineages
- `Operaattori` / `OperaattoriJako` — morphology compiles transport; nonlinear operating state changes the tangent

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evolving_axon.py --seeds 12 --generations 100 --output results.json
```

The script prints the compact battery summary and stores all per-seed curves and one-step prediction errors in `results.json`.

## Claim boundary

Low-rank trajectory models, DMD/Koopman methods, genetic algorithms, momentum, evolution paths, adaptive filtering, short-term synaptic plasticity and activity-dependent axonal excitability are established subjects.

This repo does **not** currently claim:

- a new general learning algorithm;
- that GA generations universally move low-frequency to high-frequency;
- that axons literally implement evolutionary search;
- that trajectory guidance beats ordinary optimization;
- that the observed evolutionary path contains information unavailable to simpler controls.

Those are questions for the gates.

The first thing worth earning is smaller:

> **Does the path by which an operator is changing contain actionable information about what its next useful change will be?**

## First receipt — the simple history wins

The first controlled battery changed the interpretation immediately.

With 12 seeds and 100 generations:

```text
Gate 0: one-step operator prediction
trajectory DMD / persistence error     1.392
trajectory DMD / velocity error        0.974
DMD beats persistence                   0 / 12 seeds
DMD beats velocity                     11 / 12 seeds

mean direction cosine to actual next step
last-step velocity                      0.063
DMD trajectory                          0.015
```

The richer predictor therefore does **not** earn the claim that it forecasts the next generation better than simply holding the current operator fixed. It is only slightly better than raw velocity in absolute prediction and its predicted displacement is less aligned with the realized next displacement.

But Gate 1 exposed a different fact:

```text
final MSE ratio to ordinary GA
last-step guided                        0.806
DMD-trajectory guided                   0.842

last-step beats ordinary GA            12 / 12 seeds
DMD trajectory beats ordinary GA       11 / 12 seeds
DMD trajectory beats last-step          5 / 12 seeds
```

So the first positive is much smaller and cleaner:

> **The displacement between successive generations is an actionable control signal in this toy GA. A more elaborate fitted trajectory has not yet improved on it.**

That immediately motivated Gate 2: replace the one-step displacement with a leaky generational path

```text
p_g = beta * p_(g-1) + (1-beta) * (W_g - W_(g-1)).
```

Across the same 12 seeds:

```text
beta       final MSE / ordinary GA    better seeds
0.00              0.806                 12/12
0.50              0.892                 10/12
0.80              0.944                  9/12
0.95              1.093                  2/12
```

Long memory is therefore **not automatically better**. In this clean stationary landscape it becomes drag.

Gate 3 made selection observations partial: each generation was scored on only 25% or 12.5% of the fixed probe bank. A short path still won in the initial six-seed exploratory battery. This is retained as a boundary, not tuned away.

The current mechanism hypothesis is now:

> **A changing operator can carry a useful temporal derivative. Whether deeper history helps depends on the correlation time of the world and the reliability of each generation's evidence. The memory timescale is itself part of the learning problem.**

See [`RESULTS.md`](RESULTS.md) and [`results/initial_receipt.json`](results/initial_receipt.json).
