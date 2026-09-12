# Gate F5 — Eligibility has a timescale

F4 established that one temporally mixed global scalar can still train separated local routes when each route retains its own eligibility trace.

F5 asks a simpler but important question:

> **Can the eligibility trace have any arbitrary decay?**

No. The useful local memory depends on how slowly the downstream consequence itself mixes information through time.

## Sweep

Downstream recurrence decays:

```text
0.20, 0.50, 0.70, 0.90
```

Local eligibility decays:

```text
0.00, 0.20, 0.50, 0.70, 0.85, 0.95, 0.98
```

Every cell is averaged over 8 matched seeds and 4,000 events.

## Best trace by downstream timescale

| downstream decay | best eligibility decay | best final score | fixed 0.70 trace | gain over fixed 0.70 |
|---:|---:|---:|---:|---:|
| 0.20 | 0.85 | 0.83244 | 0.82826 | +0.00418 |
| 0.50 | 0.95 | 0.81643 | 0.78961 | +0.02681 |
| 0.70 | 0.95 | 0.80460 | 0.72357 | +0.08103 |
| 0.90 | 0.98 | 0.75274 | 0.47707 | +0.27566 |

The best eligibility decay is **monotone non-decreasing** as the downstream consequence becomes slower.

The important feature is not the exact numerical optimum. In this synthetic rule the best local trace is often substantially slower than the downstream recurrence itself. Those values depend on learning rate, perturbation scale, recurrence, and task geometry.

The robust qualitative result is:

> **Slow consequences require longer-lived local causal state. A fixed medium-timescale trace increasingly loses credit as downstream mixing becomes slower.**

The slowest world makes this particularly obvious: a fixed `0.70` trace reaches `0.477`, while the best tested trace (`0.98`) reaches `0.753`.

## Biological interpretation

This makes the V25 architecture explicitly multiscale:

```text
fast event / perturbation
        ↓
local eligibility trace
        ↓
slower mixed downstream consequence
        ↓
plastic consolidation
```

A branch cannot know in advance how long its causal mark must survive. If consequences return on several timescales, one eligibility constant is unlikely to be sufficient.

That reconnects V25 to the earlier V24 fast/medium/slow idea in a much less decorative way. The different timescales now have a job:

```text
fast   = current event
medium = recent local causal state
slow   = eligibility that survives long consequence mixing
```

## What F5 does not yet solve

The best trace is selected **after the fact** by the experimenter. The neuron/learner does not yet choose among traces itself.

So F5 is a map, not the adaptive mechanism.

The next gate should give every branch a bank of fast/medium/slow eligibility traces simultaneously and make trace selection itself depend only on experienced consequences.

That is the point where V25 would begin learning not only *what* changed successfully, but **how long the cause of change should remain eligible**.
