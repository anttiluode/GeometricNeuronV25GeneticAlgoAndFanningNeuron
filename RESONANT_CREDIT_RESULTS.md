# Gate F7 — Frequency-addressed causal eligibility

F6 established that branches can choose among several **decay-only** causal timescales. F7 asks whether temporal address can include a resonant component as well as a lifetime.

The branch trace is now

```math
z_{t+1}=\rho e^{i\omega}z_t+u_t.
```

Each candidate trace therefore has two temporal coordinates:

```text
rho    persistence / lifetime
omega  temporal frequency / phase progression
```

This is implemented with complex state for convenience. It is equivalent to a real two-dimensional damped rotation and is not a claim that neurons literally use complex arithmetic.

## Hidden world

There are 12 separated 6-D branches. Three branches perturb on each event.

Each branch is assigned one hidden temporal channel from

```text
0
pi/12
pi/6
pi/3
```

The local antithetic consequence from that branch is injected into a damped oscillator with that hidden frequency. The 12 oscillator outputs are then averaged into **one global scalar modulation**.

The learner receives neither the local consequence nor the branch frequency.

So the world is deliberately a mixed temporal-credit problem:

```text
branch-local perturbations
       ↓
hidden branch-specific resonators
       ↓
all branches mixed
       ↓
one scalar modulation
```

## Learner

Every branch carries the same four candidate resonant eligibility traces.

For each candidate frequency, a tiny local linear predictor sees only that branch's own complex eligibility coordinates and tries to predict the observed global modulation. The trace with the lowest recent prediction error controls plasticity for that branch.

No predictor is given:

- the hidden branch target;
- the true temporal frequency;
- a source-labelled reward;
- another branch's local consequence.

## 8-seed result

5,000 events per seed.

| policy | mean final score | mean worst branch | mean curve score | frequency ID |
|---|---:|---:|---:|---:|
| **adaptive resonant selector** | **0.65490** | **0.51437** | **0.37821** | **1.000** |
| oracle true frequency | 0.71906 | 0.65020 | 0.44946 | 1.000 |
| fixed channel 0 | 0.31801 | -0.15169 | 0.18939 | 0.250 |
| fixed channel 1 | 0.34120 | 0.08167 | 0.18562 | 0.250 |
| fixed channel 2 | 0.29200 | 0.00580 | 0.16515 | 0.250 |
| fixed channel 3 | 0.25992 | 0.02205 | 0.14790 | 0.250 |
| shuffled branch↔frequency assignment | 0.39009 | 0.23339 | 0.20160 | 0.281 |

The adaptive selector beat every fixed channel and the shuffled assignment in **8/8 matched seeds**.

The striking diagnostic is that the learned dominant channel identifies the hidden branch frequency with **100% branch accuracy** over these runs.

That is not mysterious: the synthetic world was deliberately constructed so that branch consequence is carried through one of the candidate resonant kernels. The relevant result is therefore not that the selector discovered an arbitrary frequency. It is that **one unlabeled global scalar still contains enough correlation structure for separated local traces to recover which temporal channel belongs to which branch**.

## What this adds to F6

F6 asked:

```text
HOW LONG should this local cause remain eligible?
```

F7 adds:

```text
WHICH TEMPORAL MODE carries evidence about this route?
```

So the branch-local state now has a two-dimensional temporal address:

```text
(route identity, temporal mode)
```

or more explicitly,

```math
(b,\rho,\omega).
```

The world can mix many branches into one scalar consequence while the branch-local resonant bank retains enough distinct temporal structure for credit assignment.

## Why this is relevant to the spectral-router discussion

This gate does **not** import GAx's operator theory or claim to be a spectral router for computational approaches.

It independently establishes the lower-level object that the third-arm synthesis suggested V25 should test:

> **frequency can function as part of the address of causality.**

The information is not sent to an explicit Fourier bin. Instead, a local damped mode is coherent with one consequence history and incoherent with others.

The modulation therefore reinforces some branch-mode pairs more effectively than others.

That is the first executable V25 object of the form

```text
branch × temporal mode.
```

## Biological interpretation

A biological implementation would not need literal complex variables. A pair of real states is sufficient:

```math
\begin{bmatrix}x_{t+1}\\y_{t+1}\end{bmatrix}
=
\rho
\begin{bmatrix}
\cos\omega & -\sin\omega\\
\sin\omega & \cos\omega
\end{bmatrix}
\begin{bmatrix}x_t\\y_t\end{bmatrix}
+B u_t.
```

Biologically, that could only be an analogy for any local process with damped oscillatory / phase-sensitive dynamics. F7 does not identify a specific ion channel, dendritic oscillator, AIS mechanism, or neuromodulatory pathway.

The architectural hypothesis is narrower:

```text
spatial separation preserves WHERE
local eligibility preserves WHAT
trace lifetime preserves HOW LONG
resonant state can preserve WHICH TEMPORAL MODE
```

## Next boundary

F7 is intentionally easy in one respect: each hidden branch frequency is chosen exactly from the learner's candidate bank.

The next attacker should break that privilege.

Useful tests include:

```text
1. hidden frequencies BETWEEN the candidate channels;
2. frequencies that drift over time;
3. two simultaneous temporal modes on one branch;
4. aperiodic / broadband consequence where no resonant channel deserves to win;
5. compare a resonant bank against an equally sized generic linear state-space bank.
```

If the mechanism only works when the answer is already present as a discrete channel, F7 is a selector. If it can interpolate, split, or reorganize under those attackers, it becomes a stronger candidate for self-organizing temporal routing.

## Claim boundary

This is a synthetic temporal demixing result with hidden branch-specific resonators. It supports only:

> **Separated local resonant eligibility can use one mixed global scalar to recover branch-specific temporal channel identity and improve credit assignment when the world's causal consequences are carried by those channels.**
