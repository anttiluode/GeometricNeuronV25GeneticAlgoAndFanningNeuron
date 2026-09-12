# Gate F8 — Off-grid temporal modes

F7 showed that a branch can recover its own temporal channel from one mixed global scalar when the true hidden frequency is already present in the candidate bank.

F8 removes that privilege.

The learner still has the four F7 channels:

```text
0
pi/12
pi/6
pi/3
```

but the hidden branch frequencies are now strictly between them:

```text
pi/24
pi/8
pi/4
```

No branch has an exact matching candidate trace.

## Why this attacker matters

F7 could have been only a lookup problem:

```text
world uses channel k
      ↓
prediction error identifies channel k
      ↓
use preloaded channel k
```

A stronger spectral mechanism should eventually be able to represent or create a temporal mode that was not predeclared.

F8 asks whether the existing bank can do that merely by selecting or blending its channels.

## Arms

- `discrete`: the F7 prediction-error selector; one candidate channel wins per branch.
- `softmix`: prediction-error-weighted mixture of all four candidate eligibility traces.
- `nearest`: privileged control told which candidate frequency is geometrically nearest the hidden one.
- `oracle`: privileged upper bound using an eligibility trace at the exact hidden off-grid frequency.

All arms see the same one global mixed scalar modulation.

## 8-seed result

4,000 events per seed.

| policy | mean final score | mean worst branch | mean curve score |
|---|---:|---:|---:|
| F7 discrete selector | 0.29704 | 0.10297 | 0.15452 |
| soft mixture of bank | 0.22768 | 0.11062 | 0.11764 |
| privileged nearest channel | 0.34488 | 0.13773 | 0.18652 |
| **oracle exact off-grid trace** | **0.61058** | **0.54255** | **0.35826** |

For comparison, the on-grid F7 adaptive selector reached about `0.65490` under its matched-channel world.

Moving the hidden frequencies off the supplied grid therefore opens a large gap:

```text
oracle - discrete = +0.31354
oracle - nearest  = +0.26571
```

A naive soft mixture does not repair the gap; it is worse than discrete selection.

## The result is negative, and useful

F8 says that the current V25 resonant mechanism is **not yet a self-organizing spectrum**.

It is a good selector among already available temporal modes.

It does not currently synthesize an unseen intermediate mode just because neighboring resonant traces are available.

That distinction matters:

```text
F7 earned: choose the correct causal channel from a bank
F8 rejects: the bank automatically interpolates into arbitrary causal frequencies
```

The failure also explains why the soft mixture is not enough. A weighted sum of two long-lived resonators is generally a superposition of two frequencies, not one resonator at the intermediate frequency. Over time their phases separate.

So if V25 is going to grow a true spectral router at the biological level, the next mechanism must alter the **dynamics of the trace itself**, not merely average the outputs of fixed traces.

## Biological interpretation

This is a useful constraint on the "brain frequencies" story.

Frequency should not be treated as a label in a lookup table. If a biological compartment can adapt temporal selectivity, some physical parameter has to move:

```text
channel kinetics
membrane / conductance time constants
coupled-state rotation / phase dynamics
local circuit resonance
```

F8 does not identify which biological mechanism does that. It only shows what the computational problem requires.

## Next gate — move the pole, not the mixture

The natural F9 experiment is a continuous local resonator:

```math
z_{t+1}=\rho e^{i\omega_b}z_t+u_t
```

where each branch owns its own `omega_b` and is allowed to change that frequency using only local trace state plus the same global scalar modulation.

A differentiable version can propagate the sensitivity

```math
s_{t+1}=\rho e^{i\omega_b}(i z_t+s_t)
```

so a local prediction error can provide a gradient on `omega_b` without revealing the hidden true frequency.

The hard controls should be:

```text
fixed nearest channel
random frozen continuous frequency
discrete F7 selector
continuous adaptive frequency
exact-frequency oracle
```

And the criterion is not merely final task score. We should directly measure whether

```math
|\hat\omega_b-\omega_b^*|
```

shrinks for held-out seeds.

If continuous adaptation fails, keep the failure. It would mean that the temporal basis probably needs developmental / structural provision rather than ordinary online plasticity.

## Claim boundary

F8 supports only:

> **The F7 branch×frequency result depends strongly on having suitable resonant modes already available. Selection or naive mixing of a coarse fixed bank does not recover hidden off-grid temporal channels.**
