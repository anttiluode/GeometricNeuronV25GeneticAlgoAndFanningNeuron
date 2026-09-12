# Gate F9 — The temporal pole itself can fan, compete, and move

F8 broke the F7 mechanism by moving the hidden causal frequencies between the predeclared candidate channels.

The obvious repair — averaging neighboring resonant traces — failed.

F9 therefore changes the object. Instead of mixing the outputs of fixed resonators, each branch is allowed to change its own resonant pole.

The local trace is

```math
z_{t+1}=\rho e^{i\omega_b}z_t+u_t,
```

and `omega_b` is no longer fixed.

## The mechanism repeats the original V25 motif

Every branch starts from the same frequency `pi/6`.

Around its current frequency it opens a tiny three-way fan:

```text
omega_b - delta
omega_b
omega_b + delta
```

All three local candidate traces observe the same branch perturbations and the same single global scalar modulation. Each candidate has a tiny local predictor.

After one epoch, the branch retains the candidate with the lowest prediction error:

```text
FAN nearby temporal poles
        ↓
COMPETE by local prediction error
        ↓
RETAIN winner
        ↓
SHRINK fan radius
        ↓
FAN again
```

No branch is shown its true hidden frequency.

This is deliberately a recursion of the generate–compete–retain motif at the level of the branch's **own temporal dynamics**.

## Hidden world

The same off-grid frequencies from F8 are used:

```text
pi/24
pi/8
pi/4
```

Three branches perturb on every event. Each branch's local consequence passes through its hidden damped oscillator. All oscillator outputs are averaged into one scalar modulation.

The branch must infer its own temporal pole only from correlations between its local traces and that mixed global scalar.

## 8-seed result

6,000 events per seed.

| policy | mean final score | mean worst branch | mean curve score |
|---|---:|---:|---:|
| F8 discrete selector | 0.41883 | 0.18243 | 0.22357 |
| privileged nearest grid channel | 0.46206 | 0.20583 | 0.25980 |
| **adaptive continuous pole fan** | **0.57950** | **0.28583** | **0.29426** |
| exact-frequency oracle | 0.73561 | 0.68632 | 0.46520 |

The adaptive pole fan beats both the discrete selector and the privileged nearest-grid channel in **8/8 matched seeds**.

```text
adaptive - discrete = +0.16067
adaptive - nearest  = +0.11745
oracle   - adaptive = +0.15611
```

So it closes roughly half of the F8 gap to the exact-frequency oracle without receiving the hidden frequencies.

## Did the poles actually move toward the hidden frequencies?

Yes.

Every branch begins at exactly `pi/6`, giving mean initial absolute frequency error

```text
0.26180 rad
```

After the repeated local pole fans, the mean diagnostic error is

```text
0.07657 rad
```

The task improvement is therefore accompanied by direct recovery of the hidden temporal parameter rather than only an accidental performance gain.

This is still imperfect. Some branches/seeds retain noticeably wrong poles, which is visible in the relatively weak worst-branch score (`0.28583`) compared with the oracle (`0.68632`).

## What F9 changes conceptually

F7 established:

```text
frequency can be part of a causal address
```

F8 established:

```text
a fixed coarse bank does not automatically create unseen frequencies
```

F9 adds:

```text
the temporal address itself can be locally searched and rewritten
```

The causal address is no longer merely

```math
(branch,\rho,\omega_k)
```

with `omega_k` selected from a predefined list.

It begins to look like

```math
(branch,\rho,\omega_b(t)),
```

where the branch's own temporal dynamics are a plastic variable.

## Why this is interesting for the original V25 idea

The same abstract operation now appears at several levels:

```text
parameter delta fan
    -> select useful local change
    -> retain route history

eligibility timescale bank
    -> select useful causal lifetime

resonant pole fan
    -> select useful temporal dynamics
    -> move pole
    -> fan again
```

So "fan → compete → retain → fan" is not tied to genomes or to one parameter space. In this synthetic system it can act on the **rules that determine what counts as causal history**.

That is much closer to metaplasticity / adaptive temporal filtering than to a literal genetic algorithm inside a neuron.

## Relation to the spectral-router arm

F9 still does not import GAx's positive-operator or computational-mode machinery.

But V25 has now independently produced a lower-level self-rewriting spectral object:

```text
local route
  ×
local temporal mode
  ×
consequence-driven pole adaptation
```

The important sentence is therefore narrower than "the neuron is a spectral router":

> **A separated local route can use mixed global consequences to move the temporal mode through which its own causal history is represented.**

If GAx independently finds that contexts select among genuinely distinct computational modes, the two results can later be compared without having forced one into the other.

## Next attackers

F9 still has substantial engineered structure:

- exactly one hidden temporal mode per branch;
- stationary hidden frequencies;
- a regular epoch boundary for pole competition;
- a hand-designed three-point frequency fan;
- a predictor-error fitness signal for the pole.

The strongest next tests are:

```text
1. drift the hidden frequency continuously during a run;
2. put TWO simultaneous temporal modes on one branch;
3. remove explicit epoch boundaries and let pole competition be continuous;
4. compare against a matched generic 2-state linear dynamical trace;
5. give a broadband/no-mode world and require the resonant mechanism not to hallucinate a stable frequency.
```

The two-mode branch is especially important. A real spectral representation should sometimes preserve multiple modes rather than purify everything to one winner.

## Claim boundary

F9 supports only:

> **In the controlled off-grid resonant-credit toy, a branch-local generate–compete–retain search over a continuous resonant pole can recover much of the performance lost by a coarse fixed frequency bank, while reducing direct error to the hidden branch frequencies.**

It does not establish a biological resonator, a universal optimizer, or a complete spectral-routing architecture.
