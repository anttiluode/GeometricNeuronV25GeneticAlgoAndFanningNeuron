# Gate F4 — Mixed global consequence × local eligibility

F3 still attached the delayed scalar consequence to the causal `(branch, delta)` pair. F4 removes that source label.

Several branches now perturb simultaneously. Their effects are compressed into **one scalar consequence**, then mixed through a recurrent downstream state. The learner receives only that scalar modulation.

Each branch is allowed to retain only its own decaying local eligibility trace.

```text
branch A perturbation ─┐
branch B perturbation ─┼─> one global consequence ─> recurrent temporal mixing
branch C perturbation ─┘                              │
                                                      ×
                                     local eligibility traces
                                                      │
                                                      v
                                             branch-specific change
```

No reward packet contains a branch ID.

## Task

There are 12 separate 6-D branch parameter vectors and 12 hidden useful directions. On each event, three branches are chosen at random and receive fixed-norm exploratory perturbations.

The world computes one antithetic global scalar consequence from the combined event. A recurrent state with decay `0.70` mixes that scalar with consequences from earlier events. The learner therefore cannot tell from the scalar alone which branch or which recent perturbation caused it.

The default run is 6,000 events per seed, 16 seeds.

## Arms

- `local_oracle`: upper bound; each active branch receives its own immediate local antithetic consequence.
- `immediate_global`: one global scalar, but no temporal mixing.
- `eligibility`: temporally mixed global scalar × each branch's own decaying eligibility trace.
- `current_only`: same mixed scalar, but only the current perturbation is available; no persistent trace.
- `shuffled_eligibility`: same traces, but branch identities are permuted before the update.
- `pooled_eligibility`: branch traces are averaged into one global trace and copied back to every branch.

## 16-seed result

| policy | mean final score | mean worst branch | reaches 0.75 | mean cycle to 0.75 |
|---|---:|---:|---:|---:|
| local oracle | 0.99315 | 0.99268 | 16/16 | 19.44 |
| immediate global scalar | 0.90019 | 0.89077 | 16/16 | 218.31 |
| **mixed scalar × local eligibility** | **0.81811** | **0.79239** | **16/16** | **369.94** |
| mixed scalar × current perturbation only | 0.64450 | 0.60952 | 0/16 | — |
| mixed scalar × shuffled eligibility | 0.06668 | -0.21892 | 0/16 | — |
| mixed scalar × pooled eligibility | 0.08556 | -0.16301 | 0/16 | — |

The local eligibility arm beat all three source-destroying controls in **16/16 matched seeds**:

```text
eligibility - current_only          +0.17361
eligibility - shuffled_eligibility  +0.75143
eligibility - pooled_eligibility    +0.73254
```

## What changed from F3

F3 looked almost magical because a delayed reward still carried an exact causal tag. Correct tagging recovered nearly all immediate-credit performance.

F4 is harder and more biologically interesting. The source label is gone. Several events overlap and the observed consequence is only one temporally mixed scalar.

The result is correspondingly less perfect:

```text
local oracle        0.993
immediate global    0.900
local eligibility   0.818
```

So local eligibility does **not** solve the mixed-credit problem completely.

But it preserves substantial learning while the source-destroying controls fail badly. In particular, simply pooling all traces into one global trace nearly eliminates learning.

That gives a sharper earned statement:

> **A global consequence can remain useful without an explicit source label when local branches preserve their own temporally extended eligibility. Destroying either the temporal trace or the branch identity sharply reduces learning in this toy.**

## Biological interpretation

This is now close in form to a three-factor plasticity rule:

```text
local activity / perturbation
        ×
local eligibility state
        ×
global delayed modulatory consequence
        -> plastic change
```

The experiment does not establish that a dendrite, AIS, axon, dopamine system, or any specific molecular pathway implements this exact rule.

What it tests is the architectural consequence of **separation**:

- a branch can keep local causal state even after the global spike/event has mixed information;
- a global downstream consequence need not carry a detailed address if local eligibility preserves the address;
- averaging those local histories destroys the mechanism.

This makes the Aizenbud-style semi-independent dendritic compartment more than an extra nonlinear feature detector in the V25 hypothesis. It becomes a possible **credit-preserving compartment**.

## Next gate

F5 should test the timescale itself rather than fixing it by hand.

The downstream consequence has a correlation time; the branch eligibility has a decay time. V25 should give each branch several traces with different decay constants and ask whether the system can select the trace that best predicts future useful modulation.

That would connect the earlier fast/medium/slow V24 idea to the new separated-route learning mechanism without importing anything from GAx.

## Claim boundary

F4 still uses a synthetic hidden objective and antithetic perturbation signal. The recurrent downstream state is only a minimal model of mixed delayed consequence.

The result therefore supports a learning-architecture claim, not a biological identity claim.
