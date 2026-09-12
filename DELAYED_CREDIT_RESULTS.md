# Gate F3 — Delayed consequence × local eligibility

F2 still gave the currently active branch an immediate scalar score. F3 removes that privilege from the learner.

A branch proposes one fixed-norm delta. The hidden world generates only a scalar consequence, adds noise, and returns that consequence **4–20 events later**. During the delay, other branches continue producing events.

The learner never receives the hidden target direction or the branch score itself.

The central test is whether a persistent local eligibility tag can bridge the temporal gap:

```text
branch delta
    ↓
local event
    ↓
DELAY / other events
    ↓
scalar consequence arrives
    ×
stored branch + delta eligibility
    ↓
retain or reject the old delta
```

## Arms

- `immediate`: consequence is available immediately; this is the upper-bound oracle.
- `tagged`: consequence is delayed, but branch identity and the causal delta remain tagged until it returns.
- `history`: same delayed tag, plus a weak bias toward that branch's recent positively consolidated deltas.
- `wrong_tag`: the correct delayed consequence is assigned to a different branch.
- `current_branch`: no persistent tag; the consequence is assigned to whatever branch happens to be active when it arrives.

## 16-seed result

| policy | mean final score | mean worst branch | cycles to 0.95 | cycles to 0.99 |
|---|---:|---:|---:|---:|
| immediate oracle | 0.99058 | 0.98316 | 85.06 | 185.50* |
| delayed + correct tag | 0.99055 | 0.98406 | 85.44 | 181.36* |
| **delayed tag + own delta history** | **0.99188** | **0.98623** | **75.44** | **168.25** |
| delayed + wrong tag | -0.86513 | -2.12278 | — | — |
| delayed + current branch | -0.55092 | -1.66520 | — | — |

`*` Immediate and tagged reached 0.99 in 14/16 seeds within 2400 events; history reached it in 16/16.

Correct delayed tagging was almost indistinguishable from immediate credit in mean final score:

```text
tagged - immediate = -0.0000296
```

The correct tag beat both mis-credit controls in **16/16 seeds**.

The weak route-history term improved mean final score by `0.001326` over tagged-only and beat tagged-only in 12/16 seeds. More importantly, it reduced mean time to 0.99 from `181.36` cycles for tagged-only to `168.25`, while reaching the threshold in all 16 seeds.

## What this says

This is still a toy. But it changes the architecture of the learning problem.

Without a persistent local tag, a delayed scalar consequence has no reliable way to know which separated route caused it. The same useful scalar reward becomes destructive when applied to the wrong route.

So the important state is not only a weight or a direction. It is a **causal route identity that persists across time**:

```text
eligibility = (route, attempted delta, time)
```

That gives a more precise bridge from V25 to modern neuron biology:

- dendritic compartmentalization can preserve local route identity;
- a global spike/AIS event need not erase which local compartment was eligible;
- delayed downstream or modulatory consequences can act globally while plasticity remains local because eligibility is stored locally;
- accepted route-specific deltas can then form the next-generation "highway" for that compartment.

## Claim boundary

The world consequence is generated from a hidden synthetic objective, not from a realistic downstream recurrent circuit. The delayed queue is therefore only a minimal temporal-credit test.

F3 earns this narrower statement:

> **With delayed scalar consequences, preserving the identity of the causal local route can recover almost all of the immediate-credit performance in this toy. Destroying route identity destroys learning.**

## Next attacker

F4 should make the consequence genuinely mixed:

```text
several branch events overlap
        ↓
recurrent downstream state
        ↓
one delayed scalar consequence
```

No reward packet should carry a source label. Each branch may keep only its own decaying eligibility trace. The question then becomes whether `global consequence × local eligibility` can solve credit assignment without an explicit queue mapping reward to source.

That is much closer to a three-factor synaptic learning rule and is the next biological boundary to cross.
