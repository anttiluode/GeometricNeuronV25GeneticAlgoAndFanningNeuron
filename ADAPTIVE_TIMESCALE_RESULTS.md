# Gate F6 — The branch learns how long to remember

F5 showed that a fixed eligibility timescale is not universally useful. Slow downstream consequences require longer-lived local causal state.

F6 removes the experimenter's choice.

Every branch simultaneously carries four local eligibility traces:

```text
0.20   fast
0.70   medium
0.95   slow
0.98   very slow
```

Each trace has a tiny local linear predictor trained only from:

```text
that branch's own eligibility trace
        ->
observed global scalar modulation
```

The branch trusts the trace with the lowest recent prediction error. Different branches may select different timescales.

No predictor receives the hidden target, branch score, downstream decay, or switch time.

## Nonstationary test

The world changes halfway through a 6,000-event run:

```text
steps 0..2999:     downstream decay = 0.00
steps 3000..5999:  downstream decay = 0.90
```

The first half is essentially immediate global consequence. The second half has strong temporal mixing.

Eight matched seeds were run. Every fixed-trace control receives the same events, perturbation scale and scalar consequence process.

## Trace selection changes after the hidden switch

Mean fraction of branch-timescale selections:

| trace decay | before switch | after switch |
|---:|---:|---:|
| 0.20 | **0.93380** | 0.08941 |
| 0.70 | 0.04733 | 0.01400 |
| 0.95 | 0.00763 | 0.08815 |
| 0.98 | 0.01125 | **0.80844** |

So the same branches move from overwhelmingly fast causal memory to overwhelmingly very slow causal memory after the downstream world becomes slow.

The switch is not hard-coded into the selector. It emerges from online prediction error driven by the same global modulation used for learning.

## Learning result

| policy | mean score at switch | mean final score | mean learning-curve area |
|---|---:|---:|---:|
| fixed 0.20 | 0.78463 | 0.80984 | 0.65815 |
| fixed 0.70 | 0.78418 | 0.83336 | 0.66398 |
| fixed 0.95 | 0.75893 | 0.86595 | 0.65694 |
| fixed 0.98 | 0.70940 | 0.85432 | 0.62218 |
| **adaptive local selector** | **0.78810** | **0.87925** | **0.68435** |

The adaptive selector beats **every fixed trace in 8/8 matched seeds** on final score.

This matters because the best static compromise is not enough. A trace long enough for the slow world gives up useful responsiveness in the fast world; a trace short enough for the fast world loses causal information after the switch.

The branch bank can preserve both possibilities and change which one controls plasticity.

## What is being learned

F2–F5 separated three questions:

```text
WHERE did the causal event occur?       -> branch identity
WHAT local change remains creditable?   -> eligibility state
HOW LONG should that cause survive?     -> eligibility timescale
```

F6 is the first gate where the last answer is chosen online rather than by the experimenter.

A compact state description is now:

```text
branch b owns:
    fast eligibility     e_b^f
    medium eligibility   e_b^m
    slow eligibility     e_b^s
    very slow eligibility e_b^v

plus local confidence / prediction error for each trace.
```

The global consequence remains one scalar. The timescale state remains separated by branch.

## Biological interpretation

This is much closer to **metaplasticity** than to a literal genetic algorithm.

The neuron-shaped hypothesis is now:

```text
local dendritic event
      ↓
several local biochemical / electrical traces with different lifetimes
      ↓
global spike / downstream / modulatory consequence
      ↓
local evidence about which causal lifetime is currently useful
      ↓
plasticity uses that timescale more strongly
```

The specific predictor used here is only a computational instrument. F6 does **not** claim that neurons run normalized LMS or explicitly minimize prediction error over four numerical decay constants.

What the gate establishes is an architectural possibility:

> **A compartment with multiple local causal timescales can infer from global consequences which lifetime is currently informative, and can change that choice when the consequence dynamics change.**

## Why this changes the V24 fast / medium / slow idea

In V24, multiple timescales were largely an architectural hypothesis.

Here they have a measured job:

```text
fast state     catches immediate consequences
slow state     survives recurrent temporal mixing
meta-state     decides which lifetime currently deserves credit
```

That is not decoration anymore. It is a solution to a changing temporal-credit problem.

## Next boundary

F6 still gives every branch the same hand-designed trace bank and a dedicated predictor for each trace.

The next biological question is whether the timescales themselves can be **grown / pruned / moved** rather than predeclared.

One possible F7 is:

```text
start with only two causal lifetimes
        ↓
measure persistent prediction mismatch
        ↓
fan a new candidate lifetime between / beyond them
        ↓
retain useful lifetime, prune redundant one
```

That would return to the original V25 generate–compete–retain motif at the level of **memory timescales themselves**.

## Claim boundary

The hidden task remains synthetic, the scalar consequence is generated by an antithetic objective probe, and the trace selector is an engineered predictor bank.

F6 therefore earns a nonstationary credit-assignment result, not a biological identity claim.
