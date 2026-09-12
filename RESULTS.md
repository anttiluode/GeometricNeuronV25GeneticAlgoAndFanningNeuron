# Initial results

## Status

The first experiments do **not** support the strongest starting story. A low-rank DMD-like model of the whole recent generational trajectory does not beat persistence at forecasting the next elite-centroid operator.

They do support a narrower mechanism: **the immediately previous generational displacement can be reused as a mutation direction and improves this toy GA under an equal mutation-norm budget.**

## Gate 0 — forecast the next operator

12 seeds, 100 generations, 10x10 linear operators, population 96.

| measure | result |
|---|---:|
| DMD / persistence mean one-step error | 1.39197 |
| DMD / velocity mean one-step error | 0.97411 |
| DMD better than persistence | 0/12 seeds |
| DMD better than velocity | 11/12 seeds |
| velocity direction cosine to realized step | 0.06312 |
| DMD direction cosine to realized step | 0.01511 |

Interpretation: persistence is a very strong absolute-error baseline because consecutive elite centroids move only a little. DMD is not a useful absolute forecaster here.

## Gate 1 — use predicted change as mutation guidance

Same 12 seeds and 100-generation budget.

| arm | final MSE / ordinary GA |
|---|---:|
| last-step / velocity guide | **0.80611** |
| DMD trajectory guide | 0.84203 |

The last-step guide beat ordinary GA in 12/12 seeds. DMD trajectory guidance beat ordinary GA in 11/12 but beat the simple last-step guide in only 5/12.

This prevents the project from claiming that the fitted future operator is the useful mechanism. The simpler generational derivative is currently better.

## Gate 2 — how much generational memory?

Use a leaky evolution path `p_g = beta p_(g-1) + (1-beta) delta W_g`.

| beta | final MSE / ordinary GA | better seeds |
|---:|---:|---:|
| 0.00 | **0.80611** | 12/12 |
| 0.50 | 0.89152 | 10/12 |
| 0.80 | 0.94440 | 9/12 |
| 0.95 | 1.09302 | 2/12 |

The clean task prefers very short memory. This is an important negative result for the neuron analogy: adding a slow trace does not magically produce better learning.

## Gate 3 — bounded fitness observation (exploratory)

Six seeds, 60 generations. Selection sees only a random fraction of the public probe bank per generation; final MSE is evaluated on the full bank.

### 25% of probes per generation

| beta | final MSE / ordinary GA | better seeds |
|---:|---:|---:|
| 0.00 | **0.87233** | 6/6 |
| 0.50 | 0.89891 | 5/6 |
| 0.80 | 1.06974 | 1/6 |
| 0.95 | 1.06464 | 2/6 |

### 12.5% of probes per generation

| beta | final MSE / ordinary GA | better seeds |
|---:|---:|---:|
| 0.00 | **0.83769** | 6/6 |
| 0.50 | 0.86458 | 6/6 |
| 0.80 | 1.00280 | 4/6 |
| 0.95 | 1.06839 | 0/6 |

This battery is exploratory and not a locked claim. Partial observation alone did not create an advantage for long history.

## What follows

The next experiment should not search beta values until one wins. It should change the causal question.

A real axonal-history model needs a world whose useful direction has a measurable correlation time. Then the test becomes whether a local mechanism can infer which history timescale is predictive, rather than being handed a fixed slow trace.

That points toward a V25 neuron with multiple candidate traces (fast / medium / slow), where only evidence determines which trace is allowed to influence the next change.
