# Multi-Group DM: Implementation and Simulation Findings

Companion to `multigroup_dm_modification.md`. The proposal is implemented in
`scquint/differential_splicing.py` (`run_differential_splicing_multigroup`), with
three corrections to the design and one negative result about its payoff.

All numbers below come from simulation: counts drawn as
`p_i ~ Dirichlet(alpha * psi_g)`, `y_i ~ Multinomial(n_i, p_i)` with a known
`alpha`, 3 junctions, 6 groups, and a target that is ~9.5% of 7,400 cells (the
IN-CTX-MGE share). Scripts are not committed; see the summary tables.

## 1. The premise is correct, and the effect is large

The doc argues that pooling heterogeneous cell types into "rest" charges
between-type variation to the overdispersion parameter. It does. Fitted
concentration as a function of the size of a shared effect (true alpha = 20):

| ΔΨ of the shared effect | two-group fitted α | K-group fitted α |
|---|---:|---:|
| 0.00 | 20.26 | 20.36 |
| 0.03 | 19.65 | 19.83 |
| 0.06 | 19.50 | 20.24 |
| 0.10 | 18.18 | 20.15 |
| 0.20 | 14.43 | 20.16 |
| 0.40 | 6.82 | 20.00 |

An 8x deflation at ΔΨ = 0.40. The K-group model recovers the true value
throughout. One correction to the doc's account: the current model has a *single*
shared `log_alpha` (`DirichletMultinomialGLM.log_alpha` is a scalar), so there is
no separate inflated `α_rest`. The shared α is dragged down for both groups. This
also means the K-group model fixes the problem *with a shared α* — per-group
`α_k` is a second-order refinement, not the mechanism. Default to
`alpha_mode="shared"`.

## 2. But it does not buy power, because the deflation is self-limiting

The deflation scales with the effect size, so it is nearly absent exactly where
power is decided. Power at nominal 5% and at p < 1e-4, weak shared effect:

| ΔΨ | two-group p<.05 | K-group p<.05 | two-group p<1e-4 | K-group p<1e-4 |
|---|---:|---:|---:|---:|
| 0.05 | 0.523 | 0.523 | 0.033 | 0.037 |
| 0.08 | 0.917 | 0.917 | 0.313 | 0.323 |
| 0.12 | 1.000 | 1.000 | 0.927 | 0.927 |

At most one percentage point, within noise at n = 300. Events that are already
significant get tighter statistics (median −log10 p 73.4 vs 72.0 at ΔΨ = 0.40),
which does not change any conclusion. No power is lost on target-specific events
where the rest pool is homogeneous.

This does not support the doc's projection of +3–5% excess over null and 150–250
significant events for MGE versus 68 currently. The mechanism the doc identifies
is real; the inference it draws from it does not follow, because the mechanism
only bites once the effect is large enough to detect anyway.

## 3. The proposed null model is wrong — two separate problems

### 3a. Dropping the target's column tests uniformity, not the contrast

The doc's Option A builds the null as `x_null = x[:, null_cols]`. With one-hot
encoding this leaves the target's cells with an all-zero design row, so their
logits are 0 and their PSI is uniform (1/n_classes). The df still works out to
`n_classes - 1`, so nothing errors — you just get a flood of tiny p-values from
the wrong hypothesis. On data simulated with all groups identical (H0 true):

| null | p-value |
|---|---:|
| dropped column (doc's version) | 8.6e-45 |
| constrained reparameterization | 0.45 |

The correct constraint is a reparameterization, not a dropped column: give the
target's cells a design row equal to the weight vector `w` over the remaining
K−1 columns. See `_build_constrained_null_design`.

### 3b. Averaging logits is ill-posed on real splicing data

Even with the constraint imposed correctly, `A[target] = Σ w_k A[k]` is
unusable. A non-target group with a zero-count junction has an unbounded logit,
so the weighted average can be driven anywhere, and the null stops restricting
the target at all. Observed behaviour:

| case | two-group p | simplex null p | logit null p |
|---|---:|---:|---:|
| target + 2 others degenerate | 1.1e-273 | 0 (underflow) | **1.00** |
| 1 non-target degenerate | 3.1e-28 | 2.3e-30 | 1.5e-42 |
| 2 non-target degenerate | 1.3e-32 | 2.1e-35 | 4.8e-119 |

In the first row the null model's coefficients run to 61.6 and 54.0 (from ~6.5)
and `ll_null` matches `ll_alt` to 4 decimals: the LRT is exactly 0 on the
strongest possible event. In the others the reference is dragged toward a corner
of the simplex, inflating the statistic by 12 to 87 orders of magnitude. Both
directions are silent — no optimizer warning fires. The simplex null tracks the
two-group test within about two orders of magnitude in every case.

The two formulations also differ enormously in calibration. With the target
sitting exactly at the weighted mean of a heterogeneous rest (H0 true), rejection
rate at nominal 5% over 300 replicates:

| test | frac(p<0.05) | frac(p<0.01) | KS p |
|---|---:|---:|---:|
| two-group | 0.030 | 0.003 | 0.70 |
| K-group, simplex null | 0.050 | 0.007 | 0.83 |
| K-group, logit null | **0.513** | 0.300 | 2.7e-91 |

The logit null rejects half of all true nulls. Had the doc's Option A been
implemented as written, the resulting flood of hits would have been very easy to
read as the predicted power gain.

Junctions with zero reads in some cell type are the norm here, so this is not an
edge case. The fix is to take the weighted average **on the simplex** instead:

```
logit space (broken):  P = softmax(X @ A)   ->  A[target] = sum_k w_k A[k]
simplex     (used):    P = X @ softmax(A)   ->  psi_target = sum_k w_k psi_k
```

One line in the forward pass, and identical for one-hot rows, so only the null
model is affected. Convex combinations are bounded, and `P` is strictly positive
by construction, which also removes the `lgamma(0)` hazard. This is also the null
the doc's prose describes (`Ψ_A = Σ w_k Ψ_k`) — the doc's implementation section
silently switched it to logits. `null_space="logit"` is retained for comparison
only.

## 4. Cost: ~2x fewer fits, not 18x

The doc claims one K-group run covers all cell types, for 18x fewer fits. Under
the LRT the null constraint depends on which group is the target, so each target
needs its own null fit: 1 + T fits per intron group rather than 2T. For T = 18
that is 19 versus 36 — about 2x. `run_regression_multigroup` fits the
unconstrained model once and reuses it across all targets, so this saving is
realized. Only a Wald test would give one-fit-covers-all; the doc's Wald sketch
has a separate error (it inverts the `A`-block Hessian rather than taking the
`A` block of the inverse, ignoring `log_alpha` as a nuisance parameter, which
understates the variance).

## 5. What was validated

- **Exact reduction at K = 2.** p-values match `run_regression` to 5 significant
  figures, log-likelihoods to ~1e-9, with and without a real effect. At K = 2 the
  constraint becomes psi_target == psi_other, i.e. the existing test.
- **Calibration.** All groups identical: 0.057 vs. 0.057 for the two-group test
  at nominal 5% (KS p = 0.77). Heterogeneous rest with the target at the weighted
  mean: 0.050, KS p = 0.83.
- **No power loss** on target-specific effects: identical to the two-group test
  at every ΔΨ tested, including 0.717 vs. 0.717 at p<1e-4, ΔΨ = 0.05.
- **End-to-end plumbing:** filters, per-target BH, `psi` rows summing to 1,
  `n_jobs > 1` matching `n_jobs = 1`, and `find_marker_introns` compatibility.

Also fixed, unrelated: `_run_differential_splicing` crashed with
`KeyError: "None of ['index'] are in the columns"` whenever `adata.var.index` had
a name. Both paths now use `rename_axis("index")`.

## 6. Recommendation

The K-group refit is now correct, calibrated, and cheap, and it is the right tool
if the goal is an unbiased overdispersion estimate or per-cell-type PSI in one
pass. It is not the right tool for the stated goal of recovering shared-IN events
— the simulation says it will not materially change what passes FDR.

For that goal the empirical result already in hand is the stronger lead: pooled
IN vs EN gave +7.9% excess over null and 606 significant events, against +1.8%
and 68 for MGE vs rest. That gap is mostly effect-size dilution, which no
variance-model change addresses, because ΔΨ is identical under both models — the
doc says so itself. Changing the *reference group* is what recovers those events.
Worth considering instead:

1. Class-level one-vs-rest (IN vs all, EN vs all) alongside subtype-level, which
   is what the +7.9% result already measures.
2. A minimum-over-pairwise or nearest-different-class contrast, using the
   K-group fit here to supply all the per-group means from a single pass.

Two caveats on the negative result. It is simulation under the model's own
assumptions — equal α across groups, 3 junctions, no cell-level covariates — and
real data may deviate. And the simulated shared effect makes the target-like
groups exactly identical; in the real data MGE, CGE and STR are similar but not
identical, which sits between the "shared" and "specific" cases tested here. A
cheap check on real data: run both tests on a few hundred intron groups and
compare `alpha_target` from the K-group fit against the two-group fitted α. If
the ratio is near 1 for the events near your significance boundary, the negative
result transfers directly.
