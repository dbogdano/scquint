# Multi-Group DM: Implementation and Simulation Findings

Companion to `multigroup_dm_modification.md` (proposal) and
`multigroup_3prime_results.md` (real data). The proposal is implemented in
`scquint/differential_splicing.py` (`run_differential_splicing_multigroup`), with
three corrections to the design.

> **Superseded conclusion.** Section 2 below predicted essentially no power gain.
> The 3prime run contradicts it: +41% more significant intron groups (2,867 vs.
> 2,034 at q<0.05), and 0 significant events for both tests under permuted cell
> type labels, so the gain is not inflation. Section 2 is kept because the
> mechanism it measures is correct and because the reason it drew the wrong
> conclusion is worth recording — see §2a. The rest of this document, in
> particular the three design corrections in §3, stands.

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
also means the K-group model fixes *this* problem with a shared α — per-group
`α_k` is not the mechanism here.

That is not an argument for defaulting to `alpha_mode="shared"`, which is what
this section originally concluded. A shared α turns out to be anti-conservative
whenever the true overdispersion varies across cell types, and the default is now
`per_group`. See §5a.

## 2. In this simulation it did not buy power — and the simulation was wrong

**Superseded by the 3prime run; see §2a for why.** Within this simulation the
deflation scaled with the effect size, so it was nearly absent exactly where
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

## 2a. Why that conclusion was wrong

The simulated "shared effect" put three groups high and three low, so the
heterogeneity of the rest pool was **coupled to the size of the target's effect**.
A small ΔΨ therefore implied a nearly homogeneous rest pool, no concentration
deflation, and nothing for the K-group structure to correct. That is what
produced the flat power curve above — an artifact of the design, not a property
of the method.

Real data does not have that coupling. Testing IN-CTX-MGE pools EN, IN, RG and
glia, which differ from one another whatever MGE does, so the deflation is
present at every effect size including at the detection margin.

Re-simulating with the rest pool drawn independently of the target (18 groups,
α_true = 2 matching the fitted 1.90, ~2 reads/cell, 12,000 covered cells,
300 reps) confirms the power gain, and turns up something more consequential.

**Power**, target offset by ΔΨ from the weighted mean of a heterogeneous rest:

| ΔΨ | two-group p<.05 | K-group p<.05 | two-group p<1e-4 | K-group p<1e-4 |
|---|---:|---:|---:|---:|
| 0.03 | 0.703 | 0.810 | 0.227 | 0.217 |
| 0.06 | 0.983 | 1.000 | 0.740 | 0.933 |
| 0.10 | 1.000 | 1.000 | 1.000 | 1.000 |

**False positive rate**, target placed exactly at the weighted mean (H0 true):

| rest heterogeneity | two-group FPR | K-group FPR | two-group KS p | K-group KS p |
|---|---:|---:|---:|---:|
| mild (0.3) | 0.040 | 0.027 | 0.43 | 0.81 |
| moderate (1.0) | **0.180** | 0.047 | 5e-16 | 0.26 |
| strong (2.0) | **0.293** | 0.057 | 3e-39 | 0.95 |

The power gain reproduces. **The two-group FPR column does not generalize and
should not be relied on** — see §2b. This simulation's conditions (~3 reads/cell,
group sizes from 25 to 2,545, rest PSI drawn from a diffuse `Dirichlet(base*1.5)`)
are far more extreme than the real data, and a parametric bootstrap built from
the real fit finds the two-group test calibrated.

## 2b. The two-group inflation above does not hold on real data — withdrawn

A parametric bootstrap on 3prime (500 intron groups, 4 target cell types, 20
replicates each, 14,680 total; target PSI replaced by the cell-count-weighted
mean of the other groups; **real per-cell coverage retained**) finds both models
calibrated:

| | multi-group FPR | two-group FPR |
|---|---:|---:|
| overall (14,680 replicates) | 0.0525 | 0.0507 |

Stratifying by rest-pool heterogeneity shows no two-group inflation in any
quartile — the top quartile (max pairwise PSI distance > 0.53) gives two-group
FPR 0.051, and the correlation between heterogeneity and the two-group-minus-
multi-group FPR gap is *negative* (Spearman ρ = −0.19).

That bootstrap outranks the simulation above: it retains real coverage, real PSI
heterogeneity, real group sizes and real α. It is also structurally generous to
the inflation hypothesis, since it simulates from a shared-α DM built on the
K-group fit — the configuration in which the K-group model is correctly specified
and the two-group model is the misspecified one. The inflation still did not
appear.

So the +41% is a **power gain**, not a correction of two-group false positives,
and the 76 events unique to the two-group model should not be presumed false.

The simulated inflation is presumably a real phenomenon at extreme sparsity and
extreme heterogeneity; it just is not the regime 3prime occupies.

Three inflation hypotheses were checked against the real data and all failed:

| hypothesis | check | result |
|---|---|---|
| p-values anti-conservative | permute cell type labels | 0 sig for both; raw p<0.05 of 4.93% (multi) and 5.37% (two-group) |
| gain is reference redefinition | `weights="reads"` vs `"cells"` | 153 vs 151 sig, ρ = 0.9995 |
| gain driven by collapsed α fits | stratify by α | gained events *depleted* for low α (0.88×) |

Sparse nuisance parameters were also ruled out in simulation: with 18 groups and
as few as 8 covered cells in the smallest group, the false positive rate matches
the two-group test to three decimals (0.053/0.053, 0.050/0.050, 0.063/0.063,
0.037/0.037). Refitting the null from perturbed starts improves the
log-likelihood by 0.0000, so the constrained null is not under-converging.

### A limitation of the permutation check

Shuffling cell type labels makes every group's PSI identical, which is precisely
the regime where **both** tests are calibrated — the identical-groups rows above
show two-group and K-group FPRs matching to three decimals. So the permutation
result establishes that the K-group test is not inflated under a global null, but
it cannot speak to behaviour under heterogeneity, because it destroys the
heterogeneity. The parametric bootstrap in §2b is what covers that gap, and it
came out clean for both models.

The diagnostic that does preserve heterogeneity is the parametric bootstrap null
reported in §2b: simulate from the K-group fit — per-cell-type PSI,
per-intron-group α, real coverage — with the target's PSI replaced by the
weighted mean of the others, so H0 holds by construction while realistic
heterogeneity, sparsity and overdispersion are retained.

Worth recording a tempting alternative that does *not* work: a split-half control
(halve one cell type, test one half vs. rest) leaves the other real cell types in
the rest pool, so the one-vs-rest null is false for that half regardless.

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

## 5a. The shared-alpha assumption is the real vulnerability

Biomni's per-group-alpha simulation found the K-group model anti-conservative
when overdispersion varies across cell types. This replicates, more strongly than
reported, and — importantly — `alpha_mode="per_group"` fixes it completely.

DGP has per-group alpha (log-uniform over the stated range, target fixed at 2);
both models otherwise 500 cells/group, Poisson(20), 3 junctions, 200 reps. FPR at
nominal 0.05:

| DGP alpha | n_rest | `alpha_mode="shared"` | `alpha_mode="per_group"` |
|---|---:|---:|---:|
| shared (2.0) | 3 | 0.020 | 0.015 |
| shared (2.0) | 5 | 0.050 | 0.040 |
| per-group (0.5–5) | 3 | **0.420** | 0.040 |
| per-group (0.5–5) | 5 | **0.255** | 0.025 |
| per-group (0.5–5) | 10 | **0.165** | 0.050 |
| per-group (0.5–5) | 17 | **0.125** | 0.035 |
| extreme (0.3–10) | 3 | **0.660** | 0.055 |
| extreme (0.3–10) | 5 | **0.450** | 0.025 |
| extreme (0.3–10) | 10 | **0.320** | 0.040 |
| extreme (0.3–10) | 17 | **0.205** | 0.065 |

Two conclusions.

**The inflation does not vanish with many groups.** It decays monotonically in
the number of rest groups but is still 0.205 and 0.125 at n_rest = 17 — the
3prime configuration — i.e. 2.5–4x nominal. Biomni's §3 point 3, that 10+ rest
groups are safe, does not replicate: n_rest = 10 gives 0.320 and 0.165.

**`per_group` is calibrated everywhere and costs nothing.** It sits at nominal
under both a shared-alpha DGP and a per-group-alpha DGP, so there is no
bias–robustness tradeoff to weigh. The earlier recommendation in §1 to default to
`alpha_mode="shared"` was wrong: it was argued from parameter count and
small-group stability without ever testing it against a per-group-alpha DGP.

Note this table's `mg shared` vs `mg per_group` contrast is internally valid —
both use the same H0 construction, which is the K-group test's own null. The
two-group column from the same runs is *not* usable, for the reason in §2b.

### At 3prime's sparsity the exposure is much smaller

The table above uses Poisson(20) coverage. Repeating it at 3prime's actual
sparsity and group-size distribution (18 groups, 25–2,545 cells, alpha drawn
log-uniform over (0.3, 10), target fixed at 2) changes the magnitude a lot:

| DGP alpha | reads/cell | `shared` FPR | `per_group` FPR | `shared` power | `per_group` power |
|---|---:|---:|---:|---:|---:|
| shared | 2 | 0.060 | 0.070 | 0.990 | 0.990 |
| shared | 20 | 0.070 | 0.055 | 1.000 | 1.000 |
| extreme | 2 | 0.100 | 0.090 | 0.980 | 0.995 |
| extreme | 20 | **0.595** | 0.080 | 0.930 | 1.000 |

Sparsity *masks* the misspecification: with ~3 reads per cell there is not enough
information to resolve the differing alphas, so `shared` sits at 0.100 rather
than 0.595. So 3prime's worst-case exposure is roughly 2x nominal, not the
4–12x the Poisson(20) table implies — and only under an extreme alpha spread that
has not been shown to exist in the data.

Two consequences:

**`per_group` is now the default.** It was never worse than `shared` on either
false positive rate or power in any configuration tested, and is dramatically
better in one. Power is equal or slightly higher (0.995 vs. 0.980, 1.000 vs.
0.930). Pass `alpha_mode="shared"` to reproduce the original 3prime run.

**`per_group` is not a complete fix at extreme sparsity.** Under extreme
per-group alpha at 2 reads/cell both modes sit near 0.09–0.10. Whatever residual
inflation exists there is a property of the information available, not of the
alpha parameterization.

Whether any of this touches 3prime still depends on the real alpha spread across
cell types, which has not been measured — see §8 item 1.

## 6. Recommendation

The K-group refit is correct, cheap, and — per the 3prime run — delivers a real
+41% gain in significant intron groups, with the largest proportional gains in
the cell types whose "rest" pool is most similar to them. Use it, but with
`alpha_mode="per_group"` rather than `"shared"` (see §5a); `weights="cells"` and
`null_space="simplex"` are fine as they stand. Pass `alpha_mode="shared"`
explicitly to reproduce the original 3prime run.

It is still not a substitute for changing the reference group. The pooled IN vs
EN test gave 606 significant events against 77 for MGE vs rest in the full
multi-group run, and that gap is effect-size dilution, which no variance-model
change addresses because ΔΨ is nearly identical under both models (ρ = 0.96 on
real data). Both remain worth doing:

1. Class-level one-vs-rest (IN vs all, EN vs all) alongside subtype-level.
2. A minimum-over-pairwise or nearest-different-class contrast, using the K-group
   fit to supply all per-group means from a single pass.

## 7. Two open items

**The near-boundary analysis in `multigroup_3prime_results.md` §3 is biased and
should not be cited.** It selects events on `1e-4 < p_two-group < 0.05`, then
scores the two-group model with the same statistic used to select. The two-group
BH threshold implied by 2,034/71,250 significant is ≈ 1.4e-3, so within that
window the two-group model can only pass the slice below 1.4e-3 — the ~7,900
events above it are guaranteed failures by construction. The multi-group
p-values in the window are unconstrained. The permutation result is the sound
evidence for the same claim and should replace it.

**Low fitted concentrations.** About 25% of 3prime tests have α < 0.01, meaning
each cell's usage is effectively a single junction. This is a plausible MLE for
sparse data rather than a broken fit, and it does not inflate anything — those
tests are *less* likely to reach significance, and gained events are depleted for
them (0.88x). But `opt_warning` only fires when `ll_null > ll_alt` and so never
flags this, which was a real gap in the diagnostics. There is now a `low_alpha`
column (threshold `LOW_ALPHA_THRESHOLD = 1e-2`) so these tests can be identified
and weighted accordingly.

## 8. Benchmarking: what has been run, and what is still needed

### Already run

| check | where | outcome |
|---|---|---|
| Exact reduction at K=2 | §5 | p-values match `run_regression` to 5 s.f. |
| Calibration, homogeneous groups | §5 | 0.057 vs. 0.057 (two-group) |
| Calibration, 18 sparse groups | §2a | FPR matches two-group to 3 decimals, down to 8 covered cells in the smallest group |
| Null under-convergence | §2a | restarts improve ll by 0.0000 |
| Label permutation on real data | §2a | 0 sig for both; raw p<0.05 of 4.93% / 5.37% |
| Parametric bootstrap on real data | §2b | both calibrated; 0.0525 / 0.0507 over 14,680 replicates |
| Weighting sensitivity | §2a | 153 vs. 151 sig, ρ = 0.9995 |
| Low-α stratification | §7 | gained events depleted for low α (0.88x) |
| Per-group-α DGP, `shared` vs `per_group` | §5a | `shared` inflated to 0.125–0.660; `per_group` at nominal |

### Still needed, in priority order

1. **Per-group α spread on real data.** Fit a DM per cell type per intron group
   and look at the dispersion of α across cell types. This is the single number
   that decides whether the 3prime run is affected by §5a; everything in §5a is
   conditional on it. Narrow spread → 3prime stands. Wide spread → the
   shared-α run is anti-conservative and the +41% needs re-deriving.

2. **Re-run 3prime with `alpha_mode="per_group"`.** One run, and it is both the
   diagnostic and the remedy: if 2,867 holds, the shared-α result was fine
   anyway; if it drops materially, `per_group` is the number to report. Cheaper
   than (1) and answers the same practical question.

3. **Per-group-α parametric bootstrap.** The bootstrap in §2b simulates from a
   shared-α DM, so it is structurally blind to §5a. Redo it drawing per-group α
   from the spread measured in (1).

4. **Cross-implementation reconciliation.** My two-group FPR (0.24–0.91) and
   Biomni's (~0.05) disagree qualitatively, not by a tunable parameter. Run both
   implementations on *identical* simulated `y`/`codes` arrays. Same p-values →
   the difference is data generation, and the arithmetic-mean-vs-KL point in §2b
   is the likely cause. Different p-values → one implementation has a bug.

5. **5prime run**, once (1)–(2) settle which `alpha_mode` to use.

6. **S4 (coverage-stratified Storey) on the multi-group p-values**, which gave
   +33% over BH on the two-group results and may compound.
