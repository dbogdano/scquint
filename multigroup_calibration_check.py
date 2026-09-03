#!/usr/bin/env python
"""
Calibration diagnostics for the multi-group DM test on real data.

The +41% gain in significant events reported in multigroup_3prime_results.md is
consistent with two very different explanations: the K-group variance structure
genuinely has more power, or its p-values are systematically shifted down. No
per-cell-type breakdown of the real results distinguishes them, because both
produce more hits concentrated in the cell types with the most heterogeneous
"rest" pool.

Permutation does distinguish them. Shuffling the cell type labels destroys all
real cell-type structure, so every rejection is a false positive by construction.
A correct test returns ~0 significant events at q<0.05 no matter how many
parameters it fits.

Three checks:

  1. PERMUTATION. Shuffle labels, run both models on the same intron groups,
     count q<0.05. Both should be ~0. If the multi-group count is materially
     higher, the gain on real labels is inflation.

  2. DEGENERATE CONCENTRATION. Count tests whose fitted alpha collapsed toward 0.
     The DM likelihood is meaningless there and the LRT is not interpretable.
     `opt_warning` does NOT catch this - it only fires when ll_null > ll_alt.

  3. WEIGHTING SENSITIVITY. The default weights="cells" builds the reference as
     sum_k (n_k / sum n) * psi_k, while the two-group pooled fit is effectively
     read-weighted. Cell types differ in reads per cell, so these are different
     reference definitions. Re-running with weights="reads" shows how much of the
     gain is reference redefinition rather than variance.

Usage
-----
    python multigroup_calibration_check.py \
        --h5ad /mnt/user-uploads/3prime_grouped.h5ad \
        --obs-key SCANVI_cell_types_simplified \
        --n-intron-groups 1500 --n-jobs 8

On ~1500 of 14352 intron groups this takes roughly a tenth of the full 39.5 min
run per model fit, so all checks together land in well under an hour.
"""
import argparse
import contextlib
import io
import sys
import time

import anndata
import numpy as np
import pandas as pd

from scquint.differential_splicing import (
    run_differential_splicing_multigroup,
    run_differential_splicing_for_each_group,
)


def subset_intron_groups(adata, n, seed):
    groups = pd.unique(adata.var.intron_group.values)
    if n is None or n >= len(groups):
        return adata, len(groups)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(groups, size=n, replace=False).tolist())
    mask = np.array([g in keep for g in adata.var.intron_group.values])
    return adata[:, mask], n


def n_sig(df, q=0.05):
    if len(df) == 0 or "p_value_adj" not in df:
        return 0
    return int((df.p_value_adj < q).sum())


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--obs-key", default="SCANVI_cell_types_simplified")
    ap.add_argument("--n-intron-groups", type=int, default=1500)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-two-group", action="store_true",
                    help="skip the two-group permutation arm (the slow half)")
    ap.add_argument("--out-prefix", default="multigroup_calibration")
    args = ap.parse_args()

    filt = dict(min_cells_per_intron_group=30, min_total_cells_per_intron=30,
                min_global_proportion=1e-3, n_jobs=args.n_jobs)
    mg_filt = dict(filt, min_cells_per_target=30, min_cells_per_group=10)

    print(f"loading {args.h5ad}")
    adata = anndata.read_h5ad(args.h5ad)
    print(f"  {adata.shape}, {adata.obs[args.obs_key].nunique()} groups in "
          f"{args.obs_key!r}")
    adata, n_ig = subset_intron_groups(adata, args.n_intron_groups, args.seed)
    print(f"  subset to {n_ig} intron groups -> {adata.shape}")

    real = adata.obs[args.obs_key].values
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(np.asarray(real))
    adata.obs["_perm_group"] = pd.Categorical(perm)
    assert not np.array_equal(np.asarray(real), perm), "permutation was a no-op"

    results = {}

    # ---------------------------------------------------------- 1. permutation
    print("\n" + "=" * 74)
    print("1. PERMUTATION: labels shuffled, so every hit is a false positive")
    print("=" * 74)

    t0 = time.time()
    ig_perm, _ = quiet(run_differential_splicing_multigroup, adata, "_perm_group",
                       **mg_filt)
    results["multigroup_perm"] = ig_perm
    tested = int(ig_perm.tested.sum()) if "tested" in ig_perm else len(ig_perm)
    print(f"  multi-group : {n_sig(ig_perm):6d} sig / {tested} tested   "
          f"({time.time()-t0:.0f}s)")

    if not args.skip_two_group:
        t0 = time.time()
        ig_perm2, _ = quiet(run_differential_splicing_for_each_group, adata,
                            "_perm_group", **filt)
        results["twogroup_perm"] = ig_perm2
        print(f"  two-group   : {n_sig(ig_perm2):6d} sig / {len(ig_perm2)} tested   "
              f"({time.time()-t0:.0f}s)")

    p = ig_perm.loc[ig_perm.tested, "p_value"].values if "tested" in ig_perm \
        else ig_perm.p_value.values
    p = p[np.isfinite(p)]
    if len(p):
        print(f"\n  multi-group permuted p-values: frac(p<0.05)={np.mean(p<0.05):.4f} "
              f"frac(p<0.01)={np.mean(p<0.01):.4f} frac(p<1e-4)={np.mean(p<1e-4):.5f}")
        print("  (expected 0.05 / 0.01 / 0.0001 for a calibrated test)")
    print("\n  VERDICT: if multi-group >> two-group here, the real-data gain is")
    print("  inflation. If both are ~0, the gain is real.")

    # ------------------------------------------------ 2. degenerate concentration
    print("\n" + "=" * 74)
    print("2. DEGENERATE CONCENTRATION on real labels")
    print("=" * 74)
    t0 = time.time()
    ig_real, in_real = quiet(run_differential_splicing_multigroup, adata,
                             args.obs_key, **mg_filt)
    results["multigroup_real"] = ig_real
    t = ig_real[ig_real.tested] if "tested" in ig_real else ig_real
    for thr in [1e-6, 0.01, 0.1]:
        bad = t[t.alpha_target < thr]
        sig_bad = int((bad.p_value_adj < 0.05).sum()) if len(bad) else 0
        print(f"  alpha_target < {thr:<8g}: {len(bad):6d} tests "
              f"({100*len(bad)/max(len(t),1):5.2f}%), {sig_bad} of them q<0.05")
    print(f"  alpha_target quantiles: "
          f"{np.nanpercentile(t.alpha_target, [1, 25, 50, 75, 99]).round(3)}")
    print(f"  real-label sig: {n_sig(ig_real)} / {len(t)} tested   "
          f"({time.time()-t0:.0f}s)")

    # ---------------------------------------------------- 3. weighting sensitivity
    print("\n" + "=" * 74)
    print("3. WEIGHTING SENSITIVITY: cells vs reads")
    print("=" * 74)
    t0 = time.time()
    ig_reads, _ = quiet(run_differential_splicing_multigroup, adata, args.obs_key,
                        weights="reads", **mg_filt)
    results["multigroup_reads"] = ig_reads
    print(f"  weights='cells': {n_sig(ig_real):6d} sig")
    print(f"  weights='reads': {n_sig(ig_reads):6d} sig   ({time.time()-t0:.0f}s)")
    key = ["intron_group", "test_group"] if "test_group" in ig_real else ["intron_group"]
    a = ig_real.reset_index()[key + ["p_value", "max_abs_delta_psi"]]
    b = ig_reads.reset_index()[key + ["p_value", "max_abs_delta_psi"]]
    m = a.merge(b, on=key, suffixes=("_cells", "_reads"))
    if len(m):
        print(f"  merged {len(m)} tests; Spearman(p) = "
              f"{m.p_value_cells.corr(m.p_value_reads, method='spearman'):.4f}")
        print(f"  median max_abs_delta_psi: cells="
              f"{m.max_abs_delta_psi_cells.median():.4f} "
              f"reads={m.max_abs_delta_psi_reads.median():.4f}")
        print("  A large gap here means part of the real-data gain is the")
        print("  reference being redefined, not the variance being better.")

    for name, df in results.items():
        path = f"{args.out_prefix}_{name}.csv"
        df.to_csv(path)
        print(f"\nwrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    sys.exit(main())
