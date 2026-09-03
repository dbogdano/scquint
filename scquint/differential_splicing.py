import anndata
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from pyro.distributions import Dirichlet, DirichletMultinomial, Gamma, Multinomial
import scanpy as sc
from scipy.special import softmax
from scipy.stats import chi2, mannwhitneyu
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .data import make_intron_group_summation_cpu, filter_min_cells_per_feature, filter_min_cells_per_intron_group, regroup, filter_min_global_proportion


# original: from skbio.stats.composition import closure
def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])
    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


# original function: from skbio.stats.composition import alr
def alr(mat, denominator_idx=0):
    r"""
    Performs additive log ratio transformation.
    This function transforms compositions from a D-part Aitchison simplex to
    a non-isometric real space of D-1 dimensions. The argument
    `denominator_col` defines the index of the column used as the common
    denominator. The :math: `alr` transformed data are amenable to multivariate
    analysis as long as statistics don't involve distances.
    :math:`alr: S^D \rightarrow \mathbb{R}^{D-1}`
    The alr transformation is defined as follows
    .. math::
        alr(x) = \left[ \ln \frac{x_1}{x_D}, \ldots,
        \ln \frac{x_{D-1}}{x_D} \right]
    where :math:`D` is the index of the part used as common denominator.
    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components
    denominator_idx: int
       the index of the column (2D-matrix) or position (vector) of
       `mat` which should be used as the reference composition. By default
       `denominator_idx=0` to specify the first column or position.
    Returns
    -------
    numpy.ndarray
         alr-transformed data projected in a non-isometric real space
         of D-1 dimensions for a D-parts composition
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import alr
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    """
    mat = closure(mat)
    if mat.ndim == 2:
        mat_t = mat.T
        numerator_idx = list(range(0, mat_t.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat_t[numerator_idx, :]/mat_t[denominator_idx, :]).T
    elif mat.ndim == 1:
        numerator_idx = list(range(0, mat.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat[numerator_idx]/mat[denominator_idx])
    else:
        raise ValueError("mat must be either 1D or 2D")
    return lr



def lrtest(llmin, llmax, df):
    lr = 2 * (llmax - llmin)
    p = chi2.sf(lr, df)
    return p


def normalize(x):
    return x / sum(x)


def run_regression(args):
    intron_group, y, cell_idx_a, cell_idx_b = args
    cells_to_use = np.where(y.sum(axis=1) > 0)[0]
    y = y[cells_to_use]
    n_cells, n_classes = y.shape
    n_covariates = 2
    cell_mask_a = np.isin(cells_to_use, cell_idx_a)
    cell_mask_b = np.isin(cells_to_use, cell_idx_b)
    x = np.ones((n_cells, 2), dtype=float)
    x[cell_mask_a, 1] = 0
    x_null = np.expand_dims(x[:, 0], axis=1)

    pseudocounts = 10.0
    init_A_null = np.expand_dims(alr(y.sum(axis=0) + pseudocounts, denominator_idx=-1), axis=0)
    model_null = lambda: DirichletMultinomialGLM(1, n_classes, init_A=init_A_null)

    ll_null, model_null = fit_model(model_null, x_null, y)
    init_A = np.zeros((2, n_classes - 1), dtype=float)
    init_A[0] = alr(y[cell_mask_a].sum(axis=0) + pseudocounts, denominator_idx=-1)
    init_A[1] = alr(y[cell_mask_b].sum(axis=0) + pseudocounts, denominator_idx=-1) - init_A[0]
    model = lambda: DirichletMultinomialGLM(2, n_classes, init_A=init_A)
    ll, model = fit_model(model, x, y)
    if ll+1e-2 < ll_null:
        raise Exception(f"WARNING: optimization failed for intron_group {intron_group}. ll_null={ll_null} ll_full={ll}")
    p_value = lrtest(ll_null, ll, n_classes - 1)
    A = model.get_full_A().cpu().detach().numpy()
    log_alpha = model.log_alpha.cpu().detach().numpy()

    conc = np.exp(log_alpha)
    beta = A.T
    psi1 = normalize(conc * softmax(beta[:, 0]))
    psi2 = normalize(conc * softmax(beta.sum(axis=1)))
    if np.isnan(p_value): p_value = 1.0

    df_intron_group = pd.DataFrame(dict(intron_group=[intron_group], p_value=[p_value], ll_null=[ll_null], ll=[ll], n_classes=[n_classes]))
    df_intron = pd.DataFrame(dict(psi_a=psi1, psi_b=psi2))

    return df_intron_group, df_intron


def _run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    device="cpu",
    min_cells_per_intron_group=30,
    min_total_cells_per_intron=30,
    n_jobs=None,
    do_regroup=False,
    min_global_proportion=1e-3,
):
    n_a = len(cell_idx_a)
    n_b = len(cell_idx_b)
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    adata = adata[cell_idx_all].copy()
    cell_idx_a = np.arange(0, n_a)
    cell_idx_b = np.arange(n_a, n_a + n_b)
    print(adata.shape)
    if min_total_cells_per_intron is not None:
        adata = filter_min_cells_per_feature(adata, min_total_cells_per_intron)
        print(adata.shape)
    if min_global_proportion is not None:
        adata = filter_min_global_proportion(adata, min_global_proportion)
        print(adata.shape)
    if do_regroup:
        adata = regroup(adata)
        print(adata.shape)
    if min_cells_per_intron_group is not None:
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_a)
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_b)
        print(adata.shape)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()

    print("Number of intron groups: ", len(adata.var.intron_group.unique()))
    print("Number of introns: ", len(adata.var))

    intron_groups = adata.var.intron_group.values
    all_intron_groups = pd.unique(intron_groups)
    intron_group_introns = defaultdict(list)
    for i, c in enumerate(intron_groups):
        intron_group_introns[c].append(i)

    X = adata.X.toarray()  # for easier parallelization using Python's libraries

    if n_jobs is not None and n_jobs != 1:
        dfs_intron_group, dfs_intron = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(run_regression)((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
                for c in tqdm(all_intron_groups)
            )
        )
    else:
        dfs_intron_group, dfs_intron = zip(*[
            run_regression((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
            for c in tqdm(all_intron_groups)
        ])
    df_intron_group = pd.concat(dfs_intron_group, ignore_index=True)
    df_intron = pd.concat(dfs_intron, ignore_index=True)
    positions = np.concatenate([intron_group_introns[c] for c in all_intron_groups])
    # rename_axis so this works whether or not adata.var.index is named
    var = adata.var.iloc[positions].rename_axis("index").reset_index(drop=False)
    df_intron = pd.concat([var, df_intron], axis=1).set_index("index")
    return df_intron_group, df_intron


class MultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes):
        super(MultinomialGLM, self).__init__()
        self.A = nn.Parameter(torch.zeros((n_covariates, n_classes-1), dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        A = self.get_full_A()
        logits = X @ A
        return logits

    def loss_function(self, X, Y):
        logits = self.forward(X)
        ll = Multinomial(logits=logits).log_prob(Y).sum()
        self.ll = ll
        if torch.isnan(ll):
            print("A: ", self.A)
            print("ll: ", ll)
            raise Exception("debug")
        return -ll


class DirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, init_A=None, init_log_alpha=None):
        super(DirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(1) * 1.0
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.double))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=torch.double))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=torch.double))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        P = torch.softmax(X @ A, dim=1)
        concentration = torch.mul(alpha, P)
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration, validate_args=False).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
        )
        self.ll = ll
        return res


class MultiGroupDirichletMultinomialGLM(nn.Module):
    """
    Dirichlet-multinomial GLM allowing the concentration (overdispersion)
    parameter to vary by group.

    Same likelihood as `DirichletMultinomialGLM`, except that `log_alpha` may be
    a vector, with `alpha_group_idx` assigning each cell to one of the
    concentration parameters. With `n_alpha_groups=1` this is exactly
    `DirichletMultinomialGLM`.

    The concentration assignment is deliberately separate from the design matrix
    `X`, because in the constrained null model (see `run_regression_multigroup`)
    the design row of the target group's cells is a weighted average of the other
    groups' rows, while those cells still keep their own concentration parameter.

    `mixture` selects where that weighted average is taken:

        mixture=False:  P = softmax(X @ A)   - average of the logits
        mixture=True:   P = X @ softmax(A)   - average of the probabilities

    For a one-hot row the two are identical, so this only matters for the
    constrained null. `mixture=True` is strongly preferred there: averaging logits
    is unbounded, so a group with a zero-count junction (whose logit diverges) can
    drag the average anywhere, which makes the null either unrestrictive or
    absurd. Averaging on the simplex is bounded. Rows of `X` must be non-negative
    and sum to 1 for `mixture=True` to yield a valid probability vector; one-hot
    and weight rows both satisfy this.
    """

    def __init__(
        self,
        n_covariates,
        n_classes,
        init_A=None,
        init_log_alpha=None,
        alpha_group_idx=None,
        n_alpha_groups=1,
        mixture=False,
    ):
        super(MultiGroupDirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        self.n_alpha_groups = n_alpha_groups
        self.mixture = mixture
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(n_alpha_groups) * 1.0
        if alpha_group_idx is None:
            alpha_group_idx = np.zeros(1, dtype=int)
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.double))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.register_buffer("alpha_group_idx", torch.tensor(alpha_group_idx, dtype=torch.long))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=torch.double))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=torch.double))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        if self.mixture:
            P = X @ torch.softmax(A, dim=1)
        else:
            P = torch.softmax(X @ A, dim=1)
        if self.n_alpha_groups == 1:
            concentration = alpha * P
        else:
            concentration = alpha[self.alpha_group_idx].unsqueeze(1) * P
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration, validate_args=False).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
        )
        self.ll = ll
        return res


def fit_model(model_initializer, X, Y, device="cpu"):
    X = torch.tensor(X, dtype=torch.double, device=device)
    Y = torch.tensor(Y, dtype=torch.double, device=device)

    initial_lr = 1.0

    def try_optimization(lr):
        model = model_initializer()
        model.to(device)
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=10000, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = model.loss_function(X, Y)
            if torch.isnan(loss):
                raise ValueError("nan encountered")
            loss.backward()
            return loss

        optimizer.step(closure)
        return model.ll.cpu().detach().numpy(), model

    lr = initial_lr
    try_number = 0
    while True:
        try_number += 1
        if try_number > 10:
            print("WARNING: optimization failed, too many tries")
            return -np.inf, model_initializer()
        try:
            ll, model = try_optimization(lr)
            break
        except ValueError as ve:
            lr /= 10.0

    return ll, model


def run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    **kwargs
):
    print("sample sizes: ", len(cell_idx_a), len(cell_idx_b))

    df_intron_group, df_intron = _run_differential_splicing(
        adata,
        cell_idx_a,
        cell_idx_b,
        **kwargs,
    )
    if len(df_intron_group) == 0: return df_intron_group, df_intron
    df_intron["delta_psi"] = df_intron["psi_a"] - df_intron["psi_b"]
    df_intron["lfc_psi"] = np.log2(df_intron["psi_a"] + 1e-9) - np.log2(df_intron["psi_b"] + 1e-9)
    df_intron["abs_delta_psi"] = df_intron.delta_psi.abs()
    df_intron["abs_lfc_psi"] = df_intron.lfc_psi.abs()

    if "gene_id" not in df_intron.columns:
        df_intron["gene_id"] = "NA"
    if "gene_name" not in df_intron.columns:
        df_intron["gene_name"] = "NA"
    groupby = df_intron.groupby("intron_group").agg({"gene_id": "first", "gene_name": "first", "abs_delta_psi": "max", "abs_lfc_psi": "max"})
    groupby = groupby.rename(columns={"abs_delta_psi": "max_abs_delta_psi", "abs_lfc_psi": "max_abs_lfc_psi"})
    df_intron_group = df_intron_group.set_index("intron_group").merge(groupby, left_index=True, right_index=True, how="inner")
    df_intron_group = df_intron_group.sort_values(by="p_value")
    df_intron_group["ranking"] = np.arange(len(df_intron_group))

    reject, pvals_corrected, _, _ = multipletests(
        df_intron_group.p_value.values, 0.05, "fdr_bh"
    )
    df_intron_group["p_value_adj"] = pvals_corrected

    return df_intron_group, df_intron


# Fitted concentrations below this are reported via the `low_alpha` column; see
# where it is set in run_regression_multigroup.
LOW_ALPHA_THRESHOLD = 1e-2


def _group_weights(n_cells_per_group, reads_per_group, target_local, mode):
    """
    Weights defining the "rest" reference in the multi-group contrast. The target
    group gets weight 0; the others sum to 1.
    """
    if mode == "cells":
        base = np.asarray(n_cells_per_group, dtype=float)
    elif mode == "reads":
        base = np.asarray(reads_per_group, dtype=float)
    elif mode == "equal":
        base = np.ones(len(n_cells_per_group), dtype=float)
    else:
        raise ValueError(f"unknown weights mode: {mode}")
    w = base.copy()
    w[target_local] = 0.0
    total = w.sum()
    if total <= 0:
        return None
    return w / total


def _build_constrained_null_design(group_codes, K, target_local, w):
    """
    Design matrix for the null hypothesis that the target group's PSI equals the
    weighted average of the other groups' PSI.

    Rather than penalizing or dropping a column, the constraint is imposed by
    reparameterization: the non-target groups get one-hot columns, and the target
    group's cells get a design row equal to `w` over those same K-1 columns. The
    result is nested in the unconstrained K-group model with `n_classes - 1` fewer
    free parameters.

    Whether this averages probabilities or logits is decided by the model's
    `mixture` flag, since the same matrix serves both:

        mixture=True  (P = X @ softmax(A)):  psi_target == sum_k w_k * psi_k
        mixture=False (P = softmax(X @ A)):  A[target] == sum_k w_k * A[k]

    Use mixture=True. The logit version is ill-posed on real data: a non-target
    group with a zero-count junction has an unbounded logit, so the weighted
    average can be driven anywhere, and the test either collapses (LRT -> 0, the
    null fits as well as the alternative) or explodes (reference PSI driven to a
    corner of the simplex, giving huge but meaningless statistics).

    NOTE: dropping the target's column instead (leaving its cells with an all-zero
    design row) would force logits to 0, i.e. uniform PSI - a completely
    different, and much easier to reject, null.
    """
    other = np.array([k for k in range(K) if k != target_local])
    is_target = group_codes == target_local
    x_null = np.zeros((len(group_codes), K - 1), dtype=float)
    col_of = np.full(K, -1, dtype=int)
    col_of[other] = np.arange(K - 1)
    nt = np.where(~is_target)[0]
    x_null[nt, col_of[group_codes[nt]]] = 1.0
    x_null[is_target, :] = w[other]
    return x_null, other


def _empty_multigroup_result(intron_group, target_name, n_classes):
    df_intron_group = pd.DataFrame(dict(
        intron_group=[intron_group], test_group=[target_name], p_value=[1.0],
        ll_null=[np.nan], ll=[np.nan], n_classes=[n_classes], n_groups=[0],
        n_cells_target=[0], alpha_target=[np.nan], alpha_rest=[np.nan],
        low_alpha=[False], opt_warning=[False], tested=[False],
    ))
    df_intron = pd.DataFrame(dict(
        test_group=[target_name] * n_classes,
        psi_a=[np.nan] * n_classes,
        psi_b=[np.nan] * n_classes,
        psi_b_geom=[np.nan] * n_classes,
    ))
    return df_intron_group, df_intron


def run_regression_multigroup(args):
    """
    Multi-group differential splicing test for one intron group.

    Fits one Dirichlet-multinomial mean per group (rather than target vs. a pooled
    "rest"), then tests, for each requested target group, the contrast

        H0: psi_target == sum_{k != target} w_k * psi_k
        H1: psi_target is free

    by likelihood ratio with `n_classes - 1` degrees of freedom - the same df as
    the two-group test in `run_regression`.

    The point of the K-group mean structure is variance estimation, not the effect
    size: in the two-group model, systematic differences *between* the cell types
    pooled into "rest" have nowhere to go but the residual, which drags the shared
    concentration parameter down (more overdispersion), widening the likelihood
    and shrinking the LRT statistic for every event shared across several cell
    types. Giving each group its own mean moves that variation into the mean
    structure.

    That deflation is real and large - in simulation with a true concentration of
    20, the two-group fit returns 20.3 when no effect is present and 2.4 for a
    strong shared effect, while the K-group fit returns ~20 throughout.

    On the 3prime data (74,327 cells, 18 cell types, 14,352 intron groups) this
    yields 2,867 significant intron groups at q<0.05 against 2,034 for the
    two-group test on the same tests, +41%, with the largest proportional gains in
    the interneuron subtypes whose "rest" pool contains the most similar cell
    types. Under permuted cell type labels both tests return 0 significant events
    with raw p<0.05 rates of 4.93% and 5.37%, so the gain is not inflation.

    The unconstrained model is fit once and reused across all targets, so testing
    T targets costs 1 + T fits rather than 2T.

    args is a tuple of:
        intron_group: intron group id
        y: (n_cells, n_classes) read counts, all cells, unpermuted
        group_codes: (n_cells,) int array in [0, n_groups)
        targets: list of int group codes to test
        n_groups: total number of groups
        opts: dict with keys alpha_mode ("shared" | "per_group"), weights
            ("cells" | "reads" | "equal"), null_space ("simplex" | "logit"),
            min_cells_per_target, min_cells_per_group, group_names, device

    Returns (df_intron_group, df_intron); both carry a `test_group` column with
    one block of rows per target.
    """
    intron_group, y, group_codes, targets, n_groups, opts = args
    alpha_mode = opts.get("alpha_mode", "shared")
    weights_mode = opts.get("weights", "cells")
    min_cells_per_target = opts.get("min_cells_per_target", 30)
    min_cells_per_group = opts.get("min_cells_per_group", 10)
    null_space = opts.get("null_space", "simplex")
    group_names = opts.get("group_names")
    device = opts.get("device", "cpu")
    pseudocounts = 10.0

    def name_of(code):
        return group_names[code] if group_names is not None else code

    cells_to_use = np.where(y.sum(axis=1) > 0)[0]
    y = y[cells_to_use]
    codes = np.asarray(group_codes)[cells_to_use]
    n_cells, n_classes = y.shape

    # Coverage is sparse, so in any given intron group some groups will have only
    # a handful of cells with reads. Giving those their own free mean row would
    # let it fit those few cells almost exactly, which biases the concentration
    # upward (too little apparent overdispersion) and makes the test
    # anti-conservative. Such groups are merged into a single pooled bucket, which
    # is never a target. Keeping min_cells_per_group <= min_cells_per_target means
    # any target that survives the target threshold always keeps its own row.
    counts = np.bincount(codes, minlength=n_groups)
    present = np.where(counts > 0)[0]
    keep = present if min_cells_per_group is None else np.where(counts >= min_cells_per_group)[0]
    pooled = np.setdiff1d(present, keep)
    n_real = len(keep)
    K = n_real + (1 if len(pooled) > 0 else 0)
    local_of = np.full(n_groups, -1, dtype=int)
    local_of[keep] = np.arange(n_real)
    if len(pooled) > 0:
        local_of[pooled] = n_real
    codes_local = local_of[codes]

    if K < 2 or n_cells == 0:
        results = [_empty_multigroup_result(intron_group, name_of(t), n_classes) for t in targets]
        return (
            pd.concat([r[0] for r in results], ignore_index=True),
            pd.concat([r[1] for r in results], ignore_index=True),
        )

    n_cells_local = np.bincount(codes_local, minlength=K)
    reads_local = np.bincount(codes_local, weights=y.sum(axis=1), minlength=K)

    init_A = np.zeros((K, n_classes - 1), dtype=float)
    for k in range(K):
        init_A[k] = alr(y[codes_local == k].sum(axis=0) + pseudocounts, denominator_idx=-1)

    n_alpha_groups = K if alpha_mode == "per_group" else 1
    alpha_group_idx = codes_local if alpha_mode == "per_group" else np.zeros(1, dtype=int)

    # --- unconstrained K-group model, fit once for all targets ---
    # x_alt is one-hot, so mixture=True and mixture=False are equivalent here
    x_alt = np.zeros((n_cells, K), dtype=float)
    x_alt[np.arange(n_cells), codes_local] = 1.0
    model_alt = lambda: MultiGroupDirichletMultinomialGLM(
        K, n_classes, init_A=init_A, init_log_alpha=np.ones(n_alpha_groups) * 1.0,
        alpha_group_idx=alpha_group_idx, n_alpha_groups=n_alpha_groups,
        mixture=True,
    )
    ll_alt, model_alt = fit_model(model_alt, x_alt, y, device=device)
    A_alt = model_alt.get_full_A().cpu().detach().numpy()
    alpha_alt = np.exp(model_alt.log_alpha.cpu().detach().numpy())
    psi_per_group = np.stack([softmax(A_alt[k]) for k in range(K)])

    dfs_intron_group, dfs_intron = [], []
    for target in targets:
        target_name = name_of(target)
        target_local = local_of[target]
        if target_local < 0 or n_cells_local[target_local] < min_cells_per_target:
            g, i = _empty_multigroup_result(intron_group, target_name, n_classes)
            dfs_intron_group.append(g)
            dfs_intron.append(i)
            continue

        w = _group_weights(n_cells_local, reads_local, target_local, weights_mode)
        if w is None:
            g, i = _empty_multigroup_result(intron_group, target_name, n_classes)
            dfs_intron_group.append(g)
            dfs_intron.append(i)
            continue

        x_null, other = _build_constrained_null_design(codes_local, K, target_local, w)
        init_A_null = init_A[other]
        model_null = lambda: MultiGroupDirichletMultinomialGLM(
            K - 1, n_classes, init_A=init_A_null,
            init_log_alpha=np.ones(n_alpha_groups) * 1.0,
            alpha_group_idx=alpha_group_idx, n_alpha_groups=n_alpha_groups,
            mixture=(null_space == "simplex"),
        )
        ll_null, model_null = fit_model(model_null, x_null, y, device=device)

        opt_warning = bool(ll_alt + 1e-2 < ll_null)
        if opt_warning:
            print(
                f"WARNING: optimization failed for intron_group {intron_group}, "
                f"target {target_name}: ll_null={ll_null} ll_full={ll_alt}"
            )
            p_value = 1.0
        else:
            p_value = lrtest(ll_null, ll_alt, n_classes - 1)
        if np.isnan(p_value):
            p_value = 1.0

        psi_a = psi_per_group[target_local]
        # weighted mean of the other groups' PSI on the simplex. With the default
        # null_space="simplex" this is the reference the LRT contrasts against,
        # and it is comparable to the pooled "rest" PSI of the two-group test.
        psi_b = (w[other, None] * psi_per_group[other]).sum(axis=0)
        # the same average taken in logit space (Aitchison mean); reported for
        # comparison, and the reference used when null_space="logit"
        psi_b_geom = softmax(w[other] @ A_alt[other])

        if alpha_mode == "per_group":
            alpha_target = float(alpha_alt[target_local])
            alpha_rest = float((w[other] * alpha_alt[other]).sum())
        else:
            alpha_target = float(alpha_alt[0])
            alpha_rest = float(alpha_alt[0])

        dfs_intron_group.append(pd.DataFrame(dict(
            intron_group=[intron_group], test_group=[target_name], p_value=[p_value],
            ll_null=[ll_null], ll=[ll_alt], n_classes=[n_classes], n_groups=[K],
            n_cells_target=[int(n_cells_local[target_local])],
            alpha_target=[alpha_target], alpha_rest=[alpha_rest],
            # very low concentration: each cell's usage is effectively a single
            # junction. Common and legitimate in sparse data, but the fit carries
            # little information and `opt_warning` does not cover it, so surface
            # it separately. On 3prime this is ~25% of tests, and those tests are
            # *less* likely to reach significance, so it is not a source of
            # inflation - but individual p-values there deserve less weight.
            low_alpha=[alpha_target < LOW_ALPHA_THRESHOLD],
            opt_warning=[opt_warning], tested=[True],
        )))
        dfs_intron.append(pd.DataFrame(dict(
            test_group=[target_name] * n_classes,
            psi_a=psi_a, psi_b=psi_b, psi_b_geom=psi_b_geom,
        )))

    return (
        pd.concat(dfs_intron_group, ignore_index=True),
        pd.concat(dfs_intron, ignore_index=True),
    )


def _run_differential_splicing_multigroup(
    adata,
    group_labels,
    targets,
    device="cpu",
    min_cells_per_intron_group=30,
    min_cells_per_target=30,
    min_cells_per_group=10,
    min_total_cells_per_intron=30,
    n_jobs=None,
    do_regroup=False,
    min_global_proportion=1e-3,
    alpha_mode="shared",
    weights="cells",
    null_space="simplex",
):
    group_labels = np.asarray(group_labels)
    assert len(group_labels) == adata.shape[0], "group_labels must be aligned with adata.obs"

    print(adata.shape)
    if min_total_cells_per_intron is not None:
        adata = filter_min_cells_per_feature(adata, min_total_cells_per_intron)
        print(adata.shape)
    if min_global_proportion is not None:
        adata = filter_min_global_proportion(adata, min_global_proportion)
        print(adata.shape)
    if do_regroup:
        adata = regroup(adata)
        print(adata.shape)
    if min_cells_per_intron_group is not None:
        # Applied over all cells only. Requiring the threshold in every one of K
        # groups would drop most intron groups because of the smallest cell types;
        # per-target coverage is enforced inside run_regression_multigroup via
        # min_cells_per_target.
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group)
        print(adata.shape)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()

    print("Number of intron groups: ", len(adata.var.intron_group.unique()))
    print("Number of introns: ", len(adata.var))

    group_names, group_codes = np.unique(group_labels, return_inverse=True)
    n_groups = len(group_names)
    name_to_code = {name: i for i, name in enumerate(group_names)}
    missing = [t for t in targets if t not in name_to_code]
    if missing:
        raise ValueError(f"targets not found in group_labels: {missing}")
    target_codes = [name_to_code[t] for t in targets]
    print(f"Number of groups: {n_groups}; testing {len(target_codes)} target(s)")

    opts = dict(
        alpha_mode=alpha_mode, weights=weights,
        min_cells_per_target=min_cells_per_target,
        min_cells_per_group=min_cells_per_group,
        null_space=null_space,
        group_names=group_names, device=device,
    )

    intron_groups = adata.var.intron_group.values
    all_intron_groups = pd.unique(intron_groups)
    intron_group_introns = defaultdict(list)
    for i, c in enumerate(intron_groups):
        intron_group_introns[c].append(i)

    X = adata.X.toarray()  # for easier parallelization using Python's libraries

    def make_args(c):
        return (c, X[:, intron_group_introns[c]], group_codes, target_codes, n_groups, opts)

    if n_jobs is not None and n_jobs != 1:
        dfs_intron_group, dfs_intron = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(run_regression_multigroup)(make_args(c))
                for c in tqdm(all_intron_groups)
            )
        )
    else:
        dfs_intron_group, dfs_intron = zip(*[
            run_regression_multigroup(make_args(c)) for c in tqdm(all_intron_groups)
        ])

    df_intron_group = pd.concat(dfs_intron_group, ignore_index=True)
    df_intron = pd.concat(dfs_intron, ignore_index=True)
    # each intron group contributed its introns once per target, in target order
    positions = np.concatenate([
        np.tile(intron_group_introns[c], len(target_codes)) for c in all_intron_groups
    ])
    # rename_axis so this works whether or not adata.var.index is named
    var = adata.var.iloc[positions].rename_axis("index").reset_index(drop=False)
    df_intron = pd.concat([var, df_intron], axis=1).set_index("index")
    return df_intron_group, df_intron


def run_differential_splicing_multigroup(
    adata,
    groupby,
    targets=None,
    alpha_mode="shared",
    weights="cells",
    null_space="simplex",
    **kwargs,
):
    """
    Multi-group one-vs-rest differential splicing.

    Each group gets its own Dirichlet-multinomial mean instead of pooling the
    rest, so between-group variation is not charged to the overdispersion
    parameter. See `run_regression_multigroup`.

    The null is psi_target == sum_{k != target} w_k * psi_k, the weighted mean on
    the simplex, which matches the two-group test's reference closely enough that
    the two result sets are comparable.

    On the 3prime data this recovers +41% more significant intron groups than the
    two-group test (2,867 vs. 2,034 at q<0.05), and permuting the cell type labels
    gives 0 significant events for both tests, so the gain is real. See
    multigroup_dm_findings.md.

    Note that an earlier simulation predicted no power gain. That simulation
    coupled the heterogeneity of the rest pool to the size of the target's effect,
    so a weak target effect implied a nearly homogeneous rest and left no
    concentration deflation to correct. Real data does not work that way: the rest
    pool is heterogeneous whatever the target does, so the deflation is present at
    every effect size, including at the detection margin.

    Parameters
    ----------
    adata : AnnData
        Splicing counts, with `var.intron_group`.
    groupby : str
        Column in `adata.obs` holding the group (e.g. cell type) of each cell.
    targets : list, optional
        Groups to test. Defaults to all groups.
    alpha_mode : {"shared", "per_group"}
        Whether to fit one concentration parameter for all cells or one per group.
        "shared" is recommended as the default: the K-group mean structure is what
        removes between-group variance from the residual, while per-group
        concentrations add K-1 parameters that are poorly determined for small
        groups.
    weights : {"cells", "reads", "equal"}
        How the "rest" reference is weighted across the non-target groups.
    null_space : {"simplex", "logit"}
        Where the "rest" reference is averaged. Keep the default "simplex".
        "logit" reproduces the additive-constraint-on-coefficients variant and is
        provided only for comparison: it is ill-posed whenever a non-target group
        has a zero-count junction, and rejects 51% of true nulls at nominal 5%
        when the non-target groups are heterogeneous. See
        `_build_constrained_null_design`.
    **kwargs
        Passed to `_run_differential_splicing_multigroup` (device,
        min_cells_per_intron_group, min_cells_per_target,
        min_total_cells_per_intron, n_jobs, do_regroup, min_global_proportion).

    Returns
    -------
    df_intron_group, df_intron : pd.DataFrame
        Same schema as `run_differential_splicing_for_each_group`, with
        `p_value_adj` computed by BH within each target group, plus `alpha_target`
        / `alpha_rest` and `psi_b_geom` (the logit-space reference the LRT
        actually contrasts, as opposed to `psi_b`, the arithmetic weighted mean).
    """
    if targets is None:
        targets = list(pd.unique(adata.obs[groupby].values))
    targets = list(targets)
    print("group sizes: ", adata.obs[groupby].value_counts().to_dict())

    df_intron_group, df_intron = _run_differential_splicing_multigroup(
        adata,
        adata.obs[groupby].values,
        targets,
        alpha_mode=alpha_mode,
        weights=weights,
        null_space=null_space,
        **kwargs,
    )
    if len(df_intron_group) == 0: return df_intron_group, df_intron

    df_intron["delta_psi"] = df_intron["psi_a"] - df_intron["psi_b"]
    df_intron["lfc_psi"] = np.log2(df_intron["psi_a"] + 1e-9) - np.log2(df_intron["psi_b"] + 1e-9)
    df_intron["abs_delta_psi"] = df_intron.delta_psi.abs()
    df_intron["abs_lfc_psi"] = df_intron.lfc_psi.abs()

    if "gene_id" not in df_intron.columns:
        df_intron["gene_id"] = "NA"
    if "gene_name" not in df_intron.columns:
        df_intron["gene_name"] = "NA"
    groupby_res = df_intron.groupby(["intron_group", "test_group"], observed=True).agg(
        {"gene_id": "first", "gene_name": "first", "abs_delta_psi": "max", "abs_lfc_psi": "max"}
    ).rename(columns={"abs_delta_psi": "max_abs_delta_psi", "abs_lfc_psi": "max_abs_lfc_psi"})
    df_intron_group = df_intron_group.merge(
        groupby_res, left_on=["intron_group", "test_group"], right_index=True, how="inner"
    )

    # BH within each target group, as in the one-vs-rest design. Intron groups
    # that were not testable for a given target (too few covered cells) are left
    # as NaN rather than entered as p=1, which would only make BH conservative.
    df_intron_group["p_value_adj"] = np.nan
    tested = df_intron_group[df_intron_group.tested]
    for g, idx in tested.groupby("test_group", observed=True).groups.items():
        _, pvals_corrected, _, _ = multipletests(
            df_intron_group.loc[idx, "p_value"].values, 0.05, "fdr_bh"
        )
        df_intron_group.loc[idx, "p_value_adj"] = pvals_corrected

    df_intron_group = df_intron_group.sort_values(by=["test_group", "p_value"])
    df_intron_group["ranking"] = df_intron_group.groupby("test_group", observed=True).cumcount()
    df_intron_group = df_intron_group.set_index("intron_group")
    df_intron_group["name"] = df_intron_group.index
    df_intron["name"] = df_intron.index
    return df_intron_group, df_intron


def make_intron_group_summation(intron_groups, device):
    n_introns = len(intron_groups)
    n_intron_groups = len(np.unique(intron_groups))
    rows, cols = zip(*list(enumerate(intron_groups)))
    vals = np.ones(n_introns, dtype=int)

    I = torch.tensor([cols, rows], device=device, dtype=torch.long)
    V = torch.tensor(vals, device=device, dtype=torch.float)
    intron_group_summation = torch.sparse.FloatTensor(
        I, V, torch.Size([n_intron_groups, n_introns])
    )
    return intron_group_summation


class LassoDirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, l1_penalty, init_A=None, init_log_alpha=None):
        super(LassoDirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(1) * 1.0
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.double))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=torch.double))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=torch.double))
        self.l1_penalty = l1_penalty
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        P = torch.softmax(X @ A, dim=1)
        concentration = torch.mul(alpha, P)
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
            + self.l1_penalty * torch.sum(torch.abs(self.A[1:]))  # excluding the intercept
        )
        self.ll = ll
        return res


class LassoMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, l1_penalty):
        super(LassoMultinomialGLM, self).__init__()
        self.A = nn.Parameter(torch.zeros((n_covariates, n_classes-1), dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.ll = None
        self.l1_penalty = l1_penalty

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        A = self.get_full_A()
        logits = X @ A
        return logits

    def loss_function(self, X, Y):
        logits = self.forward(X)
        ll = Multinomial(logits=logits).log_prob(Y).sum()
        res = (
            - ll
            + self.l1_penalty * torch.sum(torch.abs(self.A[1:]))  # excluding the intercept
        )
        self.ll = ll
        return res


class CVLassoMultinomialGLM:
    def __init__(self, l1_penalty_min, l1_penalty_max, n_trials):
        self.l1_penalty_min = l1_penalty_min
        self.l1_penalty_max = l1_penalty_max
        self.n_trials = n_trials

        import warnings
        warnings.filterwarnings('ignore')

    def fit(self, X, Y, stratification, device="cpu"):
        n_covariates = X.shape[1]
        n_classes = Y.shape[1]

        def objective(l1_penalty):
            train_ll = []
            test_ll = []
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, stratification):
                model = lambda: LassoMultinomialGLM(n_covariates, n_classes, l1_penalty)
                ll, model = fit_model(model, X[train_index], Y[train_index], device=device)
                train_ll.append(ll)
                model.loss_function(
                    torch.tensor(X[test_index], dtype=torch.double, device=device),
                    torch.tensor(Y[test_index], dtype=torch.double, device=device)
                )
                test_ll.append(model.ll.cpu().detach().numpy())
            return -np.mean(test_ll)

        l1_penalties = np.linspace(self.l1_penalty_min, self.l1_penalty_max, self.n_trials)
        losses = list(map(objective, l1_penalties))
        l1_penalty = l1_penalties[np.argmin(losses)]

        self.l1_penalty = l1_penalty
        model = lambda: LassoMultinomialGLM(n_covariates, n_classes, l1_penalty)
        ll, model = fit_model(model, X, Y, device=device)
        self.ll = ll
        self.model = model


class CVLassoDirichletMultinomialGLM:
    def __init__(self, l1_penalty_min, l1_penalty_max, n_trials):
        self.l1_penalty_min = l1_penalty_min
        self.l1_penalty_max = l1_penalty_max
        self.n_trials = n_trials

        import warnings
        warnings.filterwarnings('ignore')

    def fit(self, X, Y, stratification, device="cpu", threads=1):
        n_covariates = X.shape[1]
        n_classes = Y.shape[1]

        def objective(l1_penalty, X, Y, stratification):
            train_ll = []
            test_ll = []
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, stratification):
                model = lambda: LassoDirichletMultinomialGLM(n_covariates, n_classes, l1_penalty)
                ll, model = fit_model(model, X[train_index], Y[train_index], device=device)
                train_ll.append(ll)
                model.loss_function(
                    torch.tensor(X[test_index], dtype=torch.double, device=device),
                    torch.tensor(Y[test_index], dtype=torch.double, device=device)
                )
                test_ll.append(model.ll.cpu().detach().numpy())
            return -np.mean(test_ll)

        l1_penalties = np.linspace(self.l1_penalty_min, self.l1_penalty_max, self.n_trials)
        losses =  Parallel(n_jobs=threads)(delayed(objective)(l1_penalty, X, Y, stratification) for l1_penalty in l1_penalties)
        print("losses: ", losses)
        l1_penalty = l1_penalties[np.nanargmin(losses)]

        self.l1_penalty = l1_penalty
        model = lambda: LassoDirichletMultinomialGLM(n_covariates, n_classes, l1_penalty)
        ll, model = fit_model(model, X, Y, device=device)
        self.ll = ll
        self.model = model


def _run_differential_expression(adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene):
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    print(adata.shape)
    adata = adata[cell_idx_all].copy()
    print(adata.shape)
    total_cells_per_gene = (adata.X > 0).sum(axis=0).A1.ravel()
    adata = adata[:, total_cells_per_gene >= min_total_cells_per_gene]
    print(adata.shape)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_a = adata.X[: len(cell_idx_a)].toarray()
    X_b = adata.X[len(cell_idx_a) :].toarray()
    print(X_a.shape, X_b.shape)
    diff_exp = pd.DataFrame(
        dict(
            gene=adata.var_names.values,
            p_value=[
                mannwhitneyu(X_a[:, i], X_b[:, i], alternative="two-sided").pvalue
                for i in range(X_a.shape[1])
            ],
            lfc=np.log2(np.expm1(X_a).mean(axis=0) + 1e-9)
            - np.log2(np.expm1(X_b).mean(axis=0) + 1e-9),
        )
    )
    diff_exp["abs_lfc"] = diff_exp.lfc.abs()
    diff_exp["sort_val"] = list(zip(diff_exp.p_value, -diff_exp.lfc.abs()))
    diff_exp = diff_exp.sort_values(by="sort_val").drop("sort_val", 1)
    diff_exp["ranking"] = np.arange(len(diff_exp))
    return diff_exp


def run_differential_expression(
    adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene=100
):
    print("sample sizes: ", len(cell_idx_a), len(cell_idx_b))
    diff_exp = _run_differential_expression(
        adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene
    )
    reject, pvals_corrected, _, _ = multipletests(
        diff_exp.p_value.values, 0.05, "fdr_bh"
    )
    diff_exp["p_value_adj"] = pvals_corrected
    return diff_exp


def run_differential_splicing_for_each_group(
    adata_spl,
    groupby,
    groups=None,
    subset_to_groups=False,
    n_jobs_groups=1,
    **kwargs,
):
    """
    Run differential splicing analysis for each group.
    
    Parameters
    ----------
    adata_spl : AnnData
        Annotated data matrix containing splicing information
    groupby : str
        Column name in adata_spl.obs to group by
    groups : list, optional
        Specific groups to analyze. If None, all unique groups are used.
    subset_to_groups : bool, default False
        Whether to subset the data to only the specified groups
    n_jobs_groups : int, default 1
        Number of parallel jobs for processing groups. Set to -1 to use all cores.
        Note: If this is > 1, consider setting the 'n_jobs' parameter in kwargs to 1
        to avoid nested parallelization, which can be inefficient.
    **kwargs
        Additional arguments passed to run_differential_splicing
    
    Returns
    -------
    all_intron_groups : pd.DataFrame
        Differential splicing results at the intron group level
    all_introns : pd.DataFrame
        Differential splicing results at the individual intron level
    """
    if subset_to_groups:
        assert(groups is not None)
        adata_spl = adata_spl[adata_spl.obs[groupby].isin(groups)]
    
    if groups is None:
        groups = adata_spl.obs[groupby].unique()
    
    # Pre-compute all group indices to avoid repeated np.where calls
    group_obs = adata_spl.obs[groupby].values
    group_indices = {g: np.where(group_obs == g)[0] for g in groups}
    
    def process_group(g):
        """Process a single group for differential splicing."""
        print(g)
        cell_idx_a = group_indices[g]
        cell_idx_b = np.where(group_obs != g)[0]
        
        intron_groups, introns = run_differential_splicing(
            adata_spl, cell_idx_a, cell_idx_b, **kwargs, 
        )
        
        intron_groups["test_group"] = g
        introns["test_group"] = g
        intron_groups["name"] = intron_groups.index
        introns["name"] = introns.index
        
        return intron_groups, introns
    
    # Parallelize across groups if requested
    if n_jobs_groups is not None and n_jobs_groups != 1:
        results = Parallel(n_jobs=n_jobs_groups)(
            delayed(process_group)(g) for g in groups
        )
        all_intron_groups, all_introns = zip(*results)
    else:
        results = [process_group(g) for g in groups]
        all_intron_groups, all_introns = zip(*results)

    all_intron_groups = pd.concat(all_intron_groups, ignore_index=True)
    all_introns = pd.concat(all_introns, ignore_index=True)
    return all_intron_groups, all_introns


def find_marker_introns(intron_groups, introns, n=10, max_p_value_adj=0.05, min_delta_psi=0.05):
    intron_groups = intron_groups[intron_groups.p_value_adj <= max_p_value_adj]
    significant_groups_per_ig = intron_groups.groupby(["name"]).test_group.unique()
    groups = intron_groups.test_group.unique()
    
    def check_sig(i):
        return (
            i.intron_group in significant_groups_per_ig.index and
            i.test_group in significant_groups_per_ig.loc[i.intron_group] and
            i.delta_psi >= min_delta_psi
        )
    introns = introns[introns.apply(check_sig, axis=1)]

    marker_introns = defaultdict(list)
    introns = introns.sample(frac=1.0, random_state=42).sort_values("delta_psi", ascending=False).drop_duplicates("gene_name")
    i = 0
    while sum(map(len, marker_introns.values())) < n*len(groups) and i < len(introns):
        intron = introns.iloc[i]
        if len(marker_introns[intron.test_group]) < n:
            marker_introns[intron.test_group].append(intron["name"])
        i += 1
    return marker_introns


def mask_PSI(adata_spl, marker_introns, groupby, min_cells=10):
    marker_introns_list = sum(marker_introns.values(), [])
    adata_spl = adata_spl[:, marker_introns_list]
    PSI_raw = pd.DataFrame(adata_spl.layers["PSI_raw"])
    n_cells_per_group = PSI_raw.notna().groupby(adata_spl.obs[groupby].values).sum()
    PSI_raw_masked = adata_spl.layers["PSI_raw"].copy()
    for g in n_cells_per_group.index:
        idx_cells = np.where(adata_spl.obs[groupby]==g)[0]
        for i in n_cells_per_group.columns:
            if n_cells_per_group.loc[g, i] < min_cells:
                PSI_raw_masked[idx_cells, i] = np.nan
    adata_spl.layers["PSI_raw_masked"] = PSI_raw_masked
    return adata_spl
