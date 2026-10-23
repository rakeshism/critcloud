"""
critcloud_core.py — single source of truth for every CRITCLOUD experiment.

All experiment scripts import from here, so weights, normalisation and
ranking are computed identically everywhere (fixes the cell-to-cell
inconsistencies found in the original notebooks).

Conventions
-----------
* Weight vectors are numpy arrays ordered as CRITERIA.
* Decision matrices are pandas DataFrames with columns == CRITERIA.
* Higher TOPSIS closeness = better; rank 1 = best.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import kendalltau, spearmanr

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
CRITERIA = ["Response Time", "Availability", "Throughput", "Successability",
            "Reliability", "Compliance", "Best Practices", "Latency", "Documentation"]
SHORT = ["RT", "Ava", "TP", "Suc", "Rel", "Com", "BP", "Lat", "Doc"]
COST = {"Response Time", "Latency"}
COLUMN_MAP = {"ResponseTime": "Response Time", "BestPractices": "Best Practices"}

# BWM vectors exactly as stated in the original paper (Table 3) - kept for the regression test
AB_PAPER = [1, 2, 8, 3, 5, 3, 2, 4, 4]
AW_PAPER = [8, 5, 1, 2, 4, 5, 6, 2, 3]

# FINAL vectors used by all experiments: Option A from bwm_revision_options.py
# (a_B of Successability 3->4 and Reliability 5->4; removes the 4 ordinal conflicts).
# To use another option, change these two lines and re-run the notebook.
AB_FINAL = [1, 2, 8, 4, 4, 3, 2, 4, 4]
AW_FINAL = [8, 5, 1, 2, 4, 5, 6, 2, 3]


def load_qws(path: str = "qws.csv") -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=COLUMN_MAP)
    return df[CRITERIA].dropna().reset_index(drop=True)


# ----------------------------------------------------------------------------
# Normalisation (Eqs. 2-3)
# ----------------------------------------------------------------------------
def normalize(X: pd.DataFrame, cost=COST) -> pd.DataFrame:
    N = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        lo, hi = X[c].min(), X[c].max()
        if hi == lo:
            N[c] = 1.0
        elif c in cost:
            N[c] = (hi - X[c]) / (hi - lo)
        else:
            N[c] = (X[c] - lo) / (hi - lo)
    return N


# ----------------------------------------------------------------------------
# Objective weights
# ----------------------------------------------------------------------------
def corr(N: pd.DataFrame) -> np.ndarray:
    R = np.corrcoef(N.values, rowvar=False)
    return np.nan_to_num(R, nan=0.0)


def critic_weights(N: pd.DataFrame) -> np.ndarray:
    sigma = N.values.std(axis=0, ddof=0)
    R = corr(N)
    C = sigma * (1 - np.abs(R)).sum(axis=1)
    return C / C.sum()


def entropy_weights(N: pd.DataFrame) -> np.ndarray:
    P = N.values / np.where(N.values.sum(axis=0) == 0, 1, N.values.sum(axis=0))
    P = np.clip(P, 1e-12, None)
    E = -(P * np.log(P)).sum(axis=0) / np.log(len(N))
    d = 1 - E
    return d / d.sum()


def sd_weights(N: pd.DataFrame) -> np.ndarray:
    s = N.values.std(axis=0, ddof=0)
    return s / s.sum()


def merec_weights(N: pd.DataFrame) -> np.ndarray:
    """MEREC (Keshavarz-Ghorabaee et al., 2021) on benefit-oriented N in (0,1]."""
    X = np.clip(N.values, 1e-6, None)
    n = X.shape[1]
    lnX = np.abs(np.log(X))
    S = np.log(1 + lnX.mean(axis=1))
    E = np.zeros(n)
    for j in range(n):
        Sj = np.log(1 + np.delete(lnX, j, axis=1).sum(axis=1) / n)
        E[j] = np.abs(Sj - S).sum()
    return E / E.sum()


def equal_weights(n: int = 9) -> np.ndarray:
    return np.full(n, 1.0 / n)


# ----------------------------------------------------------------------------
# BWM (linear model, Rezaei 2016) + consistency
# ----------------------------------------------------------------------------
def bwm_weights(AB, AW, best=None, worst=None):
    """Linear BWM. Returns (w, xi_L). best/worst indices default to the
    position of a_BB = 1 / a_WW = 1."""
    AB, AW = list(AB), list(AW)
    n = len(AB)
    b = AB.index(1) if best is None else best
    wst = AW.index(1) if worst is None else worst
    c = np.zeros(n + 1); c[-1] = 1
    A, ub = [], []
    for j in range(n):
        for s in (1, -1):
            r = np.zeros(n + 1); r[b] += s; r[j] -= s * AB[j]; r[-1] = -1
            A.append(r); ub.append(0)
            r = np.zeros(n + 1); r[j] += s; r[wst] -= s * AW[j]; r[-1] = -1
            A.append(r); ub.append(0)
    res = linprog(c, A_ub=A, b_ub=ub, A_eq=[np.r_[np.ones(n), 0]], b_eq=[1],
                  bounds=[(0, 1)] * n + [(0, None)], method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:n], res.x[-1]


def input_based_cr(AB, AW, worst=None):
    """Input-based consistency ratio (Liang, Brunelli & Rezaei, 2020).
    CR^I = max_j |a_Bj * a_jW - a_BW| / (a_BW^2 - a_BW); CR^I = 0 when a_BW = 1.
    `worst` = index of the worst criterion (defaults to the position of a_WW = 1).
    Returns (CR^I, per-criterion values)."""
    AB, AW = np.array(AB, float), np.array(AW, float)
    aBW = AB[list(AW).index(1) if worst is None else worst]
    if aBW <= 1:
        return 0.0, np.zeros(len(AB))
    per = np.abs(AB * AW - aBW) / (aBW ** 2 - aBW)
    return per.max(), per


def blend(w_obj, w_subj, alpha):
    return alpha * np.asarray(w_obj) + (1 - alpha) * np.asarray(w_subj)


# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------
def topsis(N: pd.DataFrame, w) -> pd.Series:
    V = N.values * np.asarray(w)
    dp = np.sqrt(((V - V.max(axis=0)) ** 2).sum(axis=1))
    dm = np.sqrt(((V - V.min(axis=0)) ** 2).sum(axis=1))
    T = dm / np.where(dp + dm == 0, 1, dp + dm)
    return pd.Series(T, index=N.index)


def rank_of(score: pd.Series) -> pd.Series:
    """Rank 1 = best (highest score)."""
    return score.rank(ascending=False, method="average")


def optcloud(N: pd.DataFrame, w_bwm, threshold=0.85, fix_sign=True):
    """OPTCLOUD (PCA-BWM) re-implementation.
    fix_sign=True aligns each eigenvector so its loadings sum to a positive
    value; numpy returns eigenvectors with arbitrary sign, which otherwise
    can invert a component's contribution to the score."""
    R = corr(N)
    vals, vecs = np.linalg.eigh(R)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    if fix_sign:
        vecs = vecs * np.where(vecs.sum(axis=0) < 0, -1, 1)
    cum = np.cumsum(vals) / vals.sum()
    m = min(int(np.searchsorted(cum, threshold) + 1), len(vals))
    Z = (N.values * np.asarray(w_bwm)) @ vecs[:, :m] @ (vals[:m] / vals[:m].sum())
    return pd.Series(Z, index=N.index), m, cum[m - 1]


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def rie(w, R) -> float:
    w = np.asarray(w); Rt = np.abs(R) - np.eye(len(w))
    return 0.5 * float(w @ Rt @ w)


def rie_norm(w, R) -> float:
    """RIE relative to equal weighting on the same correlation matrix (<1 = less redundant)."""
    n = len(w)
    return rie(w, R) / rie(np.full(n, 1 / n), R)


def shannon(w) -> float:
    w = np.asarray(w); w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def hhi(w) -> float:
    return float((np.asarray(w) ** 2).sum())


# ----------------------------------------------------------------------------
# Agreement metrics
# ----------------------------------------------------------------------------
def tau(a, b) -> float:
    return float(kendalltau(a, b)[0])


def rho(a, b) -> float:
    return float(spearmanr(a, b)[0])


def topk_hit(score_pred, score_true, k=5) -> float:
    """Share of the predicted top-k that is in the true top-k."""
    p = set(np.argsort(-np.asarray(score_pred))[:k])
    t = set(np.argsort(-np.asarray(score_true))[:k])
    return len(p & t) / k


def regret(score_pred, utility_true) -> float:
    """Utility lost by picking the predicted best instead of the true best."""
    u = np.asarray(utility_true)
    return float(u.max() - u[int(np.argmax(np.asarray(score_pred)))])


def ndcg_at_k(score_pred, utility_true, k=10) -> float:
    u = np.asarray(utility_true, float)
    g = u - u.min()  # non-negative gains
    order = np.argsort(-np.asarray(score_pred))[:k]
    ideal = np.argsort(-u)[:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = (g[ideal] * disc).sum()
    return float((g[order] * disc).sum() / idcg) if idcg > 0 else 1.0


# ----------------------------------------------------------------------------
# Rank reversal: pairwise-order definition (standard)
# ----------------------------------------------------------------------------
def pairwise_reversal_rate(score_before: pd.Series, score_after: pd.Series) -> float:
    """Share of provider pairs (present in both) whose relative order flips."""
    common = score_before.index.intersection(score_after.index)
    a = score_before.loc[common].values
    b = score_after.loc[common].values
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(common), 1)
    return float((da[iu] * db[iu] < 0).mean())


# ----------------------------------------------------------------------------
# Additional comparison methods (requested by Reviewer 1 / standard in the literature)
# ----------------------------------------------------------------------------
SAATY = np.array([1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
RI_SAATY = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def to_saaty(r):
    """Round a positive ratio to the nearest value on Saaty's 1/9..9 scale (in log space)."""
    return SAATY[np.argmin(np.abs(np.log(SAATY) - np.log(r)))]


def ahp_matrix_from_pref(p, noise_sd=0.0, rng=None):
    """Pairwise comparison matrix a_ij ~ p_i / p_j on Saaty's scale.
    Each upper-triangle judgement gets its own log-normal noise (independent human errors)."""
    n = len(p); A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = p[i] / p[j]
            if noise_sd > 0:
                r *= np.exp(noise_sd * rng.standard_normal())
            A[i, j] = to_saaty(r); A[j, i] = 1 / A[i, j]
    return A


def ahp_weights(A):
    """Principal-eigenvector AHP weights and Saaty consistency ratio."""
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real); w = w / w.sum()
    n = len(A)
    ci = (vals[k].real - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RI_SAATY.get(n, 1.49) if RI_SAATY.get(n, 0) > 0 else 0.0
    return w, cr


def multiplicative(w1, w2):
    """Multiplicative synthesis w_j = w1_j*w2_j / sum (the usual CRITIC-AHP / CRITIC-DEMATEL rule)."""
    w = np.asarray(w1) * np.asarray(w2)
    return w / w.sum()


def dematel_weights(R, scale=4):
    """Data-driven DEMATEL: direct-relation matrix = scale * |correlation| (continuous; no
    expert influence matrix exists for QWS). Rounding to the 0..4 grid is avoided because it
    sets weakly correlated criteria to zero influence and hence zero weight.
    Weights from sqrt(prominence^2 + relation^2)."""
    n = len(R)
    D = scale * np.abs(R); np.fill_diagonal(D, 0)
    s = max(D.sum(axis=1).max(), D.sum(axis=0).max())
    if s == 0:
        return np.full(n, 1 / n)
    Nn = D / s
    T = Nn @ np.linalg.inv(np.eye(n) - Nn)
    r, c = T.sum(axis=1), T.sum(axis=0)
    imp = np.sqrt((r + c) ** 2 + (r - c) ** 2)
    return imp / imp.sum()


def game_theory_combination(w1, w2):
    """Game-theory combination weighting: find a1, a2 minimising the deviation of
    a1*w1 + a2*w2 from both w1 and w2; coefficients normalised to sum to 1."""
    W = np.vstack([w1, w2])
    a = np.linalg.solve(W @ W.T, np.diag(W @ W.T))
    a = np.abs(a) / np.abs(a).sum()
    w = a @ W
    return w / w.sum(), a


def mahalanobis_topsis(N: pd.DataFrame, w, ridge=1e-6) -> pd.Series:
    """TOPSIS with Mahalanobis distance (e.g. Qian et al., Expert Systems 2019):
    d = sqrt((x - x*)^T Omega S^-1 Omega (x - x*)), Omega = diag(w), S = covariance of N.
    A small ridge keeps S invertible when criteria are almost collinear (e.g. Ava/Suc)."""
    X = N.values
    S = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
    Om = np.diag(np.asarray(w))
    M = Om @ np.linalg.pinv(S) @ Om
    best, worst = X.max(axis=0), X.min(axis=0)
    dp = np.sqrt(np.einsum("ij,jk,ik->i", X - best, M, X - best).clip(0))
    dm = np.sqrt(np.einsum("ij,jk,ik->i", X - worst, M, X - worst).clip(0))
    return pd.Series(dm / np.where(dp + dm == 0, 1, dp + dm), index=N.index)


def ordinal_violations(AB, AW):
    """Pairs (i, j) where the two BWM vectors disagree on which criterion is more important:
    a_Bi < a_Bj (i more important) but a_iW < a_jW (i less important), or vice versa."""
    AB, AW = np.asarray(AB), np.asarray(AW); out = []
    for i in range(len(AB)):
        for j in range(i + 1, len(AB)):
            if (AB[i] - AB[j]) * (AW[j] - AW[i]) < 0:
                out.append((i, j))
    return out


# ----------------------------------------------------------------------------
# Alternative normalisations (for heavy-tailed QoS such as response time / latency)
# ----------------------------------------------------------------------------
def normalize_log(X: pd.DataFrame, cost=COST, skew_limit=2.0) -> pd.DataFrame:
    """log(1 + x - min) on criteria with |skewness| > skew_limit, then max-min.
    The monotone transform keeps each criterion's order but stops a few extreme
    values from squeezing everyone else into a narrow band."""
    Y = X.copy().astype(float)
    for c in Y.columns:
        if abs(Y[c].skew()) > skew_limit:
            Y[c] = np.log1p(Y[c] - Y[c].min())
    return normalize(Y, cost)


def normalize_rank(X: pd.DataFrame, cost=COST) -> pd.DataFrame:
    """Percentile ranks in (0, 1]; fully robust to outliers but discards magnitudes."""
    N = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        N[c] = X[c].rank(ascending=c not in cost, method="average") / len(X)
    return N


NORMALIZERS = {"max-min": normalize, "log + max-min": normalize_log, "rank": normalize_rank}


def wmr(w, R) -> float:
    """Weighted Mean Redundancy: expected |r| between two distinct criteria drawn with
    probability proportional to their weights. WMR = 2*RIE / (1 - HHI), always in [0, 1].
    Undefined (returned as nan) only when all weight sits on one criterion."""
    d = 1 - hhi(w)
    return 2 * rie(w, R) / d if d > 1e-12 else float("nan")
