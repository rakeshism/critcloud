"""
Synthetic ground-truth simulator for correlation-aware weighting.

Generative model
----------------
  z_ik ~ N(0,1)                    K independent latent quality factors
  theta ~ Dirichlet(1_K)           user's true importance of each factor
  u_i = sum_k theta_k z_ik         TRUE utility  -> TRUE ranking
  x_ij = rho * z_{i,pi(j)} + sqrt(1-rho^2) * eps_ij      observed criterion j

pi(j) maps criteria to factors; mapping several criteria to one factor creates
controlled redundancy (the situation CRITCLOUD claims to handle).

User preference (input to BWM): the user judges each criterion by the importance
of the factor it measures, unaware of redundancy, with log-normal judgement noise:
  p_j = theta_{pi(j)} * exp(sigma_pref * xi_j)
p is converted to integer 1..9 BWM vectors and solved with the linear BWM model.

The oracle weight vector spreads theta_k evenly across the copies of factor k,
i.e. the redundancy-corrected weights a perfectly informed analyst would use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from critcloud_core import (normalize, critic_weights, entropy_weights, sd_weights,
                            merec_weights, equal_weights, bwm_weights, blend, topsis,
                            optcloud, tau, topk_hit, regret, ndcg_at_k, rie, corr,
                            ahp_matrix_from_pref, ahp_weights, multiplicative,
                            dematel_weights, game_theory_combination, mahalanobis_topsis)

STRUCTURES = {
    "none":     [0, 1, 2, 3, 4, 5, 6, 7, 8],   # 9 independent factors
    "moderate": [0, 0, 1, 1, 2, 2, 3, 4, 5],   # 3 duplicated pairs
    "high":     [0, 0, 0, 0, 1, 1, 2, 3, 4],   # one factor measured 4 times
}


def bwm_vectors_from_pref(p):
    """Integer 1..9 BWM vectors a user with latent preferences p would state."""
    b, w = int(np.argmax(p)), int(np.argmin(p))
    AB = np.clip(np.rint(p[b] / p), 1, 9).astype(int)
    AW = np.clip(np.rint(p / p[w]), 1, 9).astype(int)
    return list(AB), list(AW), b, w


SLA_QUANTILE = 0.25   # providers below this quantile of the SLA factor violate the SLA


def true_utility(Z, theta, mode, rng):
    """True utility under three decision models.
    linear  : u = sum_k theta_k z_k                       (compensatory, main model)
    sla     : linear utility, but providers whose SLA factor (drawn with prob. theta)
              falls below its 25th percentile are infeasible and ranked below every
              feasible provider (hard, non-compensatory constraint)
    concave : u = sum_k theta_k (1 - exp(-z_k))           (diminishing returns; a weak
              factor costs more than a strong factor gains)
    Returns (u, feasible-mask or None)."""
    if mode == "linear":
        return Z @ theta, None
    if mode == "concave":
        return (1.0 - np.exp(-Z)) @ theta, None
    if mode == "sla":
        k = rng.choice(len(theta), p=theta)
        feas = Z[:, k] >= np.quantile(Z[:, k], SLA_QUANTILE)
        u = Z @ theta
        return np.where(feas, u, u - 100.0), feas
    raise ValueError(mode)


def one_replication(rng, m, structure, rho_load, sigma_pref=0.3, skewed=False,
                    alphas=(0.25, 0.5, 0.75), theta_mode="dirichlet",
                    pref_mode="per_criterion", utility="linear"):
    """pref_mode:
       'per_criterion' (main): the user judges each criterion by the importance of the
            quality it measures, p_j = theta_{pi(j)}; a quality measured c times is
            therefore counted c times (the user does not see the redundancy).
       'split': the user already corrects for redundancy, p_j = theta_{pi(j)} / c_{pi(j)},
            so every latent quality receives its importance exactly once."""
    pi = np.array(STRUCTURES[structure])
    K, n = pi.max() + 1, len(pi)
    Z = rng.standard_normal((m, K))
    # 'dirichlet': user has strong, random factor preferences
    # 'uniform'  : every latent factor matters equally (pure redundancy test)
    theta = rng.dirichlet(np.ones(K)) if theta_mode == "dirichlet" else np.full(K, 1.0 / K)
    u, feas = true_utility(Z, theta, utility, rng)
    Xv = rho_load * Z[:, pi] + np.sqrt(1 - rho_load ** 2) * rng.standard_normal((m, n))
    if skewed:  # heavy right tail on two criteria, mimicking response time / latency
        Xv[:, [0, n - 1]] = np.exp(1.5 * Xv[:, [0, n - 1]])
    X = pd.DataFrame(Xv, columns=[f"c{j}" for j in range(n)])
    N = normalize(X, cost=set())

    counts = np.bincount(pi)
    base_pref = theta[pi] if pref_mode == "per_criterion" else theta[pi] / counts[pi]
    p = base_pref * np.exp(sigma_pref * rng.standard_normal(n))
    AB, AW, b, w = bwm_vectors_from_pref(p)
    wb, _ = bwm_weights(AB, AW, best=b, worst=w)
    w_oracle = theta[pi] / counts[pi]
    wc = critic_weights(N)

    W = {"Equal": equal_weights(n), "Entropy": entropy_weights(N), "SD": sd_weights(N),
         "MEREC": merec_weights(N), "CRITIC": wc, "BWM": wb}
    for a in alphas:
        W[f"CRITCLOUD({a})"] = blend(wc, wb, a)
    W["Naive-oracle (theta/copies)"] = w_oracle
    # Control blends: is the gain from CRITIC's correlation term, or from shrinkage alone?
    W["BWM+Equal(0.5)"] = blend(equal_weights(n), wb, 0.5)
    W["BWM+Entropy(0.5)"] = blend(entropy_weights(N), wb, 0.5)
    # Best linear oracle: w* = Cov(x)^-1 Cov(x,u) on standardised criteria.
    # Accounts for the fact that repeated noisy measurements of one factor carry signal.
    P = np.eye(K)[pi]                                   # n x K loading indicator
    Sx = rho_load ** 2 * P @ P.T + (1 - rho_load ** 2) * np.eye(n)
    w_blp = np.linalg.solve(Sx, rho_load * P @ theta)
    w_blp = np.clip(w_blp, 0, None); w_blp = w_blp / w_blp.sum()
    W["Oracle (best linear)"] = w_blp
    # Literature hybrids. AHP: the same user answers all 36 pairwise questions,
    # each with its own judgement error (sigma_pref), so AHP sees more (noisy) input than BWM.
    w_ahp, _ = ahp_weights(ahp_matrix_from_pref(base_pref, sigma_pref, rng))
    W["AHP"] = w_ahp
    W["CRITIC-AHP"] = multiplicative(wc, w_ahp)
    W["CRITIC-DEMATEL"] = multiplicative(wc, dematel_weights(corr(N)))
    W["GameTheory(CRITIC,BWM)"] = game_theory_combination(wc, wb)[0]

    scores = {k: topsis(N, w).values for k, w in W.items()}
    scores["OPTCLOUD"] = optcloud(N, wb, fix_sign=True)[0].values
    scores["BWM+Mahalanobis-TOPSIS"] = mahalanobis_topsis(N, wb).values

    R = corr(N)
    rows = []
    for k, s in scores.items():
        rows.append({"method": k, "tau": tau(s, u), "top5": topk_hit(s, u, 5),
                     "ndcg10": ndcg_at_k(s, u, 10), "regret": regret(s, u),
                     "rie": rie(W[k], R) if k in W else np.nan,
                     "top1_feasible": float(feas[int(np.argmax(s))]) if feas is not None else np.nan})
    return rows


def run_grid(reps=300, seed=7, ms=(40, 200), structures=("none", "moderate", "high"),
             rhos=(0.6, 0.9), skews=(False, True), sigma_prefs=(0.3,),
             theta_modes=("dirichlet", "uniform"), pref_mode="per_criterion", utility="linear"):
    rng = np.random.default_rng(seed)
    out = []
    for tm in theta_modes:
        for sp in sigma_prefs:
            for m in ms:
                for st in structures:
                    for r in rhos:
                        for sk in skews:
                            for rep in range(reps):
                                for row in one_replication(rng, m, st, r, sp, sk, theta_mode=tm,
                                                           pref_mode=pref_mode, utility=utility):
                                    row.update(theta=tm, sigma_pref=sp, m=m, structure=st,
                                               rho=r, skewed=sk, rep=rep)
                                    out.append(row)
    return pd.DataFrame(out)
