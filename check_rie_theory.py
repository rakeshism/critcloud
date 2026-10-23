"""
Numerical verification of the RIE propositions in RIE_theory.tex.
Every check uses random weight vectors / correlation matrices plus the real QWS matrix.
"""
import os
import numpy as np
import pandas as pd
from critcloud_core import *

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(0)


def rand_corr(n):
    A = rng.standard_normal((n, n + 3)); C = A @ A.T
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)


ok = True
for _ in range(2000):
    n = int(rng.integers(3, 15)); R = rand_corr(n); w = rng.dirichlet(np.ones(n) * rng.uniform(0.1, 3))
    r = rie(w, R)
    # P1 bounds
    ok &= -1e-12 <= r <= 0.5 * (1 - hhi(w)) + 1e-12 <= (n - 1) / (2 * n) + 1e-12
    # P2 exact mass-transfer identity
    j, k = rng.choice(n, 2, replace=False); eps = rng.uniform(0, w[j])
    g = (np.abs(R) - np.eye(n)) @ w
    w2 = w.copy(); w2[j] -= eps; w2[k] += eps
    ok &= abs(rie(w2, R) - (r + eps * (g[k] - g[j]) - eps ** 2 * abs(R[j, k]))) < 1e-12
    # P3 permutation invariance
    p = rng.permutation(n)
    ok &= abs(rie(w[p], R[np.ix_(p, p)]) - r) < 1e-12
    # P4 RIE at equal weights = (n-1)/(2n) * mean off-diagonal |r|
    rbar = (np.abs(R).sum() - n) / (n * (n - 1))
    ok &= abs(rie(np.full(n, 1 / n), R) - (n - 1) / (2 * n) * rbar) < 1e-12
    # P6 decomposition RIE = 1/2 (1 - HHI) * WMR, WMR in [0,1]
    m = wmr(w, R)
    ok &= abs(r - 0.5 * (1 - hhi(w)) * m) < 1e-12 and -1e-12 <= m <= 1 + 1e-12
print("P1-P4, P6 identities hold on 2,000 random cases:", bool(ok))

# P5: RIE_adj -> 0 as weights concentrate on one criterion (QWS correlation matrix)
X = load_qws("qws.csv"); N = normalize(X); R = corr(N); n = 9
print("\nP5: RIE_adj along w = (1-eps, eps/8, ..., eps/8) on QWS (Entropy-TOPSIS has 0.0165)")
for eps in [0.5, 0.2, 0.1, 0.01, 1e-3, 1e-4, 1e-6]:
    w = np.r_[1 - eps, np.full(n - 1, eps / (n - 1))]
    print(f"  eps={eps:<8g} RIE={rie(w, R):.6f}  H={shannon(w):.6f}  RIE_adj={rie(w, R) / shannon(w):.6f}  WMR={wmr(w, R):.4f}")

# Decomposition table for QWS methods
wb, _ = bwm_weights(AB_FINAL, AW_FINAL); wc = critic_weights(N)
W = {"Equal": equal_weights(), "Entropy": entropy_weights(N), "SD": sd_weights(N), "MEREC": merec_weights(N),
     "CRITIC": wc, "BWM": wb, "CRITCLOUD(0.5)": blend(wc, wb, 0.5)}
dec = pd.DataFrame({m: {"RIE": rie(w, R), "1/2(1-HHI)": 0.5 * (1 - hhi(w)), "WMR": wmr(w, R),
                        "N_eff=1/HHI": 1 / hhi(w), "RIE_adj": rie(w, R) / shannon(w)} for m, w in W.items()}).T
dec.round(4).to_csv("results/R1_rie_decomposition.csv")
print("\n== RIE = 1/2(1-HHI) x WMR on QWS (concentration part x redundancy part)\n", dec.round(4).to_string())
