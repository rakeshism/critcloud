"""
Week 2 - does a robust normalisation fix the heavy-tail problem?

Part A (QWS): CRITIC weights, latency/response-time weight and compression under
  max-min, log + max-min and rank normalisation.
Part B (simulation, skewed and non-skewed data): ranking accuracy (tau to the true
  ranking) of the main methods under each normalisation.
"""
import os
import numpy as np
import pandas as pd
from critcloud_core import *
from simulator import STRUCTURES, bwm_vectors_from_pref

os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------ Part A: QWS
X = load_qws("qws.csv")
wb, _ = bwm_weights(AB_FINAL, AW_FINAL)
viol = ((X["Response Time"] > 200).astype(int) + (X["Availability"] < 95).astype(int)
        + (X["Successability"] < 90).astype(int))
sla_rank = (viol * 1000 + X["Response Time"]).rank(method="average")
wtab, rows, ranks = {}, [], {}
for name, f in NORMALIZERS.items():
    N = f(X); wc = critic_weights(N); wtab[name] = wc
    s = topsis(N, blend(wc, wb, 0.5)); ranks[name] = rank_of(s)
    rows.append({"normalisation": name,
                 "share_RT_norm>0.9": (N["Response Time"] > 0.9).mean(),
                 "share_Lat_norm>0.9": (N["Latency"] > 0.9).mean(),
                 "CRITIC_w_RT": wc[0], "CRITIC_w_Lat": wc[7],
                 "CRITCLOUD_tau_vs_SLA": tau(ranks[name], sla_rank)})
qa = pd.DataFrame(rows).round(4)
qa["CRITCLOUD_tau_vs_max-min_ranking"] = [round(tau(ranks[n], ranks["max-min"]), 4) for n in NORMALIZERS]
qa.to_csv("results/N1_normalisation_qws.csv", index=False)
pd.DataFrame(wtab, index=SHORT).round(4).to_csv("results/N2_critic_weights_by_normalisation.csv")
print("== Part A: QWS\n", qa.to_string(index=False))
print("\nCRITIC weights by normalisation\n", pd.DataFrame(wtab, index=SHORT).round(4).to_string(), "\n")

# ------------------------------------------------------------------ Part B: simulation


def rep(rng, structure, skewed, theta_mode, m=40, rho_load=0.9, sigma=0.3):
    pi = np.array(STRUCTURES[structure]); K, n = pi.max() + 1, len(pi)
    Z = rng.standard_normal((m, K))
    theta = rng.dirichlet(np.ones(K)) if theta_mode == "dirichlet" else np.full(K, 1 / K)
    u = Z @ theta
    Xv = rho_load * Z[:, pi] + np.sqrt(1 - rho_load ** 2) * rng.standard_normal((m, n))
    if skewed:
        Xv[:, [0, n - 1]] = np.exp(1.5 * Xv[:, [0, n - 1]])
    Xs = pd.DataFrame(Xv, columns=[f"c{j}" for j in range(n)])
    p = theta[pi] * np.exp(sigma * rng.standard_normal(n))
    AB, AW, b, w = bwm_vectors_from_pref(p)
    wb_, _ = bwm_weights(AB, AW, best=b, worst=w)
    out = []
    for name, f in NORMALIZERS.items():
        N = f(Xs, cost=set()); wc_ = critic_weights(N)
        for meth, wv in [("BWM", wb_), ("CRITCLOUD(0.5)", blend(wc_, wb_, 0.5)),
                         ("CRITIC", wc_), ("Equal", equal_weights(n))]:
            out.append({"normalisation": name, "method": meth, "tau": tau(topsis(N, wv).values, u)})
    return out


rng = np.random.default_rng(23)
recs = []
for th in ("dirichlet", "uniform"):
    for sk in (False, True):
        for st in ("none", "moderate", "high"):
            for _ in range(200):
                for r in rep(rng, st, sk, th):
                    r.update(theta=th, skewed=sk, structure=st); recs.append(r)
sim = pd.DataFrame(recs)
tab = sim.groupby(["skewed", "method", "normalisation"])["tau"].mean().unstack("normalisation").round(4)
tab = tab[list(NORMALIZERS)]
tab.to_csv("results/N3_normalisation_sim.csv")
print("== Part B: mean tau to TRUE ranking by normalisation (1,200 reps per row)\n", tab.to_string())
