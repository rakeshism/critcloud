"""
Week 2 - alpha guidance (Reviewer 1, point 1).

Part A (real QWS data): how alpha trades off fidelity to the user's BWM ranking,
  stability under BWM answer errors, and redundancy (RIE / effective number of criteria).
Part B (simulation): which alpha gives the most accurate ranking, depending on
  (i) how noisy the user's judgements are and (ii) how redundant the criteria are.
  Because true noise is unknown in practice, we also report the best alpha by the
  observable BWM consistency ratio CR^I, which gives a usable rule for practitioners.
"""
import os
import numpy as np
import pandas as pd
from critcloud_core import *
from simulator import STRUCTURES, bwm_vectors_from_pref

os.makedirs("results", exist_ok=True)
ALPHAS_REAL = np.round(np.arange(0, 1.0001, 0.05), 2)
ALPHAS_SIM = np.round(np.arange(0, 1.0001, 0.1), 1)

# ------------------------------------------------------------------ Part A: QWS
X = load_qws("qws.csv"); N = normalize(X); R = corr(N)
wc = critic_weights(N)
wb, _ = bwm_weights(AB_FINAL, AW_FINAL)
B, Wst = AB_FINAL.index(1), AW_FINAL.index(1)
rank_bwm, rank_critic = rank_of(topsis(N, wb)), rank_of(topsis(N, wc))

rng = np.random.default_rng(5)
perturbed = []
for _ in range(300):
    AB = np.clip(np.array(AB_FINAL) + rng.integers(-1, 2, 9), 1, 9)
    AW = np.clip(np.array(AW_FINAL) + rng.integers(-1, 2, 9), 1, 9)
    AB[B], AW[Wst] = 1, 1
    aBW = int(np.clip(AB_FINAL[Wst] + rng.integers(-1, 2), 2, 9)); AB[Wst], AW[B] = aBW, aBW
    perturbed.append(bwm_weights(list(AB), list(AW), best=B, worst=Wst)[0])

rows = []
for a in ALPHAS_REAL:
    w = blend(wc, wb, a)
    s = topsis(N, w); top = s.idxmax()
    stable = np.mean([topsis(N, blend(wc, wp, a)).idxmax() == top for wp in perturbed])
    rows.append({"alpha": a, "tau_to_user_BWM": tau(rank_of(s), rank_bwm),
                 "tau_to_CRITIC": tau(rank_of(s), rank_critic),
                 "top1_stable_%": 100 * stable, "RIE": rie(w, R), "RIE_norm": rie_norm(w, R),
                 "N_eff(1/HHI)": 1 / hhi(w), "H(w)": shannon(w)})
qa = pd.DataFrame(rows).round(4)
qa.to_csv("results/A1_alpha_tradeoff_qws.csv", index=False)
print("== Part A: alpha trade-off on QWS\n", qa.to_string(index=False), "\n")

# ------------------------------------------------------------------ Part B: simulation


def sim_rep(rng, structure, sigma, m=40, rho_load=0.9):
    pi = np.array(STRUCTURES[structure]); K, n = pi.max() + 1, len(pi)
    Z = rng.standard_normal((m, K)); theta = rng.dirichlet(np.ones(K)); u = Z @ theta
    Xv = rho_load * Z[:, pi] + np.sqrt(1 - rho_load ** 2) * rng.standard_normal((m, n))
    Ns = normalize(pd.DataFrame(Xv), cost=set())
    p = theta[pi] * np.exp(sigma * rng.standard_normal(n))
    AB, AW, b, w = bwm_vectors_from_pref(p)
    wb_, _ = bwm_weights(AB, AW, best=b, worst=w)
    cr, _ = input_based_cr(AB, AW, worst=w)
    wc_ = critic_weights(Ns)
    out = {"CR_I": cr}
    for a in ALPHAS_SIM:
        out[f"a{a}"] = tau(topsis(Ns, blend(wc_, wb_, a)).values, u)
        out[f"eq{a}"] = tau(topsis(Ns, blend(equal_weights(n), wb_, a)).values, u)
    return out


rng = np.random.default_rng(17)
recs = []
for st in ("none", "moderate", "high"):
    for sg in (0.0, 0.3, 0.6, 1.0):
        for _ in range(300):
            r = sim_rep(rng, st, sg); r.update(structure=st, sigma=sg); recs.append(r)
sim = pd.DataFrame(recs)
sim.to_csv("results/A2_alpha_sim_raw.csv.gz", index=False, compression="gzip")

acols = [f"a{a}" for a in ALPHAS_SIM]
g = sim.groupby(["structure", "sigma"])[acols].mean()
best = pd.DataFrame({"best_alpha": g.idxmax(axis=1).str[1:].astype(float),
                     "tau_at_best": g.max(axis=1),
                     "tau_at_0.5": g["a0.5"], "tau_BWM_only(a=0)": g["a0.0"]})
# range of alphas within 0.005 of the best (flat optimum)
best["near_best_range"] = [f"{min(a for a in ALPHAS_SIM if g.loc[i, f'a{a}'] >= g.loc[i].max() - 0.005)}"
                           f"-{max(a for a in ALPHAS_SIM if g.loc[i, f'a{a}'] >= g.loc[i].max() - 0.005)}"
                           for i in g.index]
best = best.round(4)
best.to_csv("results/A3_best_alpha_by_noise_redundancy.csv")
print("== Part B: best alpha by judgement noise and redundancy (300 reps each)\n", best.to_string(), "\n")

# observable rule: best alpha by BWM consistency ratio (what a practitioner can see)
sim["CR_bin"] = pd.cut(sim["CR_I"], [-0.001, 0.05, 0.15, 0.30, 1.0],
                       labels=["0-0.05", "0.05-0.15", "0.15-0.30", ">0.30"])
gc = sim.groupby("CR_bin", observed=True)[acols].mean()
rule = pd.DataFrame({"n_cases": sim.groupby("CR_bin", observed=True).size(),
                     "best_alpha": gc.idxmax(axis=1).str[1:].astype(float),
                     "tau_at_best": gc.max(axis=1), "tau_at_0.5": gc["a0.5"], "tau_BWM_only": gc["a0.0"]}).round(4)
rule.to_csv("results/A4_alpha_rule_by_CR.csv")
print("== Practical rule: best alpha by observed BWM consistency ratio CR^I\n", rule.to_string(), "\n")

# CRITIC vs Equal as the objective partner, across alpha (pooled)
comp = pd.DataFrame({"CRITIC_partner": sim[acols].mean().values,
                     "Equal_partner": sim[[f"eq{a}" for a in ALPHAS_SIM]].mean().values},
                    index=ALPHAS_SIM).round(4)
comp.index.name = "alpha"
comp.to_csv("results/A5_partner_comparison.csv")
print("== Mean tau by alpha: CRITIC vs Equal as the objective partner (all cells pooled)\n", comp.T.to_string())
