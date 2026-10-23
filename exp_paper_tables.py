"""
Re-runs every real-data table of the CRITCLOUD paper with the corrected,
consolidated code. Writes CSVs to results/.

Fixes applied relative to the original notebooks:
  * BWM weights are the true LP solution of the stated A_B / A_W vectors.
  * CRITIC / entropy weights always from the full dataset (as the paper states).
  * OPTCLOUD eigenvector signs fixed (deterministic).
  * Rank reversal = share of provider pairs whose order flips; removal AND addition.
  * No Friedman test on closeness coefficients (invalid comparison).
"""
import os
import numpy as np
import pandas as pd
from critcloud_core import *

os.makedirs("results", exist_ok=True)
X = load_qws("qws.csv")
N = normalize(X)
R = corr(N)

# ---------------------------------------------------------------- BWM check
wb, xi = bwm_weights(AB_FINAL, AW_FINAL)
cr, per = input_based_cr(AB_FINAL, AW_FINAL)
bwm_tab = pd.DataFrame({"a_Bj": AB_FINAL, "a_jW": AW_FINAL,
                        "a_Bj*a_jW": np.array(AB_FINAL) * np.array(AW_FINAL),
                        "CR_I_j": per.round(3), "w_LP": wb.round(4)}, index=SHORT)
bwm_tab.to_csv("results/T1_bwm_consistency.csv")
print("== BWM (linear) xi* =", round(xi, 4), " input-based CR =", round(cr, 3))
print(bwm_tab, "\n")

# ---------------------------------------------------------------- weights
W = {
    "Equal": equal_weights(),
    "Entropy": entropy_weights(N),
    "SD": sd_weights(N),
    "MEREC": merec_weights(N),
    "CRITIC": critic_weights(N),
    "BWM": wb,
}
W["CRITCLOUD(0.5)"] = blend(W["CRITIC"], wb, 0.5)
# Additional baselines. QWS has no separate AHP or DEMATEL expert data, so:
#  - AHP matrix is built from the same user preferences as BWM (Saaty scale)
#  - DEMATEL direct-relation matrix is derived from |correlation| (data-driven DEMATEL)
W_AHP, CR_AHP = ahp_weights(ahp_matrix_from_pref(wb))
W["CRITIC-AHP"] = multiplicative(W["CRITIC"], W_AHP)
W["CRITIC-DEMATEL"] = multiplicative(W["CRITIC"], dematel_weights(R))
W["GameTheory(CRITIC,BWM)"], GT_COEF = game_theory_combination(W["CRITIC"], wb)
print("AHP CR =", round(CR_AHP, 3), "| game-theory coefficients (CRITIC, BWM) =", GT_COEF.round(3))
wt = pd.DataFrame(W, index=SHORT)
wt.loc["HHI"] = [hhi(w) for w in W.values()]
wt.round(4).to_csv("results/T2_weights.csv")
print("== Weights\n", wt.round(4), "\n")

diag = pd.DataFrame({m: {"RIE": rie(w, R), "RIE_norm": rie_norm(w, R),
                         "H(w)": shannon(w), "RIE_adj": rie(w, R) / shannon(w),
                         "HHI": hhi(w)} for m, w in W.items()}).T
diag.round(5).to_csv("results/T3_redundancy.csv")
print("== Redundancy diagnostics\n", diag.round(5), "\n")

# ---------------------------------------------------------------- rankings, full data
scores = {m: topsis(N, w) for m, w in W.items()}
scores["OPTCLOUD(sign-fixed)"] = optcloud(N, wb, fix_sign=True)[0]
scores["BWM+Mahalanobis-TOPSIS"] = mahalanobis_topsis(N, wb)
ranks = {m: rank_of(s) for m, s in scores.items()}

# SLA reference exactly as in the original notebook (for transparency)
viol = ((X["Response Time"] > 200).astype(int) + (X["Availability"] < 95).astype(int)
        + (X["Successability"] < 90).astype(int))
sla_rank = (viol * 1000 + X["Response Time"]).rank(method="average")

agree = pd.DataFrame({m: {"tau_vs_Equal": tau(r, ranks["Equal"]),
                          "tau_vs_SLA": tau(r, sla_rank)} for m, r in ranks.items()}).T
agree.round(4).to_csv("results/T4_agreement_full.csv")
print("== Kendall tau, full dataset (n=%d)\n" % len(X), agree.round(4), "\n")

# pairwise method agreement matrix (shows how close CRITCLOUD is to Equal by construction)
M = pd.DataFrame({a: {b: tau(ranks[a], ranks[b]) for b in ranks} for a in ranks})
M.round(3).to_csv("results/T5_method_tau_matrix.csv")

# distance of each weight vector to equal weights vs its tau to Equal-TOPSIS
dist = pd.DataFrame({m: {"L2_to_equal": float(np.linalg.norm(w - 1 / 9)),
                         "tau_vs_Equal": agree.loc[m, "tau_vs_Equal"]}
                     for m, w in W.items() if m != "Equal"}).T
dist.round(4).to_csv("results/T6_circularity.csv")
print("== Circularity check: closer to equal weights -> higher tau\n", dist.round(4),
      "\n   Spearman(L2, tau) =", round(rho(dist["L2_to_equal"], dist["tau_vs_Equal"]), 3), "\n")

# ---------------------------------------------------------------- rank reversal (n=40)
rng = np.random.default_rng(2024)
methods = ["Equal", "Entropy", "CRITIC", "BWM", "CRITCLOUD(0.5)", "CRITIC-AHP", "CRITIC-DEMATEL",
           "GameTheory(CRITIC,BWM)", "BWM+Mahalanobis-TOPSIS", "OPTCLOUD(sign-fixed)"]


def score_set(S, m):
    NS = normalize(S)
    if m == "OPTCLOUD(sign-fixed)":
        return optcloud(NS, wb, fix_sign=True)[0]
    if m == "BWM+Mahalanobis-TOPSIS":
        return mahalanobis_topsis(NS, wb)
    return topsis(NS, W[m])  # weights fixed from full dataset


TRIALS, SIZE = 300, 40
rr = {m: {"remove": [], "add": [], "add_copy": []} for m in methods}
for t in range(TRIALS):
    idx = rng.choice(len(X), SIZE + 1, replace=False)
    S, extra = X.iloc[idx[:SIZE]], X.iloc[[idx[SIZE]]]
    drop = rng.choice(S.index)
    copy = S.loc[[rng.choice(S.index)]].copy(); copy.index = [-1]
    for m in methods:
        base = score_set(S, m)
        rr[m]["remove"].append(pairwise_reversal_rate(base, score_set(S.drop(index=drop), m)))
        rr[m]["add"].append(pairwise_reversal_rate(base, score_set(pd.concat([S, extra]), m)))
        rr[m]["add_copy"].append(pairwise_reversal_rate(base, score_set(pd.concat([S, copy]), m)))
rrt = pd.DataFrame({m: {k: 100 * np.mean(v) for k, v in d.items()} for m, d in rr.items()}).T
rrt.columns = ["remove_%", "add_new_%", "add_duplicate_%"]
rrt.round(3).to_csv("results/T7_rank_reversal.csv")
print("== Pairwise rank-reversal rate (%%), %d random 40-provider sets\n" % TRIALS, rrt.round(3), "\n")

# ---------------------------------------------------------------- normalisation skew check
skew = pd.DataFrame({"skewness": X.skew(), "share_norm>0.9": (N > 0.9).mean()}, index=CRITERIA)
skew.index = SHORT
skew.round(3).to_csv("results/T8_skew.csv")
print("== Skew and max-min compression\n", skew.round(3))
