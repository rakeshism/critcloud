"""
Main simulation study. Produces results/S*.csv and figures/.
Runtime: about 5-8 minutes on a laptop.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from simulator import run_grid

os.makedirs("results", exist_ok=True)
REPS = 500
KEY = "CRITCLOUD(0.5)"  # main method under test


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p)
    running = 0
    for r, i in enumerate(o):
        running = max(running, (len(p) - r) * p[i]); adj[i] = min(1, running)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


# -------------------------------------------------------------- main grid
df = run_grid(reps=REPS, seed=7)
df.to_csv("results/S0_raw.csv.gz", index=False, compression="gzip")
cells = ["theta", "m", "structure", "rho", "skewed"]

summ = (df.groupby(cells + ["method"])[["tau", "top5", "ndcg10", "regret", "rie"]]
          .mean().round(3).reset_index())
summ.to_csv("results/S1_cell_means.csv", index=False)

# overall view per preference regime and redundancy level
view = (df.groupby(["theta", "structure", "method"])["tau"].mean().unstack("method").round(3))
view.to_csv("results/S2_tau_by_structure.csv")
print("== Mean Kendall tau to TRUE ranking (averaged over m, rho, skew)\n", view.T, "\n")

reg = (df.groupby(["theta", "structure", "method"])["regret"].mean().unstack("method").round(3))
reg.to_csv("results/S3_regret_by_structure.csv")
print("== Mean regret (utility units; lower = better)\n", reg.T, "\n")

# mean rank of each method across all replications (Friedman), per theta regime
ranks_out = {}
for th, g in df.groupby("theta"):
    piv = g.pivot_table(index=cells + ["rep"], columns="method", values="tau")
    stat, p = friedmanchisquare(*[piv[c] for c in piv.columns])
    mean_rank = piv.rank(axis=1, ascending=False).mean().sort_values()
    ranks_out[th] = mean_rank
    print(f"== Friedman on tau, theta={th}: chi2={stat:.1f}, p={p:.2e}; mean rank (1=best)")
    print(mean_rank.round(2).to_string(), "\n")

    # Wilcoxon: KEY vs each competitor, Holm-corrected, with Cliff's delta
    rows = []
    for c in piv.columns:
        if c == KEY: continue
        d = piv[KEY] - piv[c]
        rows.append({"vs": c, "mean_diff_tau": d.mean(), "p_raw": wilcoxon(piv[KEY], piv[c]).pvalue,
                     "cliffs_delta": cliffs_delta(piv[KEY].values, piv[c].values)})
    w = pd.DataFrame(rows); w["p_holm"] = holm(w["p_raw"])
    w.round(4).to_csv(f"results/S4_wilcoxon_{th}.csv", index=False)
    print(f"== {KEY} vs others (theta={th})\n", w.round(4).to_string(index=False), "\n")
pd.DataFrame(ranks_out).round(3).to_csv("results/S5_friedman_mean_ranks.csv")

# -------------------------------------------------------------- preference-noise sweep
dfn = run_grid(reps=REPS, seed=11, ms=(40,), structures=("moderate",), rhos=(0.9,),
               skews=(False,), sigma_prefs=(0.0, 0.3, 0.6, 1.0), theta_modes=("dirichlet",))
noise = dfn.groupby(["sigma_pref", "method"])["tau"].mean().unstack("method").round(3)
noise.to_csv("results/S6_pref_noise.csv")
print("== Tau vs BWM judgement noise (moderate redundancy, rho=0.9, m=40)\n", noise.T)
