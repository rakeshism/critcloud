"""
Reviewer round 2: sensitivity of the simulation conclusions to
  (C1) how the user's stated importance treats duplicated criteria  (pref_mode)
  (C4) the form of the true utility                                 (utility)
Design: 2 preference designs x 3 utility models; each condition is the non-skewed
24-cell grid (theta x structure x m x rho) with 300 replications = 7,200 problems.
Outputs: results/V1_raw.csv.gz, V2_tau_summary.csv, V3_key_differences.csv (bootstrap CIs),
         V4_mean_ranks.csv, V5_best_alpha.csv, V6_sla_feasible.csv
"""
import os, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from simulator import run_grid

os.makedirs("results", exist_ok=True)
REPS, B, SEED = 300, 2000, 101
PREFS = ("per_criterion", "split")
UTILS = ("linear", "sla", "concave")
CELL = ["pref_mode", "utility", "theta", "m", "structure", "rho", "rep"]


def run_condition(args):
    i, (pm, ut) = args
    d = run_grid(reps=REPS, seed=SEED + i, skews=(False,), pref_mode=pm, utility=ut)
    d["pref_mode"], d["utility"] = pm, ut
    return d


def boot_ci(x, rng, b=B):
    x = np.asarray(x); idx = rng.integers(0, len(x), (b, len(x)))
    bm = x[idx].mean(1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


if __name__ == "__main__":
    conds = list(enumerate(itertools.product(PREFS, UTILS)))
    with Pool(min(6, os.cpu_count())) as p:
        df = pd.concat(p.map(run_condition, conds), ignore_index=True)
    df.to_csv("results/V1_raw.csv.gz", index=False, compression="gzip")

    # mean tau of the main methods
    keep = ["BWM", "CRITIC", "Equal", "AHP", "CRITCLOUD(0.5)", "BWM+Equal(0.5)",
            "GameTheory(CRITIC,BWM)", "BWM+Mahalanobis-TOPSIS", "Oracle (best linear)"]
    summ = (df[df.method.isin(keep)].groupby(["pref_mode", "utility", "theta", "structure", "method"])["tau"]
              .mean().unstack("method")[keep].round(3))
    summ.to_csv("results/V2_tau_summary.csv"); print(summ.to_string(), "\n")

    # paired differences with bootstrap 95% CI
    piv = df.pivot_table(index=CELL, columns="method", values="tau")
    rng = np.random.default_rng(SEED)
    pairs = [("CRITCLOUD(0.5)", "BWM+Equal(0.5)"), ("CRITIC", "Equal"), ("CRITCLOUD(0.5)", "BWM")]
    rows = []
    for (pm, ut, th, st), g in piv.groupby(level=["pref_mode", "utility", "theta", "structure"]):
        for a, b in pairs:
            mu, lo, hi = boot_ci((g[a] - g[b]).values, rng)
            rows.append(dict(pref_mode=pm, utility=ut, theta=th, structure=st, contrast=f"{a} - {b}",
                             mean=mu, ci_lo=lo, ci_hi=hi, n=len(g)))
    v3 = pd.DataFrame(rows).round(4); v3.to_csv("results/V3_key_differences.csv", index=False)
    print(v3[v3.contrast.str.startswith("CRITCLOUD(0.5) - BWM+")].to_string(index=False), "\n")

    # Friedman mean ranks per condition and preference regime
    ranks = {}
    for (pm, ut, th), g in piv.groupby(level=["pref_mode", "utility", "theta"]):
        ranks[(pm, ut, th)] = g.rank(axis=1, ascending=False).mean()
    v4 = pd.DataFrame(ranks).round(2); v4.to_csv("results/V4_mean_ranks.csv")
    print(v4.loc[keep].to_string(), "\n")

    # best alpha on the grid {0 (BWM), .25, .5, .75, 1 (CRITIC)}
    amap = {"BWM": 0.0, "CRITCLOUD(0.25)": 0.25, "CRITCLOUD(0.5)": 0.5, "CRITCLOUD(0.75)": 0.75, "CRITIC": 1.0}
    a = (df[df.method.isin(amap)].groupby(["pref_mode", "utility", "theta", "structure", "method"])["tau"].mean()
           .unstack("method").rename(columns=amap))
    a = a[sorted(a.columns)]
    a["best_alpha"] = a.idxmax(axis=1); a["gap_0.5_to_best"] = a[sorted(amap.values())].max(axis=1) - a[0.5]
    a.round(3).to_csv("results/V5_best_alpha.csv"); print(a.round(3).to_string(), "\n")

    # SLA: share of problems where the selected provider satisfies the hard SLA
    s = (df[df.utility == "sla"].groupby(["pref_mode", "theta", "method"])["top1_feasible"].mean()
           .unstack("method")[keep].round(3))
    s.to_csv("results/V6_sla_feasible.csv"); print(s.to_string())
