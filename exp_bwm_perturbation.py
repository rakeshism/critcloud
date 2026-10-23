"""
Task 11 - BWM perturbation (anti-noise) test on the real QWS data.

Each trial adds a random integer in [-delta, +delta] to every free entry of A_B and A_W
(clipped to 1..9; a_BB = a_WW = 1 kept; a_BW kept equal in both vectors), re-solves BWM,
and measures how far each method's ranking moves from its own unperturbed ranking.
Answers Reviewer 1: "perturb BWM comparison entries and observe ranking fluctuation".
"""
import os
import numpy as np
import pandas as pd
from critcloud_core import *

os.makedirs("results", exist_ok=True)
TRIALS, DELTAS, SEED = 1000, (1, 2), 31

X = load_qws("qws.csv"); N = normalize(X)
wc, we = critic_weights(N), equal_weights()
B, Wst = AB_FINAL.index(1), AW_FINAL.index(1)


def method_scores(wb):
    return {
        "BWM": topsis(N, wb),
        "CRITCLOUD(0.25)": topsis(N, blend(wc, wb, 0.25)),
        "CRITCLOUD(0.5)": topsis(N, blend(wc, wb, 0.5)),
        "CRITCLOUD(0.75)": topsis(N, blend(wc, wb, 0.75)),
        "BWM+Equal(0.5)": topsis(N, blend(we, wb, 0.5)),
        "GameTheory(CRITIC,BWM)": topsis(N, game_theory_combination(wc, wb)[0]),
        "BWM+Mahalanobis-TOPSIS": mahalanobis_topsis(N, wb),
    }


def perturb(rng, delta):
    AB, AW = np.array(AB_FINAL), np.array(AW_FINAL)
    AB = np.clip(AB + rng.integers(-delta, delta + 1, len(AB)), 1, 9)
    AW = np.clip(AW + rng.integers(-delta, delta + 1, len(AW)), 1, 9)
    AB[B], AW[Wst] = 1, 1
    aBW = int(np.clip(AB_FINAL[Wst] + rng.integers(-delta, delta + 1), 2, 9))
    AB[Wst], AW[B] = aBW, aBW
    return list(AB), list(AW)


wb0, _ = bwm_weights(AB_FINAL, AW_FINAL)
base = method_scores(wb0)
base_rank = {m: rank_of(s) for m, s in base.items()}
base_top10 = {m: set(s.nlargest(10).index) for m, s in base.items()}
base_top1 = {m: s.idxmax() for m, s in base.items()}

rng = np.random.default_rng(SEED)
rows = []
for d in DELTAS:
    for t in range(TRIALS):
        AB, AW = perturb(rng, d)
        wb, _ = bwm_weights(AB, AW, best=B, worst=Wst)
        cr, _ = input_based_cr(AB, AW)
        for m, s in method_scores(wb).items():
            rows.append({"delta": d, "trial": t, "method": m,
                         "tau_to_base": tau(rank_of(s), base_rank[m]),
                         "top10_overlap": len(set(s.nlargest(10).index) & base_top10[m]) / 10,
                         "top1_same": float(s.idxmax() == base_top1[m]),
                         "CR_I": cr})
df = pd.DataFrame(rows)
summ = df.groupby(["delta", "method"]).agg(
    mean_tau=("tau_to_base", "mean"),
    worst5pct_tau=("tau_to_base", lambda x: np.percentile(x, 5)),
    top10_overlap=("top10_overlap", "mean"),
    top1_unchanged_pct=("top1_same", lambda x: 100 * x.mean())).round(4)
summ.to_csv("results/T9_bwm_perturbation.csv")
print("== BWM perturbation on QWS (%d trials per delta; 1.0 = ranking unchanged)" % TRIALS)
print(summ.to_string())
