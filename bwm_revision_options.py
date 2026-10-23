"""
Task 12 - BWM vector check and minimal-change revision options.

Cardinal consistency: input-based CR^I (Liang, Brunelli & Rezaei, 2020) compared with
their threshold for a_BW = 8, n = 9 criteria: 0.3657 (Table 3 of that paper, as coded in
github.com/IgorKollar-EFUMB/classical-bwm-r; verify against the original before citing).

Ordinal consistency: the two vectors must agree on which of two criteria is more important
(a_Bi < a_Bj  =>  a_iW >= a_jW).

The search finds integer vectors with zero ordinal violations that change the stated
judgements as little as possible (sum of absolute changes), keeping best = RT, worst = TP
and a_BW = 8 fixed.
"""
import itertools
import os
import numpy as np
import pandas as pd
from critcloud_core import *

os.makedirs("results", exist_ok=True)
THRESHOLD = 0.3657
AB0, AW0 = np.array(AB_PAPER), np.array(AW_PAPER)
B, Wst, aBW = 0, 2, 8
free = [j for j in range(9) if j not in (B, Wst)]


def candidates(j, radius):
    out = []
    for db in range(-radius, radius + 1):
        for dw in range(-radius, radius + 1):
            if abs(db) + abs(dw) > radius:
                continue
            b, w = AB0[j] + db, AW0[j] + dw
            if 1 <= b <= 9 and 1 <= w <= 9:
                out.append((abs(db) + abs(dw), b, w))
    return sorted(out)


def search(max_cr, radius=3, keep=3):
    """Branch-and-bound over free criteria; returns the `keep` lowest-change solutions."""
    cands = {j: [c for c in candidates(j, radius) if abs(c[1] * c[2] - aBW) / (aBW**2 - aBW) <= max_cr]
             for j in free}
    best = []

    def ok(assign):
        items = list(assign.items()) + [(B, (1, aBW)), (Wst, (aBW, 1))]
        for (i, (bi, wi)), (k, (bk, wk)) in itertools.combinations(items, 2):
            if (bi - bk) * (wk - wi) < 0:
                return False
        return True

    def rec(idx, assign, cost):
        if len(best) == keep and cost >= best[-1][0]:
            return
        if idx == len(free):
            best.append((cost, dict(assign))); best.sort(key=lambda x: x[0])
            del best[keep:]
            return
        j = free[idx]
        for c, b, w in cands[j]:
            assign[j] = (b, w)
            if ok(assign):
                rec(idx + 1, assign, cost + c)
            del assign[j]

    rec(0, {}, 0)
    return best


def vectors(assign):
    AB, AW = AB0.copy(), AW0.copy()
    for j, (b, w) in assign.items():
        AB[j], AW[j] = b, w
    return list(AB), list(AW)


rows, weights = [], {}
cr0, _ = input_based_cr(AB_PAPER, AW_PAPER)
w0, xi0 = bwm_weights(AB_PAPER, AW_PAPER)
rows.append({"option": "Original (paper)", "A_B": AB_PAPER, "A_W": AW_PAPER, "changes": 0,
             "CR_I": round(cr0, 3), "passes_threshold": cr0 <= THRESHOLD,
             "ordinal_violations": len(ordinal_violations(AB_PAPER, AW_PAPER)), "xi_L": round(xi0, 4)})
weights["Original (paper)"] = w0

opts = [("A: minimal change, ordinal fixed", search(max_cr=THRESHOLD)),
        ("B: minimal change, ordinal fixed, CR_I <= 0.10", search(max_cr=0.10))]
for name, sols in opts:
    cost, assign = sols[0]
    AB, AW = vectors(assign)
    cr, _ = input_based_cr(AB, AW); w, xi = bwm_weights(AB, AW, best=B, worst=Wst)
    rows.append({"option": name, "A_B": AB, "A_W": AW, "changes": cost, "CR_I": round(cr, 3),
                 "passes_threshold": cr <= THRESHOLD,
                 "ordinal_violations": len(ordinal_violations(AB, AW)), "xi_L": round(xi, 4)})
    weights[name] = w

# Option C: keep the decision-maker's best-to-others answers, derive others-to-worst from them
AB = list(AB0); AW = [int(np.clip(round(aBW / a), 1, 9)) for a in AB0]; AW[Wst] = 1
cr, _ = input_based_cr(AB, AW); w, xi = bwm_weights(AB, AW, best=B, worst=Wst)
rows.append({"option": "C: keep A_B, derive A_W = a_BW / a_Bj", "A_B": AB, "A_W": AW,
             "changes": int(np.abs(np.array(AW) - AW0).sum()), "CR_I": round(cr, 3),
             "passes_threshold": cr <= THRESHOLD,
             "ordinal_violations": len(ordinal_violations(AB, AW)), "xi_L": round(xi, 4)})
weights["C: keep A_B, derive A_W = a_BW / a_Bj"] = w

opt = pd.DataFrame(rows)
opt.to_csv("results/T10_bwm_options.csv", index=False)
wt = pd.DataFrame(weights, index=SHORT).round(4)
wt.to_csv("results/T11_bwm_option_weights.csv")
pd.set_option("display.width", 220)
print("Threshold for a_BW = 8, n = 9:", THRESHOLD)
print("Ordinal violations in original:", [(SHORT[i], SHORT[j]) for i, j in ordinal_violations(AB_PAPER, AW_PAPER)])
print(opt.to_string(index=False)); print("\nBWM weights for each option\n", wt.to_string())
