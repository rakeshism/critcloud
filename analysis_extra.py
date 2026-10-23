"""Targeted follow-up: where (if anywhere) does CRITIC's correlation term add value?"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
df = pd.read_csv("results/S0_raw.csv.gz")
key = ["theta","m","structure","rho","skewed","rep"]
piv = df.pivot_table(index=key, columns="method", values="tau").reset_index()
rows=[]
for (th, st), g in piv.groupby(["theta","structure"]):
    d = g["CRITCLOUD(0.5)"] - g["BWM+Equal(0.5)"]
    rows.append({"theta":th,"structure":st,"dtau_CRITCLOUD_minus_BWM+Equal":d.mean(),
                 "p":wilcoxon(g["CRITCLOUD(0.5)"], g["BWM+Equal(0.5)"]).pvalue,
                 "dtau_CRITIC_minus_Equal":(g["CRITIC"]-g["Equal"]).mean()})
t = pd.DataFrame(rows); t.round(4).to_csv("results/S7_critic_vs_shrinkage.csv", index=False)
print("== Does CRITIC's correlation term add value beyond shrinkage to equal weights?\n", t.round(4).to_string(index=False))
sk = df.groupby(["skewed","method"])["tau"].mean().unstack(0).round(3)
sk["drop"] = (sk[True]-sk[False]).round(3)
sk.to_csv("results/S8_skew_effect.csv")
print("\n== Effect of heavy-tailed criteria (max-min normalisation)\n", sk.sort_values("drop").to_string())
mm = df.groupby(["m","method"])["tau"].mean().unstack(0).round(3)
print("\n== tau by number of providers m\n", mm.to_string())

# Does lower RIE predict better ranking accuracy? (validity check for the RIE metric)
from scipy.stats import spearmanr
d = df.dropna(subset=["rie"])
out=[]
for (th, st), g in d.groupby(["theta","structure"]):
    gm = g.groupby("method")[["rie","tau"]].mean()
    out.append({"theta":th,"structure":st,"spearman(RIE,tau) across methods":spearmanr(gm.rie, gm.tau)[0]})
o = pd.DataFrame(out); o.round(3).to_csv("results/S9_rie_validity.csv", index=False)
print("\n== Is lower RIE associated with higher accuracy? (negative = yes)\n", o.round(3).to_string(index=False))
