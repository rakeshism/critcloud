"""
Reviewer round 2, Concern 2: uncertainty around the reported differences.
  U1: bootstrap 95% CI of the mean paired difference in Kendall tau (main simulation, S0_raw)
      for CRITCLOUD(0.5) vs every competitor, per preference regime.
  U2: the same for CRITCLOUD - (BWM+Equal) and CRITIC - Equal by redundancy level (Table shrinkage).
  U3: Wilson 95% CI for the 'same best provider' rates of the QWS perturbation study (T9),
      plus a CI for the difference CRITCLOUD(0.5) - competitor (unpaired, conservative).
"""
import numpy as np, pandas as pd

B, SEED = 2000, 2024
rng = np.random.default_rng(SEED)
cells = ["theta", "m", "structure", "rho", "skewed", "rep"]
df = pd.read_csv("results/S0_raw.csv.gz")
piv = df.pivot_table(index=cells, columns="method", values="tau")


def boot(x):
    x = np.asarray(x); bm = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


rows = []
for th, g in piv.groupby(level="theta"):
    for c in g.columns:
        if c == "CRITCLOUD(0.5)": continue
        mu, lo, hi = boot(g["CRITCLOUD(0.5)"] - g[c])
        rows.append(dict(theta=th, vs=c, mean_diff=mu, ci_lo=lo, ci_hi=hi, n=len(g)))
u1 = pd.DataFrame(rows).round(4); u1.to_csv("results/U1_pairwise_ci.csv", index=False)
print(u1.to_string(index=False), "\n")

rows = []
for (th, st), g in piv.groupby(level=["theta", "structure"]):
    for a, b in [("CRITCLOUD(0.5)", "BWM+Equal(0.5)"), ("CRITIC", "Equal")]:
        mu, lo, hi = boot(g[a] - g[b])
        rows.append(dict(theta=th, structure=st, contrast=f"{a} - {b}", mean=mu, ci_lo=lo, ci_hi=hi, n=len(g)))
u2 = pd.DataFrame(rows).round(4); u2.to_csv("results/U2_shrinkage_ci.csv", index=False)
print(u2.to_string(index=False), "\n")


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


t9 = pd.read_csv("results/T9_bwm_perturbation.csv"); n = 1000
rows = []
for d, g in t9.groupby("delta"):
    ref = g.set_index("method").loc["CRITCLOUD(0.5)", "top1_unchanged_pct"] / 100
    for _, r in g.iterrows():
        p = r.top1_unchanged_pct / 100; lo, hi = wilson(p, n)
        diff = ref - p; se = np.sqrt(ref * (1 - ref) / n + p * (1 - p) / n)
        rows.append(dict(delta=d, method=r.method, pct=100 * p, ci_lo=100 * lo, ci_hi=100 * hi,
                         diff_vs_critcloud=100 * diff, diff_lo=100 * (diff - 1.96 * se), diff_hi=100 * (diff + 1.96 * se)))
u3 = pd.DataFrame(rows).round(1); u3.to_csv("results/U3_perturbation_ci.csv", index=False)
print(u3.to_string(index=False))
