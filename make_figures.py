"""Paper figures from results/*.csv (same code as Step 9 of the notebook). Output: paper/figures/"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("paper/figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

d = pd.read_csv("results/T6_circularity.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(d["L2_to_equal"], d["tau_vs_Equal"], color="black")
for m, r in d.iterrows():
    ax.annotate(m.replace("(0.5)", "").replace("GameTheory(CRITIC,BWM)", "Game theory"),
                (r["L2_to_equal"], r["tau_vs_Equal"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("Distance of weight vector from equal weights (L2)")
ax.set_ylabel(r"Kendall $\tau$ vs Equal-TOPSIS")
fig.tight_layout(); fig.savefig("paper/figures/fig_circularity.png", dpi=300); plt.close(fig)

n = pd.read_csv("results/S6_pref_noise.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6, 4))
for m, lab, st in [("BWM", "BWM", "o-"), ("CRITCLOUD(0.5)", r"CRITCLOUD ($\alpha=0.5$)", "s-"),
                   ("BWM+Equal(0.5)", "BWM + Equal", "^--"), ("AHP", "AHP", "d-."),
                   ("CRITIC", "CRITIC", "v:"), ("Oracle (best linear)", "Best-linear oracle", "k-")]:
    ax.plot(n.index, n[m], st, label=lab)
ax.set_xlabel(r"Judgement noise $\sigma$"); ax.set_ylabel(r"Kendall $\tau$ to true ranking")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("paper/figures/fig_pref_noise.png", dpi=300); plt.close(fig)

s = pd.read_csv("results/S2_tau_by_structure.csv", index_col=[0, 1])
sel = ["BWM", "CRITCLOUD(0.5)", "BWM+Equal(0.5)", "CRITIC", "Equal", "OPTCLOUD"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, th, title in zip(axes, ["dirichlet", "uniform"], ["(a) Heterogeneous preferences", "(b) Uniform preferences"]):
    s.loc[th].reindex(["none", "moderate", "high"])[sel].plot(kind="bar", ax=ax, rot=0, legend=False)
    ax.set_title(title); ax.set_xlabel("Redundancy level"); ax.set_ylabel(r"Kendall $\tau$ to true ranking")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, [x.replace("(0.5)", "") for x in l], loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("paper/figures/fig_tau_by_redundancy.png", dpi=300); plt.close(fig)

a = pd.read_csv("results/A1_alpha_tradeoff_qws.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(a["alpha"], a["tau_to_user_BWM"], "o-", label=r"Agreement with user's BWM ranking ($\tau$)")
ax.plot(a["alpha"], a["top1_stable_%"] / 100, "s-", label="Same best provider under BWM errors (share)")
ax.set_xlabel(r"$\alpha$ (0 = user only, 1 = CRITIC only)"); ax.set_ylabel("Value")
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig("paper/figures/fig_alpha_tradeoff.png", dpi=300); plt.close(fig)
print("figures written")
