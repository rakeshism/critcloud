"""
Builds every numeric LaTeX table of the paper directly from results/*.csv,
so no number is ever typed by hand. Output: paper/tables/*.tex
Run after the notebook (or the individual experiment scripts).
"""
import os
import pandas as pd
import numpy as np
from critcloud_core import AB_FINAL, AW_FINAL, input_based_cr, bwm_weights

OUT = "paper/tables"
os.makedirs(OUT, exist_ok=True)
FULL = {"RT": "Response Time", "Ava": "Availability", "TP": "Throughput", "Suc": "Successability",
        "Rel": "Reliability", "Com": "Compliance", "BP": "Best Practices", "Lat": "Latency",
        "Doc": "Documentation"}


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", os.path.join(OUT, name))


# ---------------------------------------------------------------- Table: BWM inputs (Option A)
t = pd.read_csv("results/T1_bwm_consistency.csv", index_col=0)
cr, _ = input_based_cr(AB_FINAL, AW_FINAL)
_, xi = bwm_weights(AB_FINAL, AW_FINAL)
rows = "\n".join(
    f"{FULL[k]} ({k}) & {int(r.a_Bj)} & {int(r.a_jW)} & {int(r['a_Bj*a_jW'])} & {r.CR_I_j:.3f} & {r.w_LP:.4f} \\\\"
    for k, r in t.iterrows())
write("tab_bwm.tex", rf"""\begin{{table}}[!htbp]
\centering
\caption{{Best-to-others ($a_{{Bj}}$) and others-to-worst ($a_{{jW}}$) judgements used for QWS, their
local input-based consistency ratios $CR^I_j$ and the resulting linear-BWM weights. Best criterion:
Response Time; worst: Throughput; $a_{{BW}}=8$.}}
\label{{tab:bwm}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Criterion & $a_{{Bj}}$ & $a_{{jW}}$ & $a_{{Bj}}a_{{jW}}$ & $CR^I_j$ & $w^{{\mathrm{{BWM}}}}_j$\\
\midrule
{rows}
\midrule
\multicolumn{{6}}{{l}}{{Global $CR^I={cr:.3f}$ (threshold $0.3657$ for $a_{{BW}}=8$, $n=9$); $\xi^{{L*}}={xi:.4f}$; no ordinal conflicts.}}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""")


# =====================================================================
# RESULTS SECTION TABLES
# =====================================================================
NAME = {"Equal": "Equal", "Entropy": "Entropy", "SD": "SD", "MEREC": "MEREC", "CRITIC": "CRITIC",
        "BWM": "BWM", "AHP": "AHP", "CRITCLOUD(0.25)": r"CRITCLOUD ($\alpha=0.25$)",
        "CRITCLOUD(0.5)": r"\textbf{CRITCLOUD ($\alpha=0.5$)}", "CRITCLOUD(0.75)": r"CRITCLOUD ($\alpha=0.75$)",
        "BWM+Equal(0.5)": r"BWM\,+\,Equal", "BWM+Entropy(0.5)": r"BWM\,+\,Entropy",
        "CRITIC-AHP": "CRITIC-AHP", "CRITIC-DEMATEL": "CRITIC-DEMATEL",
        "GameTheory(CRITIC,BWM)": "Game-theory (CRITIC, BWM)",
        "BWM+Mahalanobis-TOPSIS": r"BWM\,+\,Mahalanobis-TOPSIS", "OPTCLOUD": "OPTCLOUD",
        "OPTCLOUD(sign-fixed)": "OPTCLOUD", "Naive-oracle (theta/copies)": "Naive oracle",
        "Oracle (best linear)": "Best-linear oracle"}
f3 = lambda v: f"{v:.3f}"

# ---- Table: circularity of the equal-weight reference (T6 + T4)
t6 = pd.read_csv("results/T6_circularity.csv", index_col=0)
t4 = pd.read_csv("results/T4_agreement_full.csv", index_col=0)
order = t6.sort_values("L2_to_equal").index
rows = "\n".join(f"{NAME.get(m, m)} & {t6.loc[m,'L2_to_equal']:.3f} & {t4.loc[m,'tau_vs_Equal']:.3f} & {t4.loc[m,'tau_vs_SLA']:.3f} \\\\" for m in order)
rows += "\n\\midrule\n" + "\n".join(f"{NAME.get(m, m)} & -- & {t4.loc[m,'tau_vs_Equal']:.3f} & {t4.loc[m,'tau_vs_SLA']:.3f} \\\\"
                                  for m in ["OPTCLOUD(sign-fixed)", "BWM+Mahalanobis-TOPSIS"])
from scipy.stats import spearmanr
rho = spearmanr(t6["L2_to_equal"], t6["tau_vs_Equal"])[0]
write("tab_circularity.tex", rf"""\begin{{table}}[!htbp]
\centering
\caption{{Distance of each weight vector from equal weights and its Kendall $\tau$ agreement with the
equal-weight (Equal-TOPSIS) and SLA-violation reference rankings on the full QWS dataset ($m=2{{,}}500$),
sorted by distance. The last two methods do not produce a single weight vector. Spearman correlation between distance and $\tau$ with the equal-weight reference:
$\rho={rho:.2f}$.}}
\label{{tab:circularity}}
\begin{{tabular}}{{lrrr}}
\toprule
Weighting scheme & $\lVert\mathbf w-\mathbf 1/n\rVert_2$ & $\tau$ vs Equal-TOPSIS & $\tau$ vs SLA ranking\\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""")

# ---- Table: simulation accuracy (S2 + S5)
s2 = pd.read_csv("results/S2_tau_by_structure.csv", index_col=[0, 1])
s5 = pd.read_csv("results/S5_friedman_mean_ranks.csv", index_col=0)
methods = s5.sort_values("dirichlet").index
rows = []
for m in methods:
    c = [f3(s2.loc[("dirichlet", st), m]) for st in ("none", "moderate", "high")] + [f"{s5.loc[m,'dirichlet']:.2f}"]
    c += [f3(s2.loc[("uniform", st), m]) for st in ("none", "moderate", "high")] + [f"{s5.loc[m,'uniform']:.2f}"]
    rows.append(f"{NAME.get(m, m)} & " + " & ".join(c) + r" \\")
write("tab_sim_accuracy.tex", r"""\begin{table*}[!tp]
\centering
\caption{Ranking accuracy against the true ranking in the ground-truth simulation: mean Kendall $\tau$ by
redundancy level (averaged over $m$, $\rho$ and distribution; 4,000 problems per cell) and mean Friedman rank
over all 12,000 problems of each preference regime (1 = best). Methods are sorted by mean rank under
heterogeneous preferences.}
\label{tab:sim_accuracy}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{Heterogeneous preferences} & \multicolumn{4}{c}{Uniform preferences}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & None & Moderate & High & Rank & None & Moderate & High & Rank\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table: CRITCLOUD(0.5) vs each method (S4)
wd = pd.read_csv("results/S4_wilcoxon_dirichlet.csv", index_col=0)
wu = pd.read_csv("results/S4_wilcoxon_uniform.csv", index_col=0)
sel = ["BWM+Equal(0.5)", "AHP", "CRITIC-AHP", "Oracle (best linear)", "BWM", "GameTheory(CRITIC,BWM)",
       "BWM+Mahalanobis-TOPSIS", "OPTCLOUD", "CRITIC", "Equal", "CRITIC-DEMATEL"]
def pfmt(p): return "$<0.001$" if p < 0.001 else f"{p:.3f}"
u1 = pd.read_csv("results/U1_pairwise_ci.csv").set_index(["theta", "vs"])
def dci(th, m):
    r = u1.loc[(th, m)]
    return f"{r.mean_diff:+.4f} & [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
rows = "\n".join(
    f"{NAME.get(m, m)} & {dci('dirichlet', m)} & {pfmt(wd.loc[m,'p_holm'])} & {wd.loc[m,'cliffs_delta']:+.3f}"
    f" & {dci('uniform', m)} & {pfmt(wu.loc[m,'p_holm'])} & {wu.loc[m,'cliffs_delta']:+.3f} \\\\" for m in sel)
write("tab_pairwise.tex", r"""\begin{table*}[!tp]
\centering
\caption{Paired comparison of CRITCLOUD ($\alpha=0.5$) with selected methods over 12,000 simulated problems per
preference regime: mean difference in Kendall $\tau$ (positive = CRITCLOUD more accurate), Holm-adjusted
Wilcoxon $p$-value and Cliff's $\delta$ ($|\delta|<0.147$ negligible, $<0.33$ small, $<0.474$ medium).
95\% CI: percentile bootstrap of the mean paired difference (2,000 resamples).}
\label{tab:pairwise}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrcrrrcrr}
\toprule
 & \multicolumn{4}{c}{Heterogeneous preferences} & \multicolumn{4}{c}{Uniform preferences}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
CRITCLOUD vs & $\Delta\tau$ & 95\% CI & $p_{\mathrm{Holm}}$ & $\delta$ & $\Delta\tau$ & 95\% CI & $p_{\mathrm{Holm}}$ & $\delta$\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table: does CRITIC add value beyond shrinkage? (S7)
u2 = pd.read_csv("results/U2_shrinkage_ci.csv").set_index(["theta", "structure", "contrast"])
lab = {"dirichlet": "Heterogeneous", "uniform": "Uniform"}
def sci(th, st, c):
    r = u2.loc[(th, st, c)]
    return f"{r['mean']:+.4f} & [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
rows = "\n".join(f"{lab[th]} & {st} & {sci(th, st, 'CRITCLOUD(0.5) - BWM+Equal(0.5)')} & {sci(th, st, 'CRITIC - Equal')} \\\\"
                 for th in ("dirichlet", "uniform") for st in ("none", "moderate", "high"))
write("tab_shrinkage.tex", r"""\begin{table}[!htbp]
\centering
\caption{Does CRITIC's correlation handling add accuracy beyond simple shrinkage towards uniform weights?
Mean difference in Kendall $\tau$ with bootstrap 95\% CI (4,000 simulated problems per row).}
\label{tab:shrinkage}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrcrc}
\toprule
 & & \multicolumn{2}{c}{CRITCLOUD $-$ (BWM\,+\,Equal)} & \multicolumn{2}{c}{CRITIC $-$ Equal}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Preferences & Redundancy & $\Delta\tau$ & 95\% CI & $\Delta\tau$ & 95\% CI\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table: redundancy decomposition on QWS (R1)
r1 = pd.read_csv("results/R1_rie_decomposition.csv", index_col=0)
rows = "\n".join(f"{NAME.get(m, m)} & {r.RIE:.4f} & {r['1/2(1-HHI)']:.4f} & {r.WMR:.4f} & {r['N_eff=1/HHI']:.2f} \\\\" for m, r in r1.iterrows())
write("tab_rie_decomp.tex", r"""\begin{table}[!htbp]
\centering
\caption{Redundancy diagnostics of the weight vectors on QWS: RIE and its decomposition
$\mathrm{RIE}=\tfrac12(1-\mathrm{HHI})\times\mathrm{WMR}$ (Proposition~\ref{prop:decomp}), with the
effective number of criteria $N_{\mathrm{eff}}=1/\mathrm{HHI}$.}
\label{tab:rie_decomp}
\begin{tabular}{lrrrr}
\toprule
Weighting scheme & RIE & $\tfrac12(1-\mathrm{HHI})$ & WMR & $N_{\mathrm{eff}}$\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table: BWM perturbation (T9)
t9 = pd.read_csv("results/T9_bwm_perturbation.csv")
order = ["BWM", "CRITCLOUD(0.25)", "CRITCLOUD(0.5)", "CRITCLOUD(0.75)", "BWM+Equal(0.5)",
         "GameTheory(CRITIC,BWM)", "BWM+Mahalanobis-TOPSIS"]
def g(d, m, c): return t9[(t9.delta == d) & (t9.method == m)][c].iloc[0]
u3 = pd.read_csv("results/U3_perturbation_ci.csv").set_index(["delta", "method"])
def sb(d, m):
    r = u3.loc[(d, m)]
    return f"{r.pct:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}]"
rows = "\n".join(f"{NAME.get(m, m)} & {sb(1,m)} & {g(1,m,'mean_tau'):.3f} & {g(1,m,'top10_overlap'):.2f}"
                 f" & {sb(2,m)} & {g(2,m,'mean_tau'):.3f} & {g(2,m,'top10_overlap'):.2f} \\\\" for m in order)
write("tab_perturbation.tex", r"""\begin{table*}[!tp]
\centering
\caption{Simulated preference robustness on QWS: stability of the ranking when the single set of BWM judgements is
perturbed. Every judgement is changed by a random integer in $\{-\delta,\dots,\delta\}$ (1,000 trials per $\delta$).
This measures stability, not selection accuracy. Same best: share of trials in which the best provider is
unchanged, with Wilson 95\% CI; $\tau$: mean Kendall $\tau$ with the unperturbed ranking; Top-10: mean overlap of the top-10 sets.}
\label{tab:perturbation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{$\delta=1$} & \multicolumn{3}{c}{$\delta=2$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & Same best (\%) & $\tau$ & Top-10 & Same best (\%) & $\tau$ & Top-10\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- Table: rank reversal (T7)
t7 = pd.read_csv("results/T7_rank_reversal.csv", index_col=0)
rows = "\n".join(f"{NAME.get(m, m)} & {r['remove_%']:.3f} & {r['add_new_%']:.3f} & {r['add_duplicate_%']:.3f} \\\\" for m, r in t7.iterrows())
write("tab_rank_reversal.tex", r"""\begin{table}[!htbp]
\centering
\caption{Pairwise rank-reversal rate (\% of provider pairs whose order flips) over 300 random sets of
40 QWS providers when one provider is removed, a new provider is added, or a duplicate is added.}
\label{tab:rank_reversal}
\begin{tabular}{lrrr}
\toprule
Method & Remove & Add new & Add duplicate\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table: normalisation (N1 + N3)
n1 = pd.read_csv("results/N1_normalisation_qws.csv", index_col=0)
n3 = pd.read_csv("results/N3_normalisation_sim.csv", index_col=[0, 1])
cols = ["max-min", "log + max-min", "rank"]
def row(label, vals): return label + " & " + " & ".join(vals) + r" \\"
rows = [row(r"QWS: Latency values above 0.9 (\%)", [f"{100*n1.loc[c,'share_Lat_norm>0.9']:.1f}" for c in cols]),
        row(r"QWS: Response Time values above 0.9 (\%)", [f"{100*n1.loc[c,'share_RT_norm>0.9']:.1f}" for c in cols]),
        row(r"QWS: CRITIC weight of Latency", [f"{n1.loc[c,'CRITIC_w_Lat']:.3f}" for c in cols]),
        row(r"Simulation, skewed data: CRITCLOUD $\tau$", [f3(n3.loc[(True, 'CRITCLOUD(0.5)'), c]) for c in cols]),
        row(r"Simulation, Gaussian data: CRITCLOUD $\tau$", [f3(n3.loc[(False, 'CRITCLOUD(0.5)'), c]) for c in cols]),
        row(r"Simulation, skewed data: BWM $\tau$", [f3(n3.loc[(True, 'BWM'), c]) for c in cols])]
write("tab_normalisation.tex", r"""\begin{table}[!htbp]
\centering
\caption{Effect of the normalization method on QWS and in the simulation (1,200 problems per cell).}
\label{tab:normalisation}
\begin{tabular}{lrrr}
\toprule
 & Max-min & Log + max-min & Rank\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table: best alpha (A3)
a3 = pd.read_csv("results/A3_best_alpha_by_noise_redundancy.csv")
rows = "\n".join(f"{r.structure} & {r.sigma:.1f} & {r.best_alpha:.1f} & {r.near_best_range.replace('-', '--')} & {r.tau_at_best:.3f} & {r['tau_at_0.5']:.3f} & {r['tau_BWM_only(a=0)']:.3f} \\\\"
                 for _, r in a3.iterrows())
write("tab_alpha.tex", r"""\begin{table}[!htbp]
\centering
\caption{Best $\alpha$ by redundancy level and judgement noise $\sigma$ (heterogeneous preferences, $\rho=0.9$,
$m=40$; 300 problems per row). Near-best range: values of $\alpha$ within 0.005 of the best mean $\tau$.}
\label{tab:alpha}
\begin{tabular}{lrrlrrr}
\toprule
Redundancy & $\sigma$ & Best $\alpha$ & Near-best & $\tau$ (best) & $\tau$ ($\alpha=0.5$) & $\tau$ (BWM)\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- Table: criterion weights on QWS (T2)
t2 = pd.read_csv("results/T2_weights.csv", index_col=0)
cols = ["Equal", "Entropy", "CRITIC", "BWM", "CRITCLOUD(0.5)", "CRITIC-AHP", "CRITIC-DEMATEL", "GameTheory(CRITIC,BWM)"]
head = {"Equal": "Equal", "Entropy": "Entropy", "CRITIC": "CRITIC", "BWM": "BWM", "CRITCLOUD(0.5)": r"\textbf{CRITCLOUD}",
        "CRITIC-AHP": "CRITIC-AHP", "CRITIC-DEMATEL": "CRITIC-DEMATEL", "GameTheory(CRITIC,BWM)": "Game theory"}
rows = "\n".join(f"{k} & " + " & ".join(f"{t2.loc[k, c]:.4f}" for c in cols) + r" \\" for k in t2.index if k != "HHI")
write("tab_weights.tex", r"""\begin{table*}[!tp]
\centering
\caption{Criterion weights on QWS ($m=2{,}500$). CRITCLOUD uses $\alpha=0.5$; the BWM judgements are those of
Table~\ref{tab:bwm}. HHI $=\sum_j w_j^2$ (minimum $1/9=0.111$ at equal weights).}
\label{tab:weights}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}
\toprule
Criterion & """ + " & ".join(head[c] for c in cols) + r"""\\
\midrule
""" + rows + r"""
\midrule
HHI & """ + " & ".join(f"{t2.loc['HHI', c]:.4f}" for c in cols) + r""" \\
\bottomrule
\end{tabular}
\end{table*}
""")


# =====================================================================
# REVIEWER ROUND 2: preference design (C1) and utility model (C4)
# =====================================================================
v3 = pd.read_csv("results/V3_key_differences.csv").set_index(["pref_mode", "utility", "theta", "structure", "contrast"])
v5 = pd.read_csv("results/V5_best_alpha.csv").set_index(["pref_mode", "utility", "theta", "structure"])
PM = {"per_criterion": "Per criterion", "split": "Split"}
UT = {"linear": "Linear", "sla": "Hard SLA", "concave": "Concave"}
CC_BE, CC_B = "CRITCLOUD(0.5) - BWM+Equal(0.5)", "CRITCLOUD(0.5) - BWM"

def cell(pm, ut, th, st, c=CC_BE):
    r = v3.loc[(pm, ut, th, st, c)]
    mark = r"$^{\dagger}$" if r.ci_lo <= 0 <= r.ci_hi else ""
    return f"{r['mean']:+.4f}{mark}"

def gain(pm, ut):
    return np.mean([v3.loc[(pm, ut, "dirichlet", st, CC_B)]["mean"] for st in ("none", "moderate", "high")])

def best_alpha(pm, ut, th):
    a = sorted(set(v5.loc[(pm, ut, th, st)]["best_alpha"] for st in ("none", "moderate", "high")))
    return "/".join(f"{x:g}" for x in a)

rows = []
for pm in ("per_criterion", "split"):
    for ut in ("linear", "sla", "concave"):
        rows.append(f"{PM[pm]} & {UT[ut]} & " + " & ".join(cell(pm, ut, "dirichlet", st) for st in ("none", "moderate", "high"))
                    + f" & {cell(pm, ut, 'uniform', 'high')} & {gain(pm, ut):+.4f} & {best_alpha(pm, ut, 'dirichlet')} & {best_alpha(pm, ut, 'uniform')} \\\\")
    if pm == "per_criterion": rows.append(r"\midrule")
write("tab_sensitivity.tex", r"""\begin{table*}[!tp]
\centering
\caption{Sensitivity of the simulation conclusions to the preference design and the utility model
(7,200 problems per design: 24 non-skewed cells $\times$ 300 runs). Columns 3--6: mean
$\Delta\tau=$ CRITCLOUD $-$ (BWM\,+\,Equal), i.e.\ the value added by CRITIC's correlation handling; $^{\dagger}$ = bootstrap
95\% CI contains zero (all other CIs exclude zero). Column 7: mean $\Delta\tau=$ CRITCLOUD $-$ BWM under heterogeneous
preferences, i.e.\ the value of blending. Columns 8--9: best $\alpha$ on the grid $\{0,0.25,0.5,0.75,1\}$ across the three
redundancy levels.}
\label{tab:sensitivity}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrcc}
\toprule
 & & \multicolumn{3}{c}{Heterogeneous: CRITIC value} & Uniform & Blend gain & \multicolumn{2}{c}{Best $\alpha$}\\
\cmidrule(lr){3-5}\cmidrule(lr){8-9}
Preference design & Utility & None & Moderate & High & High & over BWM & Het. & Uni.\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

v6 = pd.read_csv("results/V6_sla_feasible.csv", index_col=[0, 1])
cols = ["BWM", "Equal", "CRITIC", "AHP", "CRITCLOUD(0.5)", "BWM+Equal(0.5)", "GameTheory(CRITIC,BWM)", "BWM+Mahalanobis-TOPSIS"]
hdr = " & ".join({"CRITCLOUD(0.5)": r"\textbf{CRITCLOUD}", "BWM+Equal(0.5)": r"BWM\,+\,Eq.",
                  "GameTheory(CRITIC,BWM)": "Game th.", "BWM+Mahalanobis-TOPSIS": "Mahal."}.get(c, c) for c in cols)
rows = "\n".join(f"{PM[pm]} & {'Heterogeneous' if th == 'dirichlet' else 'Uniform'} & "
                 + " & ".join(f"{100 * v6.loc[(pm, th), c]:.1f}" for c in cols) + r" \\" for pm, th in v6.index)
write("tab_sla.tex", r"""\begin{table*}[!tp]
\centering
\caption{Hard-SLA utility model: share (\%) of simulated problems in which the provider ranked first satisfies the
SLA (3,600 problems per row). Providers below the 25th percentile of one latent quality are infeasible, so a random
choice would be feasible in 75\% of cases.}
\label{tab:sla}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll""" + "r" * len(cols) + r"""}
\toprule
Preference design & Preferences & """ + hdr + r"""\\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
