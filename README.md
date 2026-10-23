# CRITCLOUD — corrected code and simulation study

## Files

| File | Purpose |
| --- | --- |
| `critcloud_core.py` | Single source of truth: normalisation, CRITIC, Entropy, SD, MEREC, linear BWM + input-based CR, α-blend, TOPSIS, OPTCLOUD (sign-fixed), RIE / RIE_norm / HHI, accuracy metrics, pairwise rank reversal |
| `test_reproduce.py` | Regression test: new module reproduces the original notebook numbers (except OPTCLOUD, see below) |
| `exp_paper_tables.py` | Re-runs all real-data (QWS) tables with corrected code → `results/T*.csv` |
| `simulator.py` | Synthetic ground-truth generator (latent factors, controlled redundancy, noisy BWM users) |
| `run_simulation.py` | Main simulation study, 500 replications × 48 cells + preference-noise sweep → `results/S0–S6` |
| `analysis_extra.py` | Follow-up: CRITIC vs shrinkage, skew effect, RIE validity → `results/S7–S9` |
| `exp_bwm_perturbation.py` | BWM perturbation (±1, ±2) on real QWS data → `results/T9` |
| `bwm_revision_options.py` | BWM consistency check + minimal-change corrected vectors → `results/T10–T11` |
| `exp_alpha.py` | α sweep: QWS trade-off curve, best α by noise and redundancy, rule by CR → `results/A1–A5` |
| `exp_normalization.py` | Max-min vs log + max-min vs rank normalisation on QWS and simulation → `results/N1–N3` |
| `check_rie_theory.py` | Numerical check of the RIE propositions; RIE = ½(1−HHI) × WMR table → `results/R1` |
| `exp_uncertainty.py` | Bootstrap 95% CIs of paired Δτ and Wilson CIs of stability rates → `results/U1–U3` |
| `exp_reviewer_sensitivity.py` | Preference design (per criterion vs split) × utility model (linear, hard SLA, concave), 43,200 problems → `results/V1–V6` |
| `make_latex_tables.py`, `make_figures.py` | Build every table and figure of the paper from `results/` |
| `run_all.sh` | Runs everything in paper order, then checks checksums |
| `requirements.txt`, `Dockerfile` | Pinned environment (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2) |
| `CHECKSUMS.sha256`, `verify_checksums.py` | SHA-256 of every published result CSV and a script to compare a rerun |
| `RIE_theory.tex` / `.pdf` | Propositions and proofs for RIE, RIE_norm, RIE_adj edge case and WMR (LaTeX-ready) |
| `CRITCLOUD_run_all.ipynb` | Jupyter notebook that runs everything in order and draws the figures |
| `results_log_*.txt` | Console output of each run |

## Run

```bash
pip install -r requirements.txt     # or: docker build -t critcloud . && docker run --rm -v "$PWD":/w critcloud
bash run_all.sh                     # about 15 minutes in total
```

## Data

The QWS dataset is not redistributed here. Download it from https://qwsdata.github.io/ and save the
2,500 services with the nine QoS attributes as `qws.csv` in this folder (columns as used in `load_qws` in
`critcloud_core.py`).

## Reproducibility

Every script fixes its random seed (shown at the top of each file). Identical numbers are guaranteed only in the
pinned environment above; `python verify_checksums.py` reports which result files match the published run.
With other library versions small numerical differences are possible, for example in the last decimal places of floating-point
results.

## Corrections relative to the original notebooks

1. **BWM weights.** The paper's Table 6 BWM column is not the LP solution of the stated A_B/A_W vectors. Correct weights: RT 0.2579, Ava 0.1517, TP 0.0265, Suc 0.0986, Rel 0.0607, Com 0.1011, BP 0.1517, Lat 0.0759, Doc 0.0759 (ξ* = 0.0455). Input-based CR = 0.214, which is **below** the Liang et al. (2020) threshold of 0.3657 for a_BW = 8, n = 9 (acceptable). However, the vectors contain 4 ordinal conflicts (Suc–Rel, Suc–Doc, Rel–Lat, Rel–Doc); `bwm_revision_options.py` proposes minimal fixes.
2. **OPTCLOUD eigenvector signs.** `numpy.linalg.eigh` returns eigenvectors with arbitrary sign. On the same 40-provider sample, τ ranges from −0.80 to +0.80 over the 32 possible sign patterns; the notebook's 0.582 was one machine-specific draw. The sign-fixed version is deterministic (τ = 0.70 on n = 40; 0.755 on the full data).
3. **CRITIC weights** now always come from the full dataset, as the paper states.
4. **Rank reversal** is now the share of provider pairs whose order flips, tested for removal, addition of a new provider and addition of a duplicate.
5. **Friedman on closeness coefficients removed.** It compared score magnitudes, not quality. Friedman/Wilcoxon are now run on accuracy against the true ranking.
6. **Dataset size** is 2,500 (not 2,507).

## Added comparison methods (Reviewer 1)

CRITIC-AHP (multiplicative), CRITIC-DEMATEL (data-driven DEMATEL from |correlation|, since QWS has no expert influence matrix), game-theory combination of CRITIC and BWM, and BWM + Mahalanobis-TOPSIS. On QWS, the AHP matrix is built from the same preferences as BWM; in the simulation, the user answers all 36 AHP questions with independent errors.

## BWM vectors used

All QWS experiments use `AB_FINAL` / `AW_FINAL` in `critcloud_core.py` = **Option A**
(a_B of Successability 3→4, Reliability 5→4; CR^I = 0.143; no ordinal conflicts).
The regression test still uses the original paper vectors (`AB_PAPER` / `AW_PAPER`).

## Week 2 findings

- **α guidance (`A1–A5`).** The best α rises with judgement noise: ≈ 0.3 (careful users), 0.4–0.5 (typical), 0.6–0.7 (very noisy). α = 0.5 is within 0.01 τ of the best in 11 of 12 settings. The observed BWM consistency ratio does not indicate which α to use (best α = 0.5 in every CR band). On QWS, α = 0.5 keeps τ = 0.68 agreement with the user's BWM ranking and the same best provider in 98.7% of ±1 perturbations.
- **Normalisation (`N1–N3`).** Log + max-min on heavy-tailed criteria cuts the share of latency values above 0.9 from 98% to 9% and raises CRITIC's latency weight from 0.031 to 0.099. On skewed simulated data it lifts CRITCLOUD from τ = 0.599 to 0.632 (non-skewed reference 0.653) and changes nothing on non-skewed data. Rank normalisation costs accuracy on non-skewed data.
- **RIE theory (`RIE_theory.pdf`, `R1`).** Bounds 0 ≤ RIE ≤ ½(1−HHI) ≤ (n−1)/2n; an exact formula for moving weight between criteria; RIE = ½(1−HHI) × WMR, where WMR ∈ [0,1] is comparable across datasets; RIE_adj → 0 as weights concentrate on one criterion, so it should be withdrawn.

## Key findings

**Real data (QWS)**
- τ against Equal-TOPSIS is almost fully explained by how close a weight vector is to uniform: Spearman(L2 distance to equal, τ) = −0.92 over 9 weighting schemes (`T6`).
- With OPTCLOUD sign-fixed, its full-data τ against Equal-TOPSIS (0.739) is close to CRITCLOUD's (0.767). The paper's large gap (0.869 vs 0.582) does not survive.
- Simulated preference robustness (`T9`, `U3`): with ±1 changes to the single set of BWM answers, CRITCLOUD(0.5) keeps the same top provider in 98.3% of trials [Wilson 95% CI 97.3–98.9] (BWM alone 66.5%, BWM+Equal 94.2%, game theory 93.7%, BWM+Mahalanobis-TOPSIS 70.9%). This is stability, not real-world accuracy: CRITCLOUD(0.75) is the most stable because it uses BWM least.
- BWM vectors: CR^I = 0.214 is within the threshold (0.3657), but 4 ordinal conflicts exist. Option A (2 changes: a_B of Suc 3→4, Rel 5→4) removes all conflicts (`T10–T11`).
- Max-min normalisation is dominated by outliers: 86.5% of response-time and 98% of latency values map above 0.9 (`T8`).

**Simulation (known true ranking, 19 methods)**
- With real user preferences, the top group is AHP (mean rank 5.68), the best-linear oracle (6.05), CRITCLOUD(0.5) (6.07) and BWM+Equal (6.08). CRITCLOUD(0.5) vs BWM+Equal: Δτ = 0.0015, Holm p = 0.22, Cliff's δ = 0.007 (no difference).
- AHP and CRITIC-AHP are the most robust to judgement noise because 36 pairwise answers average out more error than BWM's 15. The simulation does not model the extra fatigue of answering 36 questions, so this is a limitation.
- CRITCLOUD(0.5) beats game-theory combination (δ = 0.026 and 0.134), BWM+Mahalanobis-TOPSIS (δ = 0.110 and 0.352) and CRITIC-DEMATEL (δ = 0.648 and 0.413) under preference and uniform regimes respectively.
- CRITIC's correlation term adds value only under high redundancy (+0.011 to +0.018 τ over shrinkage to equal weights).
- Lower RIE is not consistently associated with higher accuracy (`S9`).
- With thousands of paired replications every p-value is tiny; judge differences by Cliff's δ (|δ| < 0.147 negligible, < 0.33 small).

## Round-2 findings (`U1–U3`, `V1–V6`)

- CRITCLOUD − (BWM+Equal), heterogeneous preferences: Δτ = +0.0015 [95% CI +0.0012, +0.0018]; inside the ±0.01 margin of practical interest.
- CRITIC adds value only under high redundancy (+0.011 [0.0105, 0.0118]) **and** only when the user weights each overlapping criterion separately (per-criterion design). When the user already splits a factor's importance across its criteria, CRITCLOUD − (BWM+Equal) is negative in every cell (−0.003 to −0.012).
- Holds under hard-SLA and concave utility; blending beats BWM alone in all six conditions (+0.014 to +0.029); α = 0.5 is best on the grid for heterogeneous preferences in all six.
- Hard SLA: 5–11% of each method's first choices violate the SLA → apply SLA limits as a filter before ranking.
