#!/usr/bin/env bash
# Reproduces every table and figure of the paper, in order.
# Requires qws.csv (2,500 rows) in this folder; see README "Data".
set -euo pipefail
python test_reproduce.py
python exp_paper_tables.py          # QWS tables T1-T8
python run_simulation.py            # main simulation S0-S6 (~6 min)
python analysis_extra.py            # S7-S9
python exp_bwm_perturbation.py      # T9
python bwm_revision_options.py      # T10-T11
python exp_alpha.py                 # A1-A5
python exp_normalization.py         # N1-N3
python check_rie_theory.py          # R1
python exp_uncertainty.py           # U1-U3 (bootstrap / Wilson CIs)
python exp_reviewer_sensitivity.py  # V1-V6 (preference design, SLA and concave utility; ~6 min on 2 cores)
python make_latex_tables.py
python make_figures.py
python verify_checksums.py          # compares results/*.csv with CHECKSUMS.sha256
