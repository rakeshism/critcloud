"""Regression test: the new module must reproduce the original notebook outputs."""
import numpy as np, pandas as pd
from critcloud_core import *

X = load_qws("qws.csv"); N = normalize(X); R = corr(N)
wc, we = critic_weights(N), entropy_weights(N)
wb, xi = bwm_weights(AB_PAPER, AW_PAPER)
wf = blend(wc, wb, 0.5)
assert np.allclose(np.round(wc,4), [0.0741,0.1164,0.1122,0.1247,0.0949,0.0995,0.1121,0.0307,0.2354]), wc
assert np.allclose(np.round(we,4), [0.0093,0.0388,0.3271,0.0419,0.0301,0.0165,0.0379,0.0014,0.4970]), we
assert np.allclose(np.round(wb,4), [0.2579,0.1517,0.0265,0.0986,0.0607,0.1011,0.1517,0.0759,0.0759]), wb
assert round(rie(wf,R),5)==0.05898 and round(rie(we,R),5)==0.02182 and round(rie(np.full(9,1/9),R),5)==0.06439
Teq = topsis(N, equal_weights())
assert round(tau(rank_of(topsis(N,wf)), rank_of(Teq)),4)==0.7694
# n=40 sample, CRITIC computed on the sample (as in the original OPTCLOUD notebook)
S = X.sample(n=40, random_state=42).reset_index(drop=True); NS = normalize(S)
wfs = blend(critic_weights(NS), wb, 0.5)
assert round(tau(rank_of(topsis(NS,wfs)), rank_of(topsis(NS,equal_weights()))),4)==0.8692
# OPTCLOUD: the original notebook reported tau=0.5821, but that value depends on the
# arbitrary eigenvector signs returned by the local LAPACK build. Sign-fixed version is
# deterministic and is what all new experiments use.
Zf,m,_ = optcloud(NS, wb, fix_sign=True)
assert m == 5 and round(tau(rank_of(Zf), rank_of(topsis(NS,equal_weights()))),2)==0.70
print("ALL REPRODUCTION CHECKS PASSED")
