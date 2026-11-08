# -*- coding: utf-8 -*-
"""Generate submission/figures/fig7_model_rho.csv by a REAL host-level run.

fig7_model_rho.csv is the [T=146, N=52] matrix of instantaneous infected
fraction rho_i(t) consumed by gen_figdata.py to build the cumulative-attack
heatmap (ca_heatmap_long.csv). Every column is the I^h_i(t)/N_i time series
produced by coupling_host_level.simulate() under the calibrated coupling,
sampled every 0.2 d over 0-29 d, in Tycho-rank order. No zeros, placeholders
or fabricated values: the matrix is a direct deterministic simulation output.

Input dependencies (both produced earlier by run_all.py, both inside
submission/):
  figures/fig3_withinhost_traj.csv  -> infectious kernel phi(tau)
  figures/fig2_tycho_field.csv      -> 52-state initial rho (Tycho-rank order)
"""
import os
import sys
import csv
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = Path(__file__).resolve().parents[1]          # submission/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, HERE)
import coupling_host_level as chl

OUT_DIRS = [SUB / "figures"]
if (ROOT / "figures").is_dir():
    OUT_DIRS.append(ROOT / "figures")

# 146 samples at 0.2 d spacing (grid stride 2 at H = 0.1 d), t = 0 .. 29.0 d.
# gen_figdata.py reads 146 rows x 52 states, then keeps every 2nd row (73).
T_SAMPLES = 146
STRIDE = 2


def main():
    ker = chl.build_infectious_kernel()
    phis, sigma = ker["phis"], ker["sigma"]
    states, rho = chl.load_rho(chl.RHO_CSV)
    W, N, beta_sp = chl.build_gravity(states)
    K = chl.K_BINS

    S0 = (1.0 - rho) * N
    I0 = rho * N
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K))
    t_grid = np.round(
        np.arange(chl.T_SPAN[0], chl.T_SPAN[1] + chl.H / 2, chl.H), 8)

    coupled = chl.simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp,
                           chl.H, t_grid)
    I = coupled["I"]            # [601, 52] infected fraction per state
    t = coupled["t"]

    idx = np.arange(T_SAMPLES) * STRIDE         # 0, 2, ..., 290
    if idx.max() >= I.shape[0]:
        raise RuntimeError("sample index %d out of range (%d rows)"
                           % (idx.max(), I.shape[0]))

    OUT_DIRS[0].mkdir(parents=True, exist_ok=True)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "fig7_model_rho.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["time_days"] + list(states))
            for i in idx:
                w.writerow(["%.4f" % t[i]] + ["%.10e" % v for v in I[i]])
        print("[fig7_model_rho] wrote %s" % (d / "fig7_model_rho.csv"))
    print("  rows=%d  cols=%d  t in [%.1f, %.1f] d  beta_sp=%.6f"
          % (T_SAMPLES, len(states), t[idx[0]], t[idx[-1]], beta_sp))
    print("  rho range [%.6e, %.6e]  (real simulate() output, no placeholder)"
          % (I[idx].min(), I[idx].max()))


if __name__ == "__main__":
    main()
