"""Canonical generator for all coupling-stress figure and table data.

Single authoritative writer of

    fig4_closedloop_dense.csv   (fig:stress data)
    fig6_band_dense.csv         (fig:crossscale data)
    percell_error_N_dense.csv   (grid independence)
    fig4_symmetric_metrics.csv  (four symmetric diagnostics)
    tab_sota_coupled.csv        (data behind Table tab:sota_coupled)
    tab_bare.csv                (three-arm positivity-treatment comparison)

into ``figures/`` in one run.

One initial condition, one integrator and one infectiousness kernel (see
``regen_fig3_ours.py``), one deterministic host-level map. There are no
seeds, no bootstrap and no uncertainty bands: the coupled map has no random
source. Every figure column is comma-separated and read by name.

E_clip for the OURS arm is a definition-consistency check, not an assigned
constant: the same pre-clip deficit functional used for the post hoc arm is
evaluated on the delivered (MPP-scaled) increment and accumulated over every
patch and step. It is measured in the loop below.
"""
import os
import csv
import math
import warnings
import numpy as np

import coupling_host_level as chl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIRS = [os.path.join(HERE, "figures")]
H = chl.H
T_SPAN = chl.T_SPAN
K = chl.K_BINS
BETA_STRESS = [15.0, 30.0, 60.0, 120.0]
N_LIST = [8, 13, 20, 26, 35, 45, 52]
STRESS_BETA = 60.0


def t_grid(h=H):
    return np.round(np.arange(T_SPAN[0], T_SPAN[1] + h / 2, h), 8)


def poisson_matrix(sigma, h, nbins=K):
    nh = np.arange(nbins)
    p = (np.exp(-sigma * h) * (sigma * h) ** nh
         / np.array([math.factorial(int(x)) for x in nh], dtype=float))
    T = np.zeros((nbins, nbins))
    for k in range(nbins):
        T[k, :k + 1] = p[np.arange(k, -1, -1)]
    return T


def setup_at_N(states_all, rho_all, N_all, W_all, n):
    """Leading n-patch subsystem with the gravity matrix re-normalised.

    A principal submatrix of a row-stochastic W is not row-stochastic; without
    re-normalisation the effective reproduction number changes with N.
    """
    idx = slice(None) if n >= len(states_all) else slice(0, n)
    rho = rho_all[idx]
    N = N_all[idx]
    W = W_all[idx][:, idx]
    W = W / W.sum(axis=1, keepdims=True)
    S0 = (1.0 - rho) * N
    I0 = rho * N
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K))
    return S0, bins0, R0, W, N


def run_arm(beta_sp, arm, S0, bins0, R0, W, N, phis, sigma, grid):
    """Advance one arm ('clip', 'mpp' or 'bare') and measure diagnostics.

    'bare' applies no positivity treatment at all: the susceptible pool is
    allowed to go negative and no state is repaired. Its finite/non-finite
    outcome is measured, never assigned.
    """
    h = grid[1] - grid[0]
    T = poisson_matrix(sigma, h)
    S, bins, R = S0.copy(), bins0.copy(), R0.copy()
    d = dict(E_clip=0.0, defcheck=0.0, max_deficit=0.0, cons_resid=0.0,
             max_abs_y=0.0, min_S=float("inf"), theta_min=1.0,
             theta_lt1=0, C_gt1=0, max_C=0.0, C_at_t0=0.0,
             eps_mpp=0.0, eps_clip=0.0, neg_S_raw=0, neg_S_stored=0,
             neg_I_stored=0, n_cellstep=0, all_finite=True)
    for step in range(len(grid) - 1):
        Phi = bins @ phis
        Lam = beta_sp * (W @ (Phi / N))
        C = h * Lam
        X = C * S                                   # unlimited transfer
        S_pre = S - X
        deficit = np.maximum(-S_pre, 0.0)
        theta = np.where(X > 0,
                         np.minimum(1.0, S / np.maximum(X, 1e-300)), 1.0)
        delivered = theta * X if arm == "mpp" else X

        if arm == "clip":
            d["E_clip"] += float(deficit.sum())
            S_new = np.maximum(S - delivered, 0.0)
        elif arm == "bare":
            S_new = S - delivered
        else:
            delivered_deficit = np.maximum(-(S - delivered), 0.0)
            d["defcheck"] += float(delivered_deficit.sum())
            d["max_deficit"] = max(d["max_deficit"],
                                   float(delivered_deficit.max()))
            S_new = S - delivered

        I_new = bins @ T.T
        removed = bins.sum(axis=1) - I_new.sum(axis=1)
        I_new[:, 0] += delivered
        R_new = R + removed
        I_total = I_new.sum(axis=1)

        d["cons_resid"] = max(d["cons_resid"],
                              float(np.max(np.abs(S_new + I_total + R_new - N))))
        d["max_abs_y"] = max(d["max_abs_y"],
                             float(max(np.abs(S_new).max(),
                                       np.abs(I_total).max(),
                                       np.abs(R_new).max())))
        d["min_S"] = min(d["min_S"], float(S_new.min()))
        d["theta_min"] = min(d["theta_min"], float(theta.min()))
        d["theta_lt1"] += int((theta < 1.0 - 1e-12).sum())
        d["C_gt1"] += int((C > 1.0).sum())
        d["max_C"] = max(d["max_C"], float(C.max()))
        if step == 0:
            d["C_at_t0"] = float(C.max())
        sel = S > 0
        if sel.any():
            d["eps_mpp"] = max(d["eps_mpp"],
                               float(np.max(np.abs(np.minimum(C[sel], 1.0)
                                                  - (1.0 - np.exp(-C[sel]))))))
            d["eps_clip"] = max(d["eps_clip"],
                                float(np.max(np.abs(C[sel] - 1.0
                                                   + np.exp(-C[sel])))))
        d["neg_S_raw"] += int((S_pre < -1e-12).sum())
        d["neg_S_stored"] += int((S_new < -1e-12).sum())
        d["neg_I_stored"] += int((I_total < -1e-12).sum())
        d["n_cellstep"] += len(S)
        d["all_finite"] = d["all_finite"] and bool(
            np.all(np.isfinite(S_new)) and np.all(np.isfinite(I_total))
            and np.all(np.isfinite(R_new)))
        S, bins, R = S_new, I_new, R_new
    d["frac_theta_lt1"] = d["theta_lt1"] / d["n_cellstep"]
    d["frac_C_gt1"] = d["C_gt1"] / d["n_cellstep"]
    d["cfl_ok_at_t0"] = d["C_at_t0"] <= 1.0
    d["cfl_ok_60d"] = d["max_C"] <= 1.0
    return d


def write_csv(name, header, rows):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
    print("  wrote %s (%d rows) to both figure directories" % (name, len(rows)))


def main():
    ker = chl.build_infectious_kernel()
    phis, sigma = ker["phis"], ker["sigma"]
    states, rho = chl.load_rho(chl.RHO_CSV)
    W_all, N_all, beta_cal = chl.build_gravity(states)
    grid = t_grid()
    S0 = (1.0 - rho) * N_all
    I0 = rho * N_all
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K))
    betas = [beta_cal] + BETA_STRESS

    print("kernel: tau*=%.3f d  sigma=%.6f /d  sum(phi)dtau=%.6f"
          % (ker["tau_star"], sigma, ker["sum_phi_dtau"]))
    print("beta list: %s" % ", ".join("%.6f" % b for b in betas))

    # ---- fig:stress data, Table tab:sota_coupled, symmetric diagnostics ----
    arms = {}
    for b in betas:
        arms[(b, "clip")] = run_arm(b, "clip", S0, bins0, R0, W_all, N_all,
                                    phis, sigma, grid)
        arms[(b, "mpp")] = run_arm(b, "mpp", S0, bins0, R0, W_all, N_all,
                                   phis, sigma, grid)

    rows4 = []
    for i, b in enumerate(betas, 1):
        c, m = arms[(b, "clip")], arms[(b, "mpp")]
        rows4.append(["stress_x%d" % i, "%.6f" % b,
                      "%.6e" % c["E_clip"], "%.6e" % m["defcheck"],
                      "%.6e" % m["max_deficit"],
                      str(c["cfl_ok_at_t0"]), str(c["cfl_ok_60d"]),
                      "%.6e" % c["max_C"], "%.6e" % c["frac_C_gt1"],
                      "%.6e" % m["frac_theta_lt1"]])
    write_csv("fig4_closedloop_dense.csv",
              ["label", "beta_sp", "E_clip_clip", "E_clip_mpp_defcheck",
               "max_deficit_mpp", "cfl_ok_at_t0", "cfl_ok_60d", "max_C_60d",
               "frac_C_gt_1_60d", "frac_theta_lt_1_60d"], rows4)

    rows_tab = []
    for b in betas:
        c, m = arms[(b, "clip")], arms[(b, "mpp")]
        rows_tab.append(["%.6f" % b, "%.6e" % c["E_clip"],
                         "%.6e" % m["defcheck"], "%.6e" % c["min_S"],
                         "%.6e" % c["cons_resid"], "%.6e" % m["cons_resid"],
                         "%.6e" % c["max_abs_y"], "%.6e" % m["max_abs_y"]])
    write_csv("tab_sota_coupled.csv",
              ["beta_sp", "E_clip_clip", "E_clip_mpp_defcheck", "min_S_clip",
               "cons_resid_clip", "cons_resid_mpp", "max_abs_y_clip",
               "max_abs_y_mpp"], rows_tab)

    rows_sym = []
    for b in betas:
        for arm in ("clip", "mpp"):
            d = arms[(b, arm)]
            rows_sym.append(["%.6f" % b, arm, "%.6e" % d["E_clip"],
                             "%.6e" % d["defcheck"], "%.6e" % d["cons_resid"],
                             "%.6e" % d["max_abs_y"], "%.6e" % d["theta_min"],
                             "%.6e" % d["frac_theta_lt1"],
                             "%.6e" % d["frac_C_gt1"], "%.6e" % d["max_C"],
                             "%.6e" % d["eps_mpp"], "%.6e" % d["eps_clip"],
                             str(d["neg_S_raw"]), str(d["neg_S_stored"]),
                             str(d["neg_I_stored"])])
    write_csv("fig4_symmetric_metrics.csv",
              ["beta_sp", "arm", "E_clip", "defcheck", "cons_resid",
               "max_abs_y", "theta_min", "frac_theta_lt_1", "frac_C_gt_1",
               "max_C", "eps_over_S_mpp", "eps_over_S_clip",
               "neg_S_raw", "neg_S_stored", "neg_I_stored"], rows_sym)

    # ---- three-arm positivity-treatment comparison (Table tab:bare) ----
    rows_bare = []
    for b in (beta_cal, STRESS_BETA):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bare = run_arm(b, "bare", S0, bins0, R0, W_all, N_all,
                           phis, sigma, grid)
        c, m = arms[(b, "clip")], arms[(b, "mpp")]
        rows_bare.append(["%.6f" % b, str(bare["all_finite"]),
                          str(bare["neg_S_stored"]),
                          "%.6e" % c["E_clip"], "%.6e" % m["defcheck"],
                          "%.6e" % c["cons_resid"], "%.6e" % m["cons_resid"]])
    write_csv("tab_bare.csv",
              ["beta_sp", "bare_all_finite", "bare_neg_S_stored",
               "E_clip_clip", "E_clip_mpp_defcheck",
               "cons_resid_clip", "cons_resid_mpp"], rows_bare)

    # ---- fig:crossscale data and grid independence (W re-normalised at each N) ----
    rows6, rows_pc = [], []
    grid_ref = t_grid(0.01)
    for n in N_LIST:
        Sn, bn, Rn, Wn, Nn = setup_at_N(states, rho, N_all, W_all, n)
        c = run_arm(STRESS_BETA, "clip", Sn, bn, Rn, Wn, Nn, phis, sigma, grid)
        m = run_arm(STRESS_BETA, "mpp", Sn, bn, Rn, Wn, Nn, phis, sigma, grid)
        rows6.append([n, "%.6e" % c["E_clip"], "%.6e" % m["defcheck"],
                      "%.6e" % c["cons_resid"], "%.6e" % m["cons_resid"],
                      "%.6e" % c["max_C"]])
        cH = chl.simulate(Sn, bn, Rn, Wn, Nn, phis, sigma, beta_cal, H, grid)
        cR = chl.simulate(Sn, bn, Rn, Wn, Nn, phis, sigma, beta_cal,
                          0.01, grid_ref)
        diff = cH["R"][-1] - cR["R"][-1]
        rows_pc.append([n, "%.6e" % float(np.mean(np.abs(diff))),
                        "%.6e" % float(np.sqrt(np.mean(diff ** 2)))])
        print("  N=%2d  E_clip_clip=%.6e  cons_mpp=%.3e  L1=%.6e"
              % (n, c["E_clip"], m["cons_resid"], float(np.mean(np.abs(diff)))))

    write_csv("fig6_band_dense.csv",
              ["N", "E_clip_clip", "E_clip_mpp_defcheck",
               "cons_resid_clip", "cons_resid_mpp", "max_C"], rows6)
    write_csv("percell_error_N_dense.csv",
              ["N", "E_cell_L1", "E_cell_L2"], rows_pc)
    print("DONE (deterministic, single run, no bands).")


if __name__ == "__main__":
    main()
