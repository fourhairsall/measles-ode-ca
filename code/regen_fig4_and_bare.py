"""Canonical generator for the coupling-stress figure and table data (M3).

Single authoritative writer of

    fig4_closedloop_dense.csv   (long record: beta x arm x mode)
    fig4_stress_wide.csv        (wide record consumed by fig:stress)
    tab_sota_coupled.csv        (tab:sota_coupled)
    fig4_symmetric_metrics.csv  (symmetric diagnostics per arm)
    tab_bare.csv                (tab:bare)     positivity treatments
    fig6_band_dense.csv         (fig:crossscale) grid independence in N
    percell_error_N_dense.csv   (companion of fig:crossscale)

into BOTH ``<repo>/figures/`` and ``submission/figures/`` in one run.

Method under test
-----------------
The production host-level macro step is M3 (``coupling_host_level.host_step``):
a p-th order Taylor force of infection, a non-negative positive-square
parametrisation, and the exact phase-type (Erlang) flow.  It imposes no CFL
condition and needs no subcycling, so it is always run at exactly one step per
macro-step.

Opponent set -- measured in BOTH coupling-resolution modes
----------------------------------------------------------
Every explicit arm is measured twice on the SAME macro grid ``t_grid(H)``:

  ``mode=one-step``  exactly one step per macro-step, i.e. the CFL-violating
                     regime when h*Lambda > 1.  No controller, no subcycling.
  ``mode=auto``      the arm's own coupling-resolution controller
                     (``coupling_host_level.adaptive_subcycle``, the multirate
                     controller of ``measles_solvers_pytorch.
                     adaptive_coupling_subcycle``, tol = 0.02) selects the
                     number of substeps; the substep count is REPORTED, never
                     fixed by hand.

Reporting one mode alone would misrepresent the baselines: the one-step numbers
show what the explicit update does when the CFL condition fails, the auto
numbers show what it does when it is allowed to pay for stability.  Both are
tabulated, and the substep count is in every row.

Arms
----
    M3-p3                production (p = 3 Taylor predictor), one step
    M3-p2                degree-2 predictor (order-3 ablation), one step
    EXP2-midpoint        frozen-Lambda midpoint + closed-form age weights
    static_w+midpoint    one static age-weight vector + midpoint Lambda
    Strang+midpoint      literal Strang (age h/2 + react + age h/2) + midpoint
    clip                 forward-Euler transfer + post-hoc non-negativity clip
    bare                 forward-Euler transfer, no positivity treatment at all

Units
-----
``coupling_host_level`` stores each patch as ``(S, I bins, R)`` in units of
10^6 persons (``N_i`` comes from ``STATE_GEO`` in millions), so every quantity
it returns internally is population-scaled.  The manuscript works with the
fractions ``s_i = S_i/N_i``, ``i_i = I_i/N_i``, ``r_i = R_i/N_i``.  BOTH scales
are therefore written out, with the unit in the column name and never mixed:

    *_1e6   population-scaled, units of 10^6 persons
    *_frac  fractions, divided by N_i per patch (residual, min_S) or by
            sum_i N_i (cumulative clipped mass)

The residual in fraction units is exactly ``max_i|s_i+i_i+r_i-1|``, which is
the quantity the manuscript prints; the population-scaled residual is that same
number multiplied back by ``N_i``, so the two differ by up to 1e7 and must never
be quoted interchangeably.

Measured, never assigned
------------------------
* ``E_preclip_deficit`` is the accumulated pre-projection deficit of the arm's
  own delivered increment, summed over patches and steps.  For M3 it is
  measured and comes out exactly 0 (S(h) = S0 exp(-H(h)) >= 0 by construction);
  for clip/bare it is the mass the projection has to destroy.
* ``L2_rel`` is the RMS relative L2 distance of the recorded (S, I, R) fraction
  series to an over-resolved M3 reference on the same macro grid
  (``h_ref = H/60``).  On this real field the epidemic exhausts the susceptibles,
  so ``L2_rel`` is a distance during the burn, not an asymptotic error: the
  companion columns ``final_attack_min`` and ``t_active_days`` (the last time any
  patch still has I_i/N_i > 0.01) state how much of the 60-day window is still
  transient.  When ``t_active_days`` is of the order of a few macro-steps the
  L2 column only measures sub-step resolution and must not be read as an order of
  accuracy claim; the well-conditioned accuracy claim is the 60-step h = 1 d
  comparison of ``tab:m3acc``.
* A baseline that is mathematically undefined at the requested coupling (the
  closed-form allocation needs sigma > Lambda) is reported as
  ``status=closed_form_undefined`` with NaN numerics; it is never silently
  repaired or dropped.

Run:  python regen_fig4_and_bare.py
"""
import os
import csv
import math
import time

import numpy as np

import coupling_host_level as chl

from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [SUB / "figures"]
if (ROOT / "figures").is_dir():
    OUT_DIRS.append(ROOT / "figures")

H = chl.H
T_SPAN = chl.T_SPAN
K = chl.K_BINS
H_REF = 0.01
BETA_STRESS = [15.0, 30.0, 60.0, 120.0]
N_LIST = [8, 13, 20, 26, 35, 45, 52]
STRESS_BETA = 60.0

# arms measured at exactly one step per macro-step (M3 always, baselines for the
# CFL-violating regime)
ONE_STEP_ARMS = ["M3-p3", "M3-p2", "EXP2-midpoint", "static_w+midpoint",
                 "Strang+midpoint", "clip"]
# arms that additionally get their own auto-substepping controller
AUTO_ARMS = ["EXP2-midpoint", "static_w+midpoint", "Strang+midpoint", "clip"]
EXPLICIT_ARMS = ["EXP2-midpoint", "static_w+midpoint", "Strang+midpoint", "clip"]


def t_grid(h=H):
    return np.round(np.arange(T_SPAN[0], T_SPAN[1] + h / 2, h), 8)


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


def run_series(arm, beta_sp, S0, bins0, R0, W, N, phis, sigma, grid,
               substeps=None):
    """Advance one arm on `grid` at one step per macro-step.

    Returns recorded fractions S, I, R per patch plus the diagnostics:
    cons_resid, min_S, neg_S, neg_I, all_finite, n_subcycle, lam_evals,
    wall_ms, status, E_preclip_deficit, max_abs_y (units 10^6 persons),
    max_C_t0, max_C_run, final_attack_min.
    """
    S, Ib, R = S0.copy(), bins0.copy(), R0.copy()
    ns = len(grid) - 1
    Srec = np.zeros((ns + 1, S.shape[0]))
    Irec = np.zeros_like(Srec)
    Rrec = np.zeros_like(Srec)
    Srec[0] = S / N
    Irec[0] = Ib.sum(axis=1) / N
    Rrec[0] = R / N
    cons = 0.0
    cons_frac = 0.0
    minS = float("inf")
    minS_frac = float("inf")
    negS = negI = 0
    finite = True
    nsub_tot = 0
    lam_tot = 0
    deficit = 0.0
    max_abs = 0.0
    maxC0 = 0.0
    maxC = 0.0
    status = "ok"

    def cnum(bb):
        r = (bb @ phis) / N
        return float(np.max(H * beta_sp * (r @ W.T))) if beta_sp > 0 else 0.0

    maxC0 = cnum(Ib)
    kdone = 0
    t_active = 0.0
    t0 = time.perf_counter()
    for k in range(ns):
        dt = float(grid[k + 1] - grid[k])
        try:
            if arm.startswith("M3"):
                S, Ib, R, st = chl.host_step(
                    S, Ib, R, W, N, phis, sigma, beta_sp, dt,
                    p=(2 if arm == "M3-p2" else chl.M3_ORDER))
                deficit += float(np.maximum(-S, 0.0).sum())
            else:
                S, Ib, R, st = chl.baseline_step(
                    arm, S, Ib, R, W, N, phis, sigma, beta_sp, dt,
                    substeps=substeps)
                deficit += float(st.get("E_clip", 0.0))
        except FloatingPointError:
            status = "closed_form_undefined"
            S = np.full_like(S, np.nan)
            Ib = np.full_like(Ib, np.nan)
            R = np.full_like(R, np.nan)
            cons = float("nan")
            minS = float("nan")
            finite = False
            break
        cons = max(cons, float(st.get("cons_resid", 0.0)))
        minS = min(minS, float(st.get("min_S", np.inf)))
        Itot = Ib.sum(axis=1)
        cons_frac = max(cons_frac, float(np.max(np.abs(S + Itot + R - N) / N)))
        minS_frac = min(minS_frac, float(np.min(S / N)))
        negS += int((S < -1e-12).sum())
        negI += int((Itot < -1e-12).sum())
        finite = finite and bool(np.all(np.isfinite(S))
                                 and np.all(np.isfinite(Ib))
                                 and np.all(np.isfinite(R)))
        if float(np.max(Itot / N)) > 0.01:
            t_active = float(grid[k + 1])
        nsub_tot += int(st.get("n_subcycle", 1))
        lam_tot += int(st.get("n_lam_eval", 0))
        if not finite:
            Srec[k + 1:] = np.nan
            Irec[k + 1:] = np.nan
            Rrec[k + 1:] = np.nan
            break
        max_abs = max(max_abs, float(np.max(np.abs(S))),
                      float(np.max(np.abs(Ib))), float(np.max(np.abs(R))))
        maxC = max(maxC, cnum(Ib))
        Srec[k + 1] = S / N
        Irec[k + 1] = Itot / N
        Rrec[k + 1] = R / N
        kdone = k + 1
    wall_ms = (time.perf_counter() - t0) * 1e3
    fR = Rrec[-1] if np.all(np.isfinite(Rrec[-1])) else np.full(S.shape, np.nan)
    return dict(S=Srec, I=Irec, R=Rrec, cons_resid=cons,
                cons_resid_frac=cons_frac, min_S=minS, min_S_frac=minS_frac,
                E_clip_frac=deficit / float(N.sum()),
                neg_S=negS, neg_I=negI, all_finite=finite,
                n_subcycle=nsub_tot, lam_evals=lam_tot,
                wall_ms=wall_ms, status=status,
                E_preclip_deficit=deficit, max_abs_y=max_abs,
                max_C_t0=maxC0, max_C_run=maxC,
                final_attack_min=(float(np.nanmin(fR))
                                  if np.all(np.isfinite(fR)) else float("nan")),
                n_steps=kdone, t_active=t_active)


def rel_l2(a, b):
    """RMS relative L2 distance between two recorded fraction series."""
    tot = 0.0
    for v in ("S", "I", "R"):
        d = np.asarray(a[v], float) - np.asarray(b[v], float)
        if not np.all(np.isfinite(d)):
            return float("nan")
        scale = max(float(np.max(np.abs(b[v]))), 1e-12)
        tot += float(np.mean(d ** 2)) / (scale ** 2)
    return math.sqrt(tot / 3.0)


def write_csv(name, header, rows):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
    print("  wrote %s (%d rows) to %s + %s"
          % (name, len(rows), OUT_DIRS[0], OUT_DIRS[-1]))


def fmt(v, spec="%.6e"):
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(fv):
        return "nan"
    return spec % fv




def main():
    ker = chl.build_infectious_kernel()
    phis, sigma = ker["phis"], ker["sigma"]
    states, rho = chl.load_rho(chl.RHO_CSV)
    W_all, N_all, beta_cal = chl.build_gravity(states)
    grid = t_grid()
    grid_ref = t_grid(H_REF)
    S0 = (1.0 - rho) * N_all
    I0 = rho * N_all
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K))
    betas = [beta_cal] + BETA_STRESS

    print("kernel: tau*=%.3f d  sigma=%.6f /d  sum(phi)dtau=%.6f"
          % (ker["tau_star"], sigma, ker["sum_phi_dtau"]))
    print("beta list: %s" % ", ".join("%.6f" % b for b in betas))
    print("reference for L2_rel: M3 at h=%.3f d (over-resolved, same grid)"
          % H_REF)
    print("modes: one-step (exactly one step per macro-step, no controller) "
          "and auto (arm's own coupling-resolution controller, tol=%.2f)"
          % chl.ADAPT_TOL)

    # ---------------- coupling-strength sweep ----------------
    print("\n[1/4] M3 reference at h=%.3f d" % H_REF)
    ref = {}
    for b in betas:
        r = run_series("M3-p3", b, S0, bins0, R0, W_all, N_all, phis, sigma,
                       grid_ref)
        stride = int(round(H / H_REF))
        for v in ("S", "I", "R"):
            r[v] = r[v][::stride][:len(grid)]
        ref[b] = r
        print("  beta=%-9.4f ref cons=%-10s neg=%d finite=%s"
              % (b, fmt(r["cons_resid"]), r["neg_S"], r["all_finite"]))

    print("\n[2/4] arms, both modes")
    arms = {}
    for b in betas:
        for arm in ONE_STEP_ARMS:
            d = run_series(arm, b, S0, bins0, R0, W_all, N_all, phis, sigma,
                           grid, substeps=None if arm.startswith("M3") else 1)
            d["mode"] = "one-step"
            arms[(b, arm, d["mode"])] = d
        for arm in AUTO_ARMS:
            d = run_series(arm, b, S0, bins0, R0, W_all, N_all, phis, sigma,
                           grid)
            d["mode"] = "auto"
            arms[(b, arm, d["mode"])] = d
        for (bb, arm, mode), d in list(arms.items()):
            if bb != b:
                continue
            d["L2_rel"] = (rel_l2(d, ref[b]) if d["status"] == "ok"
                           else float("nan"))

            print("  beta=%-9.4f %-18s %-9s sub/step=%-6.1f L2=%-10s "
                  "cons=%-10s E=%-10s t_act=%-6.2f %s"
                  % (b, arm, mode, d["n_subcycle"] / max(1, d["n_steps"]),
                     fmt(d["L2_rel"]), fmt(d["cons_resid"]),
                     fmt(d["E_preclip_deficit"]), d["t_active"], d["status"]))

    def order(b):
        out = []
        for arm in ONE_STEP_ARMS:
            out.append((arm, "one-step"))
        for arm in AUTO_ARMS:
            out.append((arm, "auto"))
        return out

    # fig4_closedloop_dense.csv : complete long record
    cols = ["label", "beta_sp", "arm", "substep_mode", "substeps_per_macro",
            "lam_evals_per_macro", "L2_rel", "final_attack_min", "t_active_days",
            "cons_resid_1e6", "cons_resid_frac", "min_S_1e6", "min_S_frac",
            "E_preclip_deficit_1e6", "E_preclip_deficit_frac",
            "neg_S", "neg_I", "all_finite", "status",
            "max_abs_y_1e6", "max_C_t0", "max_C_run"]
    rows4 = []
    for b in betas:
        for arm, mode in order(b):
            d = arms[(b, arm, mode)]
            rows4.append(["stress_x%d" % (betas.index(b) + 1), "%.6f" % b,
                          arm, mode,
                          "%.3f" % (d["n_subcycle"] / max(1, d["n_steps"])),
                          "%.1f" % (d["lam_evals"] / max(1, d["n_steps"])),
                          fmt(d["L2_rel"]), fmt(d["final_attack_min"]),
                          "%.2f" % d["t_active"],
                          fmt(d["cons_resid"]), fmt(d["cons_resid_frac"]),
                          fmt(d["min_S"], "%.3e"), fmt(d["min_S_frac"], "%.3e"),
                          fmt(d["E_preclip_deficit"]), fmt(d["E_clip_frac"]),
                          str(d["neg_S"]), str(d["neg_I"]),
                          str(d["all_finite"]), d["status"],
                          fmt(d["max_abs_y"], "%.4f"),
                          fmt(d["max_C_t0"], "%.4f"),
                          fmt(d["max_C_run"], "%.4f")])
    print()
    write_csv("fig4_closedloop_dense.csv", cols, rows4)

    # fig4_stress_wide.csv : the wide record fig:stress plots from
    print()
    rows_w = []
    for b in betas:
        c1 = arms[(b, "clip", "one-step")]
        ca = arms[(b, "clip", "auto")]
        m3 = arms[(b, "M3-p3", "one-step")]
        rows_w.append(["%.6f" % b,
                       fmt(c1["E_preclip_deficit"]),
                       fmt(ca["E_preclip_deficit"]),
                       fmt(c1["cons_resid"]),
                       fmt(ca["cons_resid"]),
                       fmt(m3["cons_resid"]),
                       "%.2f" % (ca["n_subcycle"] / max(1, d["n_steps"])),
                       fmt(c1["n_subcycle"] / max(1, d["n_steps"]), "%.2f"),
                       fmt(m3["L2_rel"]), fmt(c1["L2_rel"]),
                       fmt(ca["L2_rel"]),
                       "%.4f" % m3["max_C_run"],
                       "%.2f" % m3["t_active"]])
    write_csv("fig4_stress_wide.csv",
              ["beta_sp", "E_clip_clip", "E_clip_clip_auto",
               "cons_resid_clip", "cons_resid_clip_auto", "cons_resid_M3",
               "substeps_clip_auto", "substeps_clip_onestep",
               "L2_M3", "L2_clip_onestep", "L2_clip_auto",
               "max_coupling_number_M3", "t_active_days_M3"], rows_w)

    # tab_sota_coupled.csv
    print()
    rows_tab = []
    for b in betas:
        for arm, mode in order(b):
            d = arms[(b, arm, mode)]
            rows_tab.append(["%.6f" % b, arm, mode,
                             "%.1f" % (d["n_subcycle"] / max(1, d["n_steps"])),
                             fmt(d["L2_rel"]),
                             fmt(d["cons_resid_frac"]), fmt(d["cons_resid"]),
                             fmt(d["min_S_frac"], "%.3e"),
                             str(d["neg_S"]), str(d["neg_I"]),
                             str(d["all_finite"]), d["status"],
                             fmt(d["E_clip_frac"]),
                             fmt(d["E_preclip_deficit"]),
                             "%.1f" % d["wall_ms"]])
    write_csv("tab_sota_coupled.csv",
              ["beta_sp", "arm", "substep_mode", "substeps_per_macro",
               "L2_rel", "cons_resid_frac", "cons_resid_1e6",
               "min_S_frac", "neg_S", "neg_I", "all_finite", "status",
               "E_preclip_deficit_frac", "E_preclip_deficit_1e6", "wall_ms"],
              rows_tab)

    # fig4_symmetric_metrics.csv
    print()
    rows_sym = []
    for b in betas:
        for arm, mode in order(b):
            d = arms[(b, arm, mode)]
            rows_sym.append(["%.6f" % b, arm, mode,
                             fmt(d["E_preclip_deficit"]), fmt(d["cons_resid"]),
                             fmt(d["E_clip_frac"]), fmt(d["cons_resid_frac"]),
                             fmt(d["max_abs_y"], "%.4f"),
                             fmt(d["min_S"], "%.3e"), str(d["neg_S"]),
                             str(d["neg_I"]), str(d["all_finite"]),
                             "%.2f" % (d["n_subcycle"] / max(1, d["n_steps"]))])
    write_csv("fig4_symmetric_metrics.csv",
              ["beta_sp", "arm", "substep_mode",
               "E_preclip_deficit_1e6", "cons_resid_1e6",
               "E_preclip_deficit_frac", "cons_resid_frac",
               "max_abs_y_1e6", "min_S_1e6", "neg_S", "neg_I",
               "all_finite", "substeps_per_macro"], rows_sym)

    # ---------------- positivity treatments (tab:bare) ----------------
    print("\n[3/4] positivity treatments, both modes")
    rows_bare = []
    for b in (beta_cal, STRESS_BETA):
        plan = [("M3-structural", "M3-p3", "one-step"),
                ("post-hoc-clip, one step", "clip", "one-step"),
                ("untreated, one step", "bare", "one-step"),
                ("post-hoc-clip, auto", "clip", "auto"),
                ("untreated, auto", "bare", "auto")]
        for label, arm, mode in plan:
            if mode == "auto" and arm == "bare":
                # the untreated arm is given the same controller for fairness
                d = run_series_bare(b, S0, bins0, R0, W_all, N_all, phis,
                                    sigma, grid, adaptive=True)
            elif arm == "bare":
                d = run_series_bare(b, S0, bins0, R0, W_all, N_all, phis,
                                    sigma, grid, adaptive=False)
            else:
                d = arms[(b, arm, mode)]
            rows_bare.append(["%.6f" % b, label, arm, mode,
                              "%.2f" % (d["n_subcycle"] / max(1, d["n_steps"])),
                              str(d["all_finite"]), str(d["neg_S"]),
                              fmt(d["cons_resid_frac"]), fmt(d["cons_resid"]),
                              fmt(d["min_S_frac"], "%.3e"),
                              fmt(d["E_clip_frac"]),
                              fmt(d["E_preclip_deficit"])])
            print("  beta=%-9.4f %-24s sub/step=%-6.1f finite=%-5s negS=%-5d "
                  "cons=%-10s E=%-10s"
                  % (b, label, d["n_subcycle"] / max(1, d["n_steps"]),
                     d["all_finite"], d["neg_S"], fmt(d["cons_resid"]),
                     fmt(d["E_preclip_deficit"])))
    print()
    write_csv("tab_bare.csv",
              ["beta_sp", "treatment", "arm", "substep_mode",
               "substeps_per_macro", "all_finite", "neg_S",
               "cons_resid_frac", "cons_resid_1e6", "min_S_frac",
               "E_preclip_deficit_frac", "E_preclip_deficit_1e6"], rows_bare)

    # ---------------- grid independence in N (fig:crossscale) ----------------
    print("\n[4/4] grid independence in N at beta_sp=%.1f" % STRESS_BETA)
    rows6, rows_pc = [], []
    for n in N_LIST:
        Sn, bn, Rn, Wn, Nn = setup_at_N(states, rho, N_all, W_all, n)
        clip1 = run_series("clip", STRESS_BETA, Sn, bn, Rn, Wn, Nn, phis, sigma,
                           grid, substeps=1)
        m3 = run_series("M3-p3", STRESS_BETA, Sn, bn, Rn, Wn, Nn, phis, sigma,
                        grid)
        r = run_series("M3-p3", beta_cal, Sn, bn, Rn, Wn, Nn, phis, sigma,
                       grid_ref)
        c = run_series("M3-p3", beta_cal, Sn, bn, Rn, Wn, Nn, phis, sigma, grid)
        stride = int(round(H / H_REF))
        rr = {v: r[v][::stride][:len(grid)] for v in ("S", "I", "R")}
        diff = np.asarray(c["R"], float) - np.asarray(rr["R"], float)
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.sqrt(np.mean(diff ** 2)))
        rel = rel_l2(c, rr)
        rows6.append([n, fmt(clip1["E_preclip_deficit"]),
                      fmt(clip1["E_preclip_deficit"] / n),
                      fmt(clip1["E_clip_frac"]),
                      fmt(clip1["cons_resid"]), fmt(clip1["cons_resid_frac"]),
                      str(clip1["neg_S"]), str(clip1["all_finite"]),
                      fmt(m3["cons_resid"]), fmt(m3["cons_resid_frac"]),
                      fmt(m3["min_S"], "%.3e"),
                      str(m3["neg_S"]), str(m3["neg_I"]),
                      str(m3["all_finite"]),
                      fmt(l1), fmt(l2), fmt(rel)])
        rows_pc.append([n, fmt(l1), fmt(l2)])
        print("  N=%2d  clip1step E_clip=%-12s per-cell=%-11s cons=%-11s | "
              "M3 cons=%-10s | E_cell(h=0.1 vs 0.01)=%-11s"
              % (n, fmt(clip1["E_preclip_deficit"]),
                 fmt(clip1["E_preclip_deficit"] / n), fmt(clip1["cons_resid"]),
                 fmt(m3["cons_resid"]), fmt(l1)))

    print()
    write_csv("fig6_band_dense.csv",
              ["N", "E_clip_clip_1e6", "E_clip_per_cell_1e6",
               "E_clip_frac", "cons_resid_clip_1e6", "cons_resid_clip_frac",
               "neg_S_clip", "all_finite_clip",
               "cons_resid_M3_1e6", "cons_resid_M3_frac", "min_S_M3_1e6",
               "neg_S_M3", "neg_I_M3", "all_finite_M3",
               "E_cell_L1", "E_cell_L2", "rel_L2_vs_h001"], rows6)
    write_csv("percell_error_N_dense.csv", ["N", "E_cell_L1", "E_cell_L2"],
              rows_pc)
    print("DONE (deterministic, single run, no random source).")


def run_series_bare(beta_sp, S0, bins0, R0, W, N, phis, sigma, grid,
                    adaptive=False):
    """Untreated forward-Euler transfer (NO positivity projection).

    ``adaptive=True`` gives the untreated arm the same coupling-resolution
    controller as the clip arm, so that "no subcycling" and "no projection" are
    never confounded in the same column.
    """
    S, Ib, R = S0.copy(), bins0.copy(), R0.copy()
    ns = len(grid) - 1
    Srec = np.zeros((ns + 1, S.shape[0]))
    Irec = np.zeros_like(Srec)
    Rrec = np.zeros_like(Srec)
    Srec[0] = S / N
    Irec[0] = Ib.sum(axis=1) / N
    Rrec[0] = R / N
    cons = 0.0
    cons_frac = 0.0
    negS = negI = 0
    finite = True
    deficit = 0.0
    minS = float("inf")
    minS_frac = float("inf")
    max_abs = 0.0
    nsub_tot = 0
    maxC = 0.0
    kdone = 0
    t_active = 0.0
    for k in range(ns):
        dt = float(grid[k + 1] - grid[k])
        nsub = (chl.adaptive_subcycle(S, Ib, R, W, N, phis, sigma, beta_sp, dt)
                if adaptive else 1)
        nsub_tot += nsub
        hs = dt / nsub
        T = chl.poisson_T(sigma, hs)
        with np.errstate(all="ignore"):
            for _ in range(nsub):
                Phi = Ib @ phis
                Lam = beta_sp * (W @ (Phi / N))
                maxC = max(maxC, float(np.nanmax(hs * Lam)))
                X = hs * Lam * S
                S = S - X
                deficit += float(np.maximum(-S, 0.0).sum())
                I_new = Ib @ T.T
                removed = Ib.sum(axis=1) - I_new.sum(axis=1)
                I_new[:, 0] += X
                Ib = I_new
                R = R + removed
            Itot = Ib.sum(axis=1)
            cons = max(cons, float(np.nanmax(np.abs(S + Itot + R - N))))
            cons_frac = max(cons_frac, float(np.nanmax(
                np.abs(S + Itot + R - N) / N)))
            minS = min(minS, float(np.nanmin(S)))
            minS_frac = min(minS_frac, float(np.nanmin(S / N)))
            negS += int((S < -1e-12).sum())
            negI += int((Itot < -1e-12).sum())
            finite = finite and bool(np.all(np.isfinite(S))
                                     and np.all(np.isfinite(Ib))
                                     and np.all(np.isfinite(R)))
            max_abs = max(max_abs, float(np.nanmax(np.abs(S))),
                          float(np.nanmax(np.abs(Ib))),
                          float(np.nanmax(np.abs(R))))
            if float(np.nanmax(Itot / N)) > 0.01:
                t_active = float(grid[k + 1])
            Srec[k + 1] = S / N
            Irec[k + 1] = Itot / N
            Rrec[k + 1] = R / N
        kdone = k + 1
        if not finite:
            Srec[k + 1:] = np.nan
            Irec[k + 1:] = np.nan
            Rrec[k + 1:] = np.nan
            break
    fR = Rrec[-1] if np.all(np.isfinite(Rrec[-1])) else np.full(S.shape, np.nan)
    return dict(S=Srec, I=Irec, R=Rrec, cons_resid=cons,
                cons_resid_frac=cons_frac, min_S=minS, min_S_frac=minS_frac,
                E_clip_frac=deficit / float(N.sum()),
                neg_S=negS, neg_I=negI, all_finite=finite, n_subcycle=nsub_tot,
                lam_evals=nsub_tot, wall_ms=float("nan"), status="ok",
                E_preclip_deficit=deficit, max_abs_y=max_abs,
                max_C_t0=float("nan"), max_C_run=maxC,
                final_attack_min=(float(np.nanmin(fR))
                                  if np.all(np.isfinite(fR)) else float("nan")),
                n_steps=kdone, t_active=t_active)


if __name__ == "__main__":
    main()
