"""p0b_experiment.py -- P0-B falsifiable numerical experiments (numpy/scipy only).

Question: at the biologically relevant R0 = 15 (beta_sp = 15/13), can we make
C_i = h*Lambda_i > 1 and activate the Pi_MPP limiter by increasing ONLY the
coupling macro step h (1-2 day population steps are common practice), and does
OURS then differ from clip in a distinguishable, physically interpretable way
and strictly dominate a set of fair baselines on at least one metric?

All numbers printed/written by this script are measured, never assigned.

Outputs (new files only, existing files untouched):
    figures/p0b_hscan.csv        + submission/figures/
    figures/p0b_baselines.csv    + submission/figures/
    figures/p0b_equivalence.csv  + submission/figures/
    (p0b_results.md is written by hand from these CSVs + stdout)

Shared setup is taken verbatim from the canonical main-chain module
coupling_host_level.py (root version) and regen_fig4_and_bare.py:
  * infectious kernel phi(k), sigma = K/tau*  from chl.build_infectious_kernel()
  * 52-state Tycho field rho               from chl.load_rho(chl.RHO_CSV)
  * gravity W, populations N, beta_sp      from chl.build_gravity(states)
  * IC: S0=(1-rho)N, bins0 = rho*N/K uniform, R0=0
  * Poisson aging matrix T(h) = exp of the linear aging chain (exact)
"""
import os
import sys
import csv
import math
import warnings
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupling_host_level as chl

OUT_DIRS = [os.path.join(os.path.dirname(HERE), "figures")]
K = chl.K_BINS
T0, T1 = chl.T_SPAN
BETA_SP = None  # resolved at runtime from chl.build_gravity (=> R0 = 15)

# ---------------------------------------------------------------------------
# 0. canonical shared setup
# ---------------------------------------------------------------------------
ker = chl.build_infectious_kernel()
phis = ker["phis"]
sigma = ker["sigma"]
states, rho = chl.load_rho(chl.RHO_CSV)
W, N, beta_sp = chl.build_gravity(states)
BETA_SP = beta_sp
n = len(states)

S0 = (1.0 - rho) * N
I0 = rho * N
R0 = np.zeros_like(S0)
bins0 = np.tile(I0[:, None] / K, (1, K))


def poisson_T(h):
    """Exact propagator of the linear aging chain over step h (same as
    regen_fig4_and_bare.poisson_matrix and chl.host_step's inline matrix)."""
    nh = np.arange(K)
    p = (np.exp(-sigma * h) * (sigma * h) ** nh
         / np.array([math.factorial(int(x)) for x in nh], dtype=float))
    T = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        T[k, :k + 1] = p[np.arange(k, -1, -1)]
    return T


def t_grid(h):
    return np.round(np.arange(T0, T1 + h / 2, h), 8)


# ---------------------------------------------------------------------------
# 1. a single generic macro step (literal copy of the FROZEN limiter step
#    `chl.host_step_limiter_reference`, with the delivered-new-infections line
#    switched by `arm`).  Everything else -- including a = Lambda*S and the
#    Poisson matrix application -- is shared.  The production host step is now
#    M3 (`chl.host_step`), which is verified separately in m3_land_verify.py.
# ---------------------------------------------------------------------------
ARMS = ("ours", "clip_trunc", "clip", "bare", "patankar", "nsfd")


def host_step_generic(S, Ibins, R, W, N, phis, sigma, beta_sp, h, T, arm):
    Phi = Ibins @ phis
    Lam = beta_sp * (W @ (Phi / N))
    C = h * Lam
    a = Lam * S
    X = h * a                                   # unlimited S->I transfer = C*S

    if arm == "ours":
        # Zhang-Shu single-factor limiter, exactly chl.host_mpp_limiter
        Itot = Ibins.sum(axis=1)
        theta, _ = chl.host_mpp_limiter(S, Itot, R, -X, X, np.zeros_like(S))
        delivered = theta * X
        S_new = S - delivered
    elif arm == "clip_trunc":
        # ONE-LINE SUBSTITUTE: delivered = min(X, S), no Zhang-Shu wrapper,
        # no theory. Mathematically theta*X == min(X,S) for X>0.
        delivered = np.minimum(X, S)
        S_new = S - delivered
    elif arm == "clip":
        # post hoc clipping: S truncated to >=0, but I receives the FULL X
        delivered = X
        S_new = np.maximum(S - delivered, 0.0)
    elif arm == "bare":
        delivered = X
        S_new = S - delivered
    elif arm == "patankar":
        # modified-Patankar-Euler (production/destruction weighted by the
        # updated state): S_new = S/(1+C), delivered = S*C/(1+C)
        delivered = S * C / (1.0 + C)
        S_new = S - delivered
    elif arm == "nsfd":
        # exponential / nonstandard-FD: exact frozen-Lambda S decay
        delivered = S * (1.0 - np.exp(-C))
        S_new = S - delivered
    else:
        raise ValueError("unknown arm %r" % arm)

    I_new = Ibins @ T.T
    removed = Ibins.sum(axis=1) - I_new.sum(axis=1)
    I_new[:, 0] += delivered
    R_new = R + removed

    theta_out = None
    if arm == "ours":
        theta_out = theta
    stat = dict(C=C, Lam=Lam, X=X, delivered=delivered, theta=theta_out,
                max_C=float(C.max()),
                frac_active=float(np.mean(theta < 1.0 - 1e-12)) if arm == "ours" else 0.0,
                min_S=float(S_new.min()))
    return S_new, I_new, R_new, stat


# ---------------------------------------------------------------------------
# 2. validate that the 'ours' branch is bit-for-bit the canonical main chain
# ---------------------------------------------------------------------------
def validate_ours_matches_mainchain():
    h = 0.7
    T = poisson_T(h)
    S, b, R = S0.copy(), bins0.copy(), R0.copy()
    S1, b1, R1, _ = chl.host_step_limiter_reference(
        S, b, R, W, N, phis, sigma, beta_sp, h)
    S2, b2, R2, _ = host_step_generic(S, b, R, W, N, phis, sigma, beta_sp, h, T, "ours")
    ok = (np.array_equal(S1, S2) and np.array_equal(b1, b2)
          and np.array_equal(R1, R2))
    print("[validate] generic('ours') == chl.host_step_limiter_reference  bitwise: %s" % ok)
    print("[validate]   max|S1-S2|=%.3e max|b1-b2|=%.3e max|R1-R2|=%.3e"
          % (np.max(np.abs(S1 - S2)), np.max(np.abs(b1 - b2)),
             np.max(np.abs(R1 - R2))))
    return ok


# ---------------------------------------------------------------------------
# 3. full-run driver with per-step diagnostics
# ---------------------------------------------------------------------------
def run_full(arm, h, want_series=False):
    """Advance one arm on grid(h). Returns diagnostics dict (and optionally
    full per-step state series)."""
    grid = t_grid(h)
    T = poisson_T(h)
    S, b, R = S0.copy(), bins0.copy(), R0.copy()
    nstep = len(grid) - 1

    C_max = 0.0
    C_t0 = None
    act_steps = 0            # macro steps with at least one active limiter cell
    act_cells = 0            # total (step,state) active cells
    cons_resid = 0.0
    neg_S = 0
    neg_I = 0
    min_S = float("inf")
    all_finite = True
    E_clip = 0.0             # clip-only: cumulative created mass

    series = None
    if want_series:
        series = dict(S=np.zeros((nstep + 1, n)),
                      I=np.zeros((nstep + 1, n)),
                      R=np.zeros((nstep + 1, n)), t=grid.copy())
        series["S"][0] = S / N
        series["I"][0] = b.sum(1) / N
        series["R"][0] = R / N

    with np.errstate(all="ignore"):
        for k in range(nstep):
            S, b, R, st = host_step_generic(S, b, R, W, N, phis, sigma,
                                            beta_sp, h, T, arm)
            Cm = st["max_C"]
            C_max = max(C_max, Cm)
            if k == 0:
                C_t0 = Cm
            if arm == "ours":
                fa = st["frac_active"]
                if fa > 0:
                    act_steps += 1
                act_cells += int((st["theta"] < 1.0 - 1e-12).sum())
            Itot = b.sum(1)
            cons_resid = max(cons_resid,
                             float(np.max(np.abs(S + Itot + R - N))))
            neg_S += int((S < -1e-9).sum())
            neg_I += int((Itot < -1e-9).sum())
            min_S = min(min_S, float(S.min()))
            if arm == "clip":
                # mass created = delivered - (S_old - S_new); tracked via
                # deficit = max(-(S_before - X), 0) = max(X - S_old, 0)
                pass
            all_finite = all_finite and bool(
                np.all(np.isfinite(S)) and np.all(np.isfinite(b))
                and np.all(np.isfinite(R)))
            if not all_finite:
                break
            if want_series:
                series["S"][k + 1] = S / N
                series["I"][k + 1] = Itot / N
                series["R"][k + 1] = R / N

    d = dict(arm=arm, h=h, n_steps=nstep, grid=grid,
             C_max_run=C_max, C_t0=C_t0,
             act_steps=act_steps, act_cells=act_cells,
             cons_resid=cons_resid, neg_S=neg_S, neg_I=neg_I,
             min_S=min_S, all_finite=all_finite,
             I_total_final=float(b.sum()), R_total_final=float(R.sum()),
             S_total_final=float(S.sum()),
             series=series)
    return d


# ---------------------------------------------------------------------------
# 4. EXPERIMENT A: h scan, OURS vs clip, 52-state real field, R0 = 15
# ---------------------------------------------------------------------------
def experiment_A():
    hs = [0.1, 0.25, 0.5, 1, 2, 5, 10]
    rows = []
    for h in hs:
        o = run_full("ours", h, want_series=True)
        c = run_full("clip", h, want_series=True)
        # final-state max abs diff over S,I,R fractions (same physical time T1)
        diffs = [np.abs(o["series"][v][-1] - c["series"][v][-1])
                 for v in ("S", "I", "R")]
        maxdiff_final = float(max(x.max() for x in diffs))
        # whole-run max abs diff over all recorded times
        maxdiff_run = 0.0
        for v in ("S", "I", "R"):
            maxdiff_run = max(maxdiff_run,
                              float(np.max(np.abs(o["series"][v] - c["series"][v]))))
        rows.append(dict(
            h=h, n_steps=o["n_steps"],
            C_at_t0=o["C_t0"], C_max_run=o["C_max_run"],
            limiter_active_steps=o["act_steps"],
            limiter_active_cells=o["act_cells"],
            max_abs_diff_final_ours_clip=maxdiff_final,
            max_abs_diff_run_ours_clip=maxdiff_run,
            cons_resid_ours=o["cons_resid"], cons_resid_clip=c["cons_resid"],
            neg_S_ours=o["neg_S"], neg_I_ours=o["neg_I"],
            neg_S_clip=c["neg_S"], neg_I_clip=c["neg_I"],
            S_min_ours=o["min_S"], S_min_clip=c["min_S"],
            I_total_final_ours=o["I_total_final"],
            I_total_final_clip=c["I_total_final"],
            all_finite_ours=o["all_finite"], all_finite_clip=c["all_finite"],
        ))
        print("[A] h=%5.2f  C0=%.4f  Cmax=%.4f  act_steps=%d/%d  "
              "cons(ours)=%.3e cons(clip)=%.3e  maxdiff_final=%.4e  "
              "negS(clip)=%d"
              % (h, o["C_t0"], o["C_max_run"], o["act_steps"], o["n_steps"],
                 o["cons_resid"], c["cons_resid"], maxdiff_final, c["neg_S"]))

    # critical h: bisection on "any limiter activation over the run"
    def activates(h):
        return run_full("ours", h)["act_steps"] > 0
    lo, hi = 0.1, 10.0
    if not activates(hi):
        print("[A] critical h: none found up to h=10 (NO activation ever)")
        crit = None
    else:
        assert not activates(lo), "unexpected: activation at h=0.1"
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if activates(mid):
                hi = mid
            else:
                lo = mid
        crit = hi
        print("[A] critical h (first limiter activation) ~ %.6f d "
              "(no activation at %.6f d)" % (crit, lo))

    header = ["h", "n_steps", "C_at_t0", "C_max_run",
              "limiter_active_steps", "limiter_active_cells",
              "max_abs_diff_final_ours_clip", "max_abs_diff_run_ours_clip",
              "cons_resid_ours", "cons_resid_clip",
              "neg_S_ours", "neg_I_ours", "neg_S_clip", "neg_I_clip",
              "S_min_ours", "S_min_clip",
              "I_total_final_ours", "I_total_final_clip",
              "all_finite_ours", "all_finite_clip"]
    write_csv("p0b_hscan.csv", header, [fmt_row(r, header) for r in rows])
    return rows, crit


# ---------------------------------------------------------------------------
# 5. reference solution: solve_ivp on the continuous ODE at coarse grid times
#    (direct evaluation, NO interpolation).  The ODE is exactly the h->0 limit
#    of the shared map: frozen-Lam forward Euler on S->I + exact aging chain.
# ---------------------------------------------------------------------------
def reference_solution(coarse_times, rtol=1e-10, atol=1e-12):
    y0 = np.concatenate([S0, bins0.ravel(), R0])
    nS = n

    def rhs(t, y):
        S = y[:nS]
        I = y[nS:nS + n * K].reshape(n, K)
        Phi = I @ phis
        Lam = beta_sp * (W @ (Phi / N))
        dS = -Lam * S
        dI = -sigma * I
        dI[:, 1:] += sigma * I[:, :-1]
        dI[:, 0] += Lam * S
        dR = sigma * I[:, K - 1]
        return np.concatenate([dS, dI.ravel(), dR])

    sol = solve_ivp(rhs, (coarse_times[0], coarse_times[-1]), y0,
                    t_eval=coarse_times, method="DOP853",
                    rtol=rtol, atol=atol)
    print("[B] solve_ivp status=%s  nfev=%d  success=%s  "
          "t_eval points=%d" % (sol.status, sol.nfev, sol.success,
                                len(coarse_times)))
    Y = sol.y.T                                     # (M+1, 52+52K+52)
    S = Y[:, :nS] / N
    I = Y[:, nS:nS + n * K].reshape(len(coarse_times), n, K).sum(2) / N
    R = Y[:, nS + n * K:] / N
    return dict(S=S, I=I, R=R)


def error_metrics(series, ref):
    """Errors on the concatenated (S/N, I/N, R/N) arrays at the SAME physical
    time points (series['t'] == coarse grid), no interpolation.

    Returns (L2_rel, max_abs, max_scale_rel, max_pointwise_rel):
      L2_rel            = ||d||_2 / ||ref||_2                     (robust)
      max_abs           = max |d|                                (fraction units)
      max_scale_rel     = max |d| / max |ref|                    (global scale,
                          bounded <= a few, avoids tiny denominators)
      max_pointwise_rel = max |d| / (|ref| + 1e-8)               (strict
                          pointwise relative error; can be huge when a patch
                          is near depletion -- reported for completeness)
    """
    num = np.concatenate([(series[v] - ref[v]).ravel() for v in ("S", "I", "R")])
    den = np.concatenate([ref[v].ravel() for v in ("S", "I", "R")])
    l2 = float(np.linalg.norm(num) / np.linalg.norm(den))
    maxabs = float(np.abs(num).max())
    scale = float(np.abs(den).max())
    max_scale_rel = maxabs / scale if scale > 0 else float("nan")
    max_pointwise_rel = float((np.abs(num) / (np.abs(den) + 1e-8)).max())
    return l2, maxabs, max_scale_rel, max_pointwise_rel


# ---------------------------------------------------------------------------
# 6. sub-cycling arm: within each macro step subdivide so C_i <= 1, using the
#    clip step (clipping is inert there).  Cost = total number of substeps.
# ---------------------------------------------------------------------------
def run_subcycling(h_macro, c_target=1.0):
    """Adaptive sub-cycling: within each macro step, choose every substep size
    from the CURRENT Lambda so that C_i <= c_target holds at that substep.
    Because C<=1 everywhere, the clip arm is inert and conserves mass exactly.
    Cost = total number of substeps actually taken."""
    grid = t_grid(h_macro)
    S, b, R = S0.copy(), bins0.copy(), R0.copy()
    nstep = len(grid) - 1
    total_substeps = 0
    C_realized_max = 0.0
    cons_resid = 0.0
    neg_S = neg_I = 0
    min_S = float("inf")
    series = dict(S=np.zeros((nstep + 1, n)), I=np.zeros((nstep + 1, n)),
                  R=np.zeros((nstep + 1, n)), t=grid.copy())
    series["S"][0] = S / N
    series["I"][0] = b.sum(1) / N
    series["R"][0] = R / N
    with np.errstate(all="ignore"):
        for k in range(nstep):
            remaining = h_macro
            guard = 0
            while remaining > 1e-12:
                Phi = b @ phis
                Lam = beta_sp * (W @ (Phi / N))
                lam_max = float(Lam.max())
                if lam_max <= 0:
                    h_sub = remaining
                else:
                    h_sub = min(remaining, c_target / lam_max)
                T_sub = poisson_T(h_sub)
                C_realized_max = max(C_realized_max, float(h_sub * lam_max))
                S, b, R, _ = host_step_generic(
                    S, b, R, W, N, phis, sigma, beta_sp, h_sub, T_sub, "clip")
                total_substeps += 1
                remaining -= h_sub
                guard += 1
                if guard > 100000:
                    raise RuntimeError("sub-cycling did not converge")
            Itot = b.sum(1)
            cons_resid = max(cons_resid,
                             float(np.max(np.abs(S + Itot + R - N))))
            neg_S += int((S < -1e-9).sum())
            neg_I += int((Itot < -1e-9).sum())
            min_S = min(min_S, float(S.min()))
            series["S"][k + 1] = S / N
            series["I"][k + 1] = Itot / N
            series["R"][k + 1] = R / N
    return dict(arm="clip+subcycling", h=h_macro, n_steps=nstep,
                n_integration_steps=total_substeps,
                C_max_realized=C_realized_max, cons_resid=cons_resid,
                neg_S=neg_S, neg_I=neg_I, min_S=min_S, series=series)


# ---------------------------------------------------------------------------
# 7. small-h arm: clip with h_small (C_i << 1), evaluate at coarse times by
#    exact index (coarse times are integer multiples of h_small), no interp.
# ---------------------------------------------------------------------------
def run_small_h(h_macro, h_small):
    stride = int(round(h_macro / h_small))
    assert abs(stride * h_small - h_macro) < 1e-9, "coarse times must be on fine grid"
    coarse = t_grid(h_macro)
    nstep_fine = stride * (len(coarse) - 1)
    T = poisson_T(h_small)
    S, b, R = S0.copy(), bins0.copy(), R0.copy()
    cons_resid = 0.0
    neg_S = neg_I = 0
    min_S = float("inf")
    C_max = 0.0
    series = dict(S=np.zeros((len(coarse), n)), I=np.zeros((len(coarse), n)),
                  R=np.zeros((len(coarse), n)), t=coarse.copy())
    series["S"][0] = S / N
    series["I"][0] = b.sum(1) / N
    series["R"][0] = R / N
    with np.errstate(all="ignore"):
        for k in range(nstep_fine):
            S, b, R, st = host_step_generic(
                S, b, R, W, N, phis, sigma, beta_sp, h_small, T, "clip")
            C_max = max(C_max, st["max_C"])
            Itot = b.sum(1)
            cons_resid = max(cons_resid,
                             float(np.max(np.abs(S + Itot + R - N))))
            neg_S += int((S < -1e-9).sum())
            neg_I += int((Itot < -1e-9).sum())
            min_S = min(min_S, float(S.min()))
            if (k + 1) % stride == 0:
                j = (k + 1) // stride
                series["S"][j] = S / N
                series["I"][j] = Itot / N
                series["R"][j] = R / N
    return dict(arm="clip+small_h", h=h_macro, n_steps=len(coarse) - 1,
                n_integration_steps=nstep_fine, h_small=h_small,
                C_max_realized=C_max, cons_resid=cons_resid,
                neg_S=neg_S, neg_I=neg_I, min_S=min_S, series=series)


# ---------------------------------------------------------------------------
# 8. EXPERIMENTS B & C: continuous reference error + fair baselines
# ---------------------------------------------------------------------------
def experiment_BC(hs=(1, 2, 5, 10)):
    """Fair-baseline comparison at every scanned h with C_i>1."""
    rows = []
    for h_big in hs:
        coarse = t_grid(h_big)
        ref = reference_solution(coarse)
        arms = {}
        for arm in ARMS:
            arms[arm] = run_full(arm, h_big, want_series=True)
        arms["clip+subcycling"] = run_subcycling(h_big)
        arms["clip+small_h"] = run_small_h(h_big, 0.01)

        for name, d in arms.items():
            ser = d["series"]
            l2, maxabs, max_scale_rel, max_pointwise_rel = error_metrics(ser, ref)
            nintegration = d.get("n_integration_steps", d["n_steps"])
            rows.append(dict(
                h_macro=h_big, arm=name,
                n_macro_steps=d["n_steps"],
                n_integration_steps=nintegration,
                C_max_run=d.get("C_max_run", d.get("C_max_realized", float("nan"))),
                cons_resid=d["cons_resid"],
                neg_S=d["neg_S"], neg_I=d["neg_I"],
                min_S=d["min_S"],
                L2_rel_err=l2,
                max_abs_err=maxabs,
                max_scale_rel_err=max_scale_rel,
                max_pointwise_rel_err=max_pointwise_rel,
                final_I_total=float(ser["I"][-1].sum()),
                final_R_total=float(ser["R"][-1].sum()),
                all_finite=d.get("all_finite", True),
            ))
            print("[BC] h=%2g arm=%-16s cost=%5d Cmax_real=%.3f "
                  "L2rel=%.4e maxabs=%.4e maxScaleRel=%.4e "
                  "cons=%.3e negS=%d negI=%d"
                  % (h_big, name, nintegration,
                     d.get("C_max_run", d.get("C_max_realized", float("nan"))),
                     l2, maxabs, max_scale_rel, d["cons_resid"],
                     d["neg_S"], d["neg_I"]))

    header = ["h_macro", "arm", "n_macro_steps", "n_integration_steps",
              "C_max_run", "cons_resid", "neg_S", "neg_I", "min_S",
              "L2_rel_err", "max_abs_err", "max_scale_rel_err",
              "max_pointwise_rel_err", "final_I_total", "final_R_total",
              "all_finite"]
    write_csv("p0b_baselines.csv", header, [fmt_row(r, header) for r in rows])
    return rows


# ---------------------------------------------------------------------------
# 9. EXPERIMENT D: decisive one-line equivalence test (entire state sequence)
# ---------------------------------------------------------------------------
def experiment_D(hs=(0.1, 1, 2, 5, 10)):
    rows = []
    for h in hs:
        o = run_full("ours", h, want_series=True)
        t = run_full("clip_trunc", h, want_series=True)
        maxabs = 0.0
        exactly_equal = True
        for v in ("S", "I", "R"):
            a, b = o["series"][v], t["series"][v]
            maxabs = max(maxabs, float(np.max(np.abs(a - b))))
            exactly_equal = exactly_equal and np.array_equal(a, b)
        raw = compare_raw_states(h)
        rows.append(dict(
            h=h, n_steps=o["n_steps"],
            C_max_run=o["C_max_run"],
            max_abs_diff_series=maxabs,
            bit_identical_series=exactly_equal,
            max_abs_diff_raw=raw["max_abs_diff"],
            bit_identical_raw=raw["bit_identical"],
            where_differs=raw["where"],
            n_steps_with_diff=raw["n_diff_steps"],
            max_diff_in_ulp_of_S=raw["max_diff_in_ulp"],
            first_diff_C_gt1_cells=raw["first_diff_cgt1_cells"],
            cons_resid_ours=o["cons_resid"],
            cons_resid_clip_trunc=t["cons_resid"],
            cons_resid_diff=abs(o["cons_resid"] - t["cons_resid"]),
        ))
        print("[D] h=%5.2f Cmax=%.4f  series: maxdiff=%.3e bit_identical=%s | "
              "raw: maxdiff=%.3e (%.2f ulp) bit_identical=%s  diff_steps=%d  (%s)"
              % (h, o["C_max_run"], maxabs, exactly_equal,
                 raw["max_abs_diff"], raw["max_diff_in_ulp"],
                 raw["bit_identical"], raw["n_diff_steps"], raw["where"]))

    header = ["h", "n_steps", "C_max_run",
              "max_abs_diff_series", "bit_identical_series",
              "max_abs_diff_raw", "bit_identical_raw", "where_differs",
              "n_steps_with_diff", "max_diff_in_ulp_of_S",
              "first_diff_C_gt1_cells",
              "cons_resid_ours", "cons_resid_clip_trunc", "cons_resid_diff"]
    write_csv("p0b_equivalence.csv", header, [fmt_row(r, header) for r in rows])
    return rows


def compare_raw_states(h):
    """Re-run ours and clip_trunc simultaneously, comparing the RAW internal
    (S, bins, R) arrays at every step, and locate the difference in ulps."""
    grid = t_grid(h)
    T = poisson_T(h)
    S_o, b_o, R_o = S0.copy(), bins0.copy(), R0.copy()
    S_t, b_t, R_t = S0.copy(), bins0.copy(), R0.copy()
    maxdiff = 0.0
    max_ulp = 0.0
    identical = True
    first = None
    first_cgt1 = -1
    n_diff_steps = 0
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            # Lambda / C of the SHARED pre-step state (same for both arms)
            Phi = b_o @ phis
            Lam = beta_sp * (W @ (Phi / N))
            C = h * Lam
            S_pre = S_o.copy()
            S_o, b_o, R_o, _ = host_step_generic(
                S_o, b_o, R_o, W, N, phis, sigma, beta_sp, h, T, "ours")
            S_t, b_t, R_t, _ = host_step_generic(
                S_t, b_t, R_t, W, N, phis, sigma, beta_sp, h, T, "clip_trunc")
            dS = np.abs(S_o - S_t)
            d = max(float(dS.max()), float(np.max(np.abs(b_o - b_t))),
                    float(np.max(np.abs(R_o - R_t))))
            eq = (np.array_equal(S_o, S_t) and np.array_equal(b_o, b_t)
                  and np.array_equal(R_o, R_t))
            if d > 0:
                n_diff_steps += 1
                # ulp measured against the PRE-step S magnitude (O(1) cells
                # only); depleted S~0 cells would otherwise inflate the ratio
                mask = (dS > 0) & (S_pre > 1e-6)
                if mask.any():
                    ulp = float(np.max(dS[mask] / np.spacing(S_pre[mask])))
                    max_ulp = max(max_ulp, ulp)
            if not eq and first is None:
                first = ("step %d (t=%.4g)" % (k + 1, grid[k + 1]))
                first_cgt1 = int((C > 1.0).sum())
            maxdiff = max(maxdiff, d)
            identical = identical and eq
    return dict(max_abs_diff=maxdiff, bit_identical=identical,
                where=first if first else "identical at every step",
                n_diff_steps=n_diff_steps, max_diff_in_ulp=max_ulp,
                first_diff_cgt1_cells=first_cgt1)


def ulp_microtest(h=5):
    """Decisive numerical proof of WHERE the residual difference comes from.

    At a cell with C>1, OURS computes delivered = theta*X with
    theta = min(1, S/X) = S/X; clip_trunc computes delivered = min(X,S) = S.
    Algebraically identical; in IEEE-754 the division+multiply introduces
    <=1 ulp.  We show that (a) the discrepancy is exactly this roundoff,
    (b) it is confined to C>1 cells, (c) regrouping the SAME expression as
    min(X,S) removes it entirely, and (d) it never amplifies into a
    macroscopic difference over the run.
    """
    grid = t_grid(h)
    T = poisson_T(h)
    S, b, R = S0.copy(), bins0.copy(), R0.copy()
    Phi = b @ phis
    Lam = beta_sp * (W @ (Phi / N))
    C = h * Lam
    a = Lam * S
    X = h * a
    theta, _ = chl.host_mpp_limiter(S, b.sum(1), R, -X, X, np.zeros_like(S))
    d_ours = theta * X
    d_trunc = np.minimum(X, S)
    sel = C > 1.0
    print("[ulp] h=%g  step-1 C>1 cells = %d / %d" % (h, int(sel.sum()), n))
    if sel.any():
        j = int(np.argmax(C))
        print("[ulp]   witness cell idx=%d  S=%.17e  C=%.6f  X=%.17e"
              % (j, S[j], C[j], X[j]))
        print("[ulp]   theta=%.17e  theta*X=%.17e  min(X,S)=%.17e"
              % (theta[j], d_ours[j], d_trunc[j]))
        print("[ulp]   |theta*X - min(X,S)| = %.3e = %.3f ulp(S)"
              % (abs(d_ours[j] - d_trunc[j]),
                 abs(d_ours[j] - d_trunc[j]) / np.spacing(S[j])))
        print("[ulp]   max over ALL cells = %.3e ; max over C<=1 cells = %.3e"
              % (float(np.max(np.abs(d_ours - d_trunc))),
                 float(np.max(np.abs((d_ours - d_trunc)[~sel]))
                       if (~sel).any() else 0.0)))
    # (c) regrouping removes it: use an algebraically identical OURS that
    #     computes delivered = min(X,S) but is otherwise the SAME main chain
    #     (this is by construction clip_trunc, so trivially bit-identical).
    # (d) no amplification: already measured by compare_raw_states at h=2,5,10.
    return


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fmt_row(d, header):
    out = []
    for k in header:
        v = d[k]
        if isinstance(v, bool):
            out.append(str(v).lower())
        elif isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif isinstance(v, (float, np.floating)):
            out.append("%.10e" % v)
        else:
            out.append(v)
    return out


def write_csv(name, header, rows):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
    print("  wrote %s (%d rows) to %s" % (name, len(rows), OUT_DIRS))


def main():
    print("=" * 78)
    print("P0-B experiment | R0=15 | beta_sp=%.10f (=15/13) | %d states | "
          "K=%d | tau*=%.4f sigma=%.6f" %
          (beta_sp, n, K, ker["tau_star"], sigma))
    print("  tau*=%.4f  sum(phi)dtau=%.6f  target Tg=%.1f" %
          (ker["tau_star"], ker["sum_phi_dtau"], chl.TG))
    print("=" * 78)
    validate_ours_matches_mainchain()
    print("-" * 78)
    print("EXPERIMENT A: h scan (R0=15, real 52-state field)")
    rowsA, crit = experiment_A()
    print("-" * 78)
    print("EXPERIMENT B/C: continuous reference (solve_ivp) + fair baselines "
          "at all h with C>1 (h=1,2,5,10 d)")
    rowsBC = experiment_BC((1, 2, 5, 10))
    print("-" * 78)
    print("EXPERIMENT D: one-line substitute equivalence (entire state sequence)")
    rowsD = experiment_D()
    print("-" * 78)
    print("ULP MICRO-TEST: exact source of the residual difference")
    ulp_microtest(5)
    print("=" * 78)
    print("SUMMARY: critical h ~ %s" % ("%.6f d" % crit if crit else "NONE (no activation)"))
    print("A   rows: %d ; BC rows: %d ; D rows: %d" %
          (len(rowsA), len(rowsBC), len(rowsD)))
    print("=" * 78)
    return dict(A=rowsA, crit=crit, BC=rowsBC, D=rowsD)


if __name__ == "__main__":
    main()
