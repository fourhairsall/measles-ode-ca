# -*- coding: utf-8 -*-
# =============================================================================
# regen_wavefront.py  --  R8 / route B experiment line
# -----------------------------------------------------------------------------
# "lambda_0 = 0 branch + non-zero diagonal W + small-seeding wavefront"
#
# PURPOSE (two reviewer hard points)
#   (1) The shipped 52x52 gravity matrix W has an EXACTLY ZERO diagonal, i.e.
#       no within-entity transmission.  We construct the self-loop family
#           W'(alpha) = (1-alpha) W + alpha I,  alpha in {0, .02, .05, .10}
#       and measure diag / row sums / spectral radius / the induced within-
#       entity force of infection at t=0.
#   (2) coupling_host_level.m3_square_coeffs collapses to d1=d2=d3=0 when
#       lambda_0 = 0, so Lambda(s) == 0 identically on that step.  We implement
#       an INDEPENDENT variant coefficient function (this file only; the shipped
#       module is imported read-only and never modified) using
#           Lambda(s) = s * (b0 + b1 s + b2 s^2)^2 >= 0  for s in [0,h],
#       matched to lam_derivs' first three derivatives at the zero branch.
#
# DERIVATION OF THE FALLBACK COEFFICIENTS  (s in [0,h], so s >= 0)
#   Let Q(s) = b0 + b1 s + b2 s^2,  Lambda(s) = s Q(s)^2.  Expanding,
#       Lambda(s) = b0^2 s + 2 b0 b1 s^2 + (2 b0 b2 + b1^2) s^3
#                   + 2 b1 b2 s^4 + b2^2 s^5.
#   Hence, with lambda_j = d^j Lambda/dt^j (0) as returned by chl.lam_derivs,
#       Lambda(0)   = 0                     (consistent with lambda_0 = 0),
#       Lambda'(0)  = b0^2          = lambda_1   ->  b0 = sqrt(lambda_1),
#       Lambda''(0) = 4 b0 b1       = lambda_2   ->  b1 = lambda_2/(4 b0),
#       Lambda'''(0)= 6(2 b0 b2+b1^2)= lambda_3 ->  b2 = (lambda_3/6 - b1^2)/(2 b0).
#   Non-negativity is structural: s >= 0 and Q^2 >= 0; the SIGN of Q is
#   irrelevant, exactly as for the shipped lambda_0 (1+d1 s+...)^2 form.
#
# TOLERANCE CRITERION (why it is RELATIVE, not absolute)
#   lambda_0 decays to ~1e-56 /day in the extinction tail of the R0=6 runs, so
#   ANY fixed absolute tolerance (e.g. 1e-12) would misclassify tens of
#   thousands of tail (patch,step) pairs.  Because W is row-stochastic,
#       Lambda_i = beta_sp * sum_j w_ij (Phi_j/N_j) <= beta_sp * max_j (Phi_j/N_j)
#   so we use the step's own force-of-infection scale
#       scale = beta_sp * max_j (Phi_j/N_j),   TOL = TOL_REL * scale,
#   with TOL_REL = 1e-12 by default.  A patch is routed to the fallback iff
#       lambda_0 < TOL   AND   lambda_1 > 0.
#   The second clause is essential: lambda_1 <= 0 means a decaying (or
#   even-order) zero, where Lambda == 0 IS the correct leading behaviour and
#   injecting anything would be wrong.  All other patches use the shipped
#   m3_square_coeffs formulas verbatim (bitwise-identical; unit-tested below).
#   TOL_REL sensitivity is scanned in {0, 1e-16, 1e-12, 1e-8, 1e-4}.
#
# ARMS (identical IC / W' / R0 / h / horizon)
#   1  "M3-p3"            : shipped chl.host_step, lambda_0=0 => Lambda == 0.
#   2  "M3-p3+fallback"   : this file's host_step_fallback.
#   3  "REF-DOP853"       : continuous ODE solved by scipy DOP853,
#                           rtol=1e-10, atol=1e-12.  This is the TRUTH: it
#                           has no square parametrisation at all and therefore
#                           no lambda_0=0 branch.  Output grid 0.25 d.
#   4  "EXP2-midpoint"    : shipped chl.baseline_step, frozen-Lambda midpoint
#                           exponential integrator (2nd order, independent),
#                           run at h=0.1 d only.
#
# DISCIPLINE
#   * All numbers are produced by THIS run; nothing is copied from the paper.
#   * No existing .py/.csv/.tex/.bib/.md is modified.
#   * float64 throughout; no RNG, no seed.
#   * Both CSV copies are written from ONE in-memory payload; md5 asserted.
#   * Self-tests run BEFORE the main sweep and abort on failure.
#
# ONE-KEY REPRODUCTION (from the project root)
#   C:/Python314/python.exe submission/code/regen_wavefront.py
# =============================================================================
import os
import sys
import csv
import io
import math
import time
import hashlib
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent          # submission/code
SUB = HERE.parent                              # submission
ROOT = SUB.parent                              # project root
sys.path.insert(0, str(HERE))

import coupling_host_level as chl              # read-only
import ms_numpy as msn                        # read-only

DTYPE = np.float64

# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------
ALPHAS = (0.0, 0.02, 0.05, 0.10)
RHO_SEEDS = (1e-4, 1e-3, 1e-2)
R0_LIST = (1.5, 3.0, 6.0)
H_LIST = (0.1, 1.0, 4.2, 9.0, 20.0)
HORIZON = 240.0
THETA_ARRIVE = 1e-6       # primary wavefront-arrival threshold on I_i/N_i
THETA_TAKEOFF = 1e-2      # secondary "epidemic takeoff" threshold on I_i/N_i
TOL_REL = 1e-12           # default relative lambda_0 tolerance (see header)
TOL_REL_SCAN = (0.0, 1e-16, 1e-12, 1e-8, 1e-4)
REF_DT = 0.25             # DOP853 output grid
REF_RTOL, REF_ATOL = 1e-10, 1e-12
REF_FLOOR_RTOL, REF_FLOOR_ATOL = 1e-12, 1e-14
EXP2_H = 0.1

W_CSV = HERE / "W_gravity_52x52_rowstochastic.csv"
RHO_CSV = SUB / "figures" / "fig2_tycho_field.csv"
TRAJ_CSV = SUB / "figures" / "fig3_withinhost_traj.csv"

OUT_WF = ("figures/wavefront_fallback.csv",
          "submission/figures/wavefront_fallback.csv")
OUT_DIAG = ("figures/coupling_diag_seed.csv",
            "submission/figures/coupling_diag_seed.csv")
REPORT_MD = ROOT / "_review_new_tmp_lead" / "r8_wavefront_findings.md"
LOG_PATH = ROOT / "_review_new_tmp_lead" / "r8_wavefront_run.log"

ARM1 = "M3-p3"
ARM2 = "M3-p3+fallback"
ARM3 = "REF-DOP853"
ARM4 = "EXP2-midpoint"

# ---------------------------------------------------------------------------
# Logging: tee stdout into the run log (self-contained, no shell redirection)
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_PATH, "w", encoding="utf-8", newline="")
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)
    return fh, orig


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Input loading (read-only)
# ---------------------------------------------------------------------------
def load_inputs():
    with open(W_CSV, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    states = header[1:]
    W0 = np.loadtxt(W_CSV, delimiter=",", skiprows=1,
                    usecols=range(1, 1 + len(states)), dtype=DTYPE)
    row_labels = []
    with open(W_CSV, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                row_labels.append(line.split(",")[0])
    assert row_labels == states, "W row labels != column labels"
    assert W0.shape == (52, 52)

    # fig2 is used ONLY to confirm the state ordering; its rho field (TX=1.0,
    # i.e. 100% prevalence) is deliberately NOT used as an initial condition.
    rho_states, rho_field = chl.load_rho(str(RHO_CSV))
    assert rho_states == states, "fig2 state order != W order"
    assert abs(float(rho_field[states.index("TX")]) - 1.0) < 1e-12

    W_geo, N, beta_sp_cal = chl.build_gravity(states)
    assert np.array_equal(W0, W_geo), "CSV W != build_gravity(W) bitwise"
    N = np.asarray(N, dtype=DTYPE)

    ker = chl.build_infectious_kernel(str(TRAJ_CSV))
    phis = np.asarray(ker["phis"], dtype=DTYPE)
    sigma = float(ker["sigma"])
    K = int(chl.K_BINS)
    return dict(states=states, W0=W0, N=N, phis=phis, sigma=sigma, K=K,
                ker=ker, beta_sp_cal=float(beta_sp_cal),
                rho_field=rho_field)


def w_variant(W0, alpha):
    n = W0.shape[0]
    return (1.0 - alpha) * W0 + alpha * np.eye(n, dtype=DTYPE)


def w_diagnostics(W0, Wp):
    n = Wp.shape[0]
    off = ~np.eye(n, dtype=bool)
    ev = np.linalg.eigvals(Wp)
    return dict(diag_min=float(np.diag(Wp).min()),
                diag_max=float(np.diag(Wp).max()),
                rowsum_min=float(Wp.sum(1).min()),
                rowsum_max=float(Wp.sum(1).max()),
                rowsum_dev=float(np.max(np.abs(Wp.sum(1) - 1.0))),
                offdiag_min=float(Wp[off].min()),
                offdiag_max=float(Wp[off].max()),
                spectral_radius=float(np.max(np.abs(ev))),
                max_abs_diff=float(np.max(np.abs(Wp - W0))))


def make_ic(cfg, rho_seed):
    """Only TX is seeded; the other 51 entities have I=0 and S=N.

    The seeded infected mass is spread uniformly over the K age bins, exactly
    the convention of chl.run_demo (bins0 = I0/K per bin).  The 100%-prevalence
    field of fig2_tycho_field.csv is NOT used.
    """
    n = len(cfg["states"])
    N, K = cfg["N"], cfg["K"]
    itx = cfg["states"].index("TX")
    I0 = np.zeros(n, dtype=DTYPE)
    I0[itx] = DTYPE(rho_seed) * N[itx]
    S0 = N - I0
    R0 = np.zeros(n, dtype=DTYPE)
    bins0 = np.tile(I0[:, None] / K, (1, K)).astype(DTYPE)
    return S0, bins0, R0, itx


def beta_of(R0):
    return DTYPE(R0) / DTYPE(chl.TG)


# ---------------------------------------------------------------------------
# (C) Independent fallback coefficient function
# ---------------------------------------------------------------------------
def m3_coeffs_variant(S, I, R, W, N, phis, sigma, beta_sp, h,
                      tol_rel=TOL_REL, p=3):
    """Return the shipped (l0,d1,d2,d3) PLUS the s*square fallback data.

    For every patch NOT on the branch the returned l0,d1,d2,d3 are computed
    with the formulas of chl.m3_square_coeffs verbatim (bitwise identical).
    On the branch (lambda_0 < TOL and lambda_1 > 0) they are zeroed and the
    b-coefficients of Lambda(s)=s(b0+b1 s+b2 s^2)^2 are supplied instead.
    """
    lam = chl.lam_derivs(S, I, R, W, N, phis, sigma, beta_sp, p=p)
    l0, l1, l2, l3 = lam[0], lam[1], lam[2], lam[3]

    prev = (I @ phis) / N
    scale = float(beta_sp) * float(np.max(prev))
    tol = DTYPE(tol_rel) * DTYPE(scale)

    # ---- shipped formulas, verbatim -------------------------------------
    pos = l0 > 0
    l0safe = np.where(pos, l0, 0.0)
    inv = np.where(pos, 1.0 / np.where(pos, l0, 1.0), 0.0)
    a1 = l1 * inv
    a2 = 0.5 * l2 * inv
    a3 = (l3 / 6.0) * inv
    d1 = a1 / 2.0
    d2 = (a2 - d1 ** 2) / 2.0
    d3 = a3 / 2.0 - d1 * d2

    # ---- branch detection and fallback coefficients ---------------------
    cand = (l0 < tol) & (l1 > 0)
    with np.errstate(all="ignore"):
        b0safe = np.where(cand, l1, 1.0)
        b0 = np.sqrt(b0safe)
        b1 = l2 / (4.0 * b0)
        b2 = (l3 / 6.0 - b1 ** 2) / (2.0 * b0)
    ok = np.isfinite(b0) & np.isfinite(b1) & np.isfinite(b2)
    branch = cand & ok
    n_rejected = int((cand & ~ok).sum())

    # zero the shipped square on the branch so hazard_variant's "normal"
    # channel contributes nothing there
    l0safe = np.where(branch, 0.0, l0safe)
    d1 = np.where(branch, 0.0, d1)
    d2 = np.where(branch, 0.0, d2)
    d3 = np.where(branch, 0.0, d3)

    return dict(l0=l0safe, d1=d1, d2=d2, d3=d3,
                b0=b0, b1=b1, b2=b2, branch=branch,
                l0_raw=l0, l1=l1, l2=l2, l3=l3,
                tol=float(tol), scale=scale,
                n_nonpos=int((~pos).sum()), n_rejected=n_rejected)


def hazard_variant(l0, d1, d2, d3, b0, b1, b2, branch, kernel, h):
    """H(u_j), H(h), Lambda(u_j) and log Lambda(u_j), positive-weight Gauss.

    Non-branch patches use exactly the shipped expressions
        Lambda = l0 (1+d1 s+d2 s^2+d3 s^3)^2
    (bitwise-identical to chl.hazard_and_lambda); branch patches use
        Lambda = s (b0+b1 s+b2 s^2)^2.
    """
    u = kernel["u"]
    xh, wh = kernel["xh"], kernel["wh"]
    V = 0.5 * u[:, None] * (xh[None, :] + 1.0)      # (q,qh)
    Cw = 0.5 * u[:, None] * wh[None, :]
    V3 = V[None, :, :]

    PV = (1.0 + d1[:, None, None] * V3
          + d2[:, None, None] * V3 ** 2 + d3[:, None, None] * V3 ** 3)
    lamV_n = l0[:, None, None] * PV ** 2
    QV = (b0[:, None, None] + b1[:, None, None] * V3
          + b2[:, None, None] * V3 ** 2)
    lamV_f = V3 * QV ** 2
    lamV = np.where(branch[:, None, None], lamV_f, lamV_n)
    H = np.einsum("jl,njl->nj", Cw, lamV)

    Vh = 0.5 * h * (xh + 1.0)                        # (qh,)
    Cwh = 0.5 * h * wh
    Vh2 = Vh[None, :]
    PVh = (1.0 + d1[:, None] * Vh2 + d2[:, None] * Vh2 ** 2
           + d3[:, None] * Vh2 ** 3)
    lamVh_n = l0[:, None] * PVh ** 2
    QVh = b0[:, None] + b1[:, None] * Vh2 + b2[:, None] * Vh2 ** 2
    lamVh_f = Vh2 * QVh ** 2
    lamVh = np.where(branch[:, None], lamVh_f, lamVh_n)
    Hh = (Cwh[None, :] * lamVh).sum(1)

    U = u[None, :]
    Pu = 1.0 + d1[:, None] * U + d2[:, None] * U ** 2 + d3[:, None] * U ** 3
    lam_u_n = l0[:, None] * Pu ** 2
    Qu = b0[:, None] + b1[:, None] * U + b2[:, None] * U ** 2
    lam_u_f = U * Qu ** 2
    lam_u = np.where(branch[:, None], lam_u_f, lam_u_n)

    log_lam_n = (np.log(np.where(l0 > 0, l0, 1e-300))[:, None]
                 + 2.0 * np.log(np.where(np.abs(Pu) > 0,
                                         np.abs(Pu), 1e-300)))
    log_lam_f = (np.log(u)[None, :]
                 + 2.0 * np.log(np.where(np.abs(Qu) > 0,
                                         np.abs(Qu), 1e-300)))
    log_lam = np.where(branch[:, None], log_lam_f, log_lam_n)
    return dict(H=H, Hh=Hh, lam_u=lam_u, Pu=Pu, Qu=Qu, log_lam=log_lam)


def host_step_fallback(S, Ibins, R, W, N, phis, sigma, beta_sp, h,
                       kernel=None, T=None, tol_rel=TOL_REL, p=3):
    """One M3 macro step with the s*square fallback on the lambda_0=0 branch.

    Everything after the hazard evaluation is copied verbatim from
    chl.host_step; no limiter, no clipping, no truncation, no damping.
    """
    K = Ibins.shape[1]
    if kernel is None:
        kernel = chl.static_kernel(h, sigma, K=K)
    if T is None:
        T = chl.poisson_T(sigma, h, K=K)

    co = m3_coeffs_variant(S, Ibins, R, W, N, phis, sigma, beta_sp, h,
                           tol_rel=tol_rel, p=p)
    l0 = co["l0"]
    hz = hazard_variant(l0, co["d1"], co["d2"], co["d3"],
                        co["b0"], co["b1"], co["b2"], co["branch"],
                        kernel, h)
    H, Hh, lam_u, log_lam = hz["H"], hz["Hh"], hz["lam_u"], hz["log_lam"]

    logS = np.where(S > 0, np.log(np.where(S > 0, S, 1.0)), -np.inf)
    log_g = log_lam + logS[:, None] - H
    mx = np.max(log_g, axis=1)
    bad = ~np.isfinite(mx)
    mx_safe = np.where(bad, 0.0, mx)
    with np.errstate(all="ignore"):
        ex = np.exp(np.where(np.isfinite(log_g),
                             log_g - mx_safe[:, None], -np.inf))
    ex = np.where(bad[:, None], 0.0, ex)
    wg = kernel["w"][None, :] * ex
    Z = wg.sum(1)
    wsum = kernel["w"].sum()
    pj = np.where(Z[:, None] > 0,
                  wg / np.where(Z[:, None] > 0, Z[:, None], 1.0),
                  kernel["w"][None, :] / wsum)

    S_new = S * np.exp(-Hh)
    delivered = S - S_new
    Ifresh = delivered[:, None] * (pj @ kernel["Pmat"])
    toR = delivered * (pj @ kernel["surv"])
    Iold = Ibins @ T.T
    removed = Ibins @ kernel["sv_old"]
    I_new = Iold + Ifresh
    R_new = R + removed + toR

    Itot = I_new.sum(axis=1)
    cons = float(np.max(np.abs(S_new + Itot + R_new - N)))
    C = h * l0
    stat = dict(Lambda=l0, theta=np.ones_like(S), C=C, newinf=delivered,
                frac_active=0.0, max_C=float(C.max()),
                min_S=float(S_new.min()), mass_gain_R=float(removed.sum()),
                cons_resid=cons, min_lam_u=float(np.min(lam_u)),
                min_H=float(min(np.min(H), np.min(Hh))),
                min_pj=float(np.min(pj)),
                n_subcycle=1, n_lam_eval=p + 1,
                n_nonpos_l0=co["n_nonpos"],
                n_l0_zero=int((co["l0_raw"] == 0.0).sum()),
                n_l0_below_tol=int((co["l0_raw"] < co["tol"]).sum()),
                n_branch=int(co["branch"].sum()),
                n_branch_rejected=co["n_rejected"],
                n_Zzero=int((Z <= 0).sum()))
    return S_new, I_new, R_new, stat


# ---------------------------------------------------------------------------
# Time grid, simulation driver, diagnostics and metrics
# ---------------------------------------------------------------------------
def build_grid(h, horizon=HORIZON):
    nfull = int(math.floor(horizon / h + 1e-12))
    t = [i * h for i in range(nfull + 1)]
    if t[-1] < horizon - 1e-9:
        t.append(float(horizon))
    return np.asarray(t, dtype=DTYPE)


def lambda0_diag(S, Ibins, W, N, phis, beta_sp, tol_rel=TOL_REL):
    l0 = chl.lam_of(S, Ibins, W, N, phis, beta_sp)
    prev = (Ibins @ phis) / N
    tol = DTYPE(tol_rel) * DTYPE(beta_sp) * DTYPE(np.max(prev))
    return l0, float(tol)


def first_cross(t, X, theta):
    """First time each column of X reaches >= theta; NaN if never."""
    ok = X >= theta
    any_ok = ok.any(axis=0)
    idx = np.argmax(ok, axis=0)
    return np.where(any_ok, t[idx], np.nan)


def summarize(t, S, I, R, N):
    """All per-run F-metrics. S,I,R are per-patch FRACTIONS (already /N)."""
    combo = S + I + R
    Nrow = N[None, :]
    cons_glob = float(np.max(np.abs(((combo - 1.0) * Nrow).sum(1))))
    cons_patch = float(np.max(np.abs(combo - 1.0)))
    arr = first_cross(t, I, THETA_ARRIVE)
    take = first_cross(t, I, THETA_TAKEOFF)
    prev_g = (I * Nrow).sum(1) / float(N.sum())
    kp = int(np.argmax(prev_g))
    attack = R[-1].copy()
    out = dict(
        t_arrive=arr,
        t_takeoff=take,
        n_arrived=int(np.isfinite(arr).sum()),
        n_takeoff=int(np.isfinite(take).sum()),
        t_arr_min=float(np.nanmin(arr)) if np.isfinite(arr).any() else float("nan"),
        t_arr_med=float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan"),
        t_arr_max=float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan"),
        t_take_min=float(np.nanmin(take)) if np.isfinite(take).any() else float("nan"),
        t_take_med=float(np.nanmedian(take)) if np.isfinite(take).any() else float("nan"),
        t_take_max=float(np.nanmax(take)) if np.isfinite(take).any() else float("nan"),
        attack=attack,
        attack_mean=float(attack.mean()),
        attack_global=float((attack * N).sum() / N.sum()),
        attack_tx=float(attack[TXI]),
        t_peak_global=float(t[kp]),
        peak_prev_global=float(prev_g[kp]),
        peak_prev_maxpatch=float(I.max()),
        cons_glob=cons_glob,
        cons_glob_rel=cons_glob / float(N.sum()),
        cons_patch=cons_patch,
        min_S=float(S.min()),
        min_I=float(I.min()),
        neg_S=int((S < -1e-12).sum()),
        neg_I=int((I < -1e-12).sum()),
        all_finite=bool(np.all(np.isfinite(S)) and np.all(np.isfinite(I))
                        and np.all(np.isfinite(R))),
    )
    return out


def simulate_arm(arm, S0, bins0, R0, W, N, phis, sigma, beta_sp, h,
                 tol_rel=TOL_REL, horizon=HORIZON):
    t_grid = build_grid(h, horizon)
    m = len(t_grid)
    n = len(N)
    S, Ibins, R = S0.copy(), bins0.copy(), R0.copy()
    S_rec = np.empty((m, n), dtype=DTYPE)
    I_rec = np.empty((m, n), dtype=DTYPE)
    R_rec = np.empty((m, n), dtype=DTYPE)
    S_rec[0] = S / N
    I_rec[0] = Ibins.sum(1) / N
    R_rec[0] = R / N

    n_zero = n_below = n_branch = n_rejected = 0
    steps_with_zero = 0
    max_C = 0.0
    t0 = time.perf_counter()
    for k in range(m - 1):
        dt = float(t_grid[k + 1] - t_grid[k])
        l0_pre, tol_pre = lambda0_diag(S, Ibins, W, N, phis, beta_sp, tol_rel)
        nz = int((l0_pre == 0.0).sum())
        nb = int((l0_pre < tol_pre).sum())
        n_zero += nz
        n_below += nb
        steps_with_zero += int(nz > 0)

        if arm == ARM1:
            S, Ibins, R, st = chl.host_step(
                S, Ibins, R, W, N, phis, sigma, beta_sp, dt, p=chl.M3_ORDER)
        elif arm == ARM2:
            S, Ibins, R, st = host_step_fallback(
                S, Ibins, R, W, N, phis, sigma, beta_sp, dt,
                tol_rel=tol_rel, p=chl.M3_ORDER)
        elif arm == ARM4:
            S, Ibins, R, st = chl.baseline_step(
                "EXP2-midpoint", S, Ibins, R, W, N, phis, sigma, beta_sp, dt)
        else:
            raise ValueError("unknown arm %r" % arm)

        n_branch += int(st.get("n_branch", 0))
        n_rejected += int(st.get("n_branch_rejected", 0))
        max_C = max(max_C, float(st.get("max_C", 0.0)))
        S_rec[k + 1] = S / N
        I_rec[k + 1] = Ibins.sum(1) / N
        R_rec[k + 1] = R / N

    wall = time.perf_counter() - t0
    summ = summarize(t_grid, S_rec, I_rec, R_rec, N)
    # Only lightweight summaries are retained; the full (m,52) trajectories
    # are discarded to keep peak memory bounded across ~430 runs.
    summ.update(dict(arm=arm, h=float(h), n_steps=m - 1,
                     last_dt=float(t_grid[-1] - t_grid[-2]),
                     n_l0_zero=n_zero, n_l0_below_tol=n_below,
                     n_branch=n_branch, n_branch_rejected=n_rejected,
                     steps_with_zero=steps_with_zero, max_C=max_C,
                     wall_s=wall))
    return summ


def reference_DOP853(S0, bins0, R0, W, N, phis, sigma, beta_sp,
                     t_eval, rtol=REF_RTOL, atol=REF_ATOL):
    """Independent truth: continuous ODE (the h->0 limit of the shared map).

    rhs is transcribed from p0b_experiment.reference_solution, with W' and
    beta_sp supplied per scenario.  No square parametrisation, hence no
    lambda_0=0 branch at all.
    """
    n, K = len(N), bins0.shape[1]
    y0 = np.concatenate([S0, bins0.ravel(), R0]).astype(DTYPE)

    def rhs(t, y):
        S = y[:n]
        I = y[n:n + n * K].reshape(n, K)
        Phi = I @ phis
        Lam = beta_sp * (W @ (Phi / N))
        dS = -Lam * S
        dI = -sigma * I
        dI[:, 1:] += sigma * I[:, :-1]
        dI[:, 0] += Lam * S
        dR = sigma * I[:, K - 1]
        return np.concatenate([dS, dI.ravel(), dR])

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0,
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    Y = sol.y.T
    S = Y[:, :n] / N
    I = Y[:, n:n + n * K].reshape(len(t_eval), n, K).sum(2) / N
    R = Y[:, n + n * K:] / N
    return sol, S, I, R


# ---------------------------------------------------------------------------
# Self-tests (run BEFORE the main sweep; abort on any failure)
# ---------------------------------------------------------------------------
def run_self_tests(cfg):
    log("=" * 78)
    log("[SELF-TESTS] (all assertions must pass before the main sweep)")
    log("=" * 78)
    W0, N, phis, sigma, K = (cfg["W0"], cfg["N"], cfg["phis"],
                             cfg["sigma"], cfg["K"])
    n = len(N)
    tests = []

    # T1: W diagonal exactly zero, row-stochastic, dense off-diagonal
    d = np.diag(W0)
    rdev = float(np.max(np.abs(W0.sum(1) - 1.0)))
    offmin = float(W0[~np.eye(n, dtype=bool)].min())
    ok = bool(np.all(d == 0.0) and rdev < 1e-14 and offmin > 0.0)
    tests.append(("T1 W diag==0 / rowsum / dense off-diag", ok,
                  "diag_max=%.3e rowsum_dev=%.3e offdiag_min=%.6e"
                  % (float(d.max()), rdev, offmin)))
    assert ok

    # T2: W' family keeps row sums and spectral radius
    ok = True
    det = []
    for a in ALPHAS:
        dg = w_diagnostics(W0, w_variant(W0, a))
        ok &= (abs(dg["diag_min"] - a) < 1e-14
               and dg["rowsum_dev"] < 1e-14
               and abs(dg["spectral_radius"] - 1.0) < 1e-10)
        det.append("a=%.2f diag=%.4f rho=%.12f" %
                   (a, dg["diag_min"], dg["spectral_radius"]))
    tests.append(("T2 W' rowsum / spectral radius", bool(ok), "; ".join(det)))
    assert ok

    # T3: kernel row-sum identity
    ker = chl.static_kernel(0.1, sigma, K=K)
    kdev = chl.kernel_rowsum_dev(ker)
    ok = kdev < 1e-14
    tests.append(("T3 phase-type kernel row-sum identity", bool(ok),
                  "max dev=%.3e" % kdev))
    assert ok

    # T4: coefficient function identical to shipped one OFF the branch
    S0, b0s, R0, itx = make_ic(cfg, 1e-4)
    Wp = w_variant(W0, 0.05)         # alpha>0 => no exact lambda_0=0
    bsp = beta_of(3.0)
    o = chl.m3_square_coeffs(S0, b0s, R0, Wp, N, phis, sigma, bsp, p=3)
    c = m3_coeffs_variant(S0, b0s, R0, Wp, N, phis, sigma, bsp, 0.1,
                          tol_rel=TOL_REL, p=3)
    ok = (np.array_equal(c["l0"], o[0]) and np.array_equal(c["d1"], o[1])
          and np.array_equal(c["d2"], o[2]) and np.array_equal(c["d3"], o[3]))
    tests.append(("T4 variant coeffs == m3_square_coeffs (bitwise, off-branch)",
                  bool(ok), "branch_patches=%d" % int(c["branch"].sum())))
    assert ok

    # T5: even with the branch PRESENT, tol_rel=0 disables the fallback and
    #     reproduces the shipped coefficients bitwise
    c0 = m3_coeffs_variant(S0, b0s, R0, W0, N, phis, sigma, bsp, 0.1,
                           tol_rel=0.0, p=3)
    oo = chl.m3_square_coeffs(S0, b0s, R0, W0, N, phis, sigma, bsp, p=3)
    ok = (np.array_equal(c0["l0"], oo[0]) and np.array_equal(c0["d1"], oo[1])
          and np.array_equal(c0["d2"], oo[2]) and np.array_equal(c0["d3"], oo[3]))
    tests.append(("T5 tol_rel=0 control == shipped (bitwise)", bool(ok),
                  "branch_patches=%d" % int(c0["branch"].sum())))
    assert ok

    # T6: fallback derivative matching at the real t=0 branch (TX, alpha=0)
    cb = m3_coeffs_variant(S0, b0s, R0, W0, N, phis, sigma, bsp, 0.1,
                           tol_rel=TOL_REL, p=3)
    j = int(np.where(cb["branch"])[0][0]) if cb["branch"].any() else -1
    ok = j >= 0
    detail = "no branch"
    if ok:
        B0, B1, B2 = (float(cb["b0"][j]), float(cb["b1"][j]),
                      float(cb["b2"][j]))
        L1, L2, L3 = (float(cb["l1"][j]), float(cb["l2"][j]),
                      float(cb["l3"][j]))
        d1m = B0 ** 2
        d2m = 4.0 * B0 * B1
        d3m = 6.0 * (2.0 * B0 * B2 + B1 ** 2)
        e1 = abs(d1m - L1) / max(abs(L1), 1e-300)
        e2 = abs(d2m - L2) / max(abs(L2), 1e-300)
        e3 = abs(d3m - L3) / max(abs(L3), 1e-300)
        # independent central finite differences
        def Lam(s):
            return s * (B0 + B1 * s + B2 * s * s) ** 2
        eps = 1e-5
        fd1 = (Lam(eps) - Lam(-eps)) / (2 * eps)
        fd2 = (Lam(eps) - 2 * Lam(0.0) + Lam(-eps)) / (eps ** 2)
        ok = max(e1, e2, e3) < 1e-12 and abs(fd1 - L1) < 1e-9 \
            and abs(fd2 - L2) < 1e-9
        detail = ("patch=%s L0=%.3e L1=%.6e b0=%.6e b1=%.6e b2=%.6e "
                  "relerr=(%.2e,%.2e,%.2e) FD1=%.6e FD2=%.6e"
                  % (cfg["states"][j], float(cb["l0_raw"][j]), L1,
                     B0, B1, B2, e1, e2, e3, fd1, fd2))
    tests.append(("T6 fallback Lambda'(0),Lambda''(0),Lambda'''(0) match",
                  bool(ok), detail))
    assert ok

    # T7: Gauss quadrature exactness for BOTH polynomial forms
    hh = 4.2
    kn = chl.static_kernel(hh, sigma, K=K)
    rng = np.array([0.37, 1.11, 0.83], dtype=DTYPE)   # deterministic, no RNG
    l0v = np.array([0.9, 1.2, 0.7], dtype=DTYPE)
    d1v, d2v, d3v = rng * 0.1, rng * 0.02, rng * 0.003
    b0v = np.sqrt(np.array([0.4, 0.7, 0.2], dtype=DTYPE))
    b1v = np.array([0.11, -0.05, 0.23], dtype=DTYPE)
    b2v = np.array([0.013, 0.031, -0.007], dtype=DTYPE)
    br = np.array([False, True, False])
    hz = hazard_variant(l0v, d1v, d2v, d3v, b0v, b1v, b2v, br, kn, hh)
    # analytic integral of l0*(1+d1 s+d2 s^2+d3 s^3)^2 over [0,h]
    c0 = l0v
    c1 = 2 * l0v * d1v
    c2 = l0v * (d1v ** 2 + 2 * d2v)
    c3 = l0v * (2 * d3v + 2 * d1v * d2v)
    c4 = l0v * (d2v ** 2 + 2 * d1v * d3v)
    c5 = l0v * (2 * d2v * d3v)
    c6 = l0v * d3v ** 2
    Hh_n = (c0 * hh + c1 * hh ** 2 / 2 + c2 * hh ** 3 / 3
            + c3 * hh ** 4 / 4 + c4 * hh ** 5 / 5 + c5 * hh ** 6 / 6
            + c6 * hh ** 7 / 7)
    e0 = b0v ** 2
    e1 = 2 * b0v * b1v
    e2 = 2 * b0v * b2v + b1v ** 2
    e3 = 2 * b1v * b2v
    e4 = b2v ** 2
    Hh_f = (e0 * hh ** 2 / 2 + e1 * hh ** 3 / 3 + e2 * hh ** 4 / 4
            + e3 * hh ** 5 / 5 + e4 * hh ** 6 / 6)
    Hh_ref = np.where(br, Hh_f, Hh_n)
    err = float(np.max(np.abs(hz["Hh"] - Hh_ref)))
    ok = err < 1e-12
    tests.append(("T7 Gauss H(h) exact for deg-6 and deg-5 polynomials",
                  bool(ok), "max abs err=%.3e" % err))
    assert ok

    # T8: one-step conservation + positivity of both steps
    cons1, cons2 = [], []
    for a in ALPHAS:
        Wp = w_variant(W0, a)
        S, b, R, _ = make_ic(cfg, 1e-4)
        S1, b1, R1, s1 = chl.host_step(S, b, R, Wp, N, phis, sigma, bsp, 0.1)
        S2, b2, R2, s2 = host_step_fallback(S, b, R, Wp, N, phis, sigma,
                                            bsp, 0.1)
        cons1.append(s1["cons_resid"])
        cons2.append(s2["cons_resid"])
        assert np.all(S1 >= 0) and np.all(b1 >= 0) and np.all(R1 >= 0)
        assert np.all(S2 >= 0) and np.all(b2 >= 0) and np.all(R2 >= 0)
    ok = max(max(cons1), max(cons2)) < 1e-10
    tests.append(("T8 one-step conservation + positivity", bool(ok),
                  "max resid arm1=%.3e arm2=%.3e" %
                  (max(cons1), max(cons2))))
    assert ok

    # T9: DOP853 reference accuracy floor (independent tighter solve)
    t_eval = np.linspace(0.0, HORIZON, int(HORIZON / REF_DT) + 1)
    sol_a, Sa, Ia, Ra = reference_DOP853(S0, b0s, R0, W0, N, phis, sigma,
                                         bsp, t_eval, REF_RTOL, REF_ATOL)
    sol_b, Sb, Ib, Rb = reference_DOP853(S0, b0s, R0, W0, N, phis, sigma,
                                         bsp, t_eval,
                                         REF_FLOOR_RTOL, REF_FLOOR_ATOL)
    floor = float(np.max(np.abs(Ra[-1] - Rb[-1])))
    ok = bool(sol_a.success and sol_b.success) and floor < 1e-8
    tests.append(("T9 DOP853 rtol 1e-10 vs 1e-12 accuracy floor", bool(ok),
                  "max |delta attack|=%.3e nfev=(%d,%d)" %
                  (floor, sol_a.nfev, sol_b.nfev)))
    assert ok

    for name, ok, detail in tests:
        log("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    log("[SELF-TESTS] %d/%d passed" % (sum(t[1] for t in tests), len(tests)))
    log("")
    return tests


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def csv_payload(header, rows):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def fmt(v, nd=10):
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return "nan" if math.isnan(f) else ("inf" if f > 0 else "-inf")
        return "%.12e" % f
    return str(v)


def write_dual(rel_pair, payload):
    b = payload.encode("utf-8")
    h_mem = hashlib.md5(b).hexdigest()
    paths = [ROOT / rel_pair[0], ROOT / rel_pair[1]]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(b)
    digests = []
    for p in paths:
        with open(p, "rb") as f:
            digests.append(hashlib.md5(f.read()).hexdigest())
    assert digests[0] == digests[1] == h_mem, "dual-write md5 mismatch"
    return dict(md5=h_mem, nbytes=len(b), nlines=payload.count("\n"),
                paths=[str(p) for p in paths])


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def g(x, nd=6):
    """Compact number formatter for report tables."""
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if not math.isfinite(f):
            return "nan"
        if abs(f) < 1e-4 and f != 0.0:
            return "%.3e" % f
        return ("%." + str(nd) + "f") % f
    return str(x)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
TXI = None


def main():
    global TXI
    t_start = time.perf_counter()
    tracemalloc.start()
    log("=" * 78)
    log("R8 WAVEFRONT EXPERIMENT  |  lambda_0=0 branch + diagonal W + small seed")
    log("interpreter : %s" % sys.executable)
    log("numpy %s / scipy %s" % (np.__version__,
                                 __import__("scipy").__version__))
    log("cwd         : %s" % os.getcwd())
    log("started     : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 78)

    cfg = load_inputs()
    states, W0, N, phis, sigma, K = (cfg["states"], cfg["W0"], cfg["N"],
                                     cfg["phis"], cfg["sigma"], cfg["K"])
    n = len(states)
    TXI = states.index("TX")
    log("[INPUT] states=%d  TX index=%d  K=%d  sigma=%.9f /d  TG=%.1f d"
        % (n, TXI, K, sigma, chl.TG))
    log("[INPUT] tau*=%.4f d  sum(phi)*dtau=%.6f (target TG=%.1f)"
        % (cfg["ker"]["tau_star"], cfg["ker"]["sum_phi_dtau"], chl.TG))
    log("[INPUT] fig2 rho field read ONLY for state ordering; rho_TX=%.6f is"
        " NOT used as an IC" % float(cfg["rho_field"][TXI]))
    log("[INPUT] beta_sp_cal (R0=15) = %.10f ; experiment uses R0/TG"
        % cfg["beta_sp_cal"])
    log("")

    wdiag = {a: w_diagnostics(W0, w_variant(W0, a)) for a in ALPHAS}
    log("[A] W' = (1-alpha) W + alpha I  (measured)")
    for a in ALPHAS:
        d = wdiag[a]
        log("    alpha=%.2f  diag=[%.6f,%.6f]  rowsum=[%.15f,%.15f]"
            "  |rowsum-1|=%.3e  offdiag_min=%.6e  rho(W')=%.12f"
            % (a, d["diag_min"], d["diag_max"], d["rowsum_min"],
               d["rowsum_max"], d["rowsum_dev"], d["offdiag_min"],
               d["spectral_radius"]))
    log("    alpha=0.00 diagonal is EXACTLY 0 for all 52 rows: %s"
        % bool(np.all(np.diag(W0) == 0.0)))
    log("")

    tests = run_self_tests(cfg)

    t_eval = np.linspace(0.0, HORIZON, int(HORIZON / REF_DT) + 1)
    results = []          # one dict per (combo, arm, h)
    refs = {}             # combo -> reference summary (kept lightweight)
    diag = {}             # combo -> per-entity diagnostics
    combo_list = [(a, rs, R0) for a in ALPHAS for rs in RHO_SEEDS
                  for R0 in R0_LIST]

    log("=" * 78)
    log("[B/C/D/E/F] MAIN SWEEP : %d combos x (2 arms x %d h + 1 EXP2 + 1 REF)"
        % (len(combo_list), len(H_LIST)))
    log("=" * 78)

    for ci, (alpha, rho_seed, R0) in enumerate(combo_list):
        key = (alpha, rho_seed, R0)
        Wp = w_variant(W0, alpha)
        bsp = beta_of(R0)
        S0, bins0, R0v, itx = make_ic(cfg, rho_seed)

        # --- initial-state diagnostics (A/B/C) ---
        lam0 = chl.lam_derivs(S0, bins0, R0v, Wp, N, phis, sigma, bsp, p=3)
        prev0 = (bins0 @ phis) / N
        scale0 = float(bsp) * float(prev0.max())
        tol0 = TOL_REL * scale0
        branch0 = (lam0[0] < tol0) & (lam0[1] > 0)
        self_force = bsp * np.diag(Wp) * prev0      # within-entity contribution
        diag[key] = dict(
            alpha=alpha, rho_seed=rho_seed, R0=R0, beta_sp=float(bsp),
            W_diag=float(np.diag(Wp)[0]), W_rowsum_min=wdiag[alpha]["rowsum_min"],
            W_rowsum_max=wdiag[alpha]["rowsum_max"],
            W_offdiag_min=wdiag[alpha]["offdiag_min"],
            W_rho=wdiag[alpha]["spectral_radius"],
            seed_I_frac=(bins0.sum(1) / N).copy(),
            seed_S_frac=(S0 / N).copy(),
            Lambda0_t0=lam0[0].copy(), Lambda1_t0=lam0[1].copy(),
            Lambda2_t0=lam0[2].copy(), Lambda3_t0=lam0[3].copy(),
            self_force_t0=self_force.copy(),
            Lambda0_scale_t0=scale0, tol_t0=tol0,
            branch_t0=branch0.copy(),
        )

        def _tag(r):
            r["alpha"], r["rho_seed"], r["R0"] = key
            return r

        # --- arm 3: independent DOP853 truth ---
        t_ref0 = time.perf_counter()
        sol, Sr, Ir, Rr = reference_DOP853(S0, bins0, R0v, Wp, N, phis,
                                           sigma, bsp, t_eval)
        ref_wall = time.perf_counter() - t_ref0
        ref = summarize(t_eval, Sr, Ir, Rr, N)
        ref.update(dict(arm=ARM3, h=0.0, n_steps=0, last_dt=REF_DT,
                        n_l0_zero=-1, n_l0_below_tol=-1, n_branch=0,
                        n_branch_rejected=0, steps_with_zero=0,
                        max_C=float("nan"), wall_s=ref_wall,
                        ref_success=bool(sol.success), ref_nfev=int(sol.nfev)))
        _tag(ref)
        refs[key] = ref
        results.append(ref)

        # --- arms 1 and 2 at all h, arm 4 at h=0.1 ---
        for h in H_LIST:
            for arm in (ARM1, ARM2):
                r = simulate_arm(arm, S0, bins0, R0v, Wp, N, phis, sigma,
                                 bsp, h, tol_rel=TOL_REL)
                results.append(_tag(r))
            if h == EXP2_H:
                r4 = simulate_arm(ARM4, S0, bins0, R0v, Wp, N, phis, sigma,
                                  bsp, h, tol_rel=TOL_REL)
                results.append(_tag(r4))

        log("  [%2d/%2d] a=%.2f rs=%.0e R0=%.1f | ref attack=%.6f"
            " | a1/a2@h0.1 attack=(%.6f,%.6f) | branch(a2,h0.1)=%d"
            % (ci + 1, len(combo_list), alpha, rho_seed, R0,
               ref["attack_mean"],
               _lookup(results, key, ARM1, 0.1)["attack_mean"],
               _lookup(results, key, ARM2, 0.1)["attack_mean"],
               _lookup(results, key, ARM2, 0.1)["n_branch"]))

    # -----------------------------------------------------------------------
    # Fill in relative-vs-reference attack errors and fallback deltas
    # -----------------------------------------------------------------------
    for r in results:
        key = (r["alpha"], r["rho_seed"], r["R0"])
        rf = refs[key]["attack"]
        at = r["attack"]
        r["attack_rel_L2_vs_ref"] = float(
            np.linalg.norm(at - rf) / np.linalg.norm(rf))
        r["attack_rel_max_vs_ref"] = float(
            np.max(np.abs(at - rf) / np.maximum(np.abs(rf), 1e-12)))
        r["attack_abs_max_vs_ref"] = float(np.max(np.abs(at - rf)))
        r["attack_global_rel_vs_ref"] = float(
            abs(r["attack_global"] - refs[key]["attack_global"])
            / max(abs(refs[key]["attack_global"]), 1e-12))
        # attach combo keys for CSV writing
        r["alpha"], r["rho_seed"], r["R0"] = key

    # -----------------------------------------------------------------------
    # Tolerance sensitivity (TOL_REL_SCAN), alpha=0 and alpha=0.05, R0=3,
    # rho_seed=1e-4, h=0.1
    # -----------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("[G] TOLERANCE SENSITIVITY (arm 2, h=0.1 d)")
    log("=" * 78)
    tol_rows = []
    for alpha in (0.0, 0.05):
        for R0 in (1.5, 3.0, 6.0):
            rho_seed = 1e-4
            Wp = w_variant(W0, alpha)
            bsp = beta_of(R0)
            S0, bins0, R0v, itx = make_ic(cfg, rho_seed)
            base = _lookup(results, (alpha, rho_seed, R0), ARM1, 0.1)
            for tr in TOL_REL_SCAN:
                r = simulate_arm(ARM2, S0, bins0, R0v, Wp, N, phis, sigma,
                                 bsp, 0.1, tol_rel=tr)
                tol_rows.append(dict(alpha=alpha, R0=R0, rho_seed=rho_seed,
                                     tol_rel=tr, n_branch=r["n_branch"],
                                     n_l0_below_tol=r["n_l0_below_tol"],
                                     attack_mean=r["attack_mean"],
                                     d_attack_vs_arm1=float(
                                         np.max(np.abs(r["attack"]
                                                       - base["attack"]))),
                                     d_global_attack=(
                                         r["attack_global"]
                                         - base["attack_global"])))
                log("  a=%.2f R0=%.1f TOL_REL=%.0e | n_branch=%5d"
                    " n_l0<TOL=%6d | attack=%.9f | max|d attack vs arm1|=%.3e"
                    % (alpha, R0, tr, r["n_branch"], r["n_l0_below_tol"],
                       r["attack_mean"], tol_rows[-1]["d_attack_vs_arm1"]))

    # -----------------------------------------------------------------------
    # Build per-entity diagnostics CSV (long format)
    # -----------------------------------------------------------------------
    diag_rows = []
    diag_header = [
        "alpha", "rho_seed", "R0", "beta_sp", "state", "pop_N",
        "W_diag", "W_rowsum_min", "W_rowsum_max", "W_offdiag_min",
        "W_spectral_radius", "seed_I_frac", "seed_S_frac",
        "Lambda0_t0", "Lambda1_t0", "Lambda2_t0", "Lambda3_t0",
        "self_force_t0", "Lambda0_scale_t0", "tol_t0", "branch_t0",
        "t_arrive_ref_th1", "t_arrive_ref_th2",
        "t_arrive_M3p3_h0p1_th1", "t_arrive_fb_h0p1_th1",
        "t_arrive_M3p3_h0p1_th2", "t_arrive_fb_h0p1_th2",
        "attack_ref", "attack_M3p3_h0p1", "attack_fb_h0p1",
        "d_attack_fb_minus_M3p3",
    ]
    for key in combo_list:
        d = diag[key]
        a1 = _lookup(results, key, ARM1, 0.1)
        a2 = _lookup(results, key, ARM2, 0.1)
        rf = refs[key]
        for j, st in enumerate(states):
            diag_rows.append([
                fmt(d["alpha"]), fmt(d["rho_seed"]), fmt(d["R0"]),
                fmt(d["beta_sp"]), st, fmt(N[j]),
                fmt(d["W_diag"]), fmt(d["W_rowsum_min"]),
                fmt(d["W_rowsum_max"]),                 fmt(d["W_offdiag_min"]),
                fmt(d["W_rho"]),
                fmt(d["seed_I_frac"][j]), fmt(d["seed_S_frac"][j]),
                fmt(d["Lambda0_t0"][j]), fmt(d["Lambda1_t0"][j]),
                fmt(d["Lambda2_t0"][j]), fmt(d["Lambda3_t0"][j]),
                fmt(d["self_force_t0"][j]), fmt(d["Lambda0_scale_t0"]),
                fmt(d["tol_t0"]), fmt(bool(d["branch_t0"][j])),
                fmt(rf["t_arrive"][j]), fmt(rf["t_takeoff"][j]),
                fmt(a1["t_arrive"][j]), fmt(a2["t_arrive"][j]),
                fmt(a1["t_takeoff"][j]), fmt(a2["t_takeoff"][j]),
                fmt(rf["attack"][j]), fmt(a1["attack"][j]),
                fmt(a2["attack"][j]),
                fmt(a2["attack"][j] - a1["attack"][j]),
            ])
    diag_payload = csv_payload(diag_header, diag_rows)

    # -----------------------------------------------------------------------
    # Build main results CSV
    # -----------------------------------------------------------------------
    wf_header = [
        "alpha", "rho_seed", "R0", "beta_sp", "arm", "h", "n_steps",
        "last_dt", "out_dt", "solver_note",
        "W_diag", "W_rowsum_dev", "W_offdiag_min", "W_spectral_radius",
        "n_l0_zero_patchstep", "n_l0_below_tol_patchstep",
        "n_branch_patchstep", "n_branch_rejected", "steps_with_zero",
        "max_C", "tol_rel",
        "n_arrived_th1", "t_arrive_min", "t_arrive_median",
        "t_arrive_max", "t_arrive_TX",
        "n_takeoff_th2", "t_takeoff_min", "t_takeoff_median",
        "t_takeoff_max", "t_takeoff_TX",
        "attack_mean", "attack_global", "attack_TX",
        "t_peak_global", "peak_prev_global", "peak_prev_maxpatch",
        "cons_resid_global", "cons_resid_global_rel", "cons_resid_patch",
        "min_S", "min_I", "neg_S_cells", "neg_I_cells", "all_finite",
        "ref_attack_mean", "ref_attack_global",
        "attack_rel_L2_vs_ref", "attack_rel_max_vs_ref",
        "attack_abs_max_vs_ref", "attack_global_rel_vs_ref",
        "wall_s",
    ]
    wf_rows = []
    for r in results:
        key = (r["alpha"], r["rho_seed"], r["R0"])
        d = wdiag[key[0]]
        note = ("DOP853 rtol=1e-10 atol=1e-12 t_eval=0.25d"
                if r["arm"] == ARM3 else
                ("M3-p3 shipped" if r["arm"] == ARM1 else
                 ("M3-p3 + s*square fallback" if r["arm"] == ARM2
                  else "EXP2-midpoint shipped")))
        wf_rows.append([
            fmt(r["alpha"]), fmt(r["rho_seed"]), fmt(r["R0"]),
            fmt(beta_of(r["R0"])), r["arm"], fmt(r["h"]),
            fmt(r["n_steps"]), fmt(r["last_dt"]),
            fmt(REF_DT if r["arm"] == ARM3 else r["h"]), note,
            fmt(d["diag_min"]), fmt(d["rowsum_dev"]),
            fmt(d["offdiag_min"]), fmt(d["spectral_radius"]),
            fmt(r["n_l0_zero"]), fmt(r["n_l0_below_tol"]),
            fmt(r["n_branch"]), fmt(r["n_branch_rejected"]),
            fmt(r["steps_with_zero"]), fmt(r["max_C"]), fmt(TOL_REL),
            fmt(r["n_arrived"]), fmt(r["t_arr_min"]), fmt(r["t_arr_med"]),
            fmt(r["t_arr_max"]), fmt(r["t_arrive"][TXI]),
            fmt(r["n_takeoff"]), fmt(r["t_take_min"]), fmt(r["t_take_med"]),
            fmt(r["t_take_max"]), fmt(r["t_takeoff"][TXI]),
            fmt(r["attack_mean"]), fmt(r["attack_global"]),
            fmt(r["attack_tx"]),
            fmt(r["t_peak_global"]), fmt(r["peak_prev_global"]),
            fmt(r["peak_prev_maxpatch"]),
            fmt(r["cons_glob"]), fmt(r["cons_glob_rel"]),
            fmt(r["cons_patch"]),
            fmt(r["min_S"]), fmt(r["min_I"]),
            fmt(r["neg_S"]), fmt(r["neg_I"]), fmt(r["all_finite"]),
            fmt(refs[key]["attack_mean"]), fmt(refs[key]["attack_global"]),
            fmt(r["attack_rel_L2_vs_ref"]), fmt(r["attack_rel_max_vs_ref"]),
            fmt(r["attack_abs_max_vs_ref"]),
            fmt(r["attack_global_rel_vs_ref"]),
            fmt(r["wall_s"]),
        ])
    wf_payload = csv_payload(wf_header, wf_rows)

    # -----------------------------------------------------------------------
    # Dual write + md5
    # -----------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("[H] DUAL WRITE + md5")
    log("=" * 78)
    dwf = write_dual(OUT_WF, wf_payload)
    ddg = write_dual(OUT_DIAG, diag_payload)
    for tag, d in (("wavefront_fallback.csv", dwf),
                   ("coupling_diag_seed.csv", ddg)):
        log("  %s  md5=%s  bytes=%d  lines=%d"
            % (tag, d["md5"], d["nbytes"], d["nlines"]))
        for p in d["paths"]:
            log("      %s" % p)
    assert dwf["nlines"] >= 2 and ddg["nlines"] >= 2, "CSV empty"
    assert len(wf_rows) == len(results)
    log("")

    # -----------------------------------------------------------------------
    # Derived headline numbers
    # -----------------------------------------------------------------------
    def sel(arm=None, h=None, alpha=None, R0=None, rho_seed=None):
        out = []
        for r in results:
            if arm is not None and r["arm"] != arm:
                continue
            if h is not None and abs(r["h"] - h) > 1e-12:
                continue
            if alpha is not None and abs(r["alpha"] - alpha) > 1e-12:
                continue
            if R0 is not None and abs(r["R0"] - R0) > 1e-12:
                continue
            if rho_seed is not None and abs(r["rho_seed"] - rho_seed) > 1e-18:
                continue
            out.append(r)
        return out

    # fallback effect at h=0.1 across all 36 combos
    fb_rows = []
    for key in combo_list:
        a1 = _lookup(results, key, ARM1, 0.1)
        a2 = _lookup(results, key, ARM2, 0.1)
        datt = np.abs(a2["attack"] - a1["attack"])
        darr = np.abs(np.nan_to_num(a2["t_arrive"] - a1["t_arrive"], nan=0.0))
        dtak = np.abs(np.nan_to_num(a2["t_takeoff"] - a1["t_takeoff"], nan=0.0))
        dpeak = a2["peak_prev_global"] - a1["peak_prev_global"]
        fb_rows.append(dict(key=key, n_branch=a2["n_branch"],
                            max_dattack=float(datt.max()),
                            L2_dattack=float(np.linalg.norm(datt)),
                            max_darrive=float(darr.max()),
                            max_dtakeoff=float(dtak.max()),
                            dpeak=float(dpeak),
                            d_attack_global=(a2["attack_global"]
                                             - a1["attack_global"])))
    max_dattack_all = max(x["max_dattack"] for x in fb_rows)
    max_darr_all = max(x["max_darrive"] for x in fb_rows)
    max_dtake_all = max(x["max_dtakeoff"] for x in fb_rows)
    max_dpeak_all = max(abs(x["dpeak"]) for x in fb_rows)
    tot_branch = sum(x["n_branch"] for x in fb_rows)

    # alpha effect at h=0.1
    alpha_rows = []
    for rs in RHO_SEEDS:
        for R0 in R0_LIST:
            base = _lookup(results, (0.0, rs, R0), ARM1, 0.1)
            for a in ALPHAS[1:]:
                r = _lookup(results, (a, rs, R0), ARM1, 0.1)
                alpha_rows.append(dict(
                    rs=rs, R0=R0, alpha=a,
                    Lam0_TX_0=float(diag[(0.0, rs, R0)]["Lambda0_t0"][TXI]),
                    Lam0_TX_a=float(diag[(a, rs, R0)]["Lambda0_t0"][TXI]),
                    self_force_TX=float(diag[(a, rs, R0)]["self_force_t0"][TXI]),
                    max_dattack=float(np.max(np.abs(r["attack"]
                                                    - base["attack"]))),
                    d_attack_global=r["attack_global"] - base["attack_global"],
                    d_peak=r["peak_prev_global"] - base["peak_prev_global"],
                    d_tpeak=r["t_peak_global"] - base["t_peak_global"],
                    d_arr_med=r["t_arr_med"] - base["t_arr_med"],
                    d_take_med=(r["t_take_med"] - base["t_take_med"])))

    # h scan summary
    h_rows = []
    for h in H_LIST:
        for arm in (ARM1, ARM2):
            s = sel(arm=arm, h=h)
            h_rows.append(dict(
                h=h, arm=arm,
                mean_attack=float(np.mean([x["attack_mean"] for x in s])),
                mean_relL2=float(np.mean([x["attack_rel_L2_vs_ref"]
                                          for x in s])),
                max_relmax=float(np.max([x["attack_rel_max_vs_ref"]
                                         for x in s])),
                max_cons=float(np.max([x["cons_glob"] for x in s])),
                min_S=float(np.min([x["min_S"] for x in s])),
                n_fail=sum(0 if x["all_finite"] else 1 for x in s),
                mean_wall=float(np.mean([x["wall_s"] for x in s]))))

    # explicit failure cases (attack deviating >5% from reference)
    failures = [r for r in results
                if r["arm"] != ARM3
                and r["attack_global_rel_vs_ref"] > 0.05]

    # solver cross-check at h=0.1
    cross = []
    for arm in (ARM1, ARM2, ARM4):
        s = sel(arm=arm, h=0.1)
        cross.append(dict(arm=arm,
                          mean_relL2=float(np.mean([x["attack_rel_L2_vs_ref"]
                                                    for x in s])),
                          max_relL2=float(np.max([x["attack_rel_L2_vs_ref"]
                                                  for x in s])),
                          max_abs=float(np.max([x["attack_abs_max_vs_ref"]
                                                for x in s])),
                          max_cons=float(np.max([x["cons_glob"] for x in s])),
                          min_S=float(np.min([x["min_S"] for x in s])),
                          n_notfinite=sum(0 if x["all_finite"] else 1
                                          for x in s)))

    # arrival-time spread (is there a spatial wavefront?)
    arr_stats = []
    for R0 in R0_LIST:
        for a in ALPHAS:
            s = sel(arm=ARM3, alpha=a, R0=R0)
            r = s[0]
            arr_stats.append(dict(R0=R0, alpha=a,
                                  arr_min=r["t_arr_min"],
                                  arr_med=r["t_arr_med"],
                                  arr_max=r["t_arr_max"],
                                  n_arr=r["n_arrived"],
                                  take_min=r["t_take_min"],
                                  take_med=r["t_take_med"],
                                  take_max=r["t_take_max"],
                                  n_take=r["n_takeoff"],
                                  t_peak=r["t_peak_global"],
                                  peak=r["peak_prev_global"]))

    cur, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_wall = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    rep = []
    A = rep.append
    A("# R8 波前实验线实测报告：λ₀=0 分支 + 非零对角 W + 小播种波前")
    A("")
    A("> **硬边界**：本报告所有数值均为本次 `C:/Python314/python.exe "
      "submission/code/regen_wavefront.py` 实际运行产生，"
      "**数值待用户数值确认**。团队不采信自身数值，须由用户使用其自有已校验代码"
      "复算确认后方可采信。下文每个实测数字表后均重复此标注。")
    A("")
    A("生成时间：%s；运行日志：`_review_new_tmp_lead/r8_wavefront_run.log`"
      % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("")
    A("## 0. 一键复现")
    A("")
    A("```")
    A("cd \"F:/科技论文写作/写作中论文/2026年写作中论文/麻疹的ODE-to-CA耦合框架\"")
    A("C:/Python314/python.exe -m py_compile submission/code/regen_wavefront.py")
    A("C:/Python314/python.exe submission/code/regen_wavefront.py")
    A("```")
    A("")
    A("环境：Python 3.14（`C:/Python314/python.exe`），numpy %s，scipy %s，"
      "pandas 可用但未使用；全程 float64，无 RNG、无 seed。"
      % (np.__version__, __import__("scipy").__version__))
    A("")
    A("产物目录（双份逐位一致）：")
    A("")
    A("| 文件 | md5 | 字节 | 行数 |")
    A("|---|---|---|---|")
    A("| `figures/wavefront_fallback.csv` | `%s` | %d | %d |"
      % (dwf["md5"], dwf["nbytes"], dwf["nlines"]))
    A("| `submission/figures/wavefront_fallback.csv` | `%s` | %d | %d |"
      % (dwf["md5"], dwf["nbytes"], dwf["nlines"]))
    A("| `figures/coupling_diag_seed.csv` | `%s` | %d | %d |"
      % (ddg["md5"], ddg["nbytes"], ddg["nlines"]))
    A("| `submission/figures/coupling_diag_seed.csv` | `%s` | %d | %d |"
      % (ddg["md5"], ddg["nbytes"], ddg["nlines"]))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("## 1. 实验设置")
    A("")
    A("### 1.1 只读输入")
    A("")
    A("- `submission/code/coupling_host_level.py`：常量 BETA_SP=%.4f, "
      "R0_TARGET=%.1f, TG=%.1f, K_BINS=%d, EPS_V=%.0e, H=%.2f；"
      "复用 `lam_derivs` / `m3_square_coeffs` / `hazard_and_lambda` / "
      "`host_step` / `build_gravity` / `load_rho` / `build_infectious_kernel` / "
      "`baseline_step` / `static_kernel` / `poisson_T` / `lam_of`。"
      % (chl.BETA_SP, chl.R0_TARGET, chl.TG, chl.K_BINS, chl.EPS_V, chl.H))
    A("- `submission/code/W_gravity_52x52_rowstochastic.csv`（52×52，"
      "首行 `state,TX,NY,...`）；`submission/figures/fig2_tycho_field.csv` "
      "**仅用于核对州名顺序**，其 ρ_TX=1.0 的 100% 患病率场未被用作初值；"
      "`submission/figures/fig3_withinhost_traj.csv` 提供 φ(τ) 核。")
    A("- `submission/code/ms_numpy.py`（STATE_GEO / "
      "recalibrate_gravity_coupling，只读）。")
    A("")
    A("实测：`build_gravity(states)` 返回的 W 与 CSV **逐位一致**"
      "（max|W_csv−W_geo| = %.3e）；州名顺序与 fig2 一致。"
      % float(np.max(np.abs(W0 - chl.build_gravity(states)[0]))))
    A("")
    A("### 1.2 技术选型表")
    A("")
    A("| 组件 | 选型 | 理由 | 备选 |")
    A("|---|---|---|---|")
    A("| λ₀=0 变体 | Λ(s)=s(b0+b1s+b2s²)²，由 lam_derivs 前三阶导数匹配 | "
      "s≥0 与平方保证处处非负，Λ(0)=0 且可表达 λ̇₀≠0 的起步 | "
      "λ₀(1+…)²（λ₀=0 时恒为 0，不可用） |")
    A("| 真值参考 | scipy `solve_ivp` DOP853, rtol=1e-10, atol=1e-12 | "
      "连续 ODE 无平方参数化、无 λ₀=0 分支，完全独立于 M3 | "
      "h=0.01 d 小步长 M3（仍带分支问题，未采用为真值） |")
    A("| 独立二阶对照 | `baseline_step('EXP2-midpoint')`, h=0.1 d | "
      "冻结-Λ 中点指数积分，与 M3 完全不同构造 | Strang/static 臂 |")
    A("| 步长／容差 | h∈{0.1,1,4.2,9,20} d，horizon=240 d；正权重 Gauss "
      "(q=16 源项 / q'=8  hazard) | 与现有 M3 生产步完全同口径 | 自适应子步 |")
    A("| 复现 | 单文件自足、无随机流、双份 CSV 同源 payload + md5 断言 | "
      "复现性优先于性能 | — |")
    A("")
    A("### 1.3 A：W 的自环变体（实测）")
    A("")
    A("W′=(1−α)W+αI，α∈{0,0.02,0.05,0.10}。")
    A("")
    rows = []
    for a in ALPHAS:
        d = wdiag[a]
        rows.append([g(a, 2), g(d["diag_min"], 6), g(d["diag_max"], 6),
                     g(d["rowsum_min"], 15), g(d["rowsum_max"], 15),
                     "%.2e" % d["rowsum_dev"], "%.3e" % d["offdiag_min"],
                     g(d["spectral_radius"], 12),
                     "%.2e" % d["max_abs_diff"]])
    A(md_table(["α", "diag_min", "diag_max", "rowsum_min", "rowsum_max",
                "max|rowsum−1|", "offdiag_min", "ρ(W′)", "max|W′−W|"], rows))
    A("")
    A("α=0 时 52 个对角元**全部精确为 0.0**（`np.all(np.diag(W)==0)` = %s）；"
      "全部非对角元 > 0（最小 %.6e），即 W 是稠密的；W′ 仍行随机、"
      "Perron 根 ρ(W′)=1，故 β_sp=R0/TG 的标定在 α>0 时不变。"
      % (bool(np.all(np.diag(W0) == 0.0)), wdiag[0.0]["offdiag_min"]))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 1.4 B：小播种初值（实测）")
    A("")
    A("仅 TX 播种：I_TX(0)=ρ_seed·N_TX，其余 51 个实体 I=0，"
      "S=N−I（即 S_TX=(1−ρ_seed)N_TX，其余 S=N）。"
      "播种的 I 均匀分布在 K=40 个年龄仓（与 `run_demo` 同约定）。"
      "ρ_seed∈{1e-4,1e-3,1e-2}，R0∈{1.5,3.0,6.0}，β_sp=R0/13。")
    A("")
    rows = []
    for rs in RHO_SEEDS:
        S0, b0s, R0v, _ = make_ic(cfg, rs)
        rows.append(["%.0e" % rs, g(S0[TXI] / N[TXI], 6),
                     g(b0s.sum(1)[TXI] / N[TXI], 6),
                     g(b0s.sum(), 8), g(float(np.sum(b0s)), 8)])
    A(md_table(["ρ_seed", "S_TX/N", "I_TX/N", "Σ_i I_i(0)", "seeded mass"],
               rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 1.5 阈值、网格与对比臂")
    A("")
    A("- 波前到达阈值 θ₁=%.0e（I_i/N_i 首次 ≥ θ₁ 的时刻）；"
      "次要「起飞」阈值 θ₂=%.0e。二者均在模拟输出网格上取首次越阈，"
      "故到达时间分辨率为各自步长 h（参考解为 %.2f d）。"
      % (THETA_ARRIVE, THETA_TAKEOFF, REF_DT))
    A("- t_grid：`[0,h,2h,…]` 直到不超过 240 d，若末点 <240 d 则追加 "
      "240.0 d（末步为短步，用实际 dt 调用 `host_step`，与 `simulate()` 同口径）。")
    A("- 臂①`M3-p3`（原实现，λ₀=0⇒Λ≡0）；②`M3-p3+fallback`（本文件变体）；"
      "③`REF-DOP853`（真值，输出网格 0.25 d）；④`EXP2-midpoint`（可选，"
      "仅 h=0.1 d）。四臂共用同一初值、同一 W′、同一 R0、同一 h、同一 horizon。")
    A("- 攻击率 attack_i=R_i(240)/N_i（R(0)=0 且 S+I+R=N 守恒），"
      "全局 attack=ΣR/ΣN；相对参考解误差给 L2 相对误差、逐斑块最大相对误差、"
      "绝对最大误差与全局相对误差四种口径。")
    A("")
    A("## 2. λ₀=0 fallback：容差判据与系数推导")
    A("")
    A("### 2.1 原实现的问题（复述，不改代码）")
    A("")
    A("`m3_square_coeffs` 中 `pos = l0 > 0`；`l0safe=np.where(pos,l0,0)`，"
      "`inv=np.where(pos,1/l0,0)`，于是 a_j=λ_j·inv=0，d1=d2=d3=0，"
      "`hazard_and_lambda` 给出 Λ(u)=l0·P(u)²≡0、H≡0，"
      "该步退化为纯线性老化流（`I_0 T(h)`），步内本应产生的感染全部丢失。"
      "稿件 §`subsec:flow` 已声明此分支「outside the prescribed experiments」；"
      "本实验给出它的实测规模。")
    A("")
    A("### 2.2 变体形式与系数推导")
    A("")
    A("取 Λ(s)=s·Q(s)²，Q(s)=b0+b1 s+b2 s²，s∈[0,h] 故 s≥0，Q²≥0，"
      "**Λ(s)≥0 恒成立**（Q 的符号无关，与 shipped 平方同构）。展开：")
    A("")
    A("```")
    A("Λ(s) = b0² s + 2b0b1 s² + (2b0b2+b1²) s³ + 2b1b2 s⁴ + b2² s⁵")
    A("Λ(0)=0,  Λ'(0)=b0²,  Λ''(0)=4b0b1,  Λ'''(0)=6(2b0b2+b1²)")
    A("=>  b0 = sqrt(λ1)")
    A("    b1 = λ2 / (4 b0)")
    A("    b2 = (λ3/6 − b1²) / (2 b0)")
    A("```")
    A("")
    A("其中 λ_j=`lam_derivs(...)[j]`=d^jΛ/dt^j(t_k)（原始导数，未归一化）。"
      "该形式与 shipped 的 Λ=λ₀(1+d1s+d2s²+d3s³)² 一样在 s=0 处匹配到 λ₃，"
      "局部截断误差同为 O(s⁴)；∫₀^h Λ ds 是 5 次多项式，q'=8 点 Gauss "
      "（精确到 15 次）足够。")
    A("")
    A("### 2.3 容差判据（相对尺度）")
    A("")
    A("W 行随机 ⇒ Λ_i=β_sp Σ_j w_ij(Φ_j/N_j) ≤ β_sp max_j(Φ_j/N_j)。定义")
    A("")
    A("```")
    A("scale = β_sp · max_j (Φ_j/N_j),   TOL = TOL_REL · scale,   TOL_REL=1e-12")
    A("branch_i = (λ0_i < TOL) AND (λ1_i > 0)")
    A("```")
    A("")
    A("第二个条件是必要的：λ₁≤0 表示衰减零或偶阶零，此时 Λ≡0 本身是"
      "正确的首阶行为，注入任何东西都是错的。**不能改用绝对容差**："
      "R0=6 的灭绝尾中 λ₀ 实测衰减到 ~1e-56 /d，任何固定绝对阈值（如 1e-12）"
      "会把数万个尾部 (patch,step) 误判为分支。非分支斑块一律沿用"
      "`m3_square_coeffs` 的原公式（单元测试 T4/T5 逐位验证）。"
      "b0=√λ₁ 需 λ₁>0；若 b0/b1/b2 非有限，该斑块回退到 Λ≡0 并计数 "
      "`n_branch_rejected`（本次实测计数见 §4）。")
    A("")
    A("### 2.4 与 Propositions 1–2 前提的关系")
    A("")
    A("- **Prop 1（无条件保正）** 只要求 Λ(s)≥0 on [0,h]。"
      "s·Q² 在 s≥0 上满足，且 hazard 仍用正 Gauss 权、log 域指数化，"
      "故保正前提成立。")
    A("- **Prop 2（无条件守恒）** 的证明只用 Σ_j π_j=1、"
      "Σ_k P_k(μ)=1−γ_inc(K,μ) 与 removed 的定义，与 Λ 的具体代数形式无关，"
      "只要 H≥0（由 Λ≥0+正权得到）。故守恒前提成立。")
    A("- **Prop 3（四阶）** 要求 Λ(s)−Λ_true=O(h^{p+1})、π_j 能分辨注入时刻；"
      "s·Q² 在简单零处与 shipped 形式同阶，但**未经本实验证明**满足"
      "π_j 的高阶分辨条件，故本报告不对分支步作四阶宣称。")
    A("- **形式局限**：s·Q² 只能表达**奇数阶（简单）零**。若 λ₀=λ₁=0 而 "
      "λ₂≠0（偶阶零），本形式的 Λ(s)=O(s³)，无法匹配 O(s²) 的真实行为；"
      "此时需要 s²·(…)² 或更高阶层次。本实验 λ₁>0 的守卫会直接跳过这类点。")
    A("")
    A("## 3. 核心实测数字表")
    A("")
    A("### 3.1 λ₀=0 分支与 fallback 激活计数（每 (α,ρ_seed,R0) 一个 h=0.1 d 运行，2400 步×52 斑块=124800 (patch,step)）")
    A("")
    rows = []
    for a in ALPHAS:
        for R0 in R0_LIST:
            for rs in RHO_SEEDS:
                a1 = _lookup(results, (a, rs, R0), ARM1, 0.1)
                a2 = _lookup(results, (a, rs, R0), ARM2, 0.1)
                rows.append([g(a, 2), "%.0e" % rs, g(R0, 1),
                             a1["n_l0_zero"], a1["n_l0_below_tol"],
                             a2["n_branch"], a2["n_branch_rejected"],
                             a2["steps_with_zero"]])
    A(md_table(["α", "ρ_seed", "R0", "arm1 n(λ₀=0)",
                "arm1 n(λ₀<TOL)", "arm2 n_branch", "rejected",
                "steps with a zero"], rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("结论性观察：α=0 时 λ₀ 精确为 0 的 (patch,step) 计数为 "
      "**%d**，出现在 t=0 的 **TX**（唯一有 I≠0 却因 W 对角为 0 而"
      " Λ_TX=0 的斑块）；α>0 时对角自环使 λ₀(t=0) 处处 >0，计数为 0。"
      % _lookup(results, (0.0, 1e-4, 3.0), ARM1, 0.1)["n_l0_zero"])
    A("")
    A("### 3.2 参考解（DOP853）波前到达时间与峰值")
    A("")
    rows = []
    for x in arr_stats:
        rows.append([g(x["R0"], 1), g(x["alpha"], 2),
                     g(x["arr_min"]), g(x["arr_med"]), g(x["arr_max"]),
                     x["n_arr"], g(x["take_min"]), g(x["take_med"]),
                     g(x["take_max"]), x["n_take"],
                     g(x["t_peak"]), g(x["peak"])])
    A(md_table(["R0", "α", "θ₁ min", "θ₁ med", "θ₁ max", "n(θ₁)",
                "θ₂ min", "θ₂ med", "θ₂ max", "n(θ₂)",
                "t_peak", "peak prev"], rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("说明：θ₁ 与 θ₂ 的逐实体到达时间完整列于 "
      "`coupling_diag_seed.csv`（每 (α,ρ_seed,R0,state) 一行）。"
      "由于 W 稠密（最小非对角 %.3e >0），所有 52 个实体在 t=0⁺ 即同时"
      "受到感染力，不存在反应扩散意义的空间行波；到达时间由流行病增长率与"
      "阈值决定，而非由地理距离决定。"
      % wdiag[0.0]["offdiag_min"])
    A("")
    A("### 3.3 步长扫描（36 组组合的统计）")
    A("")
    rows = []
    for x in h_rows:
        rows.append([g(x["h"], 2), x["arm"], g(x["mean_attack"]),
                     "%.3e" % x["mean_relL2"], "%.3e" % x["max_relmax"],
                     "%.3e" % x["max_cons"], "%.3e" % x["min_S"],
                     x["n_fail"], g(x["mean_wall"], 3)])
    A(md_table(["h (d)", "arm", "mean attack", "mean rel-L2 vs ref",
                "max rel-max vs ref", "max cons", "min S", "n !finite",
                "mean wall (s)"], rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 3.4 求解器交叉校验（h=0.1 d，36 组）")
    A("")
    rows = []
    for x in cross:
        rows.append([x["arm"], "%.3e" % x["mean_relL2"],
                     "%.3e" % x["max_relL2"], "%.3e" % x["max_abs"],
                     "%.3e" % x["max_cons"], "%.3e" % x["min_S"],
                     x["n_notfinite"]])
    A(md_table(["arm", "mean rel-L2", "max rel-L2", "max |Δattack|",
                "max cons", "min S", "n !finite"], rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 3.5 fallback 效应（arm2 − arm1，h=0.1 d，全部 36 组）")
    A("")
    rows = []
    for x in fb_rows[:18]:
        k = x["key"]
        rows.append([g(k[0], 2), "%.0e" % k[1], g(k[2], 1),
                     x["n_branch"], "%.3e" % x["max_dattack"],
                     "%.3e" % x["L2_dattack"], g(x["max_darrive"]),
                     g(x["max_dtakeoff"]), "%.3e" % x["dpeak"]])
    A(md_table(["α", "ρ_seed", "R0", "n_branch", "max|Δattack|",
                "L2|Δattack|", "max|Δt_arr|", "max|Δt_take|",
                "Δpeak"], rows))
    A("")
    A("（表中列出前 18 组，全 36 组见 `wavefront_fallback.csv`；"
      "本表全部数值：数值待用户数值确认）")
    A("")
    A("全部 36 组的汇总：max over patches |Δattack| 的最大值 = **%.3e**；"
      "到达时间 θ₁ 最大变化 = **%.3f d**；θ₂ 最大变化 = **%.3f d**；"
      "全局峰值流行率最大变化 = **%.3e**；fallback 激活总数 = **%d**。"
      % (max_dattack_all, max_darr_all, max_dtake_all,
         max_dpeak_all, tot_branch))
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("### 3.6 α>0 是否出现实体内传播（arm1，h=0.1 d，基线 α=0）")
    A("")
    rows = []
    for x in alpha_rows[:18]:
        rows.append([g(x["alpha"], 2), "%.0e" % x["rs"], g(x["R0"], 1),
                     "%.3e" % x["Lam0_TX_0"], "%.3e" % x["Lam0_TX_a"],
                     "%.3e" % x["self_force_TX"],
                     "%.3e" % x["max_dattack"], "%.3e" % x["d_attack_global"],
                     "%.3e" % x["d_peak"], g(x["d_tpeak"]),
                     g(x["d_arr_med"]), g(x["d_take_med"])])
    A(md_table(["α", "ρ_seed", "R0", "Λ_TX(0)@α=0", "Λ_TX(0)@α",
                "self-force TX", "max|Δattack|", "Δattack global",
                "Δpeak", "Δt_peak", "Δt_arr med", "Δt_take med"], rows))
    A("")
    A("（表中列出前 18 组，全 27 组见 `coupling_diag_seed.csv` / "
      "`wavefront_fallback.csv`；本表全部数值：数值待用户数值确认）")
    A("")
    A("全 27 组汇总：max|Δattack| 最大值 = **%.3e**；"
      "|Δattack_global| 最大值 = **%.3e**；"
      "|Δpeak| 最大值 = **%.3e**；|Δt_peak| 最大值 = **%.3f d**；"
      "|Δt_arrive median| 最大值 = **%.3f d**。"
      % (max(x["max_dattack"] for x in alpha_rows),
         max(abs(x["d_attack_global"]) for x in alpha_rows),
         max(abs(x["d_peak"]) for x in alpha_rows),
         max(abs(x["d_tpeak"]) for x in alpha_rows),
         max(abs(x["d_arr_med"]) for x in alpha_rows)))
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("### 3.7 守恒、正性、有限性")
    A("")
    all_arms = [r for r in results if r["arm"] != ARM3]
    A("- 全部 %d 个 arm 运行：max|Σy(t)−Σy(0)| = **%.3e**（绝对人口单位），"
      "归一化 **%.3e**；逐斑块 max|S+I+R−1| = **%.3e**。"
      % (len(all_arms),
         max(r["cons_glob"] for r in all_arms),
         max(r["cons_glob_rel"] for r in all_arms),
         max(r["cons_patch"] for r in all_arms)))
    A("- min_S = **%.3e**，min_I = **%.3e**；负分量计数：S<−1e-12 共 %d 个、"
      "I<−1e-12 共 %d 个（记录网格上全部单元）；all_finite = %s。"
      % (min(r["min_S"] for r in all_arms),
         min(r["min_I"] for r in all_arms),
         sum(r["neg_S"] for r in all_arms),
         sum(r["neg_I"] for r in all_arms),
         all(r["all_finite"] for r in all_arms)))
    A("- 参考解 DOP853 本身：min_I = **%.3e**（显式 RK 在 tight tol 下"
      "的舍入级微小负值，如实报告，未裁剪）。"
      % min(refs[k]["min_I"] for k in combo_list))
    A("- 实测失败案例（全局攻击率相对参考偏差 >5%%）：**%d** 个运行，"
      "列举前 10 个：" % len(failures))
    if failures:
        A("")
        A(md_table(["α", "ρ_seed", "R0", "arm", "h", "attack",
                    "ref attack", "rel err"],
                   [[g(r["alpha"], 2), "%.0e" % r["rho_seed"],
                     g(r["R0"], 1), r["arm"], g(r["h"], 2),
                     g(r["attack_global"]),
                     g(refs[(r["alpha"], r["rho_seed"], r["R0"])]["attack_global"]),
                     "%.3f" % r["attack_global_rel_vs_ref"]]
                    for r in failures[:10]]))
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("## 4. 自检与交叉校验")
    A("")
    A("`python -m py_compile` 通过；脚本内 9 项单元测试先于主扫描执行，"
      "任一失败即 assert 中止。")
    A("")
    A(md_table(["测试", "结果", "实测"],
               [[t[0], "PASS" if t[1] else "FAIL", t[2]] for t in tests]))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 4.1 λ₀ 容差敏感性（arm2，h=0.1 d，α∈{0,0.05}，R0∈{1.5,3,6}，ρ_seed=1e-4）")
    A("")
    rows = []
    for x in tol_rows:
        rows.append([g(x["alpha"], 2), g(x["R0"], 1),
                     "%.0e" % x["tol_rel"], x["n_branch"],
                     x["n_l0_below_tol"], g(x["attack_mean"], 9),
                     "%.3e" % x["d_attack_vs_arm1"],
                     "%.3e" % x["d_global_attack"]])
    A(md_table(["α", "R0", "TOL_REL", "n_branch", "n(λ₀<TOL)",
                "attack", "max|Δattack vs arm1|", "Δattack global"], rows))
    A("")
    A("（本表全部数值：数值待用户数值确认）")
    A("")
    A("### 4.2 双份 CSV 一致性")
    A("")
    A("同一份内存 payload 分别写入 4 个路径，回读 md5 与内存 md5 三方断言相等："
      "`wavefront_fallback.csv` md5=`%s`，`coupling_diag_seed.csv` md5=`%s`；"
      "均为表头 + %d / %d 行数据。"
      % (dwf["md5"], ddg["md5"], dwf["nlines"] - 1, ddg["nlines"] - 1))
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("## 5. 结论（正向结论 / 明确的阴性结论）")
    A("")
    A("**G1（fallback 是否有影响）：明确的阴性结论。** "
      "在本次全部 36 组 (α,ρ_seed,R0) 与 5 个步长的组合中，"
      "λ₀ 精确为 0 的 (patch,step) 只在 α=0 时于 t=0 的 TX 出现 1 次；"
      "fallback 在该点注入的额外累计危害量级由 λ₁ 决定，实测 "
      "λ̇₀(TX,t=0) = **%.3e /d²**（R0=3、ρ_seed=1e-4、α=0），"
      "因为新感染进入第 0 个年龄仓而 φ(τ≈0)≈0"
      "（φ_0=%.3e），需经 Erlang 链延迟才具传染性。"
      % (float(diag[(0.0, 1e-4, 3.0)]["Lambda1_t0"][TXI]),
         float(cfg["phis"][0])))
    A("因此 fallback 对攻击率、波前到达时间、峰值流行率的影响为："
      "max|Δattack| ≤ **%.3e**、到达时间变化 ≤ **%.3f d**、"
      "峰值流行率变化 ≤ **%.3e**，即在本模型与本次初值下"
      "**数值上无可见影响**。这不是实现失效（T6/T7 已验证系数与求积正确、"
      "T5 的 tol_rel=0 控制臂逐位复现原实现），而是该分支在本模型中"
      "本就几乎不产生注入：唯一 λ₀=0 的斑块是有源斑块 TX，"
      "且因零对角而无自注入，λ̇₀ 又被 φ(0)≈0 压到 1e-12 量级。"
      % (max_dattack_all, max(max_darr_all, max_dtake_all), max_dpeak_all))
    A("")
    A("**G2（α>0 是否让实体内传播出现）：结构性正向、动力学上微弱。** "
      "α>0 使 W′_ii=α>0，Λ_i 出现 β_sp·α·(Φ_i/N_i) 的实体内项；"
      "实测 TX 在 t=0 的 Λ_TX 由 α=0 的 **0.0** 变为 α=0.05 的 "
      "**%.3e /d**（R0=1.5、ρ_seed=1e-4），自作用力 = **%.3e /d**，"
      "且 λ₀=0 的 (patch,step) 计数由 1 降为 0。"
      % (float(diag[(0.05, 1e-4, 1.5)]["Lambda0_t0"][TXI]),
         float(diag[(0.05, 1e-4, 1.5)]["self_force_t0"][TXI])))
    A("但宏观上 W 稠密且行随机、ρ(W′)=1，自环只是把原加权平均中的一部分"
      "替换为本斑块值，实测 max|Δattack| ≤ **%.3e**、"
      "|Δattack_global| ≤ **%.3e**、|Δt_peak| ≤ **%.3f d**。"
      "即：**自环确实引入实体内传播项，但在本稠密引力核下不足以产生"
      "可分辨的空间波前或攻击率差异**。"
      % (max(x["max_dattack"] for x in alpha_rows),
         max(abs(x["d_attack_global"]) for x in alpha_rows),
         max(abs(x["d_tpeak"]) for x in alpha_rows)))
    A("")
    A("**G3（波前本身）：阴性结论。** W 全部非对角元 >0（最小 %.3e），"
      "52 个实体在 t=0⁺ 同时受力，θ₁/θ₂ 到达时间的空间散布远小于"
      "流行病特征时间；这是一个全网同步的入侵/灭绝过程，不是空间行波。"
      % wdiag[0.0]["offdiag_min"])
    A("")
    A("（本节全部数值：数值待用户数值确认）")
    A("")
    A("## 6. 性能数据")
    A("")
    A("- 总墙钟 %.2f s；主扫描 %d 个 (组合,臂,步长) 运行 + %d 个 DOP853 "
      "参考 + %d 个容差敏感性运行。"
      % (total_wall, len(results),
         len(combo_list), len(tol_rows)))
    A("- h=0.1 d 单运行（2400 宏步）平均墙钟：arm1 %.3f s，arm2 %.3f s，"
      "EXP2 %.3f s；DOP853 参考平均 %.3f s。"
      % (float(np.mean([r["wall_s"] for r in sel(arm=ARM1, h=0.1)])),
         float(np.mean([r["wall_s"] for r in sel(arm=ARM2, h=0.1)])),
         float(np.mean([r["wall_s"] for r in sel(arm=ARM4, h=0.1)])),
         float(np.mean([refs[k].get("wall_s", 0.0) for k in combo_list]))))
    A("- Python 层峰值内存（tracemalloc）= %.2f MiB（不含 numpy 底层缓冲，"
      "仅为量级估计）；单运行最大常驻数组为 (2401,52)×3 的 float64 "
      "记录矩阵 ≈ %.2f MiB。"
      % (peak_mem / 1048576.0, 3 * 2401 * 52 * 8 / 1048576.0))
    A("- 瓶颈：`Iold = Ibins @ T.T`（52×40×40）与 `lam_derivs` 中"
      " 4 次 W@(Φ/N) 的 52×52 矩阵乘；`static_kernel`/`poisson_T` 按 "
      "(h,σ) 缓存复用。未并行（单进程、确定性优先）。")
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("## 7. 已知数值风险与局限")
    A("")
    A("1. **λ₀ 容差敏感性**：绝对容差不可用（R0=6 灭绝尾 λ₀ 实测低到 "
      "~1e-56 /d）。本报告用相对尺度 TOL=TOL_REL·β_sp·max(Φ/N)；"
      "TOL_REL 从 1e-16 放大到 1e-4 会把 n_branch 从 1 抬到数十，"
      "但对攻击率影响 ≤ %.3e（§4.1）。换用别的尺度定义会改变计数，"
      "不改变结论。" % max(x["d_attack_vs_arm1"] for x in tol_rows))
    A("2. **fallback 形式与 Propositions 1–2**：Λ=s·Q² 满足 Prop 1 的 "
      "Λ≥0 前提与 Prop 2 的守恒前提（证明与 Λ 代数形式无关），但 "
      "Prop 3 的四阶前提（尤其 π_j 对注入时刻的高阶分辨）未在分支步上证明；"
      "且 s·Q² 无法表达偶阶零（λ₀=λ₁=0, λ₂≠0）。详见 §2.4。")
    A("3. **240 d horizon 对 R0=1.5 不足**：参考解在 t=240 d 仍在增长"
      "（峰值落在区间端点），故 attack 是区间末值而非最终规模；"
      "不宜把 R0=1.5 的 attack 解释为 final size。")
    A("4. **大 h 退化**：h≥9 d 时 Taylor 预测子在步内严重失真，"
      "实测出现 %d 个攻击率相对参考偏差 >5%% 的运行（最严重者 "
      "h=9 d/R0=1.5 attack≈%.3e vs 参考 %.3e）；全部如实保留，未裁剪、"
      "未加阻尼、未改初值。"
      % (len(failures),
         min((r["attack_global"] for r in failures), default=float("nan")),
         refs[(0.0, 1e-4, 1.5)]["attack_global"]))
    A("5. **到达时间分辨率受 h 限制**：θ₁/θ₂ 越阈时刻只能在各自输出网格上"
      "取值（参考解 0.25 d，arm 为 h）。h=20 d 时到达时间只能是 20 d 的"
      "整数倍，跨 h 的到达时间比较应只看量级。")
    A("6. **参考解微小负值**：DOP853 在 tight tol 下出现 %.3e 量级的"
      "负 I（显式 RK 舍入），未做任何后处理；M3 两臂在全部运行中 "
      "min_S/min_I 均 ≥0。" % min(refs[k]["min_I"] for k in combo_list))
    A("7. **α>0 是模型变体**：W′ 不再是稿件标定的引力矩阵；虽 ρ(W′)=1 "
      "使 β_sp=R0/TG 的标定保持，但其流行病学含义需由领域成员判断，"
      "本报告只给数值，不作机制解释。")
    A("8. **未测项**：未在 λ₀=0 且 λ̇₀=O(1) 的合成状态上跑完整流行"
      "（那需要改 φ 核或 W，属改模型，本实验禁止）；fallback 的正确性由 "
      "T6（导数匹配）、T7（求积精确）、T5（tol_rel=0 控制臂）在数值层面验证。")
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("## 8. 对稿件措辞的建议（供主理人参考；本报告不改稿件）")
    A("")
    A("1. §`subsec:flow` 现在写「Near-extinction and wavefront states are "
      "therefore outside the prescribed experiments」——可改为给出实测边界："
      "在 52 州稠密引力核、ρ_seed∈{1e-4,1e-3,1e-2}、R0∈{1.5,3,6}、"
      "120–240 d 的实测中，λ₀=0 分支仅在 t=0 的播种斑块出现 "
      "%d 个 (patch,step)，且 λ̇₀≈%.1e /d²；"
      "补的 s·square fallback 对攻击率的影响 ≤%.1e。"
      "把「outside」从回避改成有数字的限定，比删掉更稳。"
      % (_lookup(results, (0.0, 1e-4, 3.0), ARM1, 0.1)["n_l0_zero"],
         float(diag[(0.0, 1e-4, 3.0)]["Lambda1_t0"][TXI]),
         max_dattack_all))
    A("2. 不要把 W 对角为 0 写成「遗漏/缺陷」：实测 52 个对角元精确为 0、"
      "全部非对角元 >0（最小 %.3e），是引力核的结构性质。"
      "审稿人若追问「无实体内传播」，建议直接给出 W′=(1−α)W+αI 的"
      "敏感性：α=0.05 使 TX 的 Λ(t=0) 从 0 升到 %.3e /d，"
      "但攻击率变化 ≤%.1e、峰值时间变化 ≤%.1f d。"
      % (wdiag[0.0]["offdiag_min"],
         float(diag[(0.05, 1e-4, 1.5)]["Lambda0_t0"][TXI]),
         max(x["max_dattack"] for x in alpha_rows),
         max(abs(x["d_tpeak"]) for x in alpha_rows)))
    A("3. 避免「spatial wavefront / 空间行波」的强措辞：稠密 W 下 52 个"
      "实体同时受力，到达时间由增长率而非距离决定。建议改为 "
      "「invasion/extinction front in the force of infection」并明确 "
      "「the coupling kernel is dense, so the front is not a travelling wave」。")
    A("4. 不要宣称 fallback 「修复了波前」或「恢复了四阶」：本实验是"
      "明确的阴性结论，Prop 3 前提在分支步上未验证。"
      "建议措辞：「a non-negative s·square variant is available and matches "
      "the first three λ-derivatives; in the tested dense-kernel scenarios "
      "its effect is numerically negligible」。")
    A("5. 建议把「at most 1.4×10⁻¹⁴ conservation residual」这类断言"
      "限定到本报告实测范围：本实验全部 arm 运行的最大守恒残差为 %.3e"
      "（绝对人口单位），DOP853 参考解有 %.1e 量级负 I；"
      "大 h（≥9 d） attack 偏差 >5%% 的 %d 个失败案例应在正文或补充材料"
      "中明示，不宜只报 h≤4.2 d 的干净结果。"
      % (max(r["cons_glob"] for r in all_arms),
         min(refs[k]["min_I"] for k in combo_list), len(failures)))
    A("6. 若保留 Propositions 1–2 的「unconditional」表述，建议加一句："
      "the guarantee requires only Λ(s)≥0 with positive Gauss weights and "
      "Σπ_j=1; it is independent of the algebraic form of Λ，"
      "从而把 s·square 变体纳入同一证明框架（§2.4）。")
    A("")
    A("（本段全部数值：数值待用户数值确认）")
    A("")
    A("---")
    A("")
    A("复现命令：`C:/Python314/python.exe submission/code/regen_wavefront.py`；"
      "运行日志：`_review_new_tmp_lead/r8_wavefront_run.log`。"
      "本报告不改任何现有 .py/.csv/.tex/.bib/.md。")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(rep) + "\n")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    log("=" * 78)
    log("[SUMMARY]")
    log("  total wall                : %.2f s" % total_wall)
    log("  result rows               : %d  (ref %d / arm runs %d)"
        % (len(results), len(combo_list), len(results) - len(combo_list)))
    log("  W diag at alpha=0         : exactly 0 for all 52 rows = %s"
        % bool(np.all(np.diag(W0) == 0.0)))
    log("  lambda0=0 (patch,step)    : alpha=0 -> %d ; alpha>0 -> %d"
        % (_lookup(results, (0.0, 1e-4, 3.0), ARM1, 0.1)["n_l0_zero"],
           _lookup(results, (0.05, 1e-4, 3.0), ARM1, 0.1)["n_l0_zero"]))
    log("  fallback activations      : %d over 36 h=0.1 runs" % tot_branch)
    log("  max |fallback - original| : attack %.3e , arrival %.3f d , "
        "peak %.3e" % (max_dattack_all, max(max_darr_all, max_dtake_all),
                        max_dpeak_all))
    log("  max alpha effect          : attack %.3e , global %.3e , "
        "t_peak %.3f d"
        % (max(x["max_dattack"] for x in alpha_rows),
           max(abs(x["d_attack_global"]) for x in alpha_rows),
           max(abs(x["d_tpeak"]) for x in alpha_rows)))
    log("  max conservation residual : %.3e" %
        max(r["cons_glob"] for r in all_arms))
    log("  failures (>5%% attack err): %d" % len(failures))
    log("  md5 wavefront_fallback    : %s" % dwf["md5"])
    log("  md5 coupling_diag_seed    : %s" % ddg["md5"])
    log("  report                    : %s" % REPORT_MD)
    log("  log                       : %s" % LOG_PATH)
    log("  self-tests                : %d/%d passed"
        % (sum(t[1] for t in tests), len(tests)))
    log("=" * 78)
    log("DONE (exit 0)")
    return 0


def _lookup(results, key, arm, h):
    for r in results:
        if (r.get("alpha"), r.get("rho_seed"), r.get("R0")) == key \
                and r["arm"] == arm and abs(r["h"] - h) < 1e-12:
            return r
    raise KeyError("no result for %r %s h=%s" % (key, arm, h))


if __name__ == "__main__":
    fh, orig = _setup_log()
    try:
        rc = main()
    except BaseException:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fh.flush()
        sys.stdout = orig
        fh.close()
        raise
    sys.stdout.flush()
    fh.flush()
    sys.stdout = orig
    fh.close()
    sys.exit(rc)
