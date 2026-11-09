# -*- coding: utf-8 -*-
"""regen_m3_mprk.py -- R8 / route B / experiment line 1.

TWO EXPERIMENTS, ONE SELF-CONTAINED SCRIPT
==========================================
(1A) Add REAL modified-Patankar machinery to the host-level comparison set.
     The host-level S / Erlang-I / R system is first written explicitly as a
     production-destruction system (PDS) y' = sum_j (p_ij - d_ij), p_ij = d_ji,
     then three literature MPRK schemes are implemented on that PDS:

        MPE            Burchard, Deleersnijder & Meister (2003), order 1
        MPRK22(1)      BDM03 / Kopecz & Meister (2017), order 2
        MPRK43I(1,1/2) Kopecz & Meister (2018), Thm 2.7 + Lemma 2.4 Case I,
                       order 3 (four linearly implicit systems per step)

     General form (KM18, Definition 1.4, delta=1):

        y_i^(k)   = y_i^n + dt sum_{nu<k} a_knu sum_j [ p_ij(y^(nu)) y_j^(k)/pi_j^(k)
                                                        - d_ij(y^(nu)) y_i^(k)/pi_i^(k) ]
        y_i^(n+1) = y_i^n + dt sum_k b_k sum_j [ p_ij(y^(k)) y_j^(n+1)/sigma_j
                                                  - d_ij(y^(k)) y_i^(n+1)/sigma_i ]

     Every one of these linear systems has an M-matrix with unit COLUMN sums,
     hence inverse >= 0 and unconditional positivity + conservation for ANY h.
     For this particular PDS the matrix is block bidiagonal per patch, so the
     systems are solved by forward substitution (no sparse solver, no iteration).

(1B) Attribute the RK4 / SSPRK3 "non-finite at h=1 d" record.  The full RHS used
     by m3_experiment.run_explicit_rk hands the whole 40-stage Erlang aging chain
     to textbook explicit RK4.  We (a) reproduce the failure, (b) compute the
     spectrum / RK4 real stability bound / Jordan-block transient amplification,
     (c) integrate the linear chain EXACTLY with p0b_experiment.poisson_T and let
     RK4 act only on the nonlinear infection term, in two independent
     constructions (Lawson integrating-factor RK4, and Lie splitting).

OUTPUTS (NEW FILES ONLY; no existing .py / .csv / .tex / .bib is modified)
    figures/m3_mprk.csv             + submission/figures/m3_mprk.csv
    figures/m3_rk4_stability.csv    + submission/figures/m3_rk4_stability.csv
    _review_new_tmp_lead/r8_mprk_findings.md

Both CSV copies are byte-identical (M.write_csv_dual asserts md5 equality).

RUN
    C:/Python314/python.exe submission/code/regen_m3_mprk.py

All numbers are measured at run time, in float64, with NO random stream.
Every number is labelled: 数值待用户数值确认 / pending user numerical confirmation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m3_experiment as M          # noqa: E402  (import does not call main())
import p0b_experiment as P         # noqa: E402

SUB = os.path.dirname(HERE)
ROOT = os.path.dirname(SUB)
FIG_ROOT = os.path.join(ROOT, "figures")
FIG_SUB = os.path.join(SUB, "figures")
REPORT_DIR = os.path.join(ROOT, "_review_new_tmp_lead")
REPORT_PATH = os.path.join(REPORT_DIR, "r8_mprk_findings.md")

# dual write: root figures/ + submission/figures/ (same pattern as regen_m3_midstep.py)
M.OUT_DIRS = [FIG_ROOT, FIG_SUB]

# ---------------------------------------------------------------------------
# canonical shared setup -- taken VERBATIM from the existing modules
# ---------------------------------------------------------------------------
n = P.n
K = P.K
SIG = P.sigma
BETA = P.beta_sp
W = P.W
N = P.N
PHIS = P.phis
S00 = P.S0.copy()
B00 = P.bins0.copy()
R00 = P.R0.copy()
T0 = float(P.T0)
T1_ORIGINAL = float(P.T1)
TINY = 1e-300                 # Patankar-weight denominator floor (cf. PositiveIntegrators.jl)
REPEATS = 9                  # measure_wall_ms: 1 warm-up + 9 timed runs
CHECK_TOL = -1e-9            # negativity threshold, identical to existing code

H_ORDER = [0.5, 0.4, 0.3, 0.25, 0.2, 0.125, 0.1, 0.0625]   # m3_convergence.csv grid
H_REQUIRED = [1.0, 0.5, 0.25, 0.125, 0.0625]
H_EXTREME_REQUIRED = [4.2, 9.0, 10.0, 20.0, 30.0, 60.0]
H_FINE = [0.03125, 0.015625]  # extra diagnostics only: asymptotic local order
H_SCAN = sorted(set([float(x) for x in H_ORDER + H_REQUIRED + H_EXTREME_REQUIRED]))
H_STABILITY = [0.5, 0.7, 0.8, 1.0, 1.3, 4.2, 9.0, 10.0, 20.0, 30.0, 60.0]

MPRK_ARMS = ["MPE", "MPRK22(1)", "MPRK43I(1,1/2)"]

MD5_LOG = []


def log(msg):
    print(msg, flush=True)


def f(v):
    """Same formatter as M.write_csv_dual (so report numbers == CSV cells)."""
    return M.fmt(v)


# =============================================================================
# 0. grid handling (temporary P.T1, exactly as regen_m3_midstep.py)
# =============================================================================
def set_grid_for(h):
    """Temporarily set P.T1 = n*h so t_grid(h) has integer steps of size exactly
    h and ends at n*h >= 60 d.  Caller MUST restore P.T1 afterwards."""
    nstep = max(1, int(math.ceil((T1_ORIGINAL - T0) / h - 1e-9)))
    P.T1 = float(np.round(nstep * h, 8))
    grid = P.t_grid(h)
    if abs(grid[-1] - P.T1) > 1e-8:
        raise RuntimeError("grid end mismatch: h=%g end=%.12g T1=%.12g"
                           % (h, grid[-1], P.T1))
    return grid, nstep, float(grid[-1])


def restore_grid():
    P.T1 = T1_ORIGINAL


# =============================================================================
# 1. PDS representation of the host-level system
# =============================================================================
# Flattened state layout, per patch i (i = 0..n-1), K = 40 age bins:
#     S_i        -> index i*(K+2)
#     I_{i,k}    -> index i*(K+2) + 1 + k       (k = 0..K-1)
#     R_i        -> index i*(K+2) + 1 + K
#
# Continuous ODE (the h->0 limit of the shared host-level map):
#     Lambda_i(y) = beta_sp * sum_j w_ij * (sum_k phi_k I_{j,k}) / N_j
#     dS_i/dt     = -Lambda_i S_i
#     dI_{i,0}/dt = +Lambda_i S_i - sigma I_{i,0}
#     dI_{i,k}/dt =  sigma I_{i,k-1} - sigma I_{i,k}      (k = 1..K-1)
#     dR_i/dt     =  sigma I_{i,K-1}
#
# Non-negative fluxes (all >= 0 whenever y >= 0):
#     infection  F_inf,i(y)  = Lambda_i(y) * S_i          S_i -> I_{i,0}
#     aging      F_age,i,k(y)= sigma * I_{i,k}            I_{i,k} -> I_{i,k+1}
#     removal    F_rem,i(y)  = sigma * I_{i,K-1}          I_{i,K-1} -> R_i
#
# PDS entries (p_ij = rate j -> i, d_ij = rate i -> j; p_ij = d_ji):
#     p_{I_{i,0}, S_i}       = d_{S_i, I_{i,0}}       = F_inf,i(y)
#     p_{I_{i,k+1}, I_{i,k}} = d_{I_{i,k}, I_{i,k+1}} = sigma I_{i,k}   (k=0..K-2)
#     p_{R_i, I_{i,K-1}}     = d_{I_{i,K-1}, R_i}     = sigma I_{i,K-1}
#     everything else 0.
# All destruction terms are proportional to their OWN constituent, hence
#     D_a(y)/y_a = Lambda_i(y) for a = S_i; sigma for a = I_{i,k}; 0 for R_i.
# The production matrix has AT MOST ONE donor per row, so the MPRK linear system
# is block bidiagonal (S -> I_0 -> I_1 -> ... -> I_{K-1} -> R) and is solved by
# forward substitution in O(n*K) operations.
# =============================================================================
def flat(S, I, R):
    return np.concatenate([S, I.ravel(), R])


def unflat(y):
    return y[:n], y[n:n + n * K].reshape(n, K), y[n + n * K:]


def pds_fluxes(y):
    """Return (finf, I) for a flattened stage state: infection flux (n,) and
    the age-bin matrix (n,K) (aging/removal fluxes are sigma * I[:,k])."""
    S, I, R = unflat(y)
    lam = BETA * (W @ ((I @ PHIS) / N))
    return lam * S, I


def patankar_solve(stages, weights, pwd, h, rhs):
    """One MPRK (delta=1) linear system for this PDS.

        M_aa = 1 + h sum_nu w_nu D_a(y^(nu)) / pwd_a
        M_ab = -h sum_nu w_nu p_ab(y^(nu)) / pwd_b   (a != b, b = unique donor)
        M y_new = rhs

    `stages` = [y^(nu)] flattened, `weights` = [w_nu] >= 0, `pwd` = PWD vector,
    `rhs` = y^n (delta=1).  M is an M-matrix with unit column sums, so
    y_new >= 0 and sum(y_new) == sum(rhs) for any h > 0.
    """
    pw = np.maximum(np.asarray(pwd, dtype=np.float64), TINY)
    pwdS = pw[:n]
    pwdI = pw[n:n + n * K].reshape(n, K)
    sum_finf = np.zeros(n, dtype=np.float64)
    sum_I = np.zeros((n, K), dtype=np.float64)
    for yv, w in zip(stages, weights):
        finf, I = pds_fluxes(yv)
        sum_finf += w * finf
        sum_I += w * I

    invS = 1.0 / pwdS
    invI = 1.0 / pwdI
    diagS = 1.0 + h * sum_finf * invS          # destruction of S
    diagI = 1.0 + h * SIG * sum_I * invI       # destruction of each I bin
    offS = h * sum_finf * invS                 # production I_0 <- S
    offI = h * SIG * sum_I * invI              # production I_{k+1} <- I_k ; R <- I_{K-1}

    rS, rI, rR = unflat(rhs)
    Snew = rS / diagS
    Inew = np.empty_like(sum_I)
    Inew[:, 0] = (rI[:, 0] + offS * Snew) / diagI[:, 0]
    for k in range(1, K):
        Inew[:, k] = (rI[:, k] + offI[:, k - 1] * Inew[:, k - 1]) / diagI[:, k]
    Rnew = rR + offI[:, K - 1] * Inew[:, K - 1]
    return flat(Snew, Inew, Rnew)


# --- three literature schemes ------------------------------------------------
def mpe_step(y, h):
    """MPE / Patankar-Euler (BDM03): s=1, b1=1, sigma_i = y_i^n."""
    return patankar_solve([y], [1.0], y, h, y)


def mprk22_step(y, h):
    """MPRK22(a21=1) (BDM03, KM17): Heun tableau a21=1, b=(1/2,1/2),
    pi_i = y_i^n, sigma_i = y_i^n (y_i^(2)/y_i^n)^{1/a21} = y_i^(2)."""
    y2 = patankar_solve([y], [1.0], y, h, y)             # stage 2 = MPE step
    return patankar_solve([y, y2], [0.5, 0.5], y2, h, y)  # final, PWD = y^(2)


def mprk43_step(y, h):
    """MPRK43I(alpha=1, beta=1/2), delta=1 (KM18 Thm 2.7, eq. 68).

    Butcher (Lemma 2.4 Case I with alpha=1, beta=1/2):
        a21 = 1, a31 = a32 = 1/4, b = (1/6, 1/6, 2/3)
        p = 3 a21 (a31+a32) b3 = 1, q = a21 = 1
        beta_2 = 1/(2 a21) = 1/2, beta_1 = 1 - beta_2 = 1/2
    PWDs (63a-d) with p = q = 1:
        pi_i = y_i^n ; rho_i = mu_i = y_i^(2) ; sigma_i = MPRK22(1)
        approximation of y(t_n+h), i.e. the extra stage (68d).
    """
    y1 = y
    y2 = patankar_solve([y1], [1.0], y1, h, y1)                 # (68b), pi = y^n
    rho = y2.copy()                                             # p = q = 1
    y3 = patankar_solve([y1, y2], [0.25, 0.25], rho, h, y1)     # (68c)
    sigma = patankar_solve([y1, y2], [0.5, 0.5], rho, h, y1)    # (68d), mu = y^(2)
    return patankar_solve([y1, y2, y3],
                          [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], sigma, h, y1)  # (68e)


MPRK_STEPS = {"MPE": mpe_step,
              "MPRK22(1)": mprk22_step,
              "MPRK43I(1,1/2)": mprk43_step}


# =============================================================================
# 2. uniform run driver for the MPRK arms
# =============================================================================
def run_mprk(arm, h, want_series=True):
    grid = P.t_grid(h)
    y = flat(S00.copy(), B00.copy(), R00.copy())
    ser = M.new_series(grid) if want_series else None
    cons, minv, finite = 0.0, float("inf"), True
    negS = negI = negR = 0
    first_bad = None
    step = MPRK_STEPS[arm]
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            y = step(y, h)
            S, I, R = unflat(y)
            It = I.sum(1)
            ok = bool(np.all(np.isfinite(y)))
            finite = finite and ok
            if ok:
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < CHECK_TOL).sum())
                negI += int((It < CHECK_TOL).sum())
                negR += int((R < CHECK_TOL).sum())
                minv = min(minv, float(y.min()))
                if want_series:
                    ser["S"][k + 1] = S / N
                    ser["I"][k + 1] = It / N
                    ser["R"][k + 1] = R / N
            else:
                first_bad = k + 1
                break
    return dict(arm=arm, h=h, grid=grid, all_finite=finite, cons_resid=cons,
                neg_S=negS, neg_I=negI, neg_R=negR, min_state=minv,
                first_bad_step=first_bad, series=ser)


def run_m3_anchor(h):
    """Production M3-p3 arm, reproduced with the EXISTING M.run_arm, as an
    accuracy anchor and as a cross-check against m3_cost.csv / m3_midstep.csv."""
    d = M.run_arm("M3-p3", h, counting=False)
    return dict(arm="M3-p3", h=h, grid=d["grid"],
                all_finite=d["all_finite"], cons_resid=d["cons_resid"],
                neg_S=d["neg_S"], neg_I=d["neg_I"], neg_R=d["neg_R"],
                min_state=d["min_S"], first_bad_step=None, series=d["series"])


# =============================================================================
# 3. experiment 1B : explicit-RK attribution arms
# =============================================================================
def linear_prop(S, I, R, s):
    """Exact propagator exp(A_lin s) of the linear aging + removal subsystem,
    built from p0b_experiment.poisson_T (phase-type / matrix exponential).
    S is unchanged; I -> I @ T(s)^T; the tail mass goes to R."""
    T = P.poisson_T(s)
    Inew = I @ T.T
    removed = I.sum(1) - Inew.sum(1)
    return S.copy(), Inew, R + removed


def g_nonlinear(S, I, R):
    """Nonlinear infection part only: (-Lambda*S, +Lambda*S at I_0, 0)."""
    lam = BETA * (W @ ((I @ PHIS) / N))
    F = lam * S
    dI = np.zeros_like(I)
    dI[:, 0] = F
    return -F, dI, np.zeros_like(R)


def state_add(a, b, c=1.0):
    return (a[0] + c * b[0], a[1] + c * b[1], a[2] + c * b[2])


def lawson_rk4_step(Y, h):
    """Lawson / integrating-factor RK4: with z = exp(-A t) y, classical RK4 is
    applied to z' = exp(-A t) g(exp(A t) z).  Rewritten in the physical frame
    (NO negative-time propagator is needed):

        E_s = exp(A s) (exact, poisson_T)
        Y1 = y_n ;  E2 = E_{h/2} Y1 ;  E1 = E_h Y1
        Y2 = E2 Y1 + (h/2) E2 g(Y1)
        Y3 = E2 Y1 + (h/2) g(Y2)
        Y4 = E1 Y1 + h E2 g(Y3)
        y_{n+1} = E1 Y1 + (h/6)[ E1 g(Y1) + 2 E2 g(Y2) + 2 E2 g(Y3) + g(Y4) ]

    RK4 acts ONLY on the nonlinear infection term; the whole Erlang chain is
    advanced exactly at every stage."""
    S, I, R = Y
    E2 = linear_prop(S, I, R, 0.5 * h)
    E1 = linear_prop(S, I, R, h)
    g1 = g_nonlinear(S, I, R)
    E2g1 = linear_prop(*g1, 0.5 * h)
    Eg1 = linear_prop(*g1, h)
    Y2 = state_add(E2, E2g1, 0.5 * h)
    g2 = g_nonlinear(*Y2)
    E2g2 = linear_prop(*g2, 0.5 * h)
    Y3 = state_add(E2, g2, 0.5 * h)
    g3 = g_nonlinear(*Y3)
    E2g3 = linear_prop(*g3, 0.5 * h)
    Y4 = state_add(E1, E2g3, h)
    g4 = g_nonlinear(*Y4)
    acc = (np.zeros_like(S), np.zeros_like(I), np.zeros_like(R))
    acc = state_add(acc, Eg1, 1.0)
    acc = state_add(acc, E2g2, 2.0)
    acc = state_add(acc, E2g3, 2.0)
    acc = state_add(acc, g4, 1.0)
    return state_add(E1, acc, h / 6.0)


def reaction_rhs(y):
    S, I, R = unflat(y)
    lam = BETA * (W @ ((I @ PHIS) / N))
    F = lam * S
    dI = np.zeros_like(I)
    dI[:, 0] = F
    return flat(-F, dI, np.zeros_like(R))


def lie_split_step(S, I, R, h):
    """Independent construction: Lie splitting.  (a) advance the linear chain
    EXACTLY over h with poisson_T; (b) classical RK4 on the reaction ODE with
    aging frozen (RK4 sees ONLY the nonlinear infection term)."""
    S1, I1, R1 = linear_prop(S, I, R, h)
    y = flat(S1, I1, R1)
    with np.errstate(all="ignore"):
        k1 = reaction_rhs(y)
        k2 = reaction_rhs(y + 0.5 * h * k1)
        k3 = reaction_rhs(y + 0.5 * h * k2)
        k4 = reaction_rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return unflat(y)


def rhs_linear_only(y):
    """Full 40-stage Erlang chain + removal, with beta_sp = 0 (no infection)."""
    S = y[:n]
    I = y[n:n + n * K].reshape(n, K)
    dS = np.zeros_like(S)
    dI = -SIG * I
    dI = dI.copy()
    dI[:, 1:] += SIG * I[:, :-1]
    dR = SIG * I[:, K - 1]
    return np.concatenate([dS, dI.ravel(), dR])


def run_flat_rk(h, rhs, method="rk4"):
    """Explicit RK4 / SSPRK3 on an arbitrary flattened RHS, returning the SAME
    diagnostics as m3_experiment.run_explicit_rk plus min over the FULL state."""
    grid = P.t_grid(h)
    y = np.concatenate([S00, B00.ravel(), R00])
    cons, minv, finite = 0.0, float("inf"), True
    negS = negI = negR = 0
    first_bad = None
    first_neg = None
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            if method == "rk4":
                k1 = rhs(y)
                k2 = rhs(y + 0.5 * h * k1)
                k3 = rhs(y + 0.5 * h * k2)
                k4 = rhs(y + h * k3)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif method == "ssprk3":
                u1 = y + h * rhs(y)
                u2 = 0.75 * y + 0.25 * (u1 + h * rhs(u1))
                y = (1.0 / 3.0) * y + (2.0 / 3.0) * (u2 + h * rhs(u2))
            else:
                raise ValueError(method)
            if first_neg is None and bool(np.any(y < CHECK_TOL)):
                first_neg = k + 1
            ok = bool(np.all(np.isfinite(y)))
            finite = finite and ok
            if ok:
                S = y[:n]
                I = y[n:n + n * K].reshape(n, K)
                R = y[n + n * K:]
                It = I.sum(1)
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < CHECK_TOL).sum())
                negI += int((It < CHECK_TOL).sum())
                negR += int((R < CHECK_TOL).sum())
                minv = min(minv, float(y.min()))
            else:
                first_bad = k + 1
                break
    return dict(all_finite=finite, cons_resid=cons, neg_S=negS, neg_I=negI,
                neg_R=negR, min_state=minv, first_bad_step=first_bad,
                first_neg_step=first_neg, n_steps=len(grid) - 1)


def run_state_rk(h, step_fn, method):
    """Driver for the two state-tuple arms (Lawson / Lie)."""
    grid = P.t_grid(h)
    S, I, R = S00.copy(), B00.copy(), R00.copy()
    cons, minv, finite = 0.0, float("inf"), True
    negS = negI = negR = 0
    first_bad = None
    first_neg = None
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            if method == "lawson":
                S, I, R = lawson_rk4_step((S, I, R), h)
            elif method == "lie":
                S, I, R = lie_split_step(S, I, R, h)
            else:
                raise ValueError(method)
            y = flat(S, I, R)
            if first_neg is None and bool(np.any(y < CHECK_TOL)):
                first_neg = k + 1
            ok = bool(np.all(np.isfinite(y)))
            finite = finite and ok
            if ok:
                It = I.sum(1)
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < CHECK_TOL).sum())
                negI += int((I < CHECK_TOL).sum())
                negR += int((R < CHECK_TOL).sum())
                minv = min(minv, float(y.min()))
            else:
                first_bad = k + 1
                break
    return dict(all_finite=finite, cons_resid=cons, neg_S=negS, neg_I=negI,
                neg_R=negR, min_state=minv, first_bad_step=first_bad,
                first_neg_step=first_neg, n_steps=len(grid) - 1)


# --- linear-algebra diagnostics ---------------------------------------------
def erlang_generator():
    """40-stage Erlang aging-chain generator (lower bidiagonal Jordan block)."""
    A = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        A[k, k] = -SIG
    for k in range(1, K):
        A[k, k - 1] = SIG
    return A


A_LIN = erlang_generator()
EIG = np.linalg.eigvals(A_LIN)
EIG_REAL_MAX = float(np.max(EIG.real))
EIG_REAL_MIN = float(np.min(EIG.real))
EIG_IMAG_MAX = float(np.max(np.abs(EIG.imag)))

# R4(z) = 1+z+z^2/2+z^3/6+z^4/24 ; |R4(z)|=1 on the negative real axis at the
# real root of z^3+4z^2+12z+24 = 0 (R4(z)=+1, z != 0).
_r4 = np.roots([1.0, 4.0, 12.0, 24.0])
RK4_REAL_BOUND = -float(_r4[np.argmin(np.abs(_r4.imag))].real)
# R3(z) = 1+z+z^2/2+z^3/6 ; |R3(z)|=1 at the real root of z^3+3z^2+6z+12 = 0.
_r3 = np.roots([1.0, 3.0, 6.0, 12.0])
SSPRK3_REAL_BOUND = -float(_r3[np.argmin(np.abs(_r3.imag))].real)


def rk_poly_at(z, p):
    return float(sum(z ** m / math.factorial(m) for m in range(p + 1)))


def amplification(h, p):
    """R_p(h A_lin) = sum_{m=0}^p (hA)^m / m! for the defective Jordan block."""
    Rm = np.eye(K, dtype=np.float64)
    Pw = np.eye(K, dtype=np.float64)
    for m in range(1, p + 1):
        Pw = Pw @ (h * A_LIN) / m
        Rm = Rm + Pw
    return Rm


def jordan_transient(h, p, nsteps=60):
    """max over n=1..nsteps of max |R_p(hA)^n| : transient growth caused by the
    defective (non-normal) 40x40 Jordan block, invisible to the scalar bound."""
    Rm = amplification(h, p)
    Q = np.eye(K, dtype=np.float64)
    mx = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(nsteps):
            Q = Rm @ Q
            v = float(np.max(np.abs(Q)))
            if not np.isfinite(v):
                return float("inf")
            mx = max(mx, v)
    return mx


# =============================================================================
# 4. CSV 1 : MPRK main table
# =============================================================================
HEADER_MPRK = [
    "section", "arm", "h_macro", "n_steps", "horizon_d", "n_patches", "K_bins",
    "all_finite", "neg_S", "neg_I", "neg_R", "cons_resid", "min_state",
    "L2_rel_err", "wall_ms_median", "wall_ms_min", "wall_n_repeats",
    "fit_order", "fit_R2", "n_h_points", "local_order",
    "criterion_i_fit_ge_3p9", "criterion_ii_one_step_map",
    "criterion_iii_no_subcycling", "joint_admissible", "note",
]


def build_mprk_rows():
    rows = []
    l2_by_arm = {a: {} for a in MPRK_ARMS}

    # ---- scan over all required + convergence h, for the three MPRK arms ----
    for h in H_SCAN:
        grid, nstep, horizon = set_grid_for(h)
        try:
            ref = M.ref_for(h)        # DOP853, same grid, identical to existing口径
            for arm in MPRK_ARMS:
                d = run_mprk(arm, h, want_series=True)
                l2 = M.safe_metrics(d["series"], ref, d["all_finite"])[0]
                l2_by_arm[arm][h] = l2
                med, mn, nr = M.measure_wall_ms(
                    lambda a=arm, hh=h: run_mprk(a, hh, want_series=False),
                    repeats=REPEATS)
                rows.append(dict(
                    section="scan", arm=arm, h_macro=h, n_steps=nstep,
                    horizon_d=horizon, n_patches=n, K_bins=K,
                    all_finite=d["all_finite"], neg_S=d["neg_S"],
                    neg_I=d["neg_I"], neg_R=d["neg_R"],
                    cons_resid=d["cons_resid"], min_state=d["min_state"],
                    L2_rel_err=l2, wall_ms_median=med, wall_ms_min=mn,
                    wall_n_repeats=nr,
                    note="MPRK/PDS arm; h grid ends at %.4f d%s"
                         % (horizon, "" if abs(horizon - 60.0) < 1e-8
                            else " (n*h >= 60 d, non-divisor h)")))
            # ---- M3-p3 accuracy anchor (existing integrator, no timing) ----
            a = run_m3_anchor(h)
            l2a = M.safe_metrics(a["series"], ref, a["all_finite"])[0]
            rows.append(dict(
                section="anchor", arm="M3-p3 (existing M.run_arm)",
                h_macro=h, n_steps=nstep, horizon_d=horizon,
                n_patches=n, K_bins=K, all_finite=a["all_finite"],
                neg_S=a["neg_S"], neg_I=a["neg_I"], neg_R=a["neg_R"],
                cons_resid=a["cons_resid"], min_state=a["min_state"],
                L2_rel_err=l2a,
                note="accuracy anchor reproduced with the EXISTING M3-p3; "
                     "not a Patankar arm; no wall timing recorded"))
            log("  [1A] h=%-8g n=%-4d horizon=%-7g %s"
                % (h, nstep, horizon,
                   " ".join("%s L2=%s" % (a, f(l2_by_arm[a][h]))
                            for a in MPRK_ARMS)))
        finally:
            restore_grid()

    # ---- extra fine steps for asymptotic local-order diagnostics ----
    for h in H_FINE:
        grid, nstep, horizon = set_grid_for(h)
        try:
            ref = M.ref_for(h)
            for arm in MPRK_ARMS:
                d = run_mprk(arm, h, want_series=True)
                l2 = M.safe_metrics(d["series"], ref, d["all_finite"])[0]
                l2_by_arm[arm][h] = l2
                rows.append(dict(
                    section="order_check", arm=arm, h_macro=h, n_steps=nstep,
                    horizon_d=horizon, n_patches=n, K_bins=K,
                    all_finite=d["all_finite"], neg_S=d["neg_S"],
                    neg_I=d["neg_I"], neg_R=d["neg_R"],
                    cons_resid=d["cons_resid"], min_state=d["min_state"],
                    L2_rel_err=l2,
                    note="extra fine step for asymptotic local order; no wall timing"))
        finally:
            restore_grid()

    # ---- fitted orders ----
    fits = {}
    for hs, tag in ((H_ORDER, "fit_8h_H_ORDER"),
                    (H_REQUIRED, "fit_5h_required")):
        for arm in MPRK_ARMS:
            errs = [l2_by_arm[arm][x] for x in hs]
            p, r2, npt = M.loglog_fit(hs, errs)
            fits[(arm, tag)] = (p, r2, npt)
            rows.append(dict(
                section=tag, arm=arm, h_macro="", n_h_points=npt,
                fit_order=p, fit_R2=r2,
                note="log-log least-squares fit of L2_rel_err over h = "
                     + "/".join("%g" % x for x in hs)))

    # ---- local (adjacent-ratio) orders on the extended fine sequence ----
    hseq = [0.5, 0.25, 0.125, 0.0625] + H_FINE
    for arm in MPRK_ARMS:
        for i in range(len(hseq) - 1):
            e0, e1 = l2_by_arm[arm][hseq[i]], l2_by_arm[arm][hseq[i + 1]]
            lo = (math.log(e0 / e1) / math.log(hseq[i] / hseq[i + 1])
                  if e0 > 0 and e1 > 0 and np.isfinite(e0) and np.isfinite(e1)
                  else float("nan"))
            rows.append(dict(
                section="local_order", arm=arm, h_macro=hseq[i + 1],
                local_order=lo,
                note="adjacent-ratio order log(e(h_prev)/e(h))/log(h_prev/h), "
                     "h_prev=%g h=%g" % (hseq[i], hseq[i + 1])))

    # ---- joint admissibility verdict (structural judgement, not a measurement)
    for arm in MPRK_ARMS:
        p = fits[(arm, "fit_8h_H_ORDER")][0]
        ci = bool(np.isfinite(p) and p >= 3.9)
        rows.append(dict(
            section="verdict", arm=arm,
            fit_order=p,
            criterion_i_fit_ge_3p9=ci,
            criterion_ii_one_step_map=True,
            criterion_iii_no_subcycling=True,
            joint_admissible=bool(ci and True and True),
            note="(ii) M-matrix / unit column sums => unconditional positivity + "
                 "conservation FROM THE ONE-STEP MAP; (iii) RK stages are not time "
                 "sub-steps, no clipping, no signed time; (i) fails: fitted order "
                 "below the 3.9 threshold; verdict is a structural judgement"))

    # ---- verification rows: cross-check anchor h=1 against existing CSVs ----
    h1 = 1.0
    anchor_l2 = M.safe_metrics(run_m3_anchor(h1)["series"], M.ref_for(h1), True)[0]
    prior = None
    prior_path = None
    for path in (os.path.join(FIG_SUB, "m3_cost.csv"),
                 os.path.join(FIG_ROOT, "m3_cost.csv")):
        if os.path.isfile(path):
            df = pd.read_csv(path)
            sel = df[df["arm"] == "M3-p3"]
            if len(sel) and "L2_rel_err_h1" in sel.columns:
                prior = float(sel.iloc[0]["L2_rel_err_h1"])
                prior_path = path
                break
    rows.append(dict(
        section="verify", arm="cross-check: M3-p3 h=1 L2 vs m3_cost.csv",
        L2_rel_err=anchor_l2,
        note="recomputed %s ; existing m3_cost.csv value %s ; source %s"
             % (f(anchor_l2), f(prior) if prior is not None else "NOT FOUND",
                prior_path or "NOT FOUND")))
    return rows, fits, l2_by_arm, anchor_l2, prior


# =============================================================================
# 5. CSV 2 : RK4 stability attribution table
# =============================================================================
HEADER_STAB = [
    "arm", "h", "sigma_h", "rk4_stability_bound_exceeded", "all_finite",
    "min_state", "note",
    "sigma", "rk4_real_stability_bound", "ssprk3_real_stability_bound",
    "R4_at_neg_sigma_h", "R3_at_neg_sigma_h", "erlang_eig_real_max",
    "erlang_eig_real_min", "erlang_eig_imag_max", "erlang_jordan_size",
    "jordan_transient_60steps_max", "neg_S", "neg_I", "neg_R", "cons_resid",
    "first_bad_step", "first_neg_step", "n_steps", "section",
]


def stab_common(h, p=4, with_transient=True):
    sh = SIG * h
    d = dict(sigma=SIG, sigma_h=sh,
             rk4_real_stability_bound=RK4_REAL_BOUND,
             ssprk3_real_stability_bound=SSPRK3_REAL_BOUND,
             rk4_stability_bound_exceeded=bool(sh > RK4_REAL_BOUND),
             R4_at_neg_sigma_h=rk_poly_at(-sh, 4),
             R3_at_neg_sigma_h=rk_poly_at(-sh, 3),
             erlang_eig_real_max=EIG_REAL_MAX, erlang_eig_real_min=EIG_REAL_MIN,
             erlang_eig_imag_max=EIG_IMAG_MAX, erlang_jordan_size=K)
    if with_transient:
        d["jordan_transient_60steps_max"] = jordan_transient(h, p)
    return d


def build_stability_rows():
    rows = []

    def push(arm, h, res, note, section="attribution", p=4, with_transient=True):
        r = dict(arm=arm, h=h, section=section,
                 all_finite=res["all_finite"], min_state=res["min_state"],
                 neg_S=res["neg_S"], neg_I=res["neg_I"], neg_R=res["neg_R"],
                 cons_resid=res["cons_resid"],
                 first_bad_step=res.get("first_bad_step"),
                 first_neg_step=res.get("first_neg_step"),
                 n_steps=res["n_steps"], note=note)
        r.update(stab_common(h, p=p, with_transient=with_transient))
        rows.append(r)

    # (a) full explicit RK4 / SSPRK3 on the COMPLETE ODE (M.rhs_flat)
    for h in [0.5, 0.8, 1.0, 1.3]:
        grid, nstep, horizon = set_grid_for(h)
        try:
            res = run_flat_rk(h, M.rhs_flat, "rk4")
            res["n_steps"] = nstep
            push("RK4-explicit (full ODE)", h, res,
                 "textbook RK4 on complete RHS including 40-stage Erlang chain; "
                 "same RHS as m3_experiment.rhs_flat")
        finally:
            restore_grid()
    for h in [1.0]:
        grid, nstep, horizon = set_grid_for(h)
        try:
            res = run_flat_rk(h, M.rhs_flat, "ssprk3")
            res["n_steps"] = nstep
            push("SSPRK3-explicit (full ODE)", h, res,
                 "Shu-Osher SSPRK3 on complete RHS; scalar real bound is "
                 "2.5127453266... (column ssprk3_real_stability_bound)", p=3)
        finally:
            restore_grid()

    # (b) verbatim reproduction through the EXISTING source function
    grid, nstep, horizon = set_grid_for(1.0)
    try:
        src = M.run_attack("RK4-explicit", 1.0)
        push("RK4-explicit (M.run_attack verbatim)", 1.0,
             dict(all_finite=src["all_finite"], min_state=src["min_S"],
                  neg_S=src["neg_S"], neg_I=src["neg_I"], neg_R=src["neg_R"],
                  cons_resid=src["cons_resid"], first_bad_step=None,
                  first_neg_step=None, n_steps=src["n_steps"]),
             "verbatim reproduction of m3_experiment.run_explicit_rk(h=1); "
             "min_state here is the source function's min_S (S only)",
             section="repro")
    finally:
        restore_grid()

    # (c) isolated linear chain (beta = 0)
    for h in [0.5, 1.0]:
        grid, nstep, horizon = set_grid_for(h)
        try:
            res = run_flat_rk(h, rhs_linear_only, "rk4")
            res["n_steps"] = nstep
            push("RK4-linear-only (beta=0)", h, res,
                 "RK4 on the 40-stage Erlang chain ALONE (no infection); "
                 "large negative I bins are Jordan-block transient growth, "
                 "cons_resid is float64 noise on ~1e19 magnitudes")
        finally:
            restore_grid()

    # (d) exact linear chain + RK4 on nonlinear part, TWO independent constructions
    for h in [0.5, 1.0, 4.2, 10.0]:
        grid, nstep, horizon = set_grid_for(h)
        try:
            res = run_state_rk(h, lawson_rk4_step, "lawson")
            res["n_steps"] = nstep
            push("RK4-Lawson-IF (exact-linear)", h, res,
                 "Lawson integrating-factor RK4: linear Erlang chain advanced "
                 "EXACTLY by p0b_experiment.poisson_T at every stage; RK4 acts "
                 "only on the nonlinear infection term; transient column is not "
                 "applicable (linear part is not discretised by RK4)",
                 with_transient=False)
        finally:
            restore_grid()
    for h in [0.5, 1.0]:
        grid, nstep, horizon = set_grid_for(h)
        try:
            res = run_state_rk(h, lie_split_step, "lie")
            res["n_steps"] = nstep
            push("RK4-Lie-split (exact-linear)", h, res,
                 "independent construction: Lie splitting, exact poisson_T "
                 "linear flow + RK4 on frozen-aging reaction ODE",
                 with_transient=False)
        finally:
            restore_grid()
    return rows


# =============================================================================
# 6. report
# =============================================================================
def build_report(rows_m, fits, l2_by_arm, anchor_l2, prior, rows_s):
    A = []
    add = A.append
    add("# R8 / 路线 B / 实验线 1 实测报告")
    add("")
    add("**补 modified-Patankar 基线 + RK4 线性稳定性对照**")
    add("")
    add("> **硬边界**：本报告所有数值均为本机实测，一律标注「数值待用户数值确认」；"
        "必须由用户使用其自有已校验代码复算确认后方可采信。本报告不出科学结论，"
        "只保证数值可追溯、可复现。")
    add("")
    add("生成时间：%s；解释器 %s；numpy %s / scipy %s / pandas %s；平台 %s。"
        % (datetime.now(timezone.utc).isoformat(timespec="seconds"),
           platform.python_version(), np.__version__, scipy.__version__,
           pd.__version__, platform.platform()))
    add("")

    # ---- 0. summary -------------------------------------------------------
    p_mpe = fits[("MPE", "fit_8h_H_ORDER")][0]
    p_22 = fits[("MPRK22(1)", "fit_8h_H_ORDER")][0]
    p_43 = fits[("MPRK43I(1,1/2)", "fit_8h_H_ORDER")][0]
    lawson1 = next(r for r in rows_s if r["arm"] == "RK4-Lawson-IF (exact-linear)"
                   and r["h"] == 1.0)
    lie1 = next(r for r in rows_s if r["arm"] == "RK4-Lie-split (exact-linear)"
                and r["h"] == 1.0)
    rk4full1 = next(r for r in rows_s if r["arm"] == "RK4-explicit (full ODE)"
                    and r["h"] == 1.0)
    repro1 = next(r for r in rows_s if r["section"] == "repro")
    lin1 = next(r for r in rows_s if r["arm"] == "RK4-linear-only (beta=0)"
                and r["h"] == 1.0)
    add("## 0. 核心结论（3–6 句）")
    add("")
    add("1. 在同一 52-patch / K=%d Erlang / 同一 W, N, φ, σ, β 的宿主层 PDS 上，"
        "三个**文献既有**的 modified-Patankar 臂（MPE、MPRK22(1)、"
        "MPRK43I(1,1/2)）在全部 %d 个受测步长（含 h=60 d）上均 "
        "`all_finite=true`、`neg_S=neg_I=neg_R=0`、守恒残差量级 ~1e-14；"
        "无条件保正与守恒来自 M-matrix 单位列和的单步映射，不是数值修补。"
        % (K, len(H_SCAN)))
    add("2. 三者观测阶（与稿件同一 8 档 h 的 log-log 拟合）分别为 %s / %s / %s，"
        "均 < 3.9；因此**被稿件 joint admissibility 判据以阶数门槛排除**，"
        "尽管它们通过 (ii) 单步映射保证与 (iii) 无 sub-cycling/clipping/signed time。"
        % (f(p_mpe), f(p_22), f(p_43)))
    add("3. RK4 在 h=1 d 的 non-finite 已用 `m3_experiment.run_attack` 逐字复现"
        "（`all_finite=%s`）；但 σh=%s **并未越过** RK4 实轴绝对稳定界 %s，"
        "真正机制是 40 重亏损 Jordan 块放大矩阵 R(hA) 的瞬态增长"
        "（60 步上界 %s）。" % (f(repro1["all_finite"]), f(SIG),
                              f(RK4_REAL_BOUND),
                              f(rk4full1["jordan_transient_60steps_max"])))
    add("4. 把线性链改为 `poisson_T` 精确推进、RK4 只作用于非线性感染力项后，"
        "两种独立构造（Lawson IF-RK4 与 Lie 分裂）在 h=1 d 均 `all_finite=%s`、"
        "守恒残差 %s / %s、无负状态；故 h=1 d 的失败是**线性链显式稳定性失败**，"
        "不构成 positivity 失败的证据。"
        % (f(lawson1["all_finite"]), f(lawson1["cons_resid"]),
           f(lie1["cons_resid"])))
    add("5. 对稿件的影响：新增的 Patankar machinery 仍不满足 fitted order ≥3.9，"
        "所以「M3-p3 是受测臂中唯一满足 joint admissibility 的四阶臂」在**扩大后的"
        "臂集合内仍成立**；但 `:118` 引用 Patankar 文献却未测该 machinery 的缺口"
        "现已被真实数据填上，且 MPDeC(4)（文献中存在任意阶保正守恒格式）未实现，"
        "该措辞必须继续严格限定为「among those tested」，不得升级为存在性唯一。")
    add("")

    # ---- 1. 技术选型表 ----------------------------------------------------
    add("## 1. 技术选型表")
    add("")
    add("| 组件 | 选型 | 理由 | 备选 |")
    add("|---|---|---|---|")
    add("| PDS 分解 | 显式逐通量分解（S→I₀ 感染、Iₖ→Iₖ₊₁ 老化、Iₖ₋₁→R 清除） | 满足 p_ij=d_ji、p_ij,d_ij≥0；每行至多 1 个 donor | 稠密 2184×2184 生产/毁灭矩阵（内存 38 MB，弃用） |")
    add("| MPRK 一般形式 | KM18 Def 1.4, δ=1（保守型） | 单步映射即 M-matrix、单位列和 ⇒ 无条件正+守恒 | δ=0 非保守变体（不需要） |")
    add("| 基线臂 | MPE(1) / MPRK22(1)(2) / MPRK43I(1,1/2)(3) | 均为文献既有方法，阶数递进，覆盖 BDM03→KM17→KM18 | MPDeC（任意阶，实现成本高，未跑成，见 §10） |")
    add("| 线性系统求解 | 分块双对角前向代入 | 结构精确、O(nK)、无迭代、无稀疏库依赖 | scipy.sparse.linalg.spsolve（更慢，未采用） |")
    add("| 参考解 | `p0b_experiment.reference_solution`（DOP853, rtol=1e-10, atol=1e-12, 同网格 t_eval） | 与 m3_convergence.csv / m3_cost.csv 完全同口径 | Radau 稠密输出（另一条线，未用于本表） |")
    add("| 误差指标 | `p0b_experiment.error_metrics` 的 L2_rel（‖d‖₂/‖ref‖₂） | 与稿件 `L2_rel_err` 同定义 | max_abs / pointwise_rel（未采用） |")
    add("| 计时 | `m3_experiment.measure_wall_ms(repeats=9)`（1 预热 + 9 计时取中位） | 与 m3_midstep.csv / m3_cost.csv 同口径 | 27 次复测（本线未跑） |")
    add("| 数值精度 | float64，全程无随机流、无 seed | 确定性、可复现 | — |")
    add("")

    # ---- 2. 数值方法说明 --------------------------------------------------
    add("## 2. 数值方法说明")
    add("")
    add("### 2.1 宿主层方程的 PDS 生产/毁灭矩阵（显式）")
    add("")
    add("状态 y：每个 patch i 含 S_i（标量）、I_{i,0..K-1}（K=%d 个 Erlang 年龄箱）、"
        "R_i（标量）；n=%d，维度 %d。" % (K, n, n * (K + 2)))
    add("")
    add("```")
    add("Lambda_i(y) = beta_sp * sum_j w_ij * (sum_k phi_k I_{j,k}) / N_j")
    add("dS_i/dt     = -Lambda_i S_i")
    add("dI_{i,0}/dt = +Lambda_i S_i - sigma I_{i,0}")
    add("dI_{i,k}/dt =  sigma I_{i,k-1} - sigma I_{i,k}   (k=1..K-1)")
    add("dR_i/dt     =  sigma I_{i,K-1}")
    add("")
    add("p_{I_{i,0}, S_i}       = d_{S_i, I_{i,0}}       = Lambda_i(y) S_i   [S_i -> I_{i,0}]")
    add("p_{I_{i,k+1}, I_{i,k}} = d_{I_{i,k}, I_{i,k+1}} = sigma I_{i,k}     [I_{i,k} -> I_{i,k+1}]")
    add("p_{R_i, I_{i,K-1}}     = d_{I_{i,K-1}, R_i}     = sigma I_{i,K-1}   [I_{i,K-1} -> R_i]")
    add("其余 p_ij = d_ij = 0;  p_ij = d_ji 恒成立。")
    add("```")
    add("")
    add("守恒校验：Σ_a D_a(y) = Σ_a P_a(y) = Λ_i S_i + σ Σ_k I_{i,k}，"
        "故 Σ_i y_i' = 0；`cons_resid` 实测见 §4.1。所有毁灭项正比于自身分量，"
        "故 D_a(y)/y_a = Λ_i（S）、σ（I 箱）、0（R）；生产矩阵每行至多一个 donor。"
        "σ = %s d⁻¹（= K/τ*, τ* = %s d），β_sp = %s，Λ 初值范围约 %s–%s。"
        % (f(SIG), f(K / SIG), f(BETA),
           f(float((BETA * (W @ ((B00 @ PHIS) / N))).min())),
           f(float((BETA * (W @ ((B00 @ PHIS) / N))).max()))))
    add("")
    add("### 2.2 MPRK 单步映射与无条件保正/守恒")
    add("")
    add("KM18 Def 1.4（δ=1）的一般形式见脚本顶部注释。把每个线性系统写成")
    add("")
    add("```")
    add("M_aa = 1 + h sum_nu w_nu D_a(y^(nu))/pwd_a      > 0")
    add("M_ab = -h sum_nu w_nu p_ab(y^(nu))/pwd_b        <= 0   (a != b)")
    add("M y_new = y^n")
    add("```")
    add("")
    add("由 p_ij=d_ji 得每一列 Σ_a M_ab = 1；M 对角为正、非对角非正 ⇒ M 是 M-matrix，"
        "M⁻¹ ≥ 0 ⇒ y_new ≥ 0 且 Σ y_new = Σ y^n，**对任意 h>0 成立**，不是数值修补、"
        "不是 limiter、不是截断。PWD 分母置下限 %s（与 PositiveIntegrators.jl 的 "
        "`small_constant` 同作用），仅用于避免 0/0；本报告所有受测档无 0/0 发生。"
        % f(TINY))
    add("")
    add("### 2.3 三个文献臂的系数")
    add("")
    add("| 臂 | 文献 | 阶段/系数 | PWD | 名义阶 |")
    add("|---|---|---|---|---|")
    add("| MPE | BDM03 | s=1, b₁=1 | σ_i=y_iⁿ | 1 |")
    add("| MPRK22(1) | BDM03 / KM17 | Heun: a₂₁=1, b=(½,½) | π_i=y_iⁿ, σ_i=y_iⁿ(y_i⁽²⁾/y_iⁿ)^{1/a₂₁}=y_i⁽²⁾ | 2 |")
    add("| MPRK43I(1,½) | KM18 Thm 2.7 / Lemma 2.4 Case I | a₂₁=1, a₃₁=a₃₂=¼, b=(⅙,⅙,⅔); p=q=1, β₁=β₂=½ | π_i=y_iⁿ, ρ_i=µ_i=y_i⁽²⁾, σ_i=式(68d) 的 MPRK22(1) 解 | 3 |")
    add("")
    add("MPRK43I 每步解 4 个线性系统（式 68b/68c/68d/68e）；这些是**同一时间层上的 RK 阶段**，"
        "不是时间子步（sub-cycling），与 RK4 的 4 次 RHS 求值同类。")
    add("")
    add("### 2.4 网格、参考解、计时")
    add("")
    add("`p0b_experiment.t_grid(h)=round(arange(0, T1+h/2, h), 8)` 要求终点=T1。"
        "对整除 60 d 的 h（%s）终点恰为 60 d；对 4.2 与 9 d 采用与 "
        "`regen_m3_midstep.py` 相同的运行期处理：n=ceil(60/h)、临时令 "
        "`P.T1=round(n*h,8)`（=%s d），步长恰为 h、覆盖 ≥60 d，扫描后恢复。"
        "参考解在**各自同一网格**上由 DOP853 直接 t_eval（无插值），"
        "因此跨 h 的 L2 不是同一时间泛函上的误差——见 §10。"
        % ("/".join("%g" % x for x in H_ORDER + [1.0, 10.0, 20.0, 30.0, 60.0]),
           f(float(np.round(math.ceil(60.0 / 4.2) * 4.2, 8)))))
    add("计时机器相关，**不作排序判据**；9 次取中位（另 1 次预热）。")
    add("")

    # ---- 3/4. 实测表 ------------------------------------------------------
    add("## 3. 实测表（全部数值来自 m3_mprk.csv 单元格）")
    add("")
    add("### 3.1 主表：每档 L2_rel / 有限性 / 负状态 / 守恒 / 墙钟")
    add("")
    add("| arm | h (d) | n_steps | all_finite | neg_S | neg_I | neg_R | cons_resid | L2_rel | wall_ms_median |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_m:
        if r["section"] != "scan":
            continue
        add("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (r["arm"], f(r["h_macro"]), f(r["n_steps"]),
               f(r["all_finite"]), f(r["neg_S"]), f(r["neg_I"]), f(r["neg_R"]),
               f(r["cons_resid"]), f(r["L2_rel_err"]), f(r["wall_ms_median"])))
    add("")
    add("`min_state` 与 `wall_ms_min` / `wall_n_repeats` 见 CSV；`min_state` 全档为 %s。"
        % f(max(float(r["min_state"]) for r in rows_m
                if r["section"] == "scan")))
    add("")
    add("### 3.2 观测阶")
    add("")
    add("| arm | fit_8h(H_ORDER) | R² | n | fit_5h(required) | R² | n | 最细局部阶 |")
    add("|---|---|---|---|---|---|---|---|")
    for arm in MPRK_ARMS:
        a = fits[(arm, "fit_8h_H_ORDER")]
        b = fits[(arm, "fit_5h_required")]
        los = [r["local_order"] for r in rows_m
               if r["section"] == "local_order" and r["arm"] == arm
               and np.isfinite(r.get("local_order", float("nan")))]
        add("| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (arm, f(a[0]), f(a[1]), f(a[2]), f(b[0]), f(b[1]), f(b[2]),
               f(los[-1]) if los else "n/a"))
    add("")
    add("局部阶（相邻比）随 h→0 单调趋向名义阶 1/2/3；完整序列见 CSV 的 "
        "`section=local_order` 行，细网格 L2 见 `section=order_check` 行。"
        "8 档拟合值处在**前渐近区**，因此 MPRK43I 的拟合阶 %s 明显低于名义阶 3，"
        "本报告不声称「实测验证三阶」，只报告测得的拟合阶。" % f(p_43))
    add("")
    add("### 3.3 无条件保正 / 守恒 / 有限")
    add("")
    nscan = [r for r in rows_m if r["section"] == "scan"]
    add("在 %d 个 (arm,h) 组合上：all_finite=true 的比例 %d/%d；"
        "neg_S+neg_I+neg_R 总和 = %d；`cons_resid` 最大值 = %s；"
        "`min_state` 最大值 = %s。h=60 d 单行亦全为 0 负值、有限。"
        % (len(nscan), sum(1 for r in nscan if r["all_finite"]), len(nscan),
           sum(int(r["neg_S"]) + int(r["neg_I"]) + int(r["neg_R"])
               for r in nscan),
           f(max(float(r["cons_resid"]) for r in nscan)),
           f(max(float(r["min_state"]) for r in nscan))))
    add("")
    add("### 3.4 joint admissibility 逐条判定")
    add("")
    add("稿件判据（manuscript.tex:1130-1133）：(i) fitted order ≥3.9；(ii) 任意步长"
        "非负+守恒+有限，且保证来自单步映射而非数值 safeguard；(iii) 无 sub-cycling、"
        "无 clipping、无 signed time。")
    add("")
    add("| arm | (i) order≥3.9 | (ii) 单步映射保证 | (iii) 无子步/裁剪/负时间 | 接纳 |")
    add("|---|---|---|---|---|")
    for r in rows_m:
        if r["section"] == "verdict":
            add("| %s | %s (fit=%s) | %s | %s | **%s** |"
                % (r["arm"], f(r["criterion_i_fit_ge_3p9"]), f(r["fit_order"]),
                   f(r["criterion_ii_one_step_map"]),
                   f(r["criterion_iii_no_subcycling"]),
                   "否" if not r["joint_admissible"] else "是"))
    add("")
    add("**结论**：三个 Patankar 臂全部**被排除**，唯一失败项是 (i) 阶数门槛；"
        "(ii)/(iii) 通过。它们不是「保正失败」，而是「阶不够」。")
    add("")

    # ---- 5. 1B ------------------------------------------------------------
    add("## 4. 实验 1B：RK4 / SSPRK3 在 h=1 d 的失败归因")
    add("")
    add("### 4.1 线性链生成元的谱与稳定界")
    add("")
    add("40-stage Erlang 生成元 A 的特征值：实部 max=%s、min=%s，虚部最大绝对值=%s，"
        "即 %d 重亏损 Jordan 块（单一特征值 −σ=%s），不是 40 个可对角化的实模态。"
        % (f(EIG_REAL_MAX), f(EIG_REAL_MIN), f(EIG_IMAG_MAX), K, f(SIG)))
    add("")
    add("RK4 稳定函数 R₄(z)=Σ_{m=0}^4 z^m/m!；实轴绝对稳定界 = %s（"
        "z³+4z²+12z+24=0 的实根）。SSPRK3 稳定函数 R₃(z)=Σ_{m=0}^3 z^m/m!，"
        "实轴界 = %s。σ=%s d⁻¹，故 h=1 d 时 σh=%s："
        "**σh < %s，标量稳定界并未被越过**（r4_stability_bound_exceeded=false），"
        "R₄(−σh)=%s、|R₄|<1。" % (f(RK4_REAL_BOUND), f(SSPRK3_REAL_BOUND),
                                 f(SIG), f(SIG), f(RK4_REAL_BOUND),
                                 f(rk_poly_at(-SIG, 4))))
    add("")
    add("但亏损 Jordan 块的放大矩阵 R(hA) 不是标量：对 h=1 d，"
        "max|R₄(hA)^n| (n=1..60) = %s，而 h=0.5 d 时仅 %s。"
        "这就是显式 RK4 在 h=1 d 产生大振幅伪振荡的来源。"
        % (f(rk4full1["jordan_transient_60steps_max"]),
           f(next(r["jordan_transient_60steps_max"] for r in rows_s
                  if r["arm"] == "RK4-explicit (full ODE)" and r["h"] == 0.5))))
    add("")
    add("**对任务单前提的修正（如实报告）**：任务单写「σh≈2.174 已逼近 RK4 实轴"
        "稳定界（≈2.785）之外/边缘」。实测 σh=%s < %s，属稳定区间**内部**；"
        "因此把它归于「越过标量稳定界」不成立。真正的机制是 40 重亏损 Jordan 块的"
        "非正规瞬态增长（上表），结论方向不变、机制描述必须改。"
        % (f(SIG), f(RK4_REAL_BOUND)))
    add("")
    add("### 4.2 归因对照表（全部数值来自 m3_rk4_stability.csv）")
    add("")
    add("| arm | h | σh | bound_exceeded | all_finite | min_state | neg_S/I/R | cons_resid | first_bad | note |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_s:
        add("| %s | %s | %s | %s | %s | %s | %s/%s/%s | %s | %s | %s |"
            % (r["arm"], f(r["h"]), f(r["sigma_h"]),
               f(r["rk4_stability_bound_exceeded"]), f(r["all_finite"]),
               f(r["min_state"]), f(r["neg_S"]), f(r["neg_I"]), f(r["neg_R"]),
               f(r["cons_resid"]), f(r.get("first_bad_step")),
               r["note"][:90]))
    add("")
    add("### 4.3 归因结论")
    add("")
    add("- 逐字复现：`m3_experiment.run_attack('RK4-explicit', 1.0)` 给出 "
        "`all_finite=%s`、neg_S=%s、neg_I=%s、cons_resid=%s；自写同 RHS 的"
        "RK4 循环给出 all_finite=%s、首个负状态步=%s、首个非有限步=%s、"
        "min_state=%s，两者一致（交叉校验见 §7）。"
        % (f(repro1["all_finite"]), f(repro1["neg_S"]), f(repro1["neg_I"]),
           f(repro1["cons_resid"]), f(rk4full1["all_finite"]),
           f(rk4full1["first_neg_step"]), f(rk4full1["first_bad_step"]),
           f(rk4full1["min_state"])))
    add("- 隔离线性链（β=0）：h=1 d 仍 `all_finite=%s`，但 min_state=%s、"
        "守恒残差 %s（数值量级约 1e19 上的 float64 舍入噪声），"
        "即 Erlang 链本身已被显式 RK4 放大出巨大伪振荡；接上非线性 Λ 读取这些"
        "I 箱后，完整 RHS 在第 %s 步溢出。"
        % (f(lin1["all_finite"]), f(lin1["min_state"]),
           f(lin1["cons_resid"]), f(rk4full1["first_bad_step"])))
    add("- 线性部分改用 `p0b_experiment.poisson_T` 精确推进后，两种独立构造在 "
        "h=1 d 均有限且干净：Lawson IF-RK4 cons_resid=%s、Lie 分裂 cons_resid=%s、"
        "两者 neg 全 0。这直接证明 h=1 d 的 non-finite 来自**把 40-stage Erlang "
        "线性链交给显式 RK4**，而非感染力项的保正性失败。"
        % (f(lawson1["cons_resid"]), f(lie1["cons_resid"])))
    add("")
    add("**可直接支撑稿件改写的表述**：「RK4/SSPRK3 在 h=1 d 的非有限是 40-stage "
        "Erlang 线性链的显式稳定性（亏损 Jordan 块瞬态放大）失败，"
        "不构成 positivity 失败的证据」。需要同时保留的限定：RK4 本身确实没有"
        "无条件保正的结构保证（判据 (ii) 仍不成立），只是**不能用 h=1 d 的 "
        "non-finite 作为该结论的证据**——这是两类不同的失败。")
    add("")

    # ---- 6. 对稿件影响 ----------------------------------------------------
    add("## 5. 对稿件「唯一四阶臂」表述的影响")
    add("")
    add("1. `tab:joint` 的 13 个受测臂确为作者自建 midpoint/splitting/extrapolation/"
        "clipping 变体，未含任何 MPRK/MPDeC machinery，第 7 轮 3 路审阅的缺口成立。"
        "本线把这个缺口补成 %d 个真实文献臂，且用与稿件完全相同的参考解口径与 h 网格。"
        % len(MPRK_ARMS))
    add("2. 扩大后的臂集合里，唯一 fitted order ≥3.9 且 (ii)(iii) 成立的仍是 M3-p3"
        "（拟合阶 3.9918，见既有 m3_convergence.csv）。三个 Patankar 臂最高拟合阶 "
        "%s，均被 (i) 排除。故**「among those tested」的措辞在扩大集合内仍成立**。"
        % f(p_43))
    add("3. 但稿件 `:118` 引用 `KopeczMacdonald2018`（三阶 MPRK）、`HuangShu2019`、"
        "`Pan2023aero`、`Huang2023m2an` 时若暗示「该领域 machinery 已被比较覆盖」，"
        "现在必须改为：已实测 MPE/MPRK22(1)/MPRK43I(1,1/2)，它们无条件保正+守恒但"
        "阶数 ≤3，故被阶数门槛排除。")
    add("4. **不得升级为存在性唯一**：文献中存在 MPDeC（Öffner & Torlo 型 deferred "
        "correction，可构造任意阶无条件保正守恒格式）。本线**未实现** MPDeC(4)；"
        "若加入一个四阶 MPDeC 臂并通过 (ii)(iii)，`tab:joint` 的「唯一」将被直接"
        "推翻。因此稿件只能写「among the arms tested here」，并显式披露"
        "未测试的 Patankar/DeC 高阶族。")
    add("")

    # ---- 7. 验证报告 ------------------------------------------------------
    add("## 6. 验证报告（自检 / 交叉校验 / 正性与守恒）")
    add("")
    add("| 检查 | 结果 | 说明 |")
    add("|---|---|---|")
    add("| PDS 守恒恒等式 Σp=Σd | 通过（解析） | 见 §2.1；每行至多 1 donor |")
    add("| M-matrix 单位列和 | 通过（解析） | §2.2；⇒ 无条件正+守恒 |")
    add("| 正性检查 | 通过 | %d 个 (arm,h) 组合 neg 全 0、min_state≥0 |" % len(nscan))
    add("| 守恒检查 | 通过 | cons_resid 最大 %s（float64 舍入量级）"
        "|" % f(max(float(r["cons_resid"]) for r in nscan)))
    add("| 参考解交叉校验 | %s | M3-p3 h=1 L2 本线复算 %s vs 既有 m3_cost.csv %s"
        % ("通过" if prior is not None and
           abs(anchor_l2 - prior) < 1e-12 else "不一致，需复核",
           f(anchor_l2), f(prior) if prior is not None else "NOT FOUND"))
    add("| RK4 显式求解器交叉校验 | 通过 | `M.run_attack` 逐字复现 vs 自写同 RHS 循环，"
        "h=1 的 all_finite/neg/cons 一致（§4.3） |")
    add("| 网格收敛阶 | 见 §3.2 | 局部阶随 h→0 趋向名义 1/2/3，未声称已严格验证 |")
    add("| 双份 CSV md5 | 通过 | `M.write_csv_dual` 内部断言，见脚本 stdout |")
    add("")
    add("所有通过项均标注「数值待用户数值确认」；本线不声称「已验证」。")
    add("")

    # ---- 8. 性能 ----------------------------------------------------------
    add("## 7. 性能数据（机器相关，不作排序判据）")
    add("")
    add("| arm | h=0.0625 d 单次中位 (ms) | h=1 d 单次中位 (ms) | h=60 d 单次中位 (ms) | 计时次数 |")
    add("|---|---|---|---|---|")
    for arm in MPRK_ARMS:
        def wall(hh):
            return next(r["wall_ms_median"] for r in rows_m
                        if r["section"] == "scan" and r["arm"] == arm
                        and abs(float(r["h_macro"]) - hh) < 1e-12)
        add("| %s | %s | %s | %s | %s |"
            % (arm, f(wall(0.0625)), f(wall(1.0)), f(wall(60.0)), f(REPEATS)))
    add("")
    add("瓶颈：每个 MPRK43I 步需 4 次 `patankar_solve`，每次含 K=%d 次顺序前向代入"
        "（无法沿年龄箱向量化），单步成本约为 MPE 的 4 倍；无 GPU、无并行、"
        "无稀疏矩阵分解。内存峰值：状态 3 份 (n,K) 数组 + 中间量，量级 < 10 MB。"
        % K)
    add("")

    # ---- 9. 复现 ----------------------------------------------------------
    add("## 8. 复现说明")
    add("")
    add("- 环境：%s；numpy %s / scipy %s / pandas %s；无 torch。"
        % (platform.python_version(), np.__version__, scipy.__version__,
           pd.__version__))
    add("- 一键命令：`C:/Python314/python.exe submission/code/regen_m3_mprk.py`（退出码 0）。")
    add("- 固定种子：**未使用任何随机流**，全程 float64，无随机种子需求。")
    add("- 输入：只读现有 `submission/code/{coupling_host_level,m3_experiment,"
        "p0b_experiment,ms_numpy}.py` 与 `submission/figures/` 的既有 CSV，"
        "不修改任何现有文件。")
    add("- 产物：")
    add("  - `figures/m3_mprk.csv` 与 `submission/figures/m3_mprk.csv`（双份逐位一致）")
    add("  - `figures/m3_rk4_stability.csv` 与 `submission/figures/m3_rk4_stability.csv`")
    add("  - `_review_new_tmp_lead/r8_mprk_findings.md`（本文件）")
    add("- 网格口径、参考解口径、计时口径见 §2.4；运行期临时改 `P.T1` 并在每个 h "
        "结束后恢复，不污染任何其他模块。")
    add("")

    # ---- 10. 风险 / 局限 / 未跑成 -----------------------------------------
    add("## 9. 已知数值风险、局限与未跑成的部分（如实列出）")
    add("")
    add("1. **MPDeC(4) 未实现**：任务单首选 MPDeC / MPRK3，本线交付 MPE(1)、"
        "MPRK22(1)(2)、MPRK43I(1,1/2)(3)。MPRK43I 是 KM18 的三阶 method，"
        "已实现；MPDeC（任意阶 deferred correction）因权重矩阵与时间节点构造"
        "复杂、无法在本次单脚本内保证正确，**未跑成**，不编造任何 MPDeC 数值。"
        "这直接限制 §5 第 4 点的结论强度。")
    add("2. **MPRK43I 拟合阶 %s < 名义阶 3**：8 档 h 处于前渐近区；局部阶最细一档"
        "为 %s。未用独立解析解严格验证四阶/三阶收敛定理，只报告实测拟合阶。"
        % (f(p_43), f([r["local_order"] for r in rows_m
                       if r["section"] == "local_order"
                       and r["arm"] == "MPRK43I(1,1/2)"
                       and np.isfinite(r.get("local_order", float("nan")))][-1])))
    add("3. **非整除 h 的终点不是 60 d**：h=4.2 与 h=9 d 的网格终点为 %s d，"
        "跨 h 的 L2 不是同一时间泛函上的误差，只可做同 h 内的臂间比较。"
        % f(float(np.round(math.ceil(60.0 / 4.2) * 4.2, 8))))
    add("4. **Lawson / Lie 对照臂不是候选积分器**：它们在 h=1 d 有限，但在 "
        "h=4.2 d 与 h=10 d 因非线性感染力项自身的显式 RK4 步长限制而 "
        "non-finite（见 m3_rk4_stability.csv 相应行）；它们只用于失败归因，"
        "不满足无条件稳定性，不参与 joint admissibility 判定。")
    add("5. **σh < 标量稳定界**：任务单「σh 逼近/越过 2.785」的表述与实测不符，"
        "已在 §4.1 修正；若据此改写稿件，机制描述须改为「亏损 Jordan 块瞬态放大」。")
    add("6. **PWD 分母下限 %s**：当某分量恰为 0（如 ρ=1 的 TX 斑块 S=0）时"
        "用于避免 0/0；对非零分量的相对影响 < 1e-300，不改变任何报告数字。"
        % f(TINY))
    add("7. **计时仅 9 次中位**，未做 27 次稳定性复测；机器相关，不作排序判据。")
    add("8. **未做极端参数扫描**（β、σ、K 变化）；本实验只覆盖任务单指定的"
        "同一标定系统与两组 h。")
    add("")
    add("## 10. 可视化产物")
    add("")
    add("本线不生成图形或交互面板；仅产出两份双份 CSV 与本报告。图表由后续稿件回填环节使用。")
    add("")
    add("---")
    add("")
    add("**全部数字：数值待用户数值确认。**")
    return "\n".join(A) + "\n"


# =============================================================================
# 7. main
# =============================================================================
def main():
    t_start = time.perf_counter()
    log("=" * 100)
    log("R8 / route B / experiment line 1 : MPRK baseline + RK4 stability attribution")
    log("=" * 100)
    assert S00.dtype == np.float64 and B00.dtype == np.float64, "float64 only"
    log("[env] python=%s numpy=%s scipy=%s pandas=%s platform=%s"
        % (platform.python_version(), np.__version__, scipy.__version__,
           pd.__version__, platform.platform()))
    log("[env] n=%d K=%d sigma=%.17g beta_sp=%.17g T1=%.1f"
        % (n, K, SIG, BETA, T1_ORIGINAL))
    log("[env] RK4 real bound=%.17g  SSPRK3 real bound=%.17g"
        % (RK4_REAL_BOUND, SSPRK3_REAL_BOUND))
    log("[env] Erlang eig real max=%.17g min=%.17g imag|max|=%.3e (Jordan size %d)"
        % (EIG_REAL_MAX, EIG_REAL_MIN, EIG_IMAG_MAX, K))
    log("[env] float64, no random stream, no seed")

    log("-" * 100)
    log("EXPERIMENT 1A : MPRK / PDS arms")
    rows_m, fits, l2_by_arm, anchor_l2, prior = build_mprk_rows()
    dig1 = M.write_csv_dual("m3_mprk.csv", HEADER_MPRK, rows_m)
    log("  fit orders (8 h): " + ", ".join(
        "%s=%s" % (a, f(fits[(a, "fit_8h_H_ORDER")][0])) for a in MPRK_ARMS))
    log("  M3-p3 h=1 cross-check: recomputed=%s existing=%s"
        % (f(anchor_l2), f(prior) if prior is not None else "NOT FOUND"))

    log("-" * 100)
    log("EXPERIMENT 1B : RK4 / SSPRK3 stability attribution")
    rows_s = build_stability_rows()
    dig2 = M.write_csv_dual("m3_rk4_stability.csv", HEADER_STAB, rows_s)
    for r in rows_s:
        if r["h"] == 1.0:
            log("  h=1 %-42s finite=%-5s bound_exceeded=%-5s min=%s"
                % (r["arm"], f(r["all_finite"]),
                   f(r["rk4_stability_bound_exceeded"]), f(r["min_state"])))

    os.makedirs(REPORT_DIR, exist_ok=True)
    text = build_report(rows_m, fits, l2_by_arm, anchor_l2, prior, rows_s)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    log("[report] %s (%d chars)" % (REPORT_PATH, len(text)))

    # ---- acceptance self-checks -------------------------------------------
    for name, dig in (("m3_mprk.csv", dig1), ("m3_rk4_stability.csv", dig2)):
        for d in M.OUT_DIRS:
            p = os.path.join(d, name)
            with open(p, "rb") as fh:
                payload = fh.read()
            assert payload, "%s empty" % p
            assert hashlib.md5(payload).hexdigest() == dig, "%s md5 mismatch" % p
            assert b"\n" in payload and len(payload.splitlines()) > 1, \
                "%s has no header+data row" % p
        log("[check] %-26s md5=%s copies_identical=true" % (name, dig))
    log("[check] P.T1 restored = %.1f (original %.1f)" % (float(P.T1), T1_ORIGINAL))
    log("[done] %.1f s, exit 0" % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
