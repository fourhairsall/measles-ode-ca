# -*- coding: utf-8 -*-
# =============================================================================
# regen_second_system.py   (R8 / route B / EIC-scope "second instance" experiment)
# -----------------------------------------------------------------------------
# PURPOSE
#   Show, by ACTUAL NUMERICAL EXPERIMENT, that the one-step map used in the
#   manuscript -- a NON-NEGATIVE SQUARE predictor of the force of infection
#   Lambda(s) together with the EXACT phase-type (Erlang) propagator -- is not
#   tied to the single measles instance (52 patches / K=40 Erlang bins /
#   frequency-dependent standard incidence / V-derived phi kernel / tau* = ...).
#
#   A structurally distinguishable SECOND INSTANCE of the SAME model class is
#   built and pushed through the SAME construction.  Every parameter of that
#   second instance is ILLUSTRATIVE / SYNTHETIC.  It does not represent any
#   real disease, uses no literature values, and is NOT a biological prediction.
#   It exists only to probe the boundary of the model class.
#
#   ALL numbers in the CSV / log / report are produced by running this script.
#   They are marked "数值待用户数值确认" (pending user numerical confirmation)
#   per the team's hard rule: the team supplies method and code, the user must
#   re-compute with their own validated code before any number is trusted.
#
# MODEL CLASS (unchanged from the manuscript)
#   y = (S_i, I_{i,0..K-1}, R_i)_{i=1..P}
#   y' = A y + (L y)(B y)
#       A     : Erlang-K phase-type aging generator (Metzler; outflow into R)
#       L y   : Lambda(y), a STATE-LINEAR force of infection
#       B y   : (-S, +S e_0, 0)
#   One macro step:
#       Lambda(s) = lambda_0 (1 + d1 s + d2 s^2 + d3 s^3)^2 >= 0,  s in [0,h]
#       d1 = a1/2, d2 = (a2 - d1^2)/2, d3 = a3/2 - d1 d2,
#       a_j = lambda_j / (j! lambda_0),  lambda_j = d^j Lambda/dt^j (current state)
#       H(s) = int_0^s Lambda,  S(h) = S_0 exp(-H(h))
#       I(h) = I_0 T(h) + (S_0-S(h)) * sum_j p_j P_k(sigma(h-u_j))
#       R(h) = R_0 + removed + (S_0-S(h)) * sum_j p_j gamma_inc(K, sigma(h-u_j))
#   Everything is a non-negative number / non-negative matrix product.
#   No limiter, no clipping, no truncation, no CFL, no subcycling, no rng.
#
# -----------------------------------------------------------------------------
# CORRESPONDENCE WITH submission/code/coupling_host_level.py (READ ONLY)
# -----------------------------------------------------------------------------
#   chl.static_kernel(h,sigma,q,qh,K) ...... REUSED VERBATIM.  Lambda-independent
#        phase-type kernels; generic in (h, sigma, K).  No K=40 / 52 hardcode.
#   chl.poisson_T(sigma,h,K) .............. REUSED VERBATIM.  Generic in K.
#   chl.hazard_and_lambda(l0,d1,d2,d3,kernel,h) .. REUSED VERBATIM.  Fully
#        incidence-independent; positive-weight Gauss quadrature.
#   chl.kernel_rowsum_dev(kernel) ......... REUSED VERBATIM (verification).
#   chl._A_of(st,sigma,K) / chl._B_of(st,K) ...... REUSED VERBATIM.  Incidence-
#        independent linear operators; K is an explicit argument.
#   chl.fresh_closed(h,S,lam,sigma,K) / chl.w_uniform(h,sigma,K) .. REUSED
#        VERBATIM in the second-order EXP2 contrast arm.
#   chl.lam_derivs(S,I,R,W,N,phis,sigma,beta_sp,p,K) .. REUSED VERBATIM AS AN
#        ORACLE: with N == 1 the documented standard-incidence L becomes the
#        density-dependent (mass-action) L, and K=8 is passed explicitly.
#        lam_derivs2() below is an equivalent implementation with an explicit
#        _L2_of switch; the two are asserted equal to round-off in [V4].
#   chl.m3_square_coeffs(...) ... NOT callable here: internally calls
#        lam_derivs(..., p=p) WITHOUT forwarding K, so its default K=40 would
#        index column 39 of an 8-column I array.  m3_square_coeffs2 below is a
#        LINE-BY-LINE copy of its body, with the single change that `lam` is
#        supplied by the caller.  This is the "equivalent implementation"
#        permitted by the task; no original file is modified.
#   chl.host_step(...) ....... host_step2 below is a LINE-BY-LINE copy with the
#        single change that m3_square_coeffs2 is called (and N is the true
#        patch population for the conservation residual).
#   chl.build_infectious_kernel(...) ... build_kernel2 below keeps the SAME
#        normalisation (int_0^{tau*} phi dtau = Tg) but uses a DIFFERENT phi
#        SHAPE (symmetric tent / triangular), a DIFFERENT tau*, a DIFFERENT K,
#        and NO trajectory CSV (purely synthetic, deterministic).
#   lam_derivs_jet(...) ... GENERALISED (non-quadratic) jet recursion for the
#        saturating boundary probe.  At kappa=0 it reduces EXACTLY to the
#        production linear recursion (verified in [V4]).
#
# -----------------------------------------------------------------------------
# SYSTEMS
#   A : measles mainline (READ ONLY).  Structural facts read at RUNTIME from
#       coupling_host_level.py / its shipped CSV; not re-integrated here.
#   B : SECOND INSTANCE, inside the class.  K=8, P=12, tau*=6 d, sigma=8/6,
#       symmetric-tent phi, deterministic 12x12 ring row-stochastic W,
#       synthetic N/rho, DENSITY-DEPENDENT (mass-action) incidence
#           Lambda_i = beta * sum_j w_ij (sum_k phi_k I_jk)      [NO 1/N_j]
#       (contrast: mainline uses frequency-dependent standard incidence
#           Lambda_i = beta_sp * sum_j w_ij (sum_k phi_k I_jk) / N_j).
#       All values illustrative / synthetic.
#   C : BOUNDARY PROBE, OUTSIDE the quadratic class.  Same geometry as B but
#       SATURATING (task's example form)
#           Lambda_i = beta * sum_j w_ij Phi_j / (N_j + kappa * T_j),
#           Phi_j = sum_k phi_k I_jk,  T_j = sum_k I_jk.
#       Lambda is nonlinear in the state => y' is no longer quadratic.  The
#       Lambda jet is therefore computed with an EXACT generalised recursion.
#       Reported as a boundary probe only; NOT covered by Propositions 1-2 as
#       stated in the manuscript, and NOT a biological prediction.
#   NC1: NEGATIVE CONTROL, non-Metzler / non-phase-type generator.  1 patch,
#       K=3, a NEGATIVE off-diagonal generator entry.  expm(G h) has negative
#       entries => no non-negative Poisson/phase-type propagator exists =>
#       positivity premise of Proposition 1 fails.
#   NC2: NEGATIVE CONTROL, incidence built from TWO INDEPENDENT state-dependent
#       scalars => vector field is cubic, not quadratic.  1 patch, K=3.
#       The production jet identity lambda_j = L a_j fails quantitatively.
#
# -----------------------------------------------------------------------------
# ARMS (all advanced on the SAME macro grid h, ONE map application per interval,
#       NO adaptive subcycling, so the comparison is an EQUAL-h comparison)
#   M3-p3             : positive square + exact phase-type propagator (production)
#   EXP2-midpoint     : second-order, frozen-Lambda midpoint exponential step
#   Lagrange3-predictor : 3-point Lagrange polynomial Lambda(s) whose
#                       coefficients may be NEGATIVE + same propagator
#   RawCubic-Taylor   : same 4 Taylor coefficients as M3-p3 but as a RAW
#                       (unsigned) cubic polynomial + same propagator.  Exact
#                       same asymptotic order as M3-p3, but no sign guarantee.
#
# -----------------------------------------------------------------------------
# OUTPUTS (ONLY these files are created; no existing file is modified)
#   figures/second_system.csv
#   submission/figures/second_system.csv   (byte-identical, md5 asserted)
#   _review_new_tmp_lead/r8_second_system_run.log
#
# ONE-CLICK REPRODUCTION
#   C:/Python314/python.exe submission/code/regen_second_system.py
# =============================================================================

import csv
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent            # submission/code
SUB = HERE.parent                                # submission
ROOT = SUB.parent                                # repo root
sys.path.insert(0, str(HERE))
import coupling_host_level as chl                 # READ-ONLY reuse

DTYPE = np.float64
T1 = 60.0
H_ORDER = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
H_POS = [0.1, 1.0, 4.2, 9.0, 20.0, 60.0]
NEG_TOL = 1e-12

OUT_DIRS = [ROOT / "figures", SUB / "figures"]
LOG_PATH = ROOT / "_review_new_tmp_lead" / "r8_second_system_run.log"
CSV_NAME = "second_system.csv"

_LOG_LINES = []


def log(msg=""):
    s = str(msg)
    _LOG_LINES.append(s)
    print(s, flush=True)


def fmt(x, spec="%.10e"):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "true" if bool(x) else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return spec % v


# =============================================================================
# 0. System definitions  (all B/C/NC parameters ILLUSTRATIVE / SYNTHETIC)
# =============================================================================
def build_kernel2(K=8, tau_star=6.0, Tg=6.0, shape="tent"):
    """Equivalent of chl.build_infectious_kernel with a DIFFERENT phi SHAPE.

    Same normalisation int_0^{tau*} phi dtau = Tg (cholera-free: here Tg is
    just an illustrative time scale, NOT a generation interval of any disease).
    shape='tent' : symmetric triangular / unimodal kernel phi(tau) proportional
                   to max(0, 1 - |tau - tau*/2| / (tau*/2)).
    No random stream, no CSV, fully deterministic float64.
    """
    K = int(K)
    dtau = float(tau_star) / K
    tc = (np.arange(K, dtype=DTYPE) + 0.5) * dtau
    half = 0.5 * float(tau_star)
    if shape == "tent":
        raw = np.maximum(0.0, 1.0 - np.abs(tc - half) / half)
    elif shape == "uniform":
        raw = np.ones(K, dtype=DTYPE)
    else:
        raise ValueError(shape)
    scale = float(Tg) / float(raw.sum() * dtau)
    phis = np.asarray(raw * scale, dtype=DTYPE)
    sigma = K / float(tau_star)
    return dict(K=K, tau_star=float(tau_star), dtau=dtau, Tg=float(Tg),
                phis=phis, sigma=float(sigma), shape=shape, tc=tc,
                sum_phi_dtau=float(phis.sum() * dtau),
                max_phi=float(phis.max()), min_phi=float(phis.min()))


def build_W_ring(P=12, ell=2.0):
    """Deterministic, symmetric, row-stochastic 12-patch ring mixing matrix.

    G_ij = exp(-d_ring(i,j)/ell), d_ring = min(|i-j|, P-|i-j|).
    No random stream.  All entries > 0, rows sum to 1.
    """
    idx = np.arange(P, dtype=DTYPE)
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, P - d)
    G = np.exp(-d / float(ell))
    return np.asarray(G / G.sum(axis=1, keepdims=True), dtype=DTYPE)


def build_system_B():
    K, P = 8, 12
    ker = build_kernel2(K=K, tau_star=6.0, Tg=6.0, shape="tent")
    W = build_W_ring(P=P, ell=2.0)
    N = np.asarray(1.0 + 0.1 * np.arange(P), dtype=DTYPE)          # 1.0 .. 2.1
    rho = np.asarray(0.02 * (1.0 + 0.5 * np.cos(2.0 * np.pi * np.arange(P) / P)),
                     dtype=DTYPE)
    beta = 0.25                                                      # illustrative
    sysd = dict(name="B", K=K, P=P, W=W, N=N, rho=rho, beta=beta,
                phis=ker["phis"], sigma=ker["sigma"], tau_star=ker["tau_star"],
                dtau=ker["dtau"], Tg=ker["Tg"], ker=ker,
                incidence="mass_action (density-dependent), Lambda=beta*W@Phi",
                illustrative=True)
    return sysd


def build_system_C(kappa=0.6):
    """Boundary probe: same B geometry, SATURATING incidence, kappa illustrative."""
    s = dict(build_system_B())
    s["name"] = "C"
    s["kappa"] = float(kappa)
    s["incidence"] = ("saturating, Lambda=beta*W@(Phi/(N+kappa*TotalI)) "
                      "[NONLINEAR in state; outside quadratic class]")
    return s


def initial_state(s):
    N, rho, K = s["N"], s["rho"], s["K"]
    S0 = np.asarray((1.0 - rho) * N, dtype=DTYPE)
    I0 = np.asarray(rho * N, dtype=DTYPE)
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K)).astype(DTYPE)
    return S0, bins0, R0


# ---------------------------------------------------------------------------
# Force of infection.  mode is explicit so the (c) structural change is visible.
# ---------------------------------------------------------------------------
def lambda_of(S, I, s, mode="mass_action", kappa=None):
    """Lambda(y) at the current state.  S is unused (kept for signature parity
    with the production L, which is likewise S-independent)."""
    Phi = I @ s["phis"]
    if mode == "mass_action":                       # B: NO 1/N_j  (new form)
        return s["beta"] * (s["W"] @ Phi)
    if mode in ("standard", "linear_sat0"):        # mainline form / kappa=0 limit
        return s["beta"] * (s["W"] @ (Phi / s["N"]))
    if mode == "saturating":                        # C: NONLINEAR (boundary probe)
        T = I.sum(axis=1)
        return s["beta"] * (s["W"] @ (Phi / (s["N"] + float(kappa) * T)))
    raise ValueError(mode)


def _L2_of(st, s, mode="mass_action"):
    """Equivalent of chl._L_of with an explicit incidence switch.  `st` is a
    state tuple (S, I, R); only its I block enters Lambda for a state-LINEAR L."""
    Phi = st[1] @ s["phis"]
    if mode == "mass_action":
        return s["beta"] * (s["W"] @ Phi)
    if mode in ("standard", "linear_sat0"):
        return s["beta"] * (s["W"] @ (Phi / s["N"]))
    raise ValueError("nonlinear incidence has no state-linear L: %r" % mode)


# =============================================================================
# 1. Equivalent implementations (K-explicit) of the production jet / step
# =============================================================================
def lam_derivs2(S, I, R, s, mode="mass_action", p=3):
    """LINE-BY-LINE equivalent of chl.lam_derivs with explicit K and _L2_of.

    chl.lam_derivs:  a = [y]; Lc = [L(a0)]; lam = [Lc0]
      for m in range(p):
        acc = _A_of(a[m]) + sum_j C(m,j) * Lc[j] * _B_of(a[m-j])
        a.append(acc); Lc.append(L(acc)); lam.append(Lc[-1])
    Here _A_of/_B_of are the ORIGINAL chl functions (reused verbatim) and only
    _L_of -> _L2_of changes.  This exact-Taylor recursion is valid ONLY when
    Lambda = L y is state-LINEAR (the quadratic model class).
    """
    K = s["K"]
    a = [(S.copy(), I.copy(), R.copy())]
    Lc = [_L2_of(a[0], s, mode)]
    lam = [Lc[0].copy()]
    for m in range(p):
        acc = chl._A_of(a[m], s["sigma"], K)
        for j in range(m + 1):
            c = math.comb(m, j) * Lc[j]
            bv = chl._B_of(a[m - j], K)
            acc = (acc[0] + c * bv[0],
                   acc[1] + c[:, None] * bv[1],
                   acc[2] + c * bv[2])
        a.append(acc)
        Lc.append(_L2_of(acc, s, mode))
        lam.append(Lc[-1].copy())
    return lam


def m3_square_coeffs2(lam):
    """LINE-BY-LINE copy of the body of chl.m3_square_coeffs (which cannot be
    called here because it does not forward K to lam_derivs).  `lam` is the
    list [lambda_0..lambda_p]; everything else is identical to the original."""
    l0 = lam[0]
    pos = l0 > 0
    l0safe = np.where(pos, l0, 0.0)
    inv = np.where(pos, 1.0 / np.where(pos, l0, 1.0), 0.0)
    a1 = lam[1] * inv
    a2 = 0.5 * lam[2] * inv
    a3 = (lam[3] / 6.0) * inv
    d1 = a1 / 2.0
    d2 = (a2 - d1 ** 2) / 2.0
    d3 = a3 / 2.0 - d1 * d2
    return l0safe, d1, d2, d3, int((~pos).sum())


def host_step2(S, I, R, s, h, l0, d1, d2, d3, kernel, T):
    """LINE-BY-LINE copy of chl.host_step with the single change that the
    square coefficients come from m3_square_coeffs2, and N is the TRUE patch
    population for the conservation residual.  Everything else -- the
    log-domain incidence density, the Gauss hazard, the Poisson aging matrix,
    the sv_old removal vector -- is identical to the production step and uses
    chl.hazard_and_lambda / chl.static_kernel / chl.poisson_T verbatim."""
    hz = chl.hazard_and_lambda(l0, d1, d2, d3, kernel, h)
    H, Hh, lam_u = hz["H"], hz["Hh"], hz["lam_u"]

    logS = np.where(S > 0, np.log(np.where(S > 0, S, 1.0)), -np.inf)
    log_lam = (np.log(np.where(l0 > 0, l0, 1e-300))[:, None]
               + 2.0 * np.log(np.where(np.abs(hz["Pu"]) > 0,
                                       np.abs(hz["Pu"]), 1e-300)))
    log_g = log_lam + logS[:, None] - H
    mx = np.max(log_g, axis=1)
    bad = ~np.isfinite(mx)
    mx_safe = np.where(bad, 0.0, mx)
    with np.errstate(all="ignore"):
        ex = np.exp(np.where(np.isfinite(log_g), log_g - mx_safe[:, None], -np.inf))
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
    Iold = I @ T.T
    removed = I @ kernel["sv_old"]
    I_new = Iold + Ifresh
    R_new = R + removed + toR

    Itot = I_new.sum(axis=1)
    cons = float(np.max(np.abs(S_new + Itot + R_new - s["N"])))
    stat = dict(cons_resid=cons, min_S=float(S_new.min()),
                min_lam_u=float(np.min(lam_u)), min_pj=float(np.min(pj)),
                min_H=float(min(np.min(H), np.min(Hh))),
                min_toR=float(np.min(toR)), n_Zzero=int((Z <= 0).sum()))
    return S_new, I_new, R_new, stat


# ---------------------------------------------------------------------------
# Generalized (NON-quadratic) Lambda jet for the saturating boundary probe.
# ---------------------------------------------------------------------------
def lam_derivs_jet(S, I, R, s, kappa, p=3):
    """EXACT Taylor coefficients lambda_j = d^j Lambda/dt^j for the SATURATING
    incidence  Lambda_i = beta * (W @ (Phi ./ (N + kappa*TotalI)))_i.

    The production recursion assumes Lambda = L y is state-linear, so
    lambda_j = L a_j.  For a nonlinear Lambda the jet must be composed with the
    state jet.  Work in Taylor COEFFICIENTS a_m = y^{(m)}/m!:
        a_{m+1} = [A a_m + sum_{j=0}^m ell_j (B a_{m-j})] / (m+1)
        ell_m   = Lambda Taylor coefficient = lambda_m / m!
    With u_m = a_m[I] @ phis  (Phi jet), v_m = a_m[I].sum(1) (TotalI jet),
    d_0 = N + kappa v_0, d_m = kappa v_m (m>=1), the quotient jet q solves
        q_0 = u_0/d_0 ; q_m = (u_m - sum_{j=1}^m q_{m-j} d_j)/d_0
    and ell_m = beta * (W @ q_m).

    AT kappa = 0 this reduces EXACTLY to the production linear recursion
    (verified numerically in [V4]).  This routine is a BOUNDARY PROBE; it is
    NOT part of the manuscript's stated model class.
    """
    K = s["K"]
    W, N, phis, beta, sigma = s["W"], s["N"], s["phis"], s["beta"], s["sigma"]
    kap = float(kappa)
    a = [(S.copy(), I.copy(), R.copy())]
    u = [a[0][1] @ phis]
    v = [a[0][1].sum(axis=1)]
    d0 = N + kap * v[0]
    ell = []
    q = []
    for m in range(p + 1):
        if m == 0:
            q.append(u[0] / d0)
        else:
            acc = np.zeros_like(u[0])
            for j in range(1, m + 1):
                acc = acc + q[m - j] * (kap * v[j])
            q.append((u[m] - acc) / d0)
        ell.append(beta * (W @ q[m]))
        if m < p:
            ac = chl._A_of(a[m], sigma, K)
            for j in range(m + 1):
                bv = chl._B_of(a[m - j], K)
                ac = (ac[0] + ell[j] * bv[0],
                      ac[1] + ell[j][:, None] * bv[1],
                      ac[2] + ell[j] * bv[2])
            an = (ac[0] / (m + 1.0), ac[1] / (m + 1.0), ac[2] / (m + 1.0))
            a.append(an)
            u.append(an[1] @ phis)
            v.append(an[1].sum(axis=1))
    return [math.factorial(j) * ell[j] for j in range(p + 1)]


# ---------------------------------------------------------------------------
# Signed-polynomial propagator (for the negative-coefficient contrast arms).
# Mirrors m3_experiment.phase_step(poly=...) : an ARBITRARY polynomial Lambda(s)
# is integrated with the SAME phase-type propagator, but g(u) is allowed to be
# SIGNED, so p_j and hence I_new may become negative.  No clipping is applied.
# ---------------------------------------------------------------------------
def host_step_poly(S, I, R, s, h, poly, kernel, T):
    m = poly.shape[1]
    u = kernel["u"]
    powers = np.arange(m, dtype=DTYPE)[:, None]
    lam_u = poly @ (u[None, :] ** powers)                       # (P,q)
    Vint = (u[None, :] ** (powers + 1.0)) / (powers + 1.0)      # (m,q)
    H = poly @ Vint                                             # (P,q)
    hp = np.asarray(h, dtype=DTYPE) ** (powers.ravel() + 1.0) / (powers.ravel() + 1.0)
    Hh = poly @ hp                                              # (P,)
    with np.errstate(all="ignore"):
        g = lam_u * S[:, None] * np.exp(-H)
    wg = kernel["w"][None, :] * g
    Z = wg.sum(1)
    den = np.where(np.abs(Z) > 0, Z, 1.0)
    pj = np.where(np.abs(Z[:, None]) > 0, wg / den[:, None],
                  kernel["w"][None, :] / kernel["w"].sum())
    S_new = S * np.exp(-Hh)
    delivered = S - S_new
    Ifresh = delivered[:, None] * (pj @ kernel["Pmat"])
    toR = delivered * (pj @ kernel["surv"])
    Iold = I @ T.T
    removed = I @ kernel["sv_old"]
    I_new = Iold + Ifresh
    R_new = R + removed + toR
    cons = float(np.max(np.abs(S_new + I_new.sum(1) + R_new - s["N"])))
    stat = dict(cons_resid=cons, min_S=float(S_new.min()),
                min_lam_u=float(np.min(lam_u)), min_pj=float(np.min(pj)),
                min_lam_poly=float(np.min(lam_u)),
                min_toR=float(np.min(toR)), n_Zzero=int((np.abs(Z) <= 0).sum()))
    return S_new, I_new, R_new, stat


# ---------------------------------------------------------------------------
# Second-order frozen-Lambda EXP2 step (reuses chl.fresh_closed / w_uniform)
# ---------------------------------------------------------------------------
def exp2_frozen(S, I, R, s, h, lam, mode="closed"):
    K = s["K"]
    T = chl.poisson_T(s["sigma"], h, K=K)
    Iold = I @ T.T
    removed = I.sum(1) - Iold.sum(1)
    S_new = S * np.exp(-lam * h)
    delivered = S - S_new
    if mode == "closed":
        if np.any(s["sigma"] - lam <= 0):
            # closed form has no continuation at sigma <= Lambda; report via
            # fallback and count it (never silently hide an undefined branch)
            If = delivered[:, None] * chl.w_uniform(h, s["sigma"], K=K)[None, :]
            used = "static_fallback"
        else:
            If = chl.fresh_closed(h, S, lam, s["sigma"], K=K)
            used = "closed"
    else:
        If = delivered[:, None] * chl.w_uniform(h, s["sigma"], K=K)[None, :]
        used = "static"
    toR = delivered - If.sum(1)
    return S_new, Iold + If, R + removed + toR, used


def lagrange3_poly(S, I, R, s, h, mode):
    """3-point (quadratic) Lagrange Lambda(s) from half/full EXP2-midpoint
    predictors -- the manuscript's Lagrange3-predictor arm.  Coefficients are
    SIGNED; Lambda(s) may be negative on [0,h]."""
    lam0 = lambda_of(S, I, s, mode)
    _, bh, _, used1 = exp2_frozen(S, I, R, s, 0.5 * h, lam0, mode="closed")
    _, bf, _, used2 = exp2_frozen(S, I, R, s, h, lam0, mode="closed")
    lm = lambda_of(S, bh, s, mode)
    le = lambda_of(S, bf, s, mode)
    A = lam0
    B = (-3.0 * lam0 + 4.0 * lm - le) / h
    C = (2.0 * lam0 - 4.0 * lm + 2.0 * le) / (h * h)
    n_fb = int(used1 == "static_fallback") + int(used2 == "static_fallback")
    return np.stack([A, B, C], axis=1), n_fb


# =============================================================================
# 2. Reference solution (continuous ODE that the map converges to as h -> 0)
# =============================================================================
def rhs_factory(s, mode="mass_action", kappa=None):
    n, K = s["P"], s["K"]
    W, N, phis, beta, sigma = s["W"], s["N"], s["phis"], s["beta"], s["sigma"]

    def rhs(t, y):
        S = y[:n]
        I = y[n:n + n * K].reshape(n, K)
        R = y[n + n * K:]
        lam = lambda_of(S, I, s, mode, kappa)
        dS = -lam * S
        dI = -sigma * I
        dI[:, 1:] += sigma * I[:, :-1]
        dI[:, 0] += lam * S
        dR = sigma * I[:, K - 1]
        return np.concatenate([dS, dI.ravel(), dR])
    return rhs


def reference_solution(s, method="DOP853", rtol=1e-10, atol=1e-12,
                       mode="mass_action", kappa=None):
    S0, B0, R0 = initial_state(s)
    y0 = np.concatenate([S0, B0.ravel(), R0]).astype(DTYPE)
    sol = solve_ivp(rhs_factory(s, mode, kappa), (0.0, T1), y0,
                    method=method, rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError("%s reference failed: %s" % (method, sol.message))
    return sol, dict(nfev=int(sol.nfev), status=int(sol.status))


def ref_at(sol, s, grid):
    """Dense-output evaluation at EXACTLY the grid times (no resampling of a
    discrete solution, no interpolation of a coarse trajectory)."""
    n, K = s["P"], s["K"]
    Y = sol.sol(grid).T                                   # (nt, n + nK + n)
    S = Y[:, :n] / s["N"]
    I = Y[:, n:n + n * K].reshape(len(grid), n, K).sum(2) / s["N"]
    R = Y[:, n + n * K:] / s["N"]
    return dict(S=S, I=I, R=R)


def rel_l2(series, ref):
    num = np.concatenate([(series[v] - ref[v]).ravel() for v in ("S", "I", "R")])
    den = np.concatenate([ref[v].ravel() for v in ("S", "I", "R")])
    return float(np.linalg.norm(num) / np.linalg.norm(den))


def max_abs(series, ref):
    return float(max(np.max(np.abs(series[v] - ref[v])) for v in ("S", "I", "R")))


def loglog_fit(hs, ds):
    hs_a = np.asarray(hs, dtype=DTYPE)
    ds_a = np.asarray(ds, dtype=DTYPE)
    keep = (ds_a > 0) & np.isfinite(ds_a) & (hs_a > 0)
    hs_a, ds_a = hs_a[keep], ds_a[keep]
    if len(hs_a) < 2:
        return float("nan"), float("nan"), int(len(hs_a))
    x, y = np.log(hs_a), np.log(ds_a)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2), int(len(hs_a))


# =============================================================================
# 3. Grids
# =============================================================================
def conv_grid(h):
    """Nested grid ending EXACTLY at T1 (all H_ORDER divide 60 in binary)."""
    g = np.round(np.arange(0.0, T1 + 0.5 * h, h), 8)
    if abs(g[-1] - T1) > 1e-9:
        raise RuntimeError("conv_grid h=%g does not land on T1" % h)
    return g


def pos_grid(h):
    """Grid for the positivity scan: equal steps h plus a SHORTER final step so
    the horizon is exactly T1 even when h does not divide 60 (h=4.2, 9)."""
    g = np.round(np.arange(0.0, T1, h), 8)
    if g[-1] < T1 - 1e-9:
        g = np.append(g, T1)
    return g


# =============================================================================
# 4. Uniform driver -- one map application per grid interval, EQUAL h
# =============================================================================
def run_grid(s, arm, h, grid, mode="mass_action", kappa=None, jet_exact=False):
    S, I, R = initial_state(s)
    K = s["K"]
    nt = len(grid)
    S_rec = np.zeros((nt, s["P"]), dtype=DTYPE)
    I_rec = np.zeros_like(S_rec)
    R_rec = np.zeros_like(S_rec)
    S_rec[0], I_rec[0], R_rec[0] = S / s["N"], I.sum(1) / s["N"], R / s["N"]

    minS, minIb, minR = float(S.min()), float(I.min()), float(R.min())
    negS = int((S < -NEG_TOL).sum())
    negIb = int((I < -NEG_TOL).sum())
    negR = int((R < -NEG_TOL).sum())
    all_finite = True
    mass0 = float(S.sum() + I.sum() + R.sum())
    maxdrift = 0.0
    lam0_min = float("inf")
    n_l0_nonpos = 0
    n_l0_zero = 0
    min_lam_poly = float("inf")
    min_pj = float("inf")
    n_fallback = 0
    max_cons = 0.0

    for k in range(nt - 1):
        dt = float(grid[k + 1] - grid[k])
        lam0 = lambda_of(S, I, s, mode, kappa)
        lam0_min = min(lam0_min, float(np.min(lam0)))
        n_l0_nonpos += int((lam0 <= 0.0).sum())
        n_l0_zero += int((lam0 == 0.0).sum())

        kernel = chl.static_kernel(dt, s["sigma"], K=K)
        T = chl.poisson_T(s["sigma"], dt, K=K)

        if arm == "M3-p3":
            lam = (lam_derivs_jet(S, I, R, s, kappa, p=3) if jet_exact
                   else lam_derivs2(S, I, R, s, mode, p=3))
            l0, d1, d2, d3, nnp = m3_square_coeffs2(lam)
            n_l0_nonpos += 0                      # counted from true lambda above
            S, I, R, st = host_step2(S, I, R, s, dt, l0, d1, d2, d3,
                                     kernel, T)
            min_lam_poly = min(min_lam_poly, float(st["min_lam_u"]))
            min_pj = min(min_pj, float(st["min_pj"]))
        elif arm == "RawCubic-Taylor":
            lam = lam_derivs2(S, I, R, s, mode, p=3)
            poly = np.stack([lam[m] / math.factorial(m) for m in range(4)],
                            axis=1)
            S, I, R, st = host_step_poly(S, I, R, s, dt, poly, kernel, T)
            min_lam_poly = min(min_lam_poly, float(st["min_lam_poly"]))
            min_pj = min(min_pj, float(st["min_pj"]))
        elif arm == "Lagrange3-predictor":
            poly, n_fb = lagrange3_poly(S, I, R, s, dt, mode)
            n_fallback += n_fb
            S, I, R, st = host_step_poly(S, I, R, s, dt, poly, kernel, T)
            min_lam_poly = min(min_lam_poly, float(st["min_lam_poly"]))
            min_pj = min(min_pj, float(st["min_pj"]))
        elif arm == "EXP2-midpoint":
            lam0v = lambda_of(S, I, s, mode, kappa)
            _, bh, _, used1 = exp2_frozen(S, I, R, s, 0.5 * dt, lam0v,
                                          mode="closed")
            n_fallback += int(used1 == "static_fallback")
            lam_mid = lambda_of(S, bh, s, mode, kappa)
            S, I, R, used2 = exp2_frozen(S, I, R, s, dt, lam_mid, mode="closed")
            n_fallback += int(used2 == "static_fallback")
            st = dict(min_lam_poly=float("inf"), min_pj=float("inf"))
            min_lam_poly = min(min_lam_poly, float(np.min(lam0v)),
                               float(np.min(lam_mid)))
        else:
            raise ValueError(arm)

        max_cons = max(max_cons,
                       float(np.max(np.abs(S + I.sum(1) + R - s["N"]))))
        fin = (bool(np.all(np.isfinite(S))) and bool(np.all(np.isfinite(I)))
               and bool(np.all(np.isfinite(R))))
        all_finite = all_finite and fin
        minS = min(minS, float(S.min()))
        minIb = min(minIb, float(I.min()))
        minR = min(minR, float(R.min()))
        negS += int((S < -NEG_TOL).sum())
        negIb += int((I < -NEG_TOL).sum())
        negR += int((R < -NEG_TOL).sum())
        mass = float(S.sum() + I.sum() + R.sum())
        maxdrift = max(maxdrift, abs(mass - mass0))

        S_rec[k + 1], I_rec[k + 1], R_rec[k + 1] = (S / s["N"], I.sum(1) / s["N"],
                                                   R / s["N"])

    return dict(t=np.asarray(grid, dtype=DTYPE), S=S_rec, I=I_rec, R=R_rec,
                all_finite=bool(all_finite), min_S=float(minS),
                min_I_bins=float(minIb), min_R=float(minR),
                neg_S_cells=int(negS), neg_I_bin_cells=int(negIb),
                neg_R_cells=int(negR), max_abs_mass_drift=float(maxdrift),
                n_l0_nonpos=int(n_l0_nonpos), n_l0_zero=int(n_l0_zero),
                lambda0_min=float(lam0_min),
                min_lambda_poly=(float(min_lam_poly)
                                 if math.isfinite(min_lam_poly) else None),
                min_pj=(float(min_pj) if math.isfinite(min_pj) else None),
                n_static_fallback=int(n_fallback),
                max_cons_resid=float(max_cons))


# =============================================================================
# 5. Negative controls
# =============================================================================
def negative_control_NC1():
    """Non-Metzler generator: a NEGATIVE off-diagonal entry destroys the
    non-negative Poisson/phase-type propagator P_k(sigma h) = e^-mu mu^k/k!."""
    K, sigma, rho = 3, 0.8, 0.35          # illustrative / synthetic
    G = np.zeros((K, K), dtype=DTYPE)
    for k in range(K):
        G[k, k] = -sigma
        if k > 0:
            G[k, k - 1] = sigma
    G[0, 1] = -rho                        # NEGATIVE off-diagonal => NOT Metzler
    # keep diagonal consistent with the signed backflow (illustrative)
    G[0, 0] = -sigma
    min_offdiag = float(np.min(G - np.diag(np.diag(G))))
    n_neg_offdiag = int(((G - np.diag(np.diag(G))) < 0).sum())
    rows = []
    I0 = np.asarray([1.0, 0.0, 0.0], dtype=DTYPE)
    for h in (0.1, 1.0, 4.2, 9.0, 20.0, 60.0):
        E = expm(G * h)
        traj = E @ I0
        rows.append(dict(h=h, min_expm_entry=float(E.min()),
                         n_neg_expm_entries=int((E < -NEG_TOL).sum()),
                         min_traj_component=float(traj.min()),
                         n_neg_traj=int((traj < -NEG_TOL).sum()),
                         all_finite=bool(np.all(np.isfinite(E)))))
    # reference phase-type propagator for the SAME (K,sigma,h) family has NO
    # negative entries by construction -> structural contrast
    pos_min = float("inf")
    for h in (0.1, 1.0, 60.0):
        T = chl.poisson_T(sigma, h, K=K)
        pos_min = min(pos_min, float(T.min()))
    return dict(G=G, min_offdiag=min_offdiag, n_neg_offdiag=n_neg_offdiag,
                rows=rows, phasetype_T_min=pos_min, sigma=sigma, rho=rho, K=K)


def nc2_rhs_factory(s, c1, c2):
    """NC2: TWO independent state-dependent scalars u = Phi/N and w = R/N.
       Lambda = c1*u + c2*u*w  -> degree 2 in the state -> y' is CUBIC,
       NOT of the quadratic form A y + (L y)(B y)."""
    n, K = s["P"], s["K"]
    phis, sigma, N = s["phis"], s["sigma"], s["N"]

    def rhs(t, y):
        S = y[:n]
        I = y[n:n + n * K].reshape(n, K)
        R = y[n + n * K:]
        u = (I @ phis) / N
        w = R / N
        lam = c1 * u + c2 * u * w
        dS = -lam * S
        dI = -sigma * I
        dI[:, 1:] += sigma * I[:, :-1]
        dI[:, 0] += lam * S
        dR = sigma * I[:, K - 1]
        return np.concatenate([dS, dI.ravel(), dR])
    return rhs


def nc2_jet_exact(S, I, R, s, c1, c2, p=3):
    """Exact Lambda jet for NC2 (boundary evidence): ell = c1*u + c2*u*w with
    jet multiplication; state recursion in Taylor coefficients."""
    K, sigma, N, phis = s["K"], s["sigma"], s["N"], s["phis"]
    a = [(S.copy(), I.copy(), R.copy())]
    u = [(a[0][1] @ phis) / N]
    w = [a[0][2] / N]
    ell = []
    for m in range(p + 1):
        conv = np.zeros_like(u[0])
        for j in range(m + 1):
            conv = conv + u[j] * w[m - j]
        ell.append(c1 * u[m] + c2 * conv)
        if m < p:
            ac = chl._A_of(a[m], sigma, K)
            for j in range(m + 1):
                bv = chl._B_of(a[m - j], K)
                ac = (ac[0] + ell[j] * bv[0],
                      ac[1] + ell[j][:, None] * bv[1],
                      ac[2] + ell[j] * bv[2])
            an = (ac[0] / (m + 1.0), ac[1] / (m + 1.0), ac[2] / (m + 1.0))
            a.append(an)
            u.append((an[1] @ phis) / N)
            w.append(an[2] / N)
    return [math.factorial(j) * ell[j] for j in range(p + 1)]


def nc2_linear_assumed_jet(S, I, R, s, c1, p=3):
    """The PRODUCTION recursion applied verbatim with L = c1*u (the linear
    part of Lambda).  It would be exact only if Lambda were state-linear."""
    K, sigma, N, phis = s["K"], s["sigma"], s["N"], s["phis"]
    a = [(S.copy(), I.copy(), R.copy())]
    Lc = [c1 * ((a[0][1] @ phis) / N)]
    lam = [Lc[0].copy()]
    for m in range(p):
        ac = chl._A_of(a[m], sigma, K)
        for j in range(m + 1):
            c = math.comb(m, j) * Lc[j]
            bv = chl._B_of(a[m - j], K)
            ac = (ac[0] + c * bv[0],
                  ac[1] + c[:, None] * bv[1],
                  ac[2] + c * bv[2])
        a.append(ac)
        Lc.append(c1 * ((ac[1] @ phis) / N))
        lam.append(Lc[-1].copy())
    return lam


def negative_control_NC2():
    K, P = 3, 1
    ker = build_kernel2(K=K, tau_star=4.5, Tg=4.5, shape="uniform")
    s = dict(name="NC2", K=K, P=P,
             W=np.ones((1, 1), dtype=DTYPE), N=np.array([1.0], dtype=DTYPE),
             rho=np.array([0.05], dtype=DTYPE), beta=0.0,
             phis=ker["phis"], sigma=ker["sigma"], tau_star=ker["tau_star"],
             dtau=ker["dtau"], Tg=ker["Tg"], ker=ker, illustrative=True)
    c1, c2 = 0.9, 1.4                    # illustrative / synthetic
    S0, B0, R0 = initial_state(s)

    lam_exact = nc2_jet_exact(S0, B0, R0, s, c1, c2, p=3)
    lam_linear = nc2_linear_assumed_jet(S0, B0, R0, s, c1, p=3)
    jet_mismatch = []
    for j in range(4):
        a = float(np.max(np.abs(lam_exact[j])))
        d = float(np.max(np.abs(lam_exact[j] - lam_linear[j])))
        jet_mismatch.append(dict(j=j, lam_exact=float(lam_exact[j][0]),
                                 lam_linear_assumed=float(lam_linear[j][0]),
                                 abs_diff=d,
                                 rel_diff=(d / a if a > 0 else float("nan"))))

    # true continuous ODE for NC2
    y0 = np.concatenate([S0, B0.ravel(), R0]).astype(DTYPE)
    sol = solve_ivp(nc2_rhs_factory(s, c1, c2), (0.0, T1), y0,
                    method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError("NC2 DOP853 failed: %s" % sol.message)

    def frac(grid):
        Y = sol.sol(grid).T
        n, K2 = s["P"], s["K"]
        return dict(S=Y[:, :n] / s["N"],
                    I=Y[:, n:n + n * K2].reshape(len(grid), n, K2).sum(2) / s["N"],
                    R=Y[:, n + n * K2:] / s["N"])

    def run_nc2(which):
        out = []
        for h in H_ORDER:
            g = conv_grid(h)
            S, I, R = initial_state(s)
            Srec = np.zeros((len(g), 1), dtype=DTYPE)
            Irec = np.zeros_like(Srec)
            Rrec = np.zeros_like(Srec)
            Srec[0] = S / s["N"]
            Irec[0] = I.sum(1) / s["N"]
            Rrec[0] = R / s["N"]
            for k in range(len(g) - 1):
                dt = float(g[k + 1] - g[k])
                kernel = chl.static_kernel(dt, s["sigma"], K=s["K"])
                T = chl.poisson_T(s["sigma"], dt, K=s["K"])
                if which == "exact_jet":
                    lam = nc2_jet_exact(S, I, R, s, c1, c2, p=3)
                else:
                    lam = nc2_linear_assumed_jet(S, I, R, s, c1, p=3)
                l0, d1, d2, d3, _ = m3_square_coeffs2(lam)
                S, I, R, _ = host_step2(S, I, R, s, dt, l0, d1, d2, d3,
                                        kernel, T)
                Srec[k + 1] = S / s["N"]
                Irec[k + 1] = I.sum(1) / s["N"]
                Rrec[k + 1] = R / s["N"]
            out.append(rel_l2(dict(S=Srec, I=Irec, R=Rrec), frac(g)))
        p, r2, npt = loglog_fit(H_ORDER, out)
        return dict(errors=out, fit_order=p, fit_R2=r2, n_points=npt)

    ex = run_nc2("exact_jet")
    li = run_nc2("linear_assumed")
    return dict(system=s, c1=c1, c2=c2, jet_mismatch=jet_mismatch,
                exact_jet=ex, linear_assumed=li,
                ref_nfev=int(sol.nfev))


# =============================================================================
# 6. CSV assembly
# =============================================================================
COLS = [
    "section", "system", "arm", "h", "h_fine", "key", "value", "unit",
    "L2_rel_err", "max_abs_err", "order_local", "fit_order", "fit_R2",
    "n_points", "d_full", "e_h_full", "L2_ref", "ratio_e_over_ref",
    "nested_ok", "all_finite", "min_S", "min_I_bins", "min_R",
    "neg_S_cells", "neg_I_bin_cells", "neg_R_cells", "max_abs_mass_drift",
    "max_cons_resid", "n_l0_nonpos", "n_l0_zero", "lambda0_min",
    "min_lambda_poly", "min_pj", "n_static_fallback", "ref_solver",
    "ref_max_abs_diff", "note",
]

ROWS = []


def add(**kw):
    row = {c: "" for c in COLS}
    for k, v in kw.items():
        if k not in row:
            raise KeyError("unknown column %r" % k)
        row[k] = fmt(v) if not isinstance(v, str) else v
    ROWS.append(row)


def cfg(system, key, value, unit, note=""):
    add(section="system_config", system=system, key=key,
        value=(fmt(value) if not isinstance(value, str) else value),
        unit=unit, note=note)


def main():
    log("=" * 100)
    log("R8 / ROUTE B / EIC-SCOPE : SECOND INSTANCE OF THE SAME MODEL CLASS")
    log("=" * 100)
    log("interpreter : %s" % sys.executable)
    log("numpy       : %s   scipy : %s" % (np.__version__, __import__("scipy").__version__))
    log("float64 only, NO random stream, NO seed, NO torch.  All outputs are NEW files.")
    log("")

    # ------------------------------------------------------------------ A
    log("[A] measles mainline (READ-ONLY structural facts, read at RUNTIME)")
    states, rho52 = chl.load_rho(chl.RHO_CSV)
    cfg("A", "K_bins", chl.K_BINS, "bins", "read from chl.K_BINS (production)")
    cfg("A", "n_patches", len(states), "patches", "len(chl.load_rho(...))")
    cfg("A", "TG", chl.TG, "d", "read from chl.TG")
    cfg("A", "EPS_V", chl.EPS_V, "1", "read from chl.EPS_V")
    cfg("A", "BETA_SP", chl.BETA_SP, "1/d", "read from chl.BETA_SP")
    cfg("A", "M3_ORDER", chl.M3_ORDER, "1", "read from chl.M3_ORDER")
    cfg("A", "Q_INCIDENCE", chl.Q_INCIDENCE, "nodes", "read from chl.Q_INCIDENCE")
    cfg("A", "Q_HAZARD", chl.Q_HAZARD, "nodes", "read from chl.Q_HAZARD")
    cfg("A", "incidence", "frequency-dependent standard incidence beta_sp*W@(Phi/N_j)",
        "-", "production form; (c) is changed in B")
    cfg("A", "phi_shape", "V(t)-derived from fig3_withinhost_traj.csv", "-",
        "production kernel from a shipped CSV")
    log("    K=%d patches=%d TG=%g EPS_V=%g BETA_SP=%g (数值待用户数值确认)"
        % (chl.K_BINS, len(states), chl.TG, chl.EPS_V, chl.BETA_SP))

    # ------------------------------------------------------------------ B
    B = build_system_B()
    log("")
    log("[B] SECOND INSTANCE (in-class, ILLUSTRATIVE / SYNTHETIC)")
    cfg("B", "K_bins", B["K"], "bins", "ILLUSTRATIVE; changed vs A (40 -> 8)")
    cfg("B", "n_patches", B["P"], "patches", "ILLUSTRATIVE; changed vs A (52 -> 12)")
    cfg("B", "tau_star", B["tau_star"], "d", "ILLUSTRATIVE; controls sigma")
    cfg("B", "sigma", B["sigma"], "1/d", "= K/tau_star")
    cfg("B", "dtau", B["dtau"], "d", "= tau_star/K")
    cfg("B", "Tg_scale", B["Tg"], "d", "ILLUSTRATIVE normalisation int phi dtau = Tg")
    cfg("B", "phi_shape", "symmetric tent (triangular/unimodal)", "-",
        "ILLUSTRATIVE; changed vs A V-derived shape")
    cfg("B", "phi_min", B["ker"]["min_phi"], "1", "all phi_k >= 0")
    cfg("B", "phi_max", B["ker"]["max_phi"], "1", "")
    cfg("B", "sum_phi_dtau", B["ker"]["sum_phi_dtau"], "d",
        "must equal Tg_scale by construction")
    cfg("B", "beta", B["beta"], "1/(d*unit)", "ILLUSTRATIVE")
    cfg("B", "W_type", "deterministic 12-patch ring, row-stochastic", "-",
        "ILLUSTRATIVE; no random stream; G_ij=exp(-d_ring/2)")
    cfg("B", "W_rowsum_max_dev",
        float(np.max(np.abs(B["W"].sum(1) - 1.0))), "1", "")
    cfg("B", "N_min", float(B["N"].min()), "units", "ILLUSTRATIVE synthetic population")
    cfg("B", "N_max", float(B["N"].max()), "units", "")
    cfg("B", "rho_min", float(B["rho"].min()), "1", "ILLUSTRATIVE seeding")
    cfg("B", "rho_max", float(B["rho"].max()), "1", "")
    cfg("B", "incidence", "mass_action (density-dependent) Lambda=beta*W@Phi", "-",
        "ILLUSTRATIVE; (c) CHANGED vs A: NO 1/N_j")
    cfg("B", "T1", T1, "d", "horizon")
    for i in range(B["K"]):
        cfg("B", "phi_k%02d" % i, float(B["phis"][i]), "1",
            "ILLUSTRATIVE tent kernel, bin centre %.4f d" % B["ker"]["tc"][i])
    log("    K=%d P=%d tau*=%.1f sigma=%.6f beta=%.2f incidence=mass_action"
        % (B["K"], B["P"], B["tau_star"], B["sigma"], B["beta"]))

    C = build_system_C(kappa=0.6)
    log("")
    log("[C] BOUNDARY PROBE (saturating incidence, OUTSIDE quadratic class)")
    cfg("C", "kappa", C["kappa"], "1", "ILLUSTRATIVE saturation coefficient")
    cfg("C", "incidence", "beta*W@(Phi/(N+kappa*TotalI))", "-",
        "NONLINEAR in state; Lambda jet computed by generalised exact recursion")
    cfg("C", "in_class", "false", "-",
        "NOT covered by Propositions 1-2 as stated; boundary probe only")
    cfg("C", "K_bins", C["K"], "bins", "same geometry as B")
    cfg("C", "n_patches", C["P"], "patches", "same geometry as B")
    for name in ("NC1", "NC2"):
        cfg(name, "purpose", "negative control / premise failure", "-",
            "ILLUSTRATIVE synthetic; not a biological prediction")
    cfg("NC1", "K_bins", 3, "bins", "ILLUSTRATIVE")
    cfg("NC1", "n_patches", 1, "patches", "ILLUSTRATIVE")
    cfg("NC1", "sigma", 0.8, "1/d", "ILLUSTRATIVE")
    cfg("NC1", "negative_offdiag_rho", 0.35, "1/d", "ILLUSTRATIVE; breaks Metzler")
    cfg("NC2", "K_bins", 3, "bins", "ILLUSTRATIVE")
    cfg("NC2", "n_patches", 1, "patches", "ILLUSTRATIVE")
    cfg("NC2", "tau_star", 4.5, "d", "ILLUSTRATIVE")
    cfg("NC2", "phi_shape", "uniform", "-", "ILLUSTRATIVE")
    cfg("NC2", "c1", 0.9, "1/d", "ILLUSTRATIVE linear coefficient")
    cfg("NC2", "c2", 1.4, "1/d", "ILLUSTRATIVE product coefficient (2nd scalar)")

    # ------------------------------------------------------------------ V
    log("")
    log("[V] VERIFICATION / UNIT TESTS (all 数值待用户数值确认)")
    S0, B0, R0 = initial_state(B)

    # V1 kernel row-sum identity
    kern_dev = {}
    for h in H_ORDER + H_POS:
        kern = chl.static_kernel(h, B["sigma"], K=B["K"])
        kern_dev[h] = chl.kernel_rowsum_dev(kern)
    worst = max(kern_dev.values())
    add(section="verification", key="V1_kernel_rowsum_max_dev", value=worst,
        note="max |sum_k P_k(mu_j)+gamma_inc(K,mu_j)-1| over all used h")
    add(section="verification", key="V1_kernel_Pmat_min",
        value=float(min(chl.static_kernel(h, B["sigma"], K=B["K"])["Pmat"].min()
                        for h in H_ORDER + H_POS)), note="must be >= 0")
    add(section="verification", key="V1_kernel_sv_old_min",
        value=float(min(chl.static_kernel(h, B["sigma"], K=B["K"])["sv_old"].min()
                        for h in H_ORDER + H_POS)), note="must be >= 0")
    log("    V1 phase-type row-sum max dev = %.3e ; Pmat/sv_old >= 0 : %s"
        % (worst, worst < 1e-12))

    # V2 removal identity  I.sum - (I@T.T).sum == I@sv_old
    dev2 = 0.0
    for h in (0.0625, 1.0, 9.0, 60.0):
        kern = chl.static_kernel(h, B["sigma"], K=B["K"])
        T = chl.poisson_T(B["sigma"], h, K=B["K"])
        Ix = np.abs(np.sin(np.outer(np.arange(B["P"], dtype=DTYPE) + 1.0,
                                    np.arange(B["K"], dtype=DTYPE) + 2.0))) * 3.0
        a = Ix.sum(1) - (Ix @ T.T).sum(1)
        b = Ix @ kern["sv_old"]
        dev2 = max(dev2, float(np.max(np.abs(a - b))))
    add(section="verification", key="V2_removal_identity_max_dev", value=dev2,
        note="Poisson propagator vs stable survival form")
    log("    V2 removal identity max dev = %.3e" % dev2)

    # V3 phi kernel normalisation / positivity
    add(section="verification", key="V3_sum_phi_dtau_minus_Tg",
        value=float(B["ker"]["sum_phi_dtau"] - B["Tg"]), note="must be ~0")
    add(section="verification", key="V3_min_phi", value=B["ker"]["min_phi"],
        note="must be >= 0")
    log("    V3 int phi dtau - Tg = %.3e ; min phi = %.3e"
        % (B["ker"]["sum_phi_dtau"] - B["Tg"], B["ker"]["min_phi"]))

    # V4 EQUIVALENT IMPLEMENTATION vs PRODUCTION ORACLE (chl.lam_derivs with
    #    N == 1 turns standard incidence into mass action; K forwarded explicitly)
    ones = np.ones(B["P"], dtype=DTYPE)
    lam_ours = lam_derivs2(S0, B0, R0, B, "mass_action", p=3)
    lam_oracle = chl.lam_derivs(S0, B0, R0, B["W"], ones, B["phis"],
                                B["sigma"], B["beta"], p=3, K=B["K"])
    dev4 = max(float(np.max(np.abs(lam_ours[j] - lam_oracle[j])))
               for j in range(4))
    add(section="verification", key="V4_lam_derivs2_vs_chl_lam_derivs",
        value=dev4, note="equivalent implementation vs production function (N=1)")
    log("    V4 lam_derivs2 vs chl.lam_derivs (N=1) max dev = %.3e" % dev4)

    # V5 square predictor reconstructs the 4 Taylor coefficients
    l0, d1, d2, d3, nnp = m3_square_coeffs2(lam_ours)
    P = np.stack([np.ones_like(l0), d1, d2, d3], axis=1)   # (P,4)
    Q = np.zeros((B["P"], 7), dtype=DTYPE)
    for i in range(4):
        for j in range(4):
            Q[:, i + j] += P[:, i] * P[:, j]
    recon = [math.factorial(j) * l0 * Q[:, j] for j in range(4)]
    dev5 = max(float(np.max(np.abs(recon[j] - lam_ours[j]))) for j in range(4))
    add(section="verification", key="V5_square_predictor_jet_max_dev",
        value=dev5, note="Lambda(s)=l0(1+d1 s+d2 s^2+d3 s^3)^2 reproduces lambda_0..3")
    add(section="verification", key="V5_n_nonpos_l0_at_t0", value=nnp,
        note="lambda_0>0 branch count at the initial state")
    log("    V5 square predictor jet max dev = %.3e (n_nonpos_l0=%d)" % (dev5, nnp))

    # V6 generalized jet at kappa=0 reduces to the linear production recursion
    lam_jet0 = lam_derivs_jet(S0, B0, R0, C, 0.0, p=3)
    lam_lin0 = lam_derivs2(S0, B0, R0, C, "standard", p=3)
    dev6 = max(float(np.max(np.abs(lam_jet0[j] - lam_lin0[j]))) for j in range(4))
    add(section="verification", key="V6_jet_kappa0_vs_linear", value=dev6,
        note="saturating jet at kappa=0 must equal production linear recursion")
    log("    V6 jet(kappa=0) vs linear recursion max dev = %.3e" % dev6)

    # ------------------------------------------------------------------ REF
    log("")
    log("[REF] continuous reference solves (dense_output=True)")
    solB, infoB = reference_solution(B, "DOP853", 1e-10, 1e-12, "mass_action")
    solBr, infoBr = reference_solution(B, "Radau", 1e-10, 1e-12, "mass_action")
    gfine = conv_grid(H_ORDER[-1])
    fB = ref_at(solB, B, gfine)
    fBr = ref_at(solBr, B, gfine)
    ref_cross = max(float(np.max(np.abs(fB[v] - fBr[v]))) for v in ("S", "I", "R"))
    add(section="verification", key="V7_DOP853_vs_Radau_max_abs_diff",
        value=ref_cross, note="same rtol/atol, dense output, finest grid")
    add(section="verification", key="V7_DOP853_nfev", value=infoB["nfev"], note="")
    add(section="verification", key="V7_Radau_nfev", value=infoBr["nfev"], note="")
    log("    V7 DOP853 vs Radau max|diff| = %.3e (nfev %d / %d)"
        % (ref_cross, infoB["nfev"], infoBr["nfev"]))

    solC, infoC = reference_solution(C, "DOP853", 1e-10, 1e-12,
                                     "saturating", C["kappa"])
    log("    C reference nfev=%d" % infoC["nfev"])

    # ------------------------------------------------------------------ 1
    log("")
    log("[1] CONVERGENCE (8 nested h, DOP853 rtol=1e-10 atol=1e-12 dense_output)")
    ARMS_B = ["M3-p3", "EXP2-midpoint", "Lagrange3-predictor", "RawCubic-Taylor"]
    conv = {}
    for arm in ARMS_B:
        errs, abss = [], []
        for h in H_ORDER:
            g = conv_grid(h)
            r = run_grid(B, arm, h, g, mode="mass_action")
            ref = ref_at(solB, B, g)
            e = rel_l2(r, ref)
            mx = max_abs(r, ref)
            errs.append(e)
            abss.append(mx)
            add(section="convergence", system="B", arm=arm, h=h,
                L2_rel_err=e, max_abs_err=mx,
                all_finite=r["all_finite"], min_I_bins=r["min_I_bins"],
                neg_I_bin_cells=r["neg_I_bin_cells"],
                max_abs_mass_drift=r["max_abs_mass_drift"],
                ref_solver="DOP853", note="数值待用户数值确认")
        orders = [float("nan")]
        for i in range(1, len(H_ORDER)):
            orders.append(math.log(errs[i - 1] / errs[i])
                          / math.log(H_ORDER[i - 1] / H_ORDER[i])
                          if (errs[i] > 0 and errs[i - 1] > 0) else float("nan"))
        # annotate local orders back onto the rows
        base = len(ROWS) - len(H_ORDER)
        for i, o in enumerate(orders):
            ROWS[base + i]["order_local"] = fmt(o, "%.6f")
        p, r2, npt = loglog_fit(H_ORDER, errs)
        add(section="convergence", system="B", arm=arm, h="",
            key="fit", fit_order=p, fit_R2=r2, n_points=npt,
            note="log-log fit over all 8 h; 数值待用户数值确认")
        conv[arm] = dict(errs=errs, orders=orders, fit=p, r2=r2, n=npt)
        log("    %-22s L2(h=1)=%.4e -> L2(h=1/128)=%.4e  fit_order=%.6f R2=%.8f"
            % (arm, errs[0], errs[-1], p, r2))

    # C boundary probe with EXACT jet
    errsC, abssC = [], []
    for h in H_ORDER:
        g = conv_grid(h)
        r = run_grid(C, "M3-p3", h, g, mode="saturating", kappa=C["kappa"],
                     jet_exact=True)
        ref = ref_at(solC, C, g)
        e = rel_l2(r, ref)
        errsC.append(e)
        abssC.append(max_abs(r, ref))
        add(section="convergence", system="C", arm="M3-p3(exact-jet)", h=h,
            L2_rel_err=e, max_abs_err=abssC[-1], all_finite=r["all_finite"],
            min_I_bins=r["min_I_bins"], neg_I_bin_cells=r["neg_I_bin_cells"],
            max_abs_mass_drift=r["max_abs_mass_drift"], ref_solver="DOP853",
            note="boundary probe, OUTSIDE quadratic class; 数值待用户数值确认")
    pC, r2C, nC = loglog_fit(H_ORDER, errsC)
    add(section="convergence", system="C", arm="M3-p3(exact-jet)", key="fit",
        fit_order=pC, fit_R2=r2C, n_points=nC,
        note="boundary probe only, NOT a Proposition claim; 数值待用户数值确认")
    log("    C  M3-p3(exact-jet)   L2(h=1)=%.4e -> %.4e  fit_order=%.6f R2=%.8f"
        % (errsC[0], errsC[-1], pC, r2C))

    # ------------------------------------------------------------------ 1b
    log("")
    log("[1b] REFERENCE-FREE SELF-CONVERGENCE (nested even indices, e_h=16/15*d_h)")
    self_runs = {}
    for h in H_ORDER:
        self_runs[h] = run_grid(B, "M3-p3", h, conv_grid(h), mode="mass_action")
    sc_rows = []
    for hc, hf in zip(H_ORDER[:-1], H_ORDER[1:]):
        tc = self_runs[hc]["t"]
        tf = self_runs[hf]["t"]
        even = tf[::2]
        nested = bool(len(even) == len(tc) and np.allclose(even, tc, atol=1e-9))
        if not nested:
            raise RuntimeError("grids not nested at h=%g / %g" % (hc, hf))
        se = {v: self_runs[hf][v][::2] for v in ("S", "I", "R")}
        d = rel_l2(self_runs[hc], se)
        e = d * 16.0 / 15.0
        l2ref = conv["M3-p3"]["errs"][H_ORDER.index(hc)]
        sc_rows.append(dict(hc=hc, hf=hf, d=d, e=e, l2ref=l2ref,
                            ratio=e / l2ref if l2ref > 0 else float("nan"),
                            nested=nested))
    for i, r in enumerate(sc_rows):
        o = (math.log(sc_rows[i - 1]["d"] / r["d"])
             / math.log(sc_rows[i - 1]["hc"] / r["hc"])) if i > 0 else float("nan")
        add(section="self_convergence", system="B", arm="M3-p3",
            h=r["hc"], h_fine=r["hf"], d_full=r["d"], e_h_full=r["e"],
            L2_ref=r["l2ref"], ratio_e_over_ref=r["ratio"],
            order_local=o, nested_ok=r["nested"],
            note="no reference, no interpolation; 数值待用户数值确认")
    pS, r2S, nS = loglog_fit([r["hc"] for r in sc_rows],
                             [r["d"] for r in sc_rows])
    add(section="self_convergence", system="B", arm="M3-p3", key="fit_full",
        fit_order=pS, fit_R2=r2S, n_points=nS,
        note="fit of d_full vs h_coarse; 数值待用户数值确认")
    log("    fit_full order=%.6f R2=%.8f n=%d ; e_h/L2_ref range %.5f .. %.5f"
        % (pS, r2S, nS,
           min(r["ratio"] for r in sc_rows), max(r["ratio"] for r in sc_rows)))

    # ------------------------------------------------------------------ 2
    log("")
    log("[2] UNCONDITIONAL POSITIVITY / CONSERVATION / FINITENESS (h scan)")
    POS_ARMS = ["M3-p3", "EXP2-midpoint", "Lagrange3-predictor",
                "RawCubic-Taylor"]
    pos_summary = {a: [] for a in POS_ARMS}
    for arm in POS_ARMS:
        for h in H_POS:
            g = pos_grid(h)
            r = run_grid(B, arm, h, g, mode="mass_action")
            note = "数值待用户数值确认"
            if r["n_l0_zero"] > 0:
                note += ("; lambda_0==0 occurred %d times: m3_square_coeffs2 "
                         "zeroes l0/inv -> d1=d2=d3=0 -> Lambda(s)=0 -> "
                         "pure aging step (identity on S), still positivity-preserving"
                         % r["n_l0_zero"])
            add(section="positivity", system="B", arm=arm, h=h,
                all_finite=r["all_finite"], min_S=r["min_S"],
                min_I_bins=r["min_I_bins"], min_R=r["min_R"],
                neg_S_cells=r["neg_S_cells"], neg_I_bin_cells=r["neg_I_bin_cells"],
                neg_R_cells=r["neg_R_cells"],
                max_abs_mass_drift=r["max_abs_mass_drift"],
                max_cons_resid=r["max_cons_resid"],
                n_l0_nonpos=r["n_l0_nonpos"], n_l0_zero=r["n_l0_zero"],
                lambda0_min=r["lambda0_min"],
                min_lambda_poly=r["min_lambda_poly"], min_pj=r["min_pj"],
                n_static_fallback=r["n_static_fallback"], note=note)
            pos_summary[arm].append(r)
        nneg = sum(int(x["neg_S_cells"] + x["neg_I_bin_cells"] + x["neg_R_cells"])
                   for x in pos_summary[arm])
        nfin = sum(1 for x in pos_summary[arm] if not x["all_finite"])
        log("    %-22s over 6 h : neg_cells=%d  non_finite_runs=%d"
            % (arm, nneg, nfin))

    # C positivity scan (boundary probe)
    for h in H_POS:
        g = pos_grid(h)
        r = run_grid(C, "M3-p3", h, g, mode="saturating", kappa=C["kappa"],
                     jet_exact=True)
        add(section="positivity", system="C", arm="M3-p3(exact-jet)", h=h,
            all_finite=r["all_finite"], min_S=r["min_S"],
            min_I_bins=r["min_I_bins"], min_R=r["min_R"],
            neg_S_cells=r["neg_S_cells"], neg_I_bin_cells=r["neg_I_bin_cells"],
            neg_R_cells=r["neg_R_cells"],
            max_abs_mass_drift=r["max_abs_mass_drift"],
            max_cons_resid=r["max_cons_resid"],
            n_l0_nonpos=r["n_l0_nonpos"], n_l0_zero=r["n_l0_zero"],
            lambda0_min=r["lambda0_min"], min_lambda_poly=r["min_lambda_poly"],
            note="boundary probe, OUTSIDE quadratic class; 数值待用户数值确认")

    # ------------------------------------------------------------------ 3
    log("")
    log("[3] CONTRAST / JOINT METRIC (same order x arbitrary-h structural guarantee)")
    for arm in POS_ARMS:
        uncond = all(x["all_finite"] and
                     (x["neg_S_cells"] + x["neg_I_bin_cells"] + x["neg_R_cells"]) == 0
                     for x in pos_summary[arm])
        add(section="joint_metric", system="B", arm=arm,
            key="order_fit", value=conv[arm]["fit"],
            note="log-log fit over the same 8 h (equal-h); 数值待用户数值确认")
        add(section="joint_metric", system="B", arm=arm,
            key="unconditional_positivity_over_6_h", value=uncond,
            note="h in {0.1,1,4.2,9,20,60} d; 数值待用户数值确认")
        add(section="joint_metric", system="B", arm=arm,
            key="L2_at_h1", value=conv[arm]["errs"][0],
            note="equal-h, no adaptive subcycling; 数值待用户数值确认")
        log("    %-22s fit_order=%.5f  uncond_positivity=%s  L2(h=1)=%.4e"
            % (arm, conv[arm]["fit"], uncond, conv[arm]["errs"][0]))

    # ------------------------------------------------------------------ 4
    log("")
    log("[4] NEGATIVE CONTROLS (systems OUTSIDE the class)")
    nc1 = negative_control_NC1()
    add(section="negative_control", system="NC1", key="min_offdiag_entry",
        value=nc1["min_offdiag"],
        note="NEGATIVE off-diagonal => generator is NOT Metzler/phase-type; "
             "Proposition 1's non-negative Poisson propagator premise fails")
    add(section="negative_control", system="NC1", key="n_negative_offdiag_entries",
        value=nc1["n_neg_offdiag"], note="")
    add(section="negative_control", system="NC1", key="phasetype_T_min_entry",
        value=nc1["phasetype_T_min"],
        note="min entry of the SAME-K chl.poisson_T family (>=0 by construction)")
    for r in nc1["rows"]:
        add(section="negative_control", system="NC1", key="expm(G*h)_trajectory",
            h=r["h"], value=r["min_traj_component"],
            min_I_bins=r["min_traj_component"],
            neg_I_bin_cells=r["n_neg_traj"],
            note="min expm entry=%.6e, n_neg_expm=%d, all_finite=%s; "
                 "a non-negative IC develops negative I components; "
                 "数值待用户数值确认"
                 % (r["min_expm_entry"], r["n_neg_expm_entries"],
                    r["all_finite"]))
    log("    NC1 min offdiag = %.3e (%d negative offdiag entries); "
        "phase-type T_min = %.3e" %
        (nc1["min_offdiag"], nc1["n_neg_offdiag"], nc1["phasetype_T_min"]))
    log("    NC1 trajectory min over h=0.1..60 : %.6e (negative components present)"
        % min(r["min_traj_component"] for r in nc1["rows"]))

    nc2 = negative_control_NC2()
    for m in nc2["jet_mismatch"]:
        add(section="negative_control", system="NC2",
            key="lambda_%d_paper_linear_vs_exact" % m["j"],
            value=m["rel_diff"], h="",
            note="lambda_exact=%.6e lambda_linear_assumed=%.6e abs_diff=%.6e; "
                 "identity lambda_j = L a_j fails because Lambda is nonlinear; "
                 "数值待用户数值确认"
                 % (m["lam_exact"], m["lam_linear_assumed"], m["abs_diff"]))
    add(section="negative_control", system="NC2", key="exact_jet_fit_order",
        value=nc2["exact_jet"]["fit_order"],
        fit_R2=nc2["exact_jet"]["fit_R2"], n_points=nc2["exact_jet"]["n_points"],
        note="boundary evidence: with an EXACT nonlinear jet the SAME propagator "
             "recovers 4th order; the paper's Propositions are NOT claimed here; "
             "数值待用户数值确认")
    add(section="negative_control", system="NC2",
        key="linear_assumed_fit_order",
        value=nc2["linear_assumed"]["fit_order"],
        fit_R2=nc2["linear_assumed"]["fit_R2"],
        n_points=nc2["linear_assumed"]["n_points"],
        note="production recursion applied verbatim outside its class: "
             "no 4th-order convergence to the true ODE; 数值待用户数值确认")
    for i, h in enumerate(H_ORDER):
        add(section="negative_control", system="NC2", key="L2_exact_jet", h=h,
            L2_rel_err=nc2["exact_jet"]["errors"][i],
            note="数值待用户数值确认")
        add(section="negative_control", system="NC2", key="L2_linear_assumed", h=h,
            L2_rel_err=nc2["linear_assumed"]["errors"][i],
            note="数值待用户数值确认")
    log("    NC2 lambda_1 jet mismatch rel = %.4f ; exact-jet fit_order=%.5f ; "
        "linear-assumed fit_order=%.5f"
        % (nc2["jet_mismatch"][1]["rel_diff"],
           nc2["exact_jet"]["fit_order"], nc2["linear_assumed"]["fit_order"]))

    # ------------------------------------------------------------------ CSV
    log("")
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(ROWS)
    payload = buf.getvalue().encode("utf-8")
    digests = []
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        with (d / CSV_NAME).open("wb") as f:
            f.write(payload)                 # SAME in-memory payload, twice
        digests.append(hashlib.md5((d / CSV_NAME).read_bytes()).hexdigest())
    if len(set(digests)) != 1:
        raise RuntimeError("md5 mismatch between the two CSV copies: %r" % digests)
    add(section="md5", key="csv_md5", value=digests[0],
        note="byte-identical payload written to both paths; asserted at runtime")
    # re-serialise with the md5 row included so the shipped CSV documents itself
    buf2 = io.StringIO(newline="")
    w2 = csv.DictWriter(buf2, fieldnames=COLS, lineterminator="\n")
    w2.writeheader()
    w2.writerows(ROWS)
    payload2 = buf2.getvalue().encode("utf-8")
    digests2 = []
    for d in OUT_DIRS:
        with (d / CSV_NAME).open("wb") as f:
            f.write(payload2)
        digests2.append(hashlib.md5((d / CSV_NAME).read_bytes()).hexdigest())
    if len(set(digests2)) != 1 or digests2[0] != hashlib.md5(payload2).hexdigest():
        raise RuntimeError("md5 mismatch on final payload: %r" % digests2)
    log("[CSV] rows=%d cols=%d bytes=%d md5=%s copies_identical=true"
        % (len(ROWS), len(COLS), len(payload2), digests2[0]))
    log("[CSV] %s" % " ; ".join(str(d / CSV_NAME) for d in OUT_DIRS))
    log("")
    log("[CROSS-CHECK] payload md5 (pre-md5-row) = %s" % digests[0])
    log("[CROSS-CHECK] final payload md5          = %s" % digests2[0])
    log("")
    log("ALL NUMBERS: 数值待用户数值确认 (pending user numerical confirmation).")
    log("System B/C/NC1/NC2 are ILLUSTRATIVE / SYNTHETIC; no biological prediction,")
    log("no literature parameter, no real-disease claim.  Propositions 1-2 are")
    log("stated for the quadratic class; C/NC1/NC2 are boundary probes / negative")
    log("controls and are NOT claimed to be covered by them.")
    log("DONE exit_code=0")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")

    return dict(md5=digests2[0], rows=len(ROWS), conv=conv,
                fitC=(pC, r2C), fitS=(pS, r2S), nc1=nc1, nc2=nc2,
                pos=pos_summary, ref_cross=ref_cross)


if __name__ == "__main__":
    main()
