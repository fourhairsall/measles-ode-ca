# -*- coding: utf-8 -*-
"""m3_experiment.py -- M3: Lambda-Taylor / phase-type macro-step integrator.

Package layout: imports p0b_experiment as P verbatim (K, sigma, beta_sp,
W/N/S0/bins0/R0/phis, t_grid, poisson_T, reference_solution, error_metrics,
host_step_generic), which in turn imports coupling_host_level.  Creates ONLY
m3_* files: the eight m3_*.csv in <package>/figures/ plus the run record
<package>/m3_results.md.  Not part of run_all.py: a full rerun takes ~52 s and
rewrites the timing columns of m3_cost.csv and m3_equal_wall.csv.

METHOD (one macro step [t_k, t_k+h], SAME grid P.t_grid(h) as every other arm)
---------------------------------------------------------------------------
Let the host-level ODE be written as a quadratic vector field

    y' = A y + (L y) (B y),     L y = Lambda(y) = beta_sp W ((I phi)/N),
    B y = (-S, +S e_0, 0),      A = Erlang aging chain + removal to R.

1.  From the CURRENT state ONLY, compute the first p derivatives of Lambda
    along the exact flow by differentiating the quadratic field:

        a_0 = y,   a_{m+1} = A a_m + sum_{j=0}^m C(m,j) (L a_j)(B a_{m-j}),
        lambda_j := L a_j = d^j Lambda/dt^j (t_k),   j = 0..p.

    No future true state, no reference solution, no sub-cycling, no iteration.

2.  Positivity-preserving predictor of Lambda(s), s in [0,h]:

        Lambda(s) = lambda_0 * (1 + d_1 s + d_2 s^2 + d_3 s^3)^2  >= 0,

    where d_1,d_2,d_3 are chosen so the Taylor expansion of the square matches
    lambda_0, lambda_1, lambda_2, lambda_3 (p=3) or lambda_0..lambda_2 (p=2,
    d_3=0).  Explicitly, with a_j = lambda_j/(j! lambda_0):

        d_1 = a_1/2,  d_2 = (a_2 - d_1^2)/2,  d_3 = a_3/2 - d_1 d_2.

    The square is the structural positivity device: no clipping, no limiter,
    no post-hoc truncation for ANY h.  Lambda(s) - Lambda_true(t_k+s) = O(s^{p+1}).

3.  Exact solution of the resulting NON-AUTONOMOUS Markov (phase-type) linear
    subsystem with prescribed nonnegative force of infection:

        H(s) = int_0^s Lambda(u) du  (evaluated by a positive Gauss rule, exact
                                     for the degree-2p polynomial Lambda),
        S(h) = S_0 exp(-H(h)),  delivered = S_0 (1 - exp(-H(h))),
        g(u) = Lambda(u) S_0 exp(-H(u))                    (incidence density),
        P_k(mu) = exp(-mu) mu^k/k!, mu = sigma (h-u)        (Erlang age pmf),
        I_k^fresh = int_0^h g(u) P_k(sigma(h-u)) du
                  = delivered * sum_j p_j P_k(sigma(h-u_j)),
        p_j = w_j g(u_j) / sum_m w_m g(u_m) >= 0, sum_j p_j = 1,
        surv_j = P(X>=K; sigma(h-u_j)) = gammainc(K, sigma(h-u_j)),
        to_R   = delivered * sum_j p_j surv_j  >= 0,
        I(h)   = I_0 T(h) + I^fresh,   R(h) = R_0 + removed + to_R,
        removed = I_0 @ [gammainc(K-j, sigma h)]_{j=0}^{K-1} >= 0,
        T(h)   = P.poisson_T(h)  (exact Erlang aging propagator).

    The (q,K) matrix [P_k(sigma(h-u_j))] and surv_j are LAMBDA-INDEPENDENT:
    precomputed ONCE per h, reused at every step and every patch.  All
    complexity is in Lambda(s), not in the age allocation.

Run:
    C:/Python314/python.exe -u m3_experiment.py > m3_run.log 2>&1
    C:/Python314/python.exe -u m3_experiment.py --quick      # fewer repeats

All numbers in stdout / CSV / report are measured at run time, never assigned.
Every measured number is labelled: 数值待用户数值确认.
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
from numpy.polynomial.legendre import leggauss
from scipy.special import gammainc, gammaln
from scipy.integrate import solve_ivp
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p0b_experiment as P

_SUB = os.path.dirname(HERE)
OUT_DIRS = [os.path.join(_SUB, "figures")]

# canonical shared setup -- never re-derived
K = P.K
SIG = P.sigma
BETA = P.beta_sp
PHIS = P.phis
W = P.W
N = P.N
n = P.n
S00 = P.S0.copy()
B00 = P.bins0.copy()
R00 = P.R0.copy()
KK = np.arange(K, dtype=float)
KINT = np.arange(K)

# established baseline benchmark cross-check target ONLY (not used as a measured result)
BASE_E2_MID_L2 = 4.3416e-4

MD5_LOG = []
REF_CACHE = {}
T_CACHE = {}
KERN_CACHE = {}
CTR = {}
COUNTING = [False]
Q_MAIN = 16          # Gauss nodes for the in-step incidence integral
Q_H = 8              # Gauss nodes for the (positive) cumulative-hazard integral


# =============================================================================
# 0. helpers
# =============================================================================
def tick(key, k=1):
    if COUNTING[0]:
        CTR[key] = CTR.get(key, 0) + k


def reset_ctr():
    CTR.clear()


def set_counting(flag):
    COUNTING[0] = bool(flag)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return "%.10e" % x
    return str(v)


def write_csv_dual(name, header, rows):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows([[fmt(r.get(k, "")) for k in header] for r in rows])
    payload = buf.getvalue().encode("utf-8")
    digests = []
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.write(payload)
        digests.append(hashlib.md5(open(path, "rb").read()).hexdigest())
    ok = all(x == digests[0] for x in digests)
    MD5_LOG.append((name, len(rows), digests[0],
                    digests[1] if len(digests) > 1 else digests[0], ok))
    log("  [csv] %-34s rows=%-4d md5=%s copies_identical=%s"
        % (name, len(rows), digests[0], ok))
    if not ok:
        raise RuntimeError("CSV md5 mismatch: " + name)
    return digests[0]


def log(msg):
    print(msg, flush=True)


def get_T(h):
    key = float(h)
    if key not in T_CACHE:
        T_CACHE[key] = P.poisson_T(key)
        tick("T_build", 1)
    return T_CACHE[key]


def ref_for(h):
    key = float(h)
    if key not in REF_CACHE:
        REF_CACHE[key] = P.reference_solution(P.t_grid(key))
    return REF_CACHE[key]


def safe_metrics(series, ref, finite=True):
    if not finite:
        return (float("nan"),) * 4
    try:
        with np.errstate(all="ignore"):
            vals = np.concatenate([series[v].ravel() for v in ("S", "I", "R")])
            if not np.all(np.isfinite(vals)):
                return (float("nan"),) * 4
            return P.error_metrics(series, ref)
    except Exception:
        return (float("nan"),) * 4


def ratio(a, b):
    try:
        if b is None or not np.isfinite(b) or b == 0 or not np.isfinite(a):
            return float("nan")
        return float(a / b)
    except Exception:
        return float("nan")


def loglog_fit(hs, errs):
    x = np.log(np.asarray(hs, dtype=float))
    y = np.log(np.asarray(errs, dtype=float))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), int(len(x))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2), int(len(x))


def measure_wall_ms(fn, repeats=9):
    fn()  # warm-up
    ts = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.min(ts)), int(repeats)


def lam_of(S, I, R):
    tick("lam_eval", 1)
    return BETA * (W @ ((I @ PHIS) / N))


# =============================================================================
# 1. static (Lambda-independent) phase-type kernel, built ONCE per h
# =============================================================================
def static_kernel(h, q=Q_MAIN):
    key = (float(h), int(q))
    if key in KERN_CACHE:
        return KERN_CACHE[key]
    xq, wq = leggauss(int(q))
    u = 0.5 * h * (xq + 1.0)
    w = 0.5 * h * wq
    mu = SIG * (h - u)
    logp = (-mu[:, None]
            + KK[None, :] * np.log(np.where(mu[:, None] > 0, mu[:, None], 1e-300))
            - gammaln(KK[None, :] + 1.0))
    Pmat = np.exp(logp)                       # (q,K), >=0
    surv = gammainc(float(K), mu)             # (q,), >=0, stable complement
    xh, wh = leggauss(Q_H)
    out = dict(u=u, w=w, Pmat=Pmat, surv=surv,
               xh=xh, wh=wh,
               surv_old=gammainc(float(K) - KK, SIG * h))  # (K,) >=0
    KERN_CACHE[key] = out
    return out


def kernel_rows_sum_dev(kernel):
    return float(np.max(np.abs(kernel["Pmat"].sum(1) + kernel["surv"] - 1.0)))


# =============================================================================
# 2. Lambda derivatives from the quadratic vector field (current state only)
# =============================================================================
def lam_derivs(S, I, R, p=3):
    """lambda_j = d^j Lambda/dt^j at t_k for j=0..p, shape (n,) each.

    y' = A y + (L y)(B y).  For a quadratic field,
        y^{(m+1)} = A y^{(m)} + sum_j binom(m,j) (L y^{(j)})(B y^{(m-j)}).
    """
    zS = np.zeros_like(S)
    zR = np.zeros_like(S)

    def L_of(st):
        tick("lam_eval", 1)
        return BETA * (W @ ((st[1] @ PHIS) / N))

    def B_of(st):
        bI = np.zeros_like(I)
        bI[:, 0] = st[0]
        return (-st[0], bI, np.zeros_like(S))

    def A_of(st):
        dI = -SIG * st[1]
        dI[:, 1:] += SIG * st[1][:, :-1]
        return (zS.copy(), dI, SIG * st[1][:, K - 1])

    a = [(S.copy(), I.copy(), R.copy())]
    Lc = [L_of(a[0])]
    lam = [Lc[0].copy()]
    for m in range(p):
        acc = A_of(a[m])
        for j in range(m + 1):
            c = math.comb(m, j) * Lc[j]
            bv = B_of(a[m - j])
            acc = (acc[0] + c * bv[0],
                   acc[1] + c[:, None] * bv[1],
                   acc[2] + c * bv[2])
        a.append(acc)
        Lc.append(L_of(acc))
        lam.append(Lc[-1].copy())
    return lam


def m3_square_coeffs(S, I, R, p=3):
    """(l0, d1, d2, d3) of Lambda(s)=l0*(1+d1 s+d2 s^2+d3 s^3)^2."""
    lam = lam_derivs(S, I, R, p=p)
    l0 = lam[0]
    pos = l0 > 0
    l0safe = np.where(pos, l0, 0.0)
    inv = np.where(pos, 1.0 / np.where(pos, l0, 1.0), 0.0)
    a1 = lam[1] * inv
    a2 = 0.5 * lam[2] * inv if p >= 2 else np.zeros_like(l0)
    a3 = (lam[3] / 6.0) * inv if p >= 3 else np.zeros_like(l0)
    d1 = a1 / 2.0
    d2 = (a2 - d1 ** 2) / 2.0
    d3 = a3 / 2.0 - d1 * d2 if p >= 3 else np.zeros_like(l0)
    return l0safe, d1, d2, d3, int((~pos).sum())


# =============================================================================
# 3. one M3 macro step: exact non-autonomous phase-type flow
# =============================================================================
def hazard_and_lambda(l0, d1, d2, d3, kernel, h, analytic_check=False):
    """H(u_j) >= 0 by POSITIVE Gauss quadrature (exact for degree<=2p),
    H(h) >= 0 by the same rule, Lambda(u_j) = l0*P(u_j)^2 >= 0 by squaring."""
    u = kernel["u"]; xh = kernel["xh"]; wh = kernel["wh"]
    V = 0.5 * u[:, None] * (xh[None, :] + 1.0)        # (q,Q_H)
    Cw = 0.5 * u[:, None] * wh[None, :]               # (q,Q_H), >=0
    PV = (1.0 + d1[:, None, None] * V[None, :, :]
          + d2[:, None, None] * V[None, :, :] ** 2
          + d3[:, None, None] * V[None, :, :] ** 3)
    lamV = l0[:, None, None] * PV ** 2                 # >=0
    H = np.einsum("jl,njl->nj", Cw, lamV)             # >=0
    Vh = 0.5 * h * (xh + 1.0)
    Cwh = 0.5 * h * wh
    PVh = (1.0 + d1[:, None] * Vh[None, :]
           + d2[:, None] * Vh[None, :] ** 2
           + d3[:, None] * Vh[None, :] ** 3)
    lamVh = l0[:, None] * PVh ** 2
    Hh = (Cwh[None, :] * lamVh).sum(1)                 # >=0
    Pu = (1.0 + d1[:, None] * u[None, :]
          + d2[:, None] * u[None, :] ** 2
          + d3[:, None] * u[None, :] ** 3)
    lam_u = l0[:, None] * Pu ** 2                      # >=0
    out = dict(H=H, Hh=Hh, lam_u=lam_u, Pu=Pu)
    if analytic_check:
        # independent cross-check: closed-form polynomial integral
        c = np.stack([np.ones_like(l0), 2 * d1, d1 ** 2 + 2 * d2,
                      2 * d3 + 2 * d1 * d2, d2 ** 2 + 2 * d1 * d3,
                      2 * d2 * d3, d3 ** 2], axis=1) * l0[:, None]
        m = c.shape[1]
        Vint = u[None, :] ** np.arange(1, m + 1)[:, None] / np.arange(1, m + 1)[:, None]
        Ha = c @ Vint
        Hha = c @ (h ** np.arange(1, m + 1) / np.arange(1, m + 1))
        scale = max(1.0, float(np.max(np.abs(Ha))), float(np.max(np.abs(Hha))))
        out["H_analytic_maxdiff"] = float(np.max(np.abs(Ha - H))) / scale
        out["Hh_analytic_maxdiff"] = float(np.max(np.abs(Hha - Hh))) / scale
    return out


def phase_step(S, I, R, h, kernel, l0, d1, d2, d3, T,
               single_w=None, diag=None, analytic_check=False, poly=None):
    """One M3 step.  `poly` (n,m) optionally overrides Lambda(s) by an ARBITRARY
    polynomial (used ONLY by the raw 3-point-Lagrange attack)."""
    tick("phase_step", 1)
    if poly is None:
        hz = hazard_and_lambda(l0, d1, d2, d3, kernel, h,
                               analytic_check=analytic_check)
        H, Hh, lam_u = hz["H"], hz["Hh"], hz["lam_u"]
        if analytic_check and diag is not None:
            diag["H_crosscheck"] = max(diag.get("H_crosscheck", 0.0),
                                       hz["H_analytic_maxdiff"],
                                       hz["Hh_analytic_maxdiff"])
    else:
        # raw polynomial Lambda(s) = sum_m poly[:,m] s^m ; H(s) analytic
        m = poly.shape[1]
        upow = kernel["u"][None, :] ** np.arange(m)[:, None]
        lam_u = poly @ upow
        Vint = (kernel["u"][None, :] ** np.arange(1, m + 1)[:, None]
                / np.arange(1, m + 1)[:, None])
        H = poly @ Vint
        Hh = poly @ (h ** np.arange(1, m + 1) / np.arange(1, m + 1))
        lam_u = lam_u
    tick("alloc_matmul", 1)
    if poly is None:
        # log-domain incidence density (log-guard only, NO state clipping)
        logS = np.where(S > 0, np.log(np.where(S > 0, S, 1.0)), -np.inf)
        log_lam = (np.log(np.where(l0 > 0, l0, 1e-300))[:, None]
                   + 2.0 * np.log(np.where(np.abs(hz["Pu"]) > 0,
                                           np.abs(hz["Pu"]), 1e-300)))
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
        pj = np.where(Z[:, None] > 0,
                      wg / np.where(Z[:, None] > 0, Z[:, None], 1.0),
                      kernel["w"][None, :] / kernel["w"].sum())
    else:
        # raw-polynomial attack: allow SIGNED g(u) so negativity is MEASURED
        with np.errstate(all="ignore"):
            g = lam_u * S[:, None] * np.exp(-H)
            wg = kernel["w"][None, :] * g
            Z = wg.sum(1)
            den = np.where(np.abs(Z) > 0, Z, 1.0)
            pj = np.where(np.abs(Z[:, None]) > 0, wg / den[:, None],
                          kernel["w"][None, :] / kernel["w"].sum())
    S_new = S * np.exp(-Hh)
    delivered = S - S_new
    if single_w is not None:
        Ifresh = delivered[:, None] * np.asarray(single_w)[None, :]
        toR = delivered - Ifresh.sum(1)
    else:
        Ifresh = delivered[:, None] * (pj @ kernel["Pmat"])
        toR = delivered * (pj @ kernel["surv"])
    tick("aging_matmul", 1)
    Iold = I @ T.T
    removed = I @ kernel["surv_old"]
    I_new = Iold + Ifresh
    R_new = R + removed + toR
    if diag is not None:
        diag["min_lam_u"] = min(diag.get("min_lam_u", np.inf),
                                float(np.min(lam_u)))
        diag["min_H"] = min(diag.get("min_H", np.inf), float(np.min(H)),
                            float(np.min(Hh)))
        diag["min_pj"] = min(diag.get("min_pj", np.inf), float(np.min(pj)))
        diag["min_toR"] = min(diag.get("min_toR", np.inf), float(np.min(toR)))
        diag["n_Zzero"] = diag.get("n_Zzero", 0) + int((Z <= 0).sum())
    return S_new, I_new, R_new


# =============================================================================
# 4. second-order exponential / splitting baselines (own implementation, P definitions reused)
# =============================================================================
def fresh_closed(h, S, lam):
    a = SIG - lam
    if np.any(a <= 0):
        # exponential-midpoint closed form has no continuation at a<=0; report rather than hide
        raise FloatingPointError("exponential-midpoint closed form: min(sigma-Lam)=%g"
                                 % float(np.min(a)))
    ratio_sa = (SIG / a)[:, None] ** KK[None, :] / a[:, None]
    G = gammainc(KK[None, :] + 1.0, (a * h)[:, None])
    tick("alloc_matmul", 1)
    return (lam * S * np.exp(-lam * h))[:, None] * ratio_sa * G


def w_uniform(h):
    return gammainc(KK + 1.0, SIG * h) / (SIG * h)


def w_fixed_pp(h, S, I, R):
    lam = lam_of(S, I, R)
    a = SIG - lam
    pref = lam / np.expm1(lam * h)
    coef = np.exp(KK[None, :] * np.log(SIG / a)[:, None]) / a[:, None]
    return pref[:, None] * coef * gammainc(KK[None, :] + 1.0, (a * h)[:, None])


def exp2_step(S, I, R, h, lam, mode="closed", w=None):
    T = get_T(h)
    tick("aging_matmul", 1)
    Iold = I @ T.T
    removed = I.sum(1) - Iold.sum(1)
    S_new = S * np.exp(-lam * h)
    delivered = S - S_new
    if mode == "closed":
        If = fresh_closed(h, S, lam)
    elif mode == "static":
        If = delivered[:, None] * w_uniform(h)[None, :]
        tick("alloc_matmul", 1)
    elif mode == "fix":
        If = delivered[:, None] * np.asarray(w)
        tick("alloc_matmul", 1)
    elif mode == "strang":
        # literal Strang: age(h/2) -> react(h) -> age(h/2)
        T2 = get_T(0.5 * h)
        tick("aging_matmul", 2)
        b1 = I @ T2.T
        rm1 = I.sum(1) - b1.sum(1)
        b1[:, 0] += delivered
        b2 = b1 @ T2.T
        rm2 = b1.sum(1) - b2.sum(1)
        return S_new, b2, R + rm1 + rm2
    else:
        raise ValueError(mode)
    toR = delivered - If.sum(1)
    return S_new, Iold + If, R + removed + toR


def midpoint_lambda(S, I, R, h, mode="closed", w=None):
    """Half-step EXP2-midpoint predictor -> Lambda at the midpoint (no future truth)."""
    lam0 = lam_of(S, I, R)
    _, b_half, _, = exp2_step(S, I, R, 0.5 * h, lam0,
                                  mode="closed", w=w)
    tick("aux_state_prop", 1)
    return lam_of(S, b_half, R)


def clip_step(S, I, R, h, T):
    S, I, R, _ = P.host_step_generic(S, I, R, W, N, PHIS, SIG, BETA,
                                     h, T, "clip")
    tick("lam_eval", 1)
    tick("aging_matmul", 1)
    tick("integration_step", 1)
    return S, I, R


# =============================================================================
# 5. uniform driver
# =============================================================================
def new_series(grid):
    m = len(grid) - 1
    ser = dict(S=np.zeros((m + 1, n)), I=np.zeros((m + 1, n)),
               R=np.zeros((m + 1, n)), t=grid.copy())
    ser["S"][0] = S00 / N
    ser["I"][0] = B00.sum(1) / N
    ser["R"][0] = R00 / N
    return ser


def finish_run(ser, cons, negS, negI, negR, minS, finite, grid, diag=None):
    d = dict(series=ser, cons_resid=cons, neg_S=negS, neg_I=negI,
             neg_R=negR, min_S=minS, all_finite=finite, n_steps=len(grid) - 1,
             grid=grid)
    if diag:
        d.update(diag)
    d.update(dict(CTR))
    return d


def drive(h, step_fn, extra=None):
    grid = P.t_grid(h)
    if abs(grid[-1] - P.T1) > 1e-8:
        raise RuntimeError("t_grid(h) does not end at T1 for h=%g" % h)
    S, I, R = S00.copy(), B00.copy(), R00.copy()
    ser = new_series(grid)
    cons, negS, negI, negR = 0.0, 0, 0, 0
    minS, finite = float("inf"), True
    diag = {}
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            S, I, R = step_fn(S, I, R, k, diag)
            It = I.sum(1)
            finite = finite and bool(np.all(np.isfinite(S))
                                     and np.all(np.isfinite(I))
                                     and np.all(np.isfinite(R)))
            if finite:
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < -1e-9).sum())
                negI += int((It < -1e-9).sum())
                negR += int((R < -1e-9).sum())
                minS = min(minS, float(S.min()))
            else:
                break
            ser["S"][k + 1] = S / N
            ser["I"][k + 1] = It / N
            ser["R"][k + 1] = R / N
    return finish_run(ser, cons, negS, negI, negR, minS, finite, grid, diag)


# --- generic arm dispatcher -------------------------------------------------
def run_arm(arm, h, counting=False):
    set_counting(counting)
    reset_ctr()
    q = Q_MAIN
    if arm.startswith("M3-q"):
        q = int(arm.split("q")[1])
    kernel = static_kernel(h, q)
    T = get_T(h)
    single = w_uniform(h) if arm.endswith("+single_w") else None
    p = 2 if arm.startswith("M3-p2") else 3
    p = 1 if arm.startswith("M3-p1") else p
    hist = []
    fixed_w = None
    if "fix_pp" in arm:
        fixed_w = w_fixed_pp(h, S00, B00, R00)

    def step(S, I, R, k, diag):
        if arm.startswith("M3"):
            l0, d1, d2, d3, ndeg = m3_square_coeffs(S, I, R, p=p)
            diag["n_l0_nonpos"] = diag.get("n_l0_nonpos", 0) + ndeg
            return phase_step(S, I, R, h, kernel, l0, d1, d2, d3, T,
                              single_w=single, diag=diag,
                              analytic_check=("crosscheck" in arm))
        if arm == "AB3-extrap+PT":
            lamk = lam_of(S, I, R)
            hist.append(lamk)
            if k < 2:
                l0, d1, d2, d3, ndeg = m3_square_coeffs(S, I, R, p=3)
            else:
                lm2, lm1, lm0 = hist[k - 2], hist[k - 1], lamk
                b1 = (3 * lm0 - 4 * lm1 + lm2) / (2 * h)
                b2 = (lm0 - 2 * lm1 + lm2) / (2 * h * h)
                l0 = np.where(lamk > 0, lamk, 0.0)
                inv = np.where(lamk > 0, 1.0 / np.where(lamk > 0, lamk, 1.0), 0.0)
                A1, A2 = b1 * inv, b2 * inv
                d1 = A1 / 2.0
                d2 = (A2 - d1 ** 2) / 2.0
                d3 = np.zeros_like(l0)
            return phase_step(S, I, R, h, kernel, l0, d1, d2, d3, T, diag=diag)
        if arm == "AB3-extrap+PT(frozen-startup)":
            lamk = lam_of(S, I, R)
            hist.append(lamk)
            if k < 2:
                l0 = np.where(lamk > 0, lamk, 0.0)
                d1 = d2 = d3 = np.zeros_like(l0)
            else:
                lm2, lm1, lm0 = hist[k - 2], hist[k - 1], lamk
                b1 = (3 * lm0 - 4 * lm1 + lm2) / (2 * h)
                b2 = (lm0 - 2 * lm1 + lm2) / (2 * h * h)
                l0 = np.where(lamk > 0, lamk, 0.0)
                inv = np.where(lamk > 0, 1.0 / np.where(lamk > 0, lamk, 1.0), 0.0)
                A1, A2 = b1 * inv, b2 * inv
                d1 = A1 / 2.0
                d2 = (A2 - d1 ** 2) / 2.0
                d3 = np.zeros_like(l0)
            return phase_step(S, I, R, h, kernel, l0, d1, d2, d3, T, diag=diag)
        if arm == "Lagrange3-predictor":
            # candidate direction 1: 3-point quadratic Lagrange of Lambda(s)
            lam0 = lam_of(S, I, R)
            _, bh, _ = exp2_step(S, I, R, 0.5 * h, lam0, mode="closed")
            _, bf, _ = exp2_step(S, I, R, h, lam0, mode="closed")
            tick("aux_state_prop", 2)
            lm = lam_of(S, bh, R)
            le = lam_of(S, bf, R)
            A = lam0
            B = (-3 * lam0 + 4 * lm - le) / h
            C = (2 * lam0 - 4 * lm + 2 * le) / (h * h)
            poly = np.stack([A, B, C], axis=1)
            return phase_step(S, I, R, h, kernel, None, None, None, None, T,
                              diag=diag, poly=poly)
        if arm.startswith("clip+"):
            m = int(arm.split("+")[1].rstrip("sub"))
            Ts = get_T(h / m)
            for _ in range(m):
                S, I, R = clip_step(S, I, R, h / m, Ts)
            return S, I, R
        # ---- EXP2-midpoint / static / Strang / fix, frozen or midpoint ----
        mid = "midpoint" in arm
        if arm.startswith("fix_pp"):
            mode, w = "fix", fixed_w
        elif arm.startswith("Strang"):
            mode, w = "strang", None
        elif arm.startswith("static_w"):
            mode, w = "static", None
        else:
            mode, w = "closed", None
        lam = (midpoint_lambda(S, I, R, h, mode="closed", w=w) if mid
               else lam_of(S, I, R))
        return exp2_step(S, I, R, h, lam, mode=mode, w=w)

    return drive(h, step)


# =============================================================================
# 6. explicit-RK / Richardson / Yoshida attacks (own loops on the same grid)
# =============================================================================
def rhs_flat(y):
    S = y[:n]
    I = y[n:n + n * K].reshape(n, K)
    lam = BETA * (W @ ((I @ PHIS) / N))
    tick("lam_eval", 1)
    dS = -lam * S
    dI = -SIG * I
    dI = dI.copy()
    dI[:, 1:] += SIG * I[:, :-1]
    dI[:, 0] += lam * S
    dR = SIG * I[:, K - 1]
    return np.concatenate([dS, dI.ravel(), dR])


def y0_flat():
    return np.concatenate([S00, B00.ravel(), R00])


def run_explicit_rk(h, method="rk4"):
    grid = P.t_grid(h)
    y = y0_flat()
    ser = new_series(grid)
    cons, negS, negI, negR = 0.0, 0, 0, 0
    minS, finite = float("inf"), True
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            if method == "rk4":
                k1 = rhs_flat(y)
                k2 = rhs_flat(y + 0.5 * h * k1)
                k3 = rhs_flat(y + 0.5 * h * k2)
                k4 = rhs_flat(y + h * k3)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            elif method == "ssprk3":
                u1 = y + h * rhs_flat(y)
                u2 = 0.75 * y + 0.25 * (u1 + h * rhs_flat(u1))
                y = (1.0 / 3.0) * y + (2.0 / 3.0) * (u2 + h * rhs_flat(u2))
            elif method == "rk3":
                k1 = rhs_flat(y)
                k2 = rhs_flat(y + 0.5 * h * k1)
                k3 = rhs_flat(y - h * k1 + 2 * h * k2)
                y = y + (h / 6.0) * (k1 + 4 * k2 + k3)
            else:
                raise ValueError(method)
            tick("integration_step", 1)
            S = y[:n]; I = y[n:n + n * K].reshape(n, K); It = I.sum(1)
            R = y[n + n * K:]
            finite = finite and bool(np.all(np.isfinite(y)))
            if finite:
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < -1e-9).sum()); negI += int((It < -1e-9).sum())
                negR += int((R < -1e-9).sum())
                minS = min(minS, float(S.min()))
            else:
                break
            ser["S"][k + 1] = S / N; ser["I"][k + 1] = It / N
            ser["R"][k + 1] = R / N
    return finish_run(ser, cons, negS, negI, negR, minS, finite, grid)


def run_richardson(h):
    """(4*y_{h/2} - y_h)/3 on EXP2-midpoint states; uses 2 half-size
    integration steps inside each macro step (sub-cycling: disclosed)."""
    grid = P.t_grid(h)
    S, I, R = S00.copy(), B00.copy(), R00.copy()
    ser = new_series(grid)
    cons, negS, negI, negR = 0.0, 0, 0, 0
    minS, finite = float("inf"), True
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            lam0 = lam_of(S, I, R)
            _, bh, _ = exp2_step(S, I, R, 0.5 * h, lam0, mode="closed")
            lm = lam_of(S, bh, R)
            Sh, Ih, Rh = exp2_step(S, I, R, h, lm, mode="closed")
            tick("aux_state_prop", 1)
            S2, I2, R2 = S.copy(), I.copy(), R.copy()
            for _ in range(2):
                l0 = lam_of(S2, I2, R2)
                _, bq, _ = exp2_step(S2, I2, R2, 0.25 * h, l0, mode="closed")
                lq = lam_of(S2, bq, R2)
                S2, I2, R2 = exp2_step(S2, I2, R2, 0.5 * h, lq,
                                           mode="closed")
                tick("aux_state_prop", 2)
            S = (4.0 * S2 - Sh) / 3.0
            I = (4.0 * I2 - Ih) / 3.0
            R = (4.0 * R2 - Rh) / 3.0
            tick("integration_step", 3)
            It = I.sum(1)
            finite = finite and bool(np.all(np.isfinite(S))
                                     and np.all(np.isfinite(I))
                                     and np.all(np.isfinite(R)))
            if finite:
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < -1e-9).sum()); negI += int((It < -1e-9).sum())
                negR += int((R < -1e-9).sum())
                minS = min(minS, float(S.min()))
            else:
                break
            ser["S"][k + 1] = S / N; ser["I"][k + 1] = It / N
            ser["R"][k + 1] = R / N
    return finish_run(ser, cons, negS, negI, negR, minS, finite, grid)


def strang_sub(S, I, R, tau):
    """Strang sub-flow with signed time tau (Yoshida needs tau<0)."""
    lam = lam_of(S, I, R)
    T2 = get_T(0.5 * tau)
    tick("aging_matmul", 2)
    b1 = I @ T2.T
    rm1 = I.sum(1) - b1.sum(1)
    Sn = S * np.exp(-lam * tau)
    delivered = S - Sn
    b1[:, 0] += delivered
    b2 = b1 @ T2.T
    rm2 = b1.sum(1) - b2.sum(1)
    return Sn, b2, R + rm1 + rm2


def run_yoshida4(h):
    c = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - c)
    w0 = -c / (2.0 - c)
    grid = P.t_grid(h)
    S, I, R = S00.copy(), B00.copy(), R00.copy()
    ser = new_series(grid)
    cons, negS, negI, negR = 0.0, 0, 0, 0
    minS, finite = float("inf"), True
    with np.errstate(all="ignore"):
        for k in range(len(grid) - 1):
            for tau in (w1 * h, w0 * h, w1 * h):
                S, I, R = strang_sub(S, I, R, tau)
                tick("integration_step", 1)
            It = I.sum(1)
            finite = finite and bool(np.all(np.isfinite(S))
                                     and np.all(np.isfinite(I))
                                     and np.all(np.isfinite(R)))
            if finite:
                cons = max(cons, float(np.max(np.abs(S + It + R - N))))
                negS += int((S < -1e-9).sum()); negI += int((It < -1e-9).sum())
                negR += int((R < -1e-9).sum())
                minS = min(minS, float(S.min()))
            else:
                break
            ser["S"][k + 1] = S / N; ser["I"][k + 1] = It / N
            ser["R"][k + 1] = R / N
    return finish_run(ser, cons, negS, negI, negR, minS, finite, grid)


def run_attack(arm, h, counting=False):
    # A standalone 60-step run builds its T(h) / static kernel ONCE.  Clear the
    # process-wide caches so both the per-run construction count and the wall
    # clock include that precomputation (no free lunch from earlier arms).
    T_CACHE.clear()
    KERN_CACHE.clear()
    set_counting(counting)
    reset_ctr()
    if arm == "Richardson(EXP2-midpoint)":
        return run_richardson(h)
    if arm == "RK4-explicit":
        return run_explicit_rk(h, "rk4")
    if arm == "SSPRK3-explicit":
        return run_explicit_rk(h, "ssprk3")
    if arm == "RK3-explicit":
        return run_explicit_rk(h, "rk3")
    if arm == "Yoshida4-split":
        return run_yoshida4(h)
    return run_arm(arm, h, counting=counting)


ARMS_ALL = ["M3-p3", "M3-p2", "M3-p1", "M3+single_w",
            "EXP2-midpoint", "static_w+midpoint", "Strang+midpoint",
            "fix_pp+midpoint", "clip+2sub", "clip+40sub",
            "AB3-extrap+PT", "AB3-extrap+PT(frozen-startup)",
            "Lagrange3-predictor", "Richardson(EXP2-midpoint)",
            "RK4-explicit", "SSPRK3-explicit", "Yoshida4-split"]

H_ORDER = [0.5, 0.4, 0.3, 0.25, 0.2, 0.125, 0.1, 0.0625]
H_EXTREME = [10.0, 20.0, 30.0, 60.0]


# =============================================================================
# EXPERIMENT 0 : self-tests / cross-checks
# =============================================================================
def experiment_0():
    log("=" * 100)
    log("EXPERIMENT 0: self-tests and independent cross-checks")
    log("=" * 100)
    rows = []
    # grid sanity
    for h in H_ORDER + [1.0] + H_EXTREME:
        g = P.t_grid(h)
        assert abs(g[-1] - P.T1) < 1e-8, (h, g[-1])
    log("[0] t_grid(h) ends at T1=%.1f for all %d scanned h : OK"
        % (P.T1, len(H_ORDER) + 1 + len(H_EXTREME)))
    # removal identity: exact Poisson propagator vs stable survival form
    rng = np.random.default_rng(20261107)
    for h in (0.0625, 1.0, 10.0, 60.0):
        kern = static_kernel(h, Q_MAIN)
        T = P.poisson_T(h)
        Ix = rng.random((n, K)) * 10.0
        a = Ix.sum(1) - (Ix @ T.T).sum(1)
        b = Ix @ kern["surv_old"]
        dev = float(np.max(np.abs(a - b)))
        rows.append(dict(section="removal_identity", h=h,
                         max_abs_diff=dev, scale=float(np.max(np.abs(a)))))
        log("[0] removal identity h=%-7g max|d|=%.3e (scale %.3e)"
            % (h, dev, float(np.max(np.abs(a)))))
    # kernel row-sum
    for h in (0.0625, 1.0, 10.0, 60.0):
        dev = kernel_rows_sum_dev(static_kernel(h, Q_MAIN))
        rows.append(dict(section="kernel_rowsum", h=h, max_abs_diff=dev))
        log("[0] static kernel row-sum dev h=%-7g = %.3e" % (h, dev))
    # H evaluator cross-check (analytic polynomial vs positive Gauss rule)
    for h in (0.0625, 1.0, 10.0):
        l0, d1, d2, d3, _ = m3_square_coeffs(S00, B00, R00, p=3)
        hz = hazard_and_lambda(l0, d1, d2, d3, static_kernel(h, Q_MAIN), h,
                               analytic_check=True)
        rows.append(dict(section="H_crosscheck", h=h,
                         max_abs_diff=max(hz["H_analytic_maxdiff"],
                                          hz["Hh_analytic_maxdiff"])))
        log("[0] H cross-check h=%-7g analytic-vs-positive-Gauss rel = %.3e"
            % (h, max(hz["H_analytic_maxdiff"], hz["Hh_analytic_maxdiff"])))
    # EXP2-midpoint baseline recomputation against the established benchmark
    d = run_attack("EXP2-midpoint", 1.0)
    l2 = safe_metrics(d["series"], ref_for(1.0), d["all_finite"])[0]
    rows.append(dict(section="E2_baseline_reproduction", h=1.0,
                     L2_rel_err=l2, prior_value=BASE_E2_MID_L2,
                     rel_dev=abs(l2 - BASE_E2_MID_L2) / BASE_E2_MID_L2))
    log("[0] EXP2-midpoint baseline h=1 L2=%.6e | benchmark %.6e | rel dev %.3e"
        % (l2, BASE_E2_MID_L2,
           abs(l2 - BASE_E2_MID_L2) / BASE_E2_MID_L2))
    # Regression guard: the baseline must reproduce its established value.
    # A silent arm-dispatch change would otherwise alter every comparison.
    if abs(l2 - BASE_E2_MID_L2) / BASE_E2_MID_L2 > 1e-3:
        raise RuntimeError("EXP2-midpoint baseline drifted from its "
                           "established benchmark (rel dev > 1e-3); "
                           "aborting rather than reporting changed numbers")
    return rows


# =============================================================================
# EXPERIMENT 1 : M3-a  (h=1, 60 macro steps, same reference)
# =============================================================================
BASE_ARMS_A = ["M3-p3", "M3-p2", "EXP2-midpoint", "static_w+midpoint",
               "Strang+midpoint", "fix_pp+midpoint", "clip+2sub",
               "clip+40sub"]


def experiment_1():
    log("")
    log("=" * 100)
    log("EXPERIMENT 1 | M3-a : h=1, 60 macro steps, identical P.reference_solution")
    log("=" * 100)
    h = 1.0
    ref = ref_for(h)
    res = {}
    for arm in BASE_ARMS_A:
        d = run_attack(arm, h, counting=True)
        l2, ma, msr, mpr = safe_metrics(d["series"], ref, d["all_finite"])
        res[arm] = dict(d=d, l2=l2, ma=ma)
        log("[1] %-20s L2=%.6e maxabs=%.4e cons=%.3e negS=%d negI=%d "
            "minS=%.3e finite=%s nLam=%d nInt=%d"
            % (arm, l2, ma, d["cons_resid"], d["neg_S"], d["neg_I"],
               d["min_S"], d["all_finite"],
               d.get("lam_eval", 0), d.get("integration_step", d["n_steps"])))
    m3 = res["M3-p3"]["l2"]
    rows = []
    for arm in BASE_ARMS_A:
        r = res[arm]
        rows.append(dict(h_macro=h, arm=arm, L2_rel_err=r["l2"],
                         max_abs_err=r["ma"],
                         L2_ratio_vs_M3p3=ratio(r["l2"], m3),
                         M3p3_better_by=ratio(r["l2"], m3),
                         better_than_E2=ratio(res["EXP2-midpoint"]["l2"], r["l2"]),
                         cons_resid=r["d"]["cons_resid"],
                         neg_S=r["d"]["neg_S"], neg_I=r["d"]["neg_I"],
                         min_S=r["d"]["min_S"], all_finite=r["d"]["all_finite"],
                         n_lambda_evals=r["d"].get("lam_eval", 0),
                         n_integration_steps=r["d"].get("integration_step",
                                                        r["d"]["n_steps"]),
                         strictly_better_than_M3p3=(r["l2"] > m3)))
    for arm in BASE_ARMS_A:
        log("[1] M3-p3 / %-20s = %.3f x  (M3 better by %.3f x)"
            % (arm, ratio(res[arm]["l2"], m3), ratio(res[arm]["l2"], m3)))
    header = ["h_macro", "arm", "L2_rel_err", "max_abs_err",
              "L2_ratio_vs_M3p3", "M3p3_better_by", "better_than_E2",
              "cons_resid", "neg_S", "neg_I", "min_S", "all_finite",
              "n_lambda_evals", "n_integration_steps",
              "strictly_better_than_M3p3"]
    write_csv_dual("m3_baselines.csv", header, rows)
    return rows, res


# =============================================================================
# EXPERIMENT 2 : M3-b convergence order (>=6 points, 0.5 -> 0.0625)
# =============================================================================
ARMS_ORDER = ["M3-p3", "M3-p2", "M3-p1", "M3+single_w",
              "EXP2-midpoint", "static_w+midpoint", "Strang+midpoint",
              "fix_pp+midpoint", "clip+2sub",
              "AB3-extrap+PT", "AB3-extrap+PT(frozen-startup)",
              "Lagrange3-predictor", "Richardson(EXP2-midpoint)",
              "RK4-explicit", "SSPRK3-explicit", "Yoshida4-split"]


def experiment_2():
    log("")
    log("=" * 100)
    log("EXPERIMENT 2 | M3-b : convergence order, h=%s" % H_ORDER)
    log("=" * 100)
    rows = []
    fits = {}
    for arm in ARMS_ORDER:
        errs = []
        for h in H_ORDER:
            ref = ref_for(h)
            try:
                d = run_attack(arm, h)
                l2, ma, _, _ = safe_metrics(d["series"], ref, d["all_finite"])
            except Exception as e:
                log("[2] %-30s h=%-8g EXCEPTION %s" % (arm, h, e))
                l2, ma, d = float("nan"), float("nan"), None
            errs.append(l2)
            rows.append(dict(section="order", arm=arm, h_macro=h,
                             L2_rel_err=l2, max_abs_err=ma,
                             cons_resid=(d["cons_resid"] if d else float("nan")),
                             neg_S=(d["neg_S"] if d else -1),
                             neg_I=(d["neg_I"] if d else -1),
                             min_S=(d["min_S"] if d else float("nan")),
                             all_finite=(d["all_finite"] if d else False),
                             fit_order=float("nan"), fit_R2=float("nan")))
            log("[2] %-30s h=%-8g L2=%.6e %s"
                % (arm, h, l2, ("finite=%s negS=%d" %
                                (d["all_finite"], d["neg_S"])) if d else ""))
        p, r2, npt = loglog_fit(H_ORDER, errs)
        fits[arm] = (p, r2, npt)
        rows.append(dict(section="fit", arm=arm, h_macro=float("nan"),
                         L2_rel_err=float("nan"), max_abs_err=float("nan"),
                         fit_order=p, fit_R2=r2, n_h_points=npt))
        log("[2:fit] %-30s order=%.4f R2=%.6f n=%d" % (arm, p, r2, npt))
    # reference-floor check: independent tighter DOP853 solve
    log("-- 2b reference accuracy floor (rtol 1e-10 vs 1e-12) --")
    floor_rows = []
    for h in (0.5, 0.125, 0.0625):
        grid = P.t_grid(h)
        r10 = ref_for(h)
        y0 = np.concatenate([S00, B00.ravel(), R00])

        def rhs(t, y):
            S = y[:n]
            I = y[n:n + n * K].reshape(n, K)
            lam = BETA * (W @ ((I @ PHIS) / N))
            dS = -lam * S
            dI = -SIG * I
            dI = dI.copy()
            dI[:, 1:] += SIG * I[:, :-1]
            dI[:, 0] += lam * S
            return np.concatenate([dS, dI.ravel(), SIG * I[:, K - 1]])

        sol = solve_ivp(rhs, (grid[0], grid[-1]), y0, t_eval=grid,
                        method="DOP853", rtol=1e-12, atol=1e-14)
        Y = sol.y.T
        r12 = dict(S=Y[:, :n] / N,
                   I=Y[:, n:n + n * K].reshape(len(grid), n, K).sum(2) / N,
                   R=Y[:, n + n * K:] / N)
        dd = max(float(np.max(np.abs(r10[v] - r12[v]))) for v in ("S", "I", "R"))
        d = run_attack("M3-p3", h)
        e10 = safe_metrics(d["series"], r10, d["all_finite"])[0]
        e12 = safe_metrics(d["series"], r12, d["all_finite"])[0]
        floor_rows.append(dict(h_macro=h, ref_max_abs_diff=dd,
                               M3_L2_vs_rtol1e10=e10, M3_L2_vs_rtol1e12=e12,
                               rel_change=abs(e12 - e10) / e10))
        log("[2b] h=%-8g |ref(1e-10)-ref(1e-12)|max=%.3e  "
            "M3 L2 vs loose=%.6e vs tight=%.6e rel.change=%.3e"
            % (h, dd, e10, e12, abs(e12 - e10) / e10))
    write_csv_dual("m3_ref_floor.csv",
                   ["h_macro", "ref_max_abs_diff", "M3_L2_vs_rtol1e10",
                    "M3_L2_vs_rtol1e12", "rel_change"], floor_rows)
    header = ["section", "arm", "h_macro", "L2_rel_err", "max_abs_err",
              "cons_resid", "neg_S", "neg_I", "min_S", "all_finite",
              "fit_order", "fit_R2", "n_h_points"]
    write_csv_dual("m3_convergence.csv", header, rows)
    return rows, fits, floor_rows


# =============================================================================
# EXPERIMENT 3 : M3-c extreme CFL
# =============================================================================
ARMS_EXT = ["M3-p3", "M3-p2", "M3-p1", "M3+single_w",
            "EXP2-midpoint", "static_w+midpoint", "Strang+midpoint",
            "fix_pp+midpoint", "clip+2sub",
            "AB3-extrap+PT", "AB3-extrap+PT(frozen-startup)",
            "Lagrange3-predictor", "Richardson(EXP2-midpoint)",
            "RK4-explicit", "SSPRK3-explicit", "Yoshida4-split"]


def experiment_3():
    log("")
    log("=" * 100)
    log("EXPERIMENT 3 | M3-c : extreme macro steps, no CFL / no limiter / no clip")
    log("=" * 100)
    rows = []
    for h in H_EXTREME:
        ref = ref_for(h)
        for arm in ARMS_EXT:
            try:
                d = run_attack(arm, h)
                l2 = safe_metrics(d["series"], ref, d["all_finite"])[0]
            except Exception as e:
                log("[3] %-26s h=%-5g EXCEPTION %s" % (arm, h, e))
                continue
            rows.append(dict(h_macro=h, arm=arm, n_steps=d["n_steps"],
                             all_finite=d["all_finite"],
                             cons_resid=d["cons_resid"],
                             neg_S=d["neg_S"], neg_I=d["neg_I"],
                             neg_R=d["neg_R"], min_S=d["min_S"],
                             min_lam_u=d.get("min_lam_u", float("nan")),
                             min_H=d.get("min_H", float("nan")),
                             min_pj=d.get("min_pj", float("nan")),
                             min_toR=d.get("min_toR", float("nan")),
                             n_Zzero=d.get("n_Zzero", 0),
                             n_l0_nonpos=d.get("n_l0_nonpos", 0),
                             L2_rel_err=l2))
            log("[3] h=%-5g %-26s finite=%-5s cons=%-10.3e negS=%-5d "
                "negI=%-5d negR=%-5d minS=%-11.3e L2=%s"
                % (h, arm, d["all_finite"], d["cons_resid"], d["neg_S"],
                   d["neg_I"], d["neg_R"], d["min_S"],
                   ("%.4e" % l2) if np.isfinite(l2) else "nan"))
    header = ["h_macro", "arm", "n_steps", "all_finite", "cons_resid",
              "neg_S", "neg_I", "neg_R", "min_S", "min_lam_u", "min_H",
              "min_pj", "min_toR", "n_Zzero", "n_l0_nonpos", "L2_rel_err"]
    write_csv_dual("m3_extreme_cfl.csv", header, rows)
    return rows


# =============================================================================
# EXPERIMENT 4 : M3-d cost fairness
# =============================================================================
ARMS_COST = ["M3-p3", "M3-p2", "EXP2-midpoint", "static_w+midpoint",
             "clip+2sub", "clip+40sub", "AB3-extrap+PT",
             "Lagrange3-predictor", "Richardson(EXP2-midpoint)",
             "RK4-explicit"]


def experiment_4(res1, quick=False):
    log("")
    log("=" * 100)
    log("EXPERIMENT 4 | M3-d : honest cost accounting, h=1, full 60-step run")
    log("=" * 100)
    repeats = 3 if quick else 9
    h = 1.0
    rows = []
    for arm in ARMS_COST:
        dc = run_attack(arm, h, counting=True)
        nst = dc["n_steps"]
        med, mn, nr = measure_wall_ms(lambda: run_attack(arm, h, counting=False),
                                      repeats=repeats)
        if arm in res1:
            l2 = res1[arm]["l2"]
        else:
            dtmp = run_attack(arm, h)
            l2 = safe_metrics(dtmp["series"], ref_for(h),
                              dtmp["all_finite"])[0]
        valid = bool(dc["all_finite"])
        fv = (lambda x: x if valid else float("nan"))
        rows.append(dict(arm=arm, n_macro_steps=nst,
                         all_finite_h1=valid,
                         lam_evals_total=fv(dc.get("lam_eval", 0)),
                         lam_evals_per_macro=fv(dc.get("lam_eval", 0) / nst),
                         aging_matmul_per_macro=fv(dc.get("aging_matmul", 0) / nst),
                         alloc_matmul_per_macro=fv(dc.get("alloc_matmul", 0) / nst),
                         aux_state_prop_per_macro=fv(dc.get("aux_state_prop", 0) / nst),
                         integration_steps_per_macro=fv(
                             dc.get("integration_step", nst) / nst),
                         T_build_per_run=fv(dc.get("T_build", 0)),
                         wall_ms_median=med, wall_ms_min=mn, wall_n_repeats=nr,
                         L2_rel_err_h1=l2))
        log("[4] %-26s lam/macro=%-6.2f aging/macro=%-6.2f alloc/macro=%-6.2f "
            "auxProp/macro=%-6.2f int/macro=%-6.2f Tbuild/run=%-3s "
            "wall=%9.3f ms L2=%.4e"
            % (arm, fv(dc.get("lam_eval", 0) / nst),
               fv(dc.get("aging_matmul", 0) / nst),
               fv(dc.get("alloc_matmul", 0) / nst),
               fv(dc.get("aux_state_prop", 0) / nst),
               fv(dc.get("integration_step", nst) / nst),
               fmt(fv(dc.get("T_build", 0))), med, l2))
    m3 = next(r for r in rows if r["arm"] == "M3-p3")
    # 4b: EQUAL-WALL-CLOCK clip sweep -- decisive cost-fairness axis.
    #     Report the most accurate clip arm whose measured wall time does not
    #     exceed M3-p3's, rather than only the favourable Lambda-eval count.
    eq_rows = []
    for m in [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64]:
        name = "clip+%dsub" % m
        d = run_attack(name, h)
        l2 = safe_metrics(d["series"], ref_for(h), d["all_finite"])[0]
        med, mn, nr = measure_wall_ms(
            lambda name=name: run_attack(name, h), repeats=repeats)
        eq_rows.append(dict(arm=name, m_substeps=m,
                            wall_ms_median=med, wall_ms_min=mn,
                            wall_n_repeats=nr, L2_rel_err_h1=l2,
                            wall_ratio_vs_M3p3=ratio(med, m3["wall_ms_median"]),
                            affordable_at_M3_wall=(med <= m3["wall_ms_median"])))
        log("[4b] %-12s wall=%9.3f ms (x%6.3f M3) L2=%.4e affordable=%s"
            % (name, med, ratio(med, m3["wall_ms_median"]), l2,
               eq_rows[-1]["affordable_at_M3_wall"]))
    aff = [r for r in eq_rows if r["affordable_at_M3_wall"]]
    best = max(aff, key=lambda r: r["m_substeps"]) if aff else None
    for r in rows:
        r["wall_ratio_vs_M3p3"] = ratio(r["wall_ms_median"], m3["wall_ms_median"])
        r["L2_ratio_vs_M3p3"] = ratio(r["L2_rel_err_h1"], m3["L2_rel_err_h1"])
        r["equal_wall_verdict"] = ("M3-p3 strictly better at LOWER wall time"
                                   if r["wall_ms_median"] >= m3["wall_ms_median"]
                                   and r["L2_rel_err_h1"] > m3["L2_rel_err_h1"]
                                   else "see raw columns")
    log("[4] M3-p3 wall=%.3f ms L2=%.4e" % (m3["wall_ms_median"], m3["L2_rel_err_h1"]))
    if best is not None:
        log("[4b] EQUAL-WALL verdict: M3-p3 %.3f ms L2=%.4e ; affordable clip %s "
            "%.3f ms L2=%.4e -> M3 better by %.2fx"
            % (m3["wall_ms_median"], m3["L2_rel_err_h1"], best["arm"],
               best["wall_ms_median"], best["L2_rel_err_h1"],
               ratio(best["L2_rel_err_h1"], m3["L2_rel_err_h1"])))
    header = ["arm", "n_macro_steps", "all_finite_h1", "lam_evals_total",
              "lam_evals_per_macro",
              "aging_matmul_per_macro", "alloc_matmul_per_macro",
              "aux_state_prop_per_macro", "integration_steps_per_macro",
              "T_build_per_run", "wall_ms_median", "wall_ms_min",
              "wall_n_repeats", "wall_ratio_vs_M3p3", "L2_rel_err_h1",
              "L2_ratio_vs_M3p3", "equal_wall_verdict"]
    write_csv_dual("m3_cost.csv", header, rows)
    write_csv_dual("m3_equal_wall.csv",
                   ["arm", "m_substeps", "wall_ms_median", "wall_ms_min",
                    "wall_n_repeats", "L2_rel_err_h1", "wall_ratio_vs_M3p3",
                    "affordable_at_M3_wall"], eq_rows)
    return rows, eq_rows, best


# =============================================================================
# EXPERIMENT 5 : M3-e Devil's Advocate / trivial substitutes / ablations
# =============================================================================
def experiment_5(fits, rows3):
    log("")
    log("=" * 100)
    log("EXPERIMENT 5 | M3-e : Devil's Advocate, trivial substitutes & ablations")
    log("=" * 100)
    ext_by_arm = {}
    for r in rows3:
        ext_by_arm.setdefault(r["arm"], []).append(r)
    attacks = [
        ("static_w+midpoint", "3-line static weights + midpoint Lambda"),
        ("Strang+midpoint", "literal Strang splitting + midpoint Lambda"),
        ("fix_pp+midpoint", "t=0 fixed per-patch weight lookup + midpoint"),
        ("Richardson(EXP2-midpoint)", "Richardson (4*y_{h/2}-y_h)/3 on EXP2-midpoint"),
        ("RK4-explicit", "classical explicit RK4 on the full ODE"),
        ("SSPRK3-explicit", "SSP-RK3 (positive coefficients) on the full ODE"),
        ("Yoshida4-split", "Yoshida 4th-order 3-stage splitting (signed times)"),
        ("AB3-extrap+PT", "Adams-type quadratic Lambda extrapolation + M3 flow"),
        ("AB3-extrap+PT(frozen-startup)", "same, but frozen 2-step startup (no Taylor)"),
        ("Lagrange3-predictor", "3-point Lagrange Lambda(s) from half/full EXP2-midpoint predictors"),
        ("M3+single_w", "M3 Lambda-predictor but ONE static age weight vector"),
        ("M3-p1", "M3 with degree-1 Lambda predictor"),
        ("M3-p2", "M3 with degree-2 Lambda predictor"),
    ]
    rows = []
    e2 = safe_metrics(run_attack("EXP2-midpoint", 1.0)["series"],
                        ref_for(1.0), True)[0]
    m3l2 = safe_metrics(run_attack("M3-p3", 1.0)["series"],
                        ref_for(1.0), True)[0]
    for arm, desc in attacks:
        p, r2, npt = fits.get(arm, (float("nan"),) * 3)
        d1 = run_attack(arm, 1.0)
        d10 = run_attack(arm, 10.0)
        l2h1 = safe_metrics(d1["series"], ref_for(1.0), d1["all_finite"])[0]
        ext = ext_by_arm.get(arm, [])
        large_h_clean = (len(ext) > 0
                         and all(r["all_finite"] and r["neg_S"] == 0
                                 and r["neg_I"] == 0
                                 and r["cons_resid"] <= 1e-13 for r in ext))
        rows.append(dict(arm=arm, description=desc,
                         fit_order=p, fit_R2=r2,
                         h1_L2=l2h1, h1_L2_vs_E2=ratio(l2h1, e2),
                         h1_L2_vs_M3p3=ratio(l2h1, m3l2),
                         h1_negS=d1["neg_S"], h1_negI=d1["neg_I"],
                         h1_cons=d1["cons_resid"], h1_finite=d1["all_finite"],
                         h10_negS=d10["neg_S"], h10_negI=d10["neg_I"],
                         h10_cons=d10["cons_resid"], h10_finite=d10["all_finite"],
                         large_h_clean=large_h_clean,
                         uses_subcycling=(arm in ("Richardson(EXP2-midpoint)",
                                                  "clip+2sub", "clip+40sub")),
                         uses_signed_time=(arm == "Yoshida4-split"),
                         needs_M3_flow=(arm in ("AB3-extrap+PT",
                                                "AB3-extrap+PT(frozen-startup)",
                                                "Lagrange3-predictor")),
                         needs_M3_startup=(arm == "AB3-extrap+PT")))
        log("[5] %-34s order=%-8s R2=%-9s h1 L2=%-11.4e h1negS=%-5d "
            "h10negS=%-6d h10finite=%-5s"
            % (arm, ("%.4f" % p) if np.isfinite(p) else "nan",
               ("%.5f" % r2) if np.isfinite(r2) else "nan",
               l2h1, d1["neg_S"], d10["neg_S"], d10["all_finite"]))
    # q sensitivity of M3 (static-kernel resolution attack)
    log("-- 5b static-kernel node sensitivity q --")
    abl = []
    for q in (4, 6, 8, 12, 16, 24, 32, 48, 64):
        for h in (1.0, 0.0625):
            d = run_attack("M3-q%d" % q, h)
            l2 = safe_metrics(d["series"], ref_for(h), d["all_finite"])[0]
            abl.append(dict(section="q_sensitivity", variant="M3-q%d" % q,
                            h_macro=h, q=q, L2_rel_err=l2,
                            cons_resid=d["cons_resid"], neg_S=d["neg_S"]))
            log("[5b] q=%-3d h=%-8g L2=%.10e" % (q, h, l2))
    for h in (1.0, 0.0625):
        vals = [r["L2_rel_err"] for r in abl if r["h_macro"] == h]
        log("[5b] h=%-8g L2 spread over q=4..64 = %.3e (max/min = %.6f)"
            % (h, max(vals) - min(vals), max(vals) / min(vals)))
    # single static weight ablation order
    p, r2, npt = fits.get("M3+single_w", (float("nan"),) * 3)
    abl.append(dict(section="single_static_weight", variant="M3+single_w",
                    h_macro=float("nan"), q=Q_MAIN,
                    L2_rel_err=float("nan"), fit_order=p, fit_R2=r2))
    log("[5b] single static age weight vector: order=%.4f R2=%.6f"
        % (p, r2))
    header5 = ["arm", "description", "fit_order", "fit_R2", "h1_L2",
               "h1_L2_vs_E2", "h1_L2_vs_M3p3", "h1_negS", "h1_negI",
               "h1_cons", "h1_finite", "h10_negS", "h10_negI", "h10_cons",
               "h10_finite", "large_h_clean", "uses_subcycling",
               "uses_signed_time", "needs_M3_flow", "needs_M3_startup"]
    write_csv_dual("m3_trivial_attacks.csv", header5, rows)
    header6 = ["section", "variant", "h_macro", "q", "L2_rel_err",
               "cons_resid", "neg_S", "fit_order", "fit_R2"]
    write_csv_dual("m3_ablations.csv", header6, abl)
    return rows, abl


# =============================================================================
# report
# =============================================================================
def table(header, rows, cols=None):
    cols = cols or header
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_report(res0, rows1, res1, rows2, fits, floor_rows, rows3,
                 rows4, rows5, abl, eq_rows, eq_best):
    m3 = res1["M3-p3"]
    m3l2 = m3["l2"]
    e2 = res1["EXP2-midpoint"]["l2"]
    static = res1["static_w+midpoint"]["l2"]
    strang = res1["Strang+midpoint"]["l2"]
    clip2 = res1["clip+2sub"]["l2"]
    clip40 = res1["clip+40sub"]["l2"]
    p3, r23, npt3 = fits["M3-p3"]
    p2, r22, _ = fits["M3-p2"]
    m3wall = next(r for r in rows4 if r["arm"] == "M3-p3")["wall_ms_median"]
    e2wall = next(r for r in rows4 if r["arm"] == "EXP2-midpoint")["wall_ms_median"]
    staticwall = next(r for r in rows4 if r["arm"] == "static_w+midpoint")["wall_ms_median"]
    c2wall = next(r for r in rows4 if r["arm"] == "clip+2sub")["wall_ms_median"]
    c40wall = next(r for r in rows4 if r["arm"] == "clip+40sub")["wall_ms_median"]
    ext_m3 = [r for r in rows3 if r["arm"] == "M3-p3"]
    cons_max = max(r["cons_resid"] for r in ext_m3)
    negS = sum(r["neg_S"] for r in ext_m3)
    negI = sum(r["neg_I"] for r in ext_m3)
    allfin = all(r["all_finite"] for r in ext_m3)
    minS = min(r["min_S"] for r in ext_m3)
    q1 = [r["L2_rel_err"] for r in abl if r["section"] == "q_sensitivity"
          and r["h_macro"] == 1.0]
    sing = fits["M3+single_w"]

    def md_table(header, rows, cols=None):
        return (table(header, rows, cols)
                + "\n\n（表中全部数值待用户数值确认）\n")

    txt = []
    A = txt.append
    A("# M3 数值实验报告：Λ-Taylor / 相位型 Markov 宏步积分器")
    A("")
    A("> **全部实测数字标注「数值待用户数值确认」**。本团队不认证数值结果，"
      "只提供可复现测量；任何数值需由用户用自有已校验代码复算确认后方可采信。")
    A("")
    A("生成时间（UTC）：%s  |  Python %s  |  numpy %s  |  scipy %s  |  %s"
      % (datetime.now(timezone.utc).isoformat(timespec="seconds"),
         platform.python_version(), np.__version__, scipy.__version__,
         platform.platform()))
    A("")
    A("复现命令：`C:/Python314/python.exe -u m3_experiment.py > m3_run.log 2>&1`；"
      "共享设定 `import p0b_experiment as P` 原样复用，未自建参考解/基线定义。")
    A("")
    A("---")
    A("")
    A("## 1. 方法定义（可直接写入论文的数学式）")
    A("")
    A("记宿主级 ODE 为二次向量场（$I$ 为 $K$ 个 Erlang 感染年龄箱，"
      "$\\Lambda(y)=\\beta_{sp} W (I\\phi/N)$ 是**状态线性泛函**）：")
    A("")
    A("$$ y' = A y + (L y)\\,(B y),\\qquad "
      "L y=\\Lambda(y),\\quad B y=(-S,\\; +S e_0,\\;0),\\quad "
      "A=\\text{Erlang 老化链}+\\sigma I_{K-1}\\partial_R . $$")
    A("")
    A("**M3 单宏步**（网格恒为 `P.t_grid(h)`，无子步、无 subcycling、无更小 h）：")
    A("")
    A("1. **Λ 的局部 Taylor 预测（唯一投入复杂度的成分）**：仅由当前状态 $y_k$ 解析地给出")
    A("$$ a_0=y_k,\\quad "
      "a_{m+1}=A a_m+\\sum_{j=0}^{m}\\binom{m}{j}(L a_j)(B a_{m-j}),"
      "\\quad \\lambda_j:=L a_j=\\frac{d^j\\Lambda}{dt^j}(t_k),\\ j=0,\\dots,p .$$")
    A("   不使用未来真值、不使用参考解、不迭代、不做整步「预测→重算」。")
    A("2. **保正的 Λ(s) 参数化**（$s\\in[0,h]$）：")
    A("$$ \\Lambda(s)=\\lambda_0\\bigl(1+d_1 s+d_2 s^2+d_3 s^3\\bigr)^2\\ge 0, "
      "\\qquad d_1=\\tfrac{a_1}{2},\\ d_2=\\tfrac{a_2-d_1^2}{2}, "
      "\\ d_3=\\tfrac{a_3}{2}-d_1d_2, \\quad a_j=\\frac{\\lambda_j}{j!\\lambda_0}. $$")
    A("   平方是结构性保正机制：**无 CFL、无 limiter、无 clip、无 post-hoc 截断**；"
      "对任意 $h$ 成立。$\\Lambda(s)-\\Lambda_{\\rm true}(t_k+s)=O(s^{p+1})$。")
    A("3. **非自治 Markov（phase-type）流的精确解**："
      "$H(s)=\\int_0^s\\Lambda(u)du$（由**正系数** Gauss 规则求值，对次数 $\\le 2p$ 多项式精确），")
    A("$$ S(h)=S_0 e^{-H(h)},\\quad D=S_0(1-e^{-H(h)}),\\quad "
      "g(u)=\\Lambda(u)S_0 e^{-H(u)}, $$")
    A("$$ I^{\\rm fresh}_k=D\\sum_j p_j P_k(\\sigma(h-u_j)),\\quad "
      "p_j=\\frac{w_j g(u_j)}{\\sum_m w_m g(u_m)}\\ge0,\\ \\sum_j p_j=1, $$")
    A("$$ \\mathrm{toR}=D\\sum_j p_j\\,\\gamma_{\\rm inc}(K,\\sigma(h-u_j)),\\quad "
      "I(h)=I_0T(h)+I^{\\rm fresh},\\quad R(h)=R_0+\\mathrm{removed}+\\mathrm{toR}. $$")
    A("   其中 $P_k(\\mu)=e^{-\\mu}\\mu^k/k!$、$T(h)=$`P.poisson_T(h)`、"
      "$\\mathrm{removed}=I_0\\,[\\gamma_{\\rm inc}(K-j,\\sigma h)]_{j=0}^{K-1}\\ge0$。")
    A("   $(q,K)$ 矩阵 $[P_k(\\sigma(h-u_j))]$ 与 $\\gamma_{\\rm inc}(K,\\cdot)$ "
      "**与 Λ 完全无关**：每个 $h$ 预计算一次，全步全 patch 复用（即「最简静态权重」）。")
    A("")
    A("**命题 1（无条件保正）**：若 $S_0,I_0,R_0\\ge0$ 且 $\\Lambda(s)\\ge0$，"
      "则 M3 每步 $S,I_k,R\\ge0$，对任意 $h>0$。证明：上式中 "
      "$\\exp(-H)\\in(0,1]$、$D\\ge0$、$p_j\\ge0$、$P_k\\ge0$、"
      "$\\gamma_{\\rm inc}\\in[0,1]$、$T(h)\\ge0$、$\\mathrm{removed}\\ge0$，"
      "全部为**非负数与非负矩阵的加/乘**；浮点实现中 $H$ 由正权 Gauss 规则计算、"
      "$\\Lambda(u_j)$ 由平方计算，故浮点路径亦无非负性破坏。"
      "（数值待用户数值确认：见 §5 极端 CFL 表）")
    A("")
    A("**命题 2（无条件守恒）**：$S(h)+\\sum_k I_k(h)+R(h)=N$ 在精确算术下成立："
      "$\\mathrm{removed}$ 与 $\\mathrm{toR}$ 分别补上 $I_0T(h)$ 与 $I^{\\rm fresh}$ "
      "相对 $D$ 的年龄箱外质量，而 $\\sum_k P_k+\\gamma_{\\rm inc}(K,\\cdot)=1$。"
      "浮点残差实测见 §5。（数值待用户数值确认）")
    A("")
    A("**命题 3（阶）**：$\\Lambda$ 的 $p$ 次 Taylor 预测给出局部误差 $O(h^{p+2})$，"
      "故全局 $L^2$ 误差阶为 $p+1$（$p=2\\Rightarrow3$，$p=3\\Rightarrow4$）。"
      "该结论要求**注入时刻分布** $p_j$ 与 $\\Lambda(s)$ 同步达到同阶；"
      "单一静态权重向量把阶压回 2（实测见 §6）。（数值待用户数值确认）")
    A("")
    A("**成本声明**：$\\lambda_0,\\dots,\\lambda_p$ 共 $p+1$ 次 $W@$ 求值/宏步"
      "（$p=3$ 时为 4 次），**无** Λ_end 重算、**无**半步状态传播、**无**隐式迭代。"
      "静态核矩阵每 $h$ 构造 1 次。与二阶指数中点基线 EXP2-midpoint 的「2 次 Λ 求值」口径不同，"
      "§4 同时给出真实墙钟。")
    A("")
    A("**对照基线 `EXP2-midpoint`（二阶指数中点积分器，本脚本独立实现）**："
      "在单个宏步内把 $\\Lambda$ 冻结于中点估计 $\\Lambda_{mid}$"
      "（由半步状态预测给出，不使用未来真值），用 `P.poisson_T(h)` "
      "精确推进 Erlang 老化链，并以闭式 $\\gamma_{\\rm inc}$ / "
      "Poisson 权重把新感染按年龄分布分配。该基线每宏步 2 次 "
      "$\\Lambda$ 求值、2 次年龄推进，其 h=1 实测值见 §2。"
      "本文全部对照臂（静态权重、Strang、clip 子步、Richardson、"
      "显式 RK、三点 Lagrange、Λ 外推）均由本脚本按同一 "
      "`P.reference_solution` 独立实现，定义与数值可由 "
      "`m3_experiment.py` 一键复现。")
    A("")
    A("### 1.1 技术选型表")
    A("")
    A("| 组件 | 选型 | 理由 | 备选（本次也实测） |")
    A("|---|---|---|---|")
    A("| 方程类型处理 | 非自治线性 phase-type 子系统**精确解** + Λ 的局部 Taylor 预测 | "
      "把全部复杂度投给唯一对指标敏感的 Λ(t)；年龄链精确推进 | 二阶指数中点（闭式卷积）、静态权重、Strang |")
    A("| Λ(s) 参数化 | $\\lambda_0(1+d_1s+d_2s^2+d_3s^3)^2$（正平方） | "
      "对任意 h 结构性非负，无需 limiter/clip | 原始多项式（三点 Lagrange，大 h 变负）、log-Taylor |")
    A("| 累积危险 $H(s)$ | 正系数 Gauss-Legendre（$q'=8$，对次数 $\\le 2p$ 精确） | "
      "浮点路径同样非负；与解析多项式积分交叉校验 | 解析多项式积分（已交叉校验，一致） |")
    A("| 注入时刻积分 | $q=16$ 点 Gauss-Legendre，正权；$p_j$ 归一化 | "
      "$q=4..64$ 实测 L2 极差仅 %s | 单一静态权重向量（阶压回 2） |"
      % fmt(max(q1) - min(q1)))
    A("| 年龄核 / 老化传播 | 预算 $(q,K)$ 静态 Erlang 核 + `P.poisson_T(h)` | "
      "与 Λ 无关，每 h 构造一次；精确 Erlang 老化矩阵 | 闭式 gamma 卷积（二阶指数中点基线） |")
    A("| 参考解 | `P.reference_solution`：DOP853, rtol=1e-10, atol=1e-12 | "
      "与全部基线同一口径，不自建 | rtol=1e-12 交叉校验（§3） |")
    A("| 步长策略 | 单一宏步 $h$ = `P.t_grid(h)`，无子步、无 subcycling、无自适应 | "
      "硬约束；与其它臂完全相同的网格 | clip+nsub、Richardson（均在攻击表中实测） |")
    A("| Jacobian / 导数 | 二次向量场解析 Taylor 递推，无数值 Jacobian、无自动微分 | "
      "4 次 $W@$ 即得 $\\lambda_0..\\lambda_3$ | 半步/整步状态预测（三点 Lagrange，2 次额外传播） |")
    A("| 事件检测 / 脉冲 / 随机性 | 无（无脉冲给药、无干预切换、无随机项） | "
      "本问题无此类结构 | 自检用固定种子 `default_rng(20261107)`，主实验全确定性 |")
    A("| 容差 | 未引入迭代容差（每步为闭式/正权求值的显式一步） | "
      "唯一容差在参考解，沿用 P | — |")
    A("")
    A("### 1.2 自检 / 交叉校验清单（全部由脚本实测，见 `m3_run.log` EXPERIMENT 0）")
    A("")
    A(md_table(["section", "h", "max_abs_diff", "L2_rel_err",
                "prior_value", "rel_dev"], res0))
    A("")
    A("含：`t_grid(h)` 终点检查、移除通量恒等式（精确 Poisson 传播子 vs 稳定 "
      "$\\gamma_{\\rm inc}$ 补集）、静态核行和、$H$ 两种独立求值器交叉校验、"
      "EXP2-midpoint 基线 h=1 对既定基准值的复现（相对偏差 %s）。"
      % fmt(next(r["rel_dev"] for r in res0
                 if r["section"] == "E2_baseline_reproduction")))
    A("")
    A("---")
    A("")
    A("## 2. (M3-a) h=1、60 宏步、同一 P.reference_solution")
    A("")
    A(md_table(["h_macro", "arm", "L2_rel_err", "max_abs_err",
                "L2_ratio_vs_M3p3", "cons_resid", "neg_S", "neg_I",
                "min_S", "all_finite", "n_lambda_evals",
                "n_integration_steps"], rows1))
    A("")
    A("**裁定：%s**。M3-p3 的 $L_2^{rel}$ = %s；"
      "严格优于二阶指数中点基线 EXP2-midpoint %s（%.2f×）、静态权重+midpoint %s（%.2f×）、"
      "Strang+midpoint %s（%.2f×）、clip+2sub %s（%.2f×）、"
      "clip+40sub %s（%.2f×）。（数值待用户数值确认）"
      % ("达标" if all(r["L2_rel_err"] > m3l2 for r in rows1
                       if r["arm"] != "M3-p3" and r["arm"] != "M3-p2")
         else "不达标",
         fmt(m3l2), fmt(e2), ratio(e2, m3l2),
         fmt(static), ratio(static, m3l2), fmt(strang), ratio(strang, m3l2),
         fmt(clip2), ratio(clip2, m3l2), fmt(clip40), ratio(clip40, m3l2)))
    A("")
    A("---")
    A("")
    A("## 3. (M3-b) 收敛阶（h=0.5→0.0625，%d 点 log-log 拟合）" % npt3)
    A("")
    fitrows = [r for r in rows2 if r["section"] == "fit"]
    A(md_table(["arm", "fit_order", "fit_R2", "n_h_points"], fitrows))
    A("")
    A("逐点原始误差见 `m3_convergence.csv`。参考解精度下限交叉检验（rtol=1e-10 "
      "vs 1e-12，独立 DOP853）：")
    A("")
    A(md_table(["h_macro", "ref_max_abs_diff", "M3_L2_vs_rtol1e10",
                "M3_L2_vs_rtol1e12", "rel_change"], floor_rows))
    A("")
    A("**裁定：%s**。M3-p3 阶 = %.4f（R²=%.6f），M3-p2 阶 = %.4f（R²=%.6f），"
      "均 > 2；EXP2-midpoint / 静态权重 / Strang / fix_pp / clip+2sub 全部 ≈2 或 ≈1。"
      "**注意：阶 > 2 并非 M3 独有**——Richardson 外推 %.4f、显式 RK4 %.4f、"
      "三点 Lagrange %.4f 也 > 2，但它们在保正或大 h 稳定上失败（见 §6）。"
      "（数值待用户数值确认）"
      % ("达标" if p3 > 2 and p2 > 2 else "不达标", p3, r23, p2, r22,
         fits["Richardson(EXP2-midpoint)"][0], fits["RK4-explicit"][0],
         fits["Lagrange3-predictor"][0]))
    A("")
    A("---")
    A("")
    A("## 4. (M3-c) 无条件保正守恒，h=10/20/30/60")
    A("")
    A(md_table(["h_macro", "arm", "n_steps", "all_finite", "cons_resid",
                "neg_S", "neg_I", "neg_R", "min_S", "min_lam_u", "min_H",
                "min_pj", "min_toR", "L2_rel_err"],
               [r for r in rows3 if r["arm"] == "M3-p3"]))
    A("")
    A("**裁定：%s**。M3-p3 在 h=%s 上 all_finite=%s、"
      "max cons_resid=%s（阈值 1e-13）、neg_S=%d、neg_I=%d、min_S=%s≥0；"
      "无 CFL、无 limiter、无 clip、无 post-hoc 截断。"
      "**诚实边界**：大 h 的**精度**不成立（h=10 的 L2 见上表，"
      "h=60 的 L2 小值来自近吸收态的指标退化，不得解读为大步长精度）。"
      "（数值待用户数值确认）"
      % ("达标" if allfin and cons_max <= 1e-13 and negS == 0 and negI == 0
         and minS >= 0 else "不达标",
         "/".join("%g" % h for h in H_EXTREME), allfin, fmt(cons_max),
         negS, negI, fmt(minS)))
    A("")
    A("---")
    A("")
    A("## 5. (M3-d) 成本公平（并列三口径，h=1 全 60 步）")
    A("")
    A(md_table(["arm", "all_finite_h1", "lam_evals_per_macro",
                "aging_matmul_per_macro", "alloc_matmul_per_macro",
                "aux_state_prop_per_macro", "integration_steps_per_macro",
                "T_build_per_run", "wall_ms_median", "wall_n_repeats",
                "wall_ratio_vs_M3p3", "L2_rel_err_h1",
                "L2_ratio_vs_M3p3"], rows4))
    A("")
    A("墙钟为 %d 次计时中位数（含 1 次预热），机器/实现相关，**非逐位可复现**。"
      "M3-p3：Λ 求值 %.0f 次/宏步，墙钟 %.3f ms；EXP2-midpoint %.3f ms、"
      "静态权重+midpoint %.3f ms、clip+2sub %.3f ms、clip+40sub %.3f ms。"
      "**不得只报对自己有利的口径**：按 Λ 求值次数 M3-p3 比 EXP2-midpoint 多 "
      "%.1f 倍；按真实墙钟 M3-p3/EXP2-midpoint = %.3f，"
      "M3-p3/clip+40sub = %.3f。（数值待用户数值确认）"
      % (next(r for r in rows4 if r["arm"] == "M3-p3")["wall_n_repeats"],
         next(r for r in rows4 if r["arm"] == "M3-p3")["lam_evals_per_macro"],
         m3wall, e2wall, staticwall, c2wall, c40wall,
         ratio(next(r for r in rows4 if r["arm"] == "M3-p3")["lam_evals_per_macro"],
               next(r for r in rows4 if r["arm"] == "EXP2-midpoint")["lam_evals_per_macro"]),
         ratio(m3wall, e2wall), ratio(m3wall, c40wall)))
    A("")
    if eq_best is not None:
        A("**等墙钟裁定（成本公平的决定性口径，必须披露）**：M3-p3 墙钟 %s ms、"
          "L2=%s；在墙钟不超过 M3-p3 的 clip 子步臂中最优的是 `%s`（%s ms，"
          "L2=%s），M3-p3 优于它 %.2f 倍。完整 `m=%s` 扫表见 "
          "`m3_equal_wall.csv`。（数值待用户数值确认）"
          % (fmt(m3wall), fmt(m3["l2"]), eq_best["arm"],
             fmt(eq_best["wall_ms_median"]), fmt(eq_best["L2_rel_err_h1"]),
             ratio(eq_best["L2_rel_err_h1"], m3["l2"]),
             "/".join(str(r["m_substeps"]) for r in eq_rows)))
    else:
        A("**等墙钟裁定**：本次 clip 子步扫表中没有任何臂的墙钟低于 M3-p3，"
          "因此无法给出「同墙钟」对照；原始扫表见 `m3_equal_wall.csv`。"
          "（数值待用户数值确认）")
    A("")
    A("---")
    A("")
    A("## 6. (M3-e) Devil's Advocate：平凡替代逐一实测")
    A("")
    A(md_table(["arm", "description", "fit_order", "fit_R2", "h1_L2",
                "h1_L2_vs_E2", "h1_L2_vs_M3p3", "h1_negS", "h1_negI",
                "h10_negS", "h10_negI", "h10_finite", "large_h_clean",
                "uses_subcycling", "uses_signed_time", "needs_M3_flow",
                "needs_M3_startup"], rows5))
    A("")
    A("**消融**：静态核节点 q=4..64 的 h=1 全跑 L2 极差 = %s（max/min=%.6f）；"
      "**单一静态权重向量**把 M3-p3 的阶压到 %.4f（R²=%.6f）。"
      "⇒ 指标对**年龄分配/卷积分辨率**几乎不敏感，"
      "唯一敏感成分是 Λ(s) 的时变处理；但「单个静态权重向量」不足以支撑 >2 阶，"
      "必须是**与 Λ 无关的静态多节点核矩阵**（每 h 预算一次）。"
      "（数值待用户数值确认）"
      % (fmt(max(q1) - min(q1)), max(q1) / min(q1), sing[0], sing[1]))
    A("")
    # verdict on trivial substitute existence
    # M3-p1 / M3-p2 / M3+single_w are ABLATIONS OF M3 ITSELF, not independent
    # trivial substitutes; they are reported in the table but excluded here.
    subst = [r for r in rows5 if not r["arm"].startswith("M3")]
    four_axis = [r for r in subst
                 if np.isfinite(r["fit_order"]) and r["fit_order"] > 2
                 and r["h1_negS"] == 0 and r["h1_negI"] == 0
                 and r["h1_cons"] <= 1e-13 and r["large_h_clean"]]
    four_names = set(r["arm"] for r in four_axis)
    order_only = [r for r in subst
                  if np.isfinite(r["fit_order"]) and r["fit_order"] > 2
                  and r["arm"] not in four_names]
    beats_m3 = [r for r in subst
                if np.isfinite(r["h1_L2_vs_M3p3"])
                and r["h1_L2_vs_M3p3"] < 1.0]
    A("### 「是否存在与 M3 打平收敛阶/守恒/保正/大步长稳定的平凡替代」的明确回答")
    A("")
    A("判定口径（全部实测）：阶 > 2；h=1 `cons≤1e-13`、零负；"
      "h=10/20/30/60 全部 `all_finite`、`cons≤1e-13`、`neg_S=neg_I=0`"
      "（表中 `large_h_clean`）。")
    A("")
    A("说明：`M3-p1` / `M3-p2` / `M3+single_w` 是 **M3 自身的消融臂**，"
      "不是独立平凡替代，故在上表列出但不计入本裁定。")
    A("")
    if four_axis:
        A("**存在（如实报告，不包装）**。满足上述四项轴的平凡替代共 %d 个："
          % len(four_axis))
        for r in four_axis:
            A("- `%s`：阶 %.4f（R²=%.5f）；h=1 L2=%s（是二阶指数中点基线 EXP2-midpoint %s 的 "
              "%.3f 倍，**不满足 M3-a 的严格优于**）；h=10 negS=%d negI=%d；"
              "复用 M3 相位型流=%s，需要 M3 Taylor 启动=%s。"
              % (r["arm"], r["fit_order"], r["fit_R2"], fmt(r["h1_L2"]),
                 fmt(e2), r["h1_L2_vs_E2"], r["h10_negS"], r["h10_negI"],
                 r["needs_M3_flow"], r["needs_M3_startup"]))
        A("")
        A("即：**四项轴（阶/守恒/保正/大 h 稳定）可以被一个更便宜的 Λ 外推替身打平**，"
          "但它（a）阶只有 %s 而非 M3-p3 的 %s；（b）h=1 精度比 EXP2-midpoint "
          "更差，故 (M3-a) 不成立；（c）复用 M3 的非自治相位型流与正平方参数化，"
          "不是「三行静态权重」级的独立构造。因此 **M3 不能声称「不存在平凡替代」，"
          "也不能声称「唯一」**。（数值待用户数值确认）"
          % (fmt(four_axis[0]["fit_order"]), fmt(p3)))
    else:
        A("**不存在**：本次构造的全部平凡替代中，没有一个同时满足上述四项轴。"
          "（数值待用户数值确认）")
    A("")
    A("**另有 %d 个替身阶 > 2 但不满足四项轴**（如实列出）：" % len(order_only))
    for r in order_only:
        A("- `%s`：阶 %.4f，h=1 L2=%s、negS=%d negI=%d，"
          "h=10 negS=%d negI=%d finite=%s，large_h_clean=%s，"
          "subcycling=%s，负时间=%s。"
          % (r["arm"], r["fit_order"], fmt(r["h1_L2"]), r["h1_negS"],
             r["h1_negI"], r["h10_negS"], r["h10_negI"], r["h10_finite"],
             r["large_h_clean"], r["uses_subcycling"], r["uses_signed_time"]))
    A("")
    if beats_m3:
        A("**特别注意（对自己不利的数字）**：%d 个替身在 h=1 的 L2 比 M3-p3 "
          "更小：" % len(beats_m3))
        for r in beats_m3:
            A("- `%s`：L2=%s，是 M3-p3 的 %.4f 倍（更准），"
              "但 large_h_clean=%s、h1_negS=%d、h1_negI=%d、subcycling=%s。"
              % (r["arm"], fmt(r["h1_L2"]), r["h1_L2_vs_M3p3"],
                 r["large_h_clean"], r["h1_negS"], r["h1_negI"],
                 r["uses_subcycling"]))
        A("⇒ M3 的贡献**不是**「h=1 最准」，而是「阶 > 2 + 无条件守恒保正 + "
          "大 h 稳定 + 无子步/无 limiter」四项在同宏步网格上的可证组合。"
          "（数值待用户数值确认）")
    A("")
    A("---")
    A("")
    A("## 7. 诚实边界 / 未检验角度 / 成本口径局限")
    A("")
    A("1. **未检验角度**：非 52 州真实场、非 $R_0=15$、非该 $\\phi$ 核/$\\sigma$/"
      "$\\beta_{sp}$ 的参数区；其它 $|\\lambda_0|$ 量级、$\\lambda_0=0$ 吸收态附近、"
      "$\\Lambda(s)$ 的 Taylor 半径小于 $h$ 的强非线性区；DAE/时滞/随机版本；"
      "K 变化；长时间（$T\\gg60$）误差累积；其它范数（$L^\\infty$、逐点相对）。")
    A("2. **成本口径局限**：墙钟只在本机单次会话、单线程 NumPy 下测量，"
      "未控制 CPU 频率/后台负载；脚本在每次独立 run 前清空 `T(h)`/静态核缓存，"
      "使每个 60 步 run 自行承担一次构造成本（另列 `T_build_per_run`），"
      "避免把预算成本转嫁给别的臂；不同臂的 Python 层调用次数不同，"
      "wall clock 不可逐位复现。"
      "按 Λ 求值次数口径 M3-p3 比 EXP2-midpoint 更贵，按墙钟口径更便宜——"
      "两种口径都列出，不择一。")
    A("3. **大 h 精度**：M3-c 只主张有限/守恒/保正，**不主张精度**；"
      "h=10 的 L2 明显大于 h=1，h=60 的 L2 小值是指标退化（近吸收态，"
      "t_grid 只含 2 个时刻），不得作为大步长精度证据。")
    A("4. **平凡替代未穷尽**：未测 IMEX / expRK / 更高阶 Patankar / "
      "更高阶 Adams 启动方案 / 自适应 Richardson / 其它正平方参数化；"
      "不能据此声称「唯一」。")
    A("5. **参考解**：沿用 `P.reference_solution`（DOP853, rtol=1e-10, atol=1e-12），"
      "未自建；§3 给出与更紧容差参考解的交叉检验，但未把参考解误差计入误差棒。")
    A("6. **log 域保护不是 limiter**：`log|Lambda|` 中的 `1e-300` 下界仅作用于"
      "对数参数以避免 `log(0)`；不改变任何状态、不截断负值。"
      "$\\lambda_0\\le0$ 只在精确吸收态出现，此时 $\\Lambda(s)\\equiv0$ 是精确解。")
    A("7. **大 h 的对数域退化**：当 $H(u_j)$ 大到令全部节点注入密度下溢时，"
      "`p_j` 退化为正的节点权重（实测触发次数 `n_Zzero`：h=%s）。"
      "该分支仍严格保正守恒，但**不再对应真实注入时刻分布**，"
      "是 numerics 退化而非方法精度；大 h 的 L2 不得据此解读。（数值待用户数值确认）"
      % "/".join("%g:%d" % (r["h_macro"], r["n_Zzero"])
                 for r in rows3 if r["arm"] == "M3-p3"))
    A("8. 所有数字：**数值待用户数值确认**。")
    A("")
    A("---")
    A("")
    A("## 8. 产物与复现")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `m3_experiment.py` | 全部实测脚本（自足，只读项目） |")
    A("| `m3_run.log` | 完整 stdout |")
    A("| `m3_results.md` | 本报告 |")
    for name, nr, d1_, d2_, ok in MD5_LOG:
        A("| `figures/%s` + `submission/figures/%s` | %d 行，md5=%s，双份一致=%s |"
          % (name, name, nr, d1_, ok))
    A("")
    A("环境：`C:/Python314/python.exe`，numpy %s / scipy %s；无 torch、无新增依赖；"
      "固定随机种子仅用于 experiment_0 的 removal identity 随机检验 "
      "（`np.random.default_rng(20261107)`），主实验全为确定性计算。"
      % (np.__version__, scipy.__version__))
    A("")
    A("一键复现：`C:/Python314/python.exe -u m3_experiment.py > m3_run.log 2>&1`")
    with open(os.path.join(os.path.dirname(HERE), "m3_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    log("  [report] wrote m3_results.md (%d lines)" % len(txt))


# =============================================================================
# main
# =============================================================================
def main():
    quick = "--quick" in sys.argv
    log("=" * 100)
    log("M3 experiment | Lambda-Taylor / phase-type integrator | %d states | K=%d"
        % (n, K))
    log("sigma=%.10f beta_sp=%.10f RHO field from P | Tspan=[%g,%g]"
        % (SIG, BETA, P.T0, P.T1))
    log("ALL MEASURED NUMBERS BELOW ARE LABELLED: 数值待用户数值确认")
    log("=" * 100)
    res0 = experiment_0()
    rows1, res1 = experiment_1()
    rows2, fits, floor_rows = experiment_2()
    rows3 = experiment_3()
    rows4, eq_rows, eq_best = experiment_4(res1, quick=quick)
    rows5, abl = experiment_5(fits, rows3)
    log("")
    log("=" * 100)
    log("SELF-TEST / CSV md5 summary")
    log("=" * 100)
    for name, nr, d1_, d2_, ok in MD5_LOG:
        log("  %-34s rows=%-4d md5=%s dual_identical=%s" % (name, nr, d1_, ok))
    write_report(res0, rows1, res1, rows2, fits, floor_rows, rows3,
                 rows4, rows5, abl, eq_rows, eq_best)
    m3 = res1["M3-p3"]["l2"]
    log("")
    log("FINAL: M3-p3 h=1 L2=%s | fit order=%.4f R2=%.6f | "
        "wall=%.3f ms | extreme CFL rows=%d"
        % (fmt(m3), fits["M3-p3"][0], fits["M3-p3"][1],
           next(r for r in rows4 if r["arm"] == "M3-p3")["wall_ms_median"],
           len(rows3)))
    log("FINAL: 数值待用户数值确认")
    log("=" * 100)


if __name__ == "__main__":
    main()
