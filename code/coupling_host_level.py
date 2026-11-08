# =============================================================================
# coupling_host_level.py
# -----------------------------------------------------------------------------
# Path-C reference implementation (numpy + scipy.special): host-level age-binned
# metapopulation + nested within-host 7-variable ODE.
#
# HOST-LEVEL MACRO STEP = M3
# --------------------------
# The coupled host-level equation is a QUADRATIC vector field driven by the
# scalar, state-linear force of infection
#
#       y' = A y + (L y)(B y),
#       L y  = Lambda(y) = beta_sp * W @ ( (I @ phi) / N )        (per patch)
#       B y  = (-S, +S e_0, 0),   A y = (0, -sigma I + sigma shift(I), sigma I_{K-1})
#
# For this structure the force of infection has exact Taylor coefficients
#
#       a_0 = y_k,  a_{m+1} = A a_m + sum_{j=0}^{m} C(m,j) (L a_j)(B a_{m-j}),
#       lambda_j := L a_j = d^j Lambda / dt^j (t_k),   j = 0..p,
#
# obtained from the CURRENT state only (no future truth, no iteration).  The
# in-step force of infection is parametrised as a NON-NEGATIVE square,
#
#       Lambda(s) = lambda_0 (1 + d1 s + d2 s^2 + d3 s^3)^2 >= 0,   s in [0,h],
#
# with d1 = a1/2, d2 = (a2 - d1^2)/2, d3 = a3/2 - d1 d2 and a_j = lambda_j/(j! lambda_0).
# The non-autonomous phase-type (Erlang aging) subsystem is then integrated in
# closed form, so one macro step is
#
#       H(s)  = int_0^s Lambda(u) du          (positive-weight Gauss, exact deg<=2p)
#       S(h)  = S_0 exp(-H(h)),   D = S_0 (1 - exp(-H(h))),
#       g(u)  = Lambda(u) S_0 exp(-H(u)),
#       I_k^new = D sum_j p_j P_k(sigma (h-u_j)),   p_j = w_j g(u_j) / sum_m w_m g(u_m) >= 0,
#       toR     = D sum_j p_j gamma_inc(K, sigma (h-u_j)),
#       I(h)  = I_0 T(h) + I^new,   R(h) = R_0 + removed + toR.
#
# Every ingredient is a non-negative number or a non-negative matrix product:
# no limiter, no clipping, no truncation, no CFL condition, no subcycling.
# `(q,K)` kernel matrices P_k(sigma(h-u_j)) and gamma_inc(K, sigma(h-u_j)) are
# Lambda-INDEPENDENT and are built once per (h, sigma), then reused by all patches.
#
# Event handling: the two within-host switches (delta_hat at t = 4.2 d, antibody
# onset at t = 10 d) live in the OFFLINE within-host kernel; at host level a
# macro step never straddles them because they are multiples of the host step.
# `simulate()` nevertheless advances with the ACTUAL grid increment t[k+1]-t[k],
# so any event-aligned grid is integrated exactly at its own step boundaries.
#
# Structure
# ---------
#   * OFFLINE within-host: read the single canonical trajectory written by
#     regen_fig3_ours.py and turn its V(tau) into the infectious kernel
#     phi(tau) = V(tau) * Tg / int_0^{tau*} V dtau  (so int phi dtau = Tg).
#   * HOST level: per patch i carry S^h_i, R^h_i (scalars) + I^h_{i,k} over K age bins.
#         Lambda_i = beta_sp * sum_j w_ij * Phi_j / N_j ,  Phi_j = sum_k phi_k I^h_{j,k}
#     advanced with one M3 macro step per grid interval.
#
# Reused (NOT modified): ms_numpy.STATE_GEO, ms_numpy.recalibrate_gravity_coupling.
#
# The trajectory, the compiled Fig.1 data and this kernel share one initial
# condition y0 = [3800, 0, 1, 0, 1e-4, 0, 0] and one within-host kernel; no second
# integration of the within-host ODE is performed here.
# =============================================================================
import os
import sys
import math
import csv
import numpy as np
from pathlib import Path
from numpy.polynomial.legendre import leggauss
from scipy.special import gammainc, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = Path(__file__).resolve().parents[1]     # submission/
sys.path.insert(0, HERE)
import ms_numpy as msn  # numpy port of the within-host solvers (read-only reuse)

DTYPE = np.float64

# ----------------------------- fixed parameters -----------------------------
BETA_SP = 1.1538      # = R0/Tg = 15/13, the calibrated per-day spatial coupling
R0_TARGET = 15.0
TG = 13.0             # generation interval (days)
K_BINS = 40           # age bins
EPS_V = 1e-3          # V cutoff for tau*
H = 0.1               # host macro step (days)
T_SPAN = (0.0, 60.0)  # demo horizon (days)

# M3 numerical parameters
M3_ORDER = 3          # Taylor degree p -> global order p+1 = 4
Q_INCIDENCE = 16      # Gauss nodes for the in-step incidence integral
Q_HAZARD = 8          # Gauss nodes for the (positive) cumulative-hazard integral
ADAPT_TOL = 0.02      # coupling-resolution tolerance of the adaptive subcycle controller

TRAJ_CSV = str(SUB / "figures" / "fig3_withinhost_traj.csv")
RHO_CSV = str(SUB / "figures" / "fig2_tycho_field.csv")
OUT_CSV = os.path.join(HERE, "coupling_host_level_results.csv")
OUT_TXT = os.path.join(HERE, "coupling_host_level_report.txt")

KIDX = np.arange(K_BINS, dtype=float)


# =============================================================================
# 1. OFFLINE within-host 7-var trajectory  ->  infectious kernel phi(k)
# =============================================================================
def build_infectious_kernel(traj_csv=TRAJ_CSV, eps_v=EPS_V, K=K_BINS, tg=TG):
    """Discretise phi(tau) from the single canonical within-host trajectory.

    The trajectory is written by regen_fig3_ours.py and is also the data source
    of compiled Fig.1; no within-host ODE is integrated here.
    """
    D = np.loadtxt(traj_csv)
    ts = D[:, 0]
    Vs = D[:, 5]

    t_peak = float(ts[int(np.argmax(Vs))])
    V_peak = float(Vs.max())

    # tau* = first tau > t_peak with V(tau) <= eps_v
    sel = ts > t_peak
    below = ts[sel][Vs[sel] <= eps_v]
    tau_star = float(below[0]) if below.size > 0 else float(ts[-1])

    # integral of V over [0, tau_star]
    m = ts <= tau_star + 1e-9
    tsel = ts[m]
    vsel = Vs[m]
    Vint = float(np.trapezoid(vsel, tsel))

    # discretise into K bins, sample phi at bin centres
    dtau = tau_star / K
    phis = np.zeros(K, dtype=DTYPE)
    for k in range(K):
        tc = (k + 0.5) * dtau
        Vk = float(np.interp(tc, tsel, vsel))
        phis[k] = Vk * tg / Vint
    sigma = K / tau_star
    sum_phi_dtau = float(np.sum(phis) * dtau)
    return dict(phis=phis, tau_star=tau_star, dtau=dtau, sigma=sigma,
                t_peak=t_peak, V_peak=V_peak, Vint=Vint,
                sum_phi_dtau=sum_phi_dtau, ts=ts, Vs=Vs)


# =============================================================================
# 2. Gravity matrix + population  (reuse existing Xia-2004 builder)
# =============================================================================
def build_gravity(state_names, r0_target=R0_TARGET, tg=TG):
    rc = msn.recalibrate_gravity_coupling(
        state_names, msn.STATE_GEO, R0_target=r0_target, Tg=tg)
    W = rc['W']
    beta_sp = float(rc['beta_sp_cal'])
    N = np.array([msn.STATE_GEO[s][2] for s in state_names], dtype=DTYPE)
    return W, N, beta_sp


def load_rho(path=RHO_CSV):
    """fig2_tycho_field.csv is space-separated:  rank  state  rho  (TX normalised=1)."""
    states, rho = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            states.append(parts[1])
            rho.append(float(parts[2]))
    return states, np.array(rho, dtype=DTYPE)


# =============================================================================
# 3. Lambda-independent phase-type kernels (built ONCE per (h, sigma))
# =============================================================================
_KERN_CACHE = {}
_T_CACHE = {}


def static_kernel(h, sigma, q=Q_INCIDENCE, qh=Q_HAZARD, K=K_BINS):
    """All h-only (Lambda-independent) ingredients of one macro step.

        Pmat[j,k] = P_k(sigma (h-u_j))            (q,K)   >= 0
        surv[j]   = gamma_inc(K, sigma (h-u_j))   (q,)    >= 0
        sv_old[k] = gamma_inc(K-k, sigma h)       (K,)    >= 0
    """
    key = ("k", float(h), float(sigma), int(q), int(qh), int(K))
    hit = _KERN_CACHE.get(key)
    if hit is not None:
        return hit
    xq, wq = leggauss(int(q))
    u = 0.5 * h * (xq + 1.0)
    w = 0.5 * h * wq
    mu = sigma * (h - u)
    kk = np.arange(K, dtype=DTYPE)
    logp = (-mu[:, None]
            + kk[None, :] * np.log(np.where(mu[:, None] > 0, mu[:, None], 1e-300))
            - gammaln(kk[None, :] + 1.0))
    Pmat = np.exp(logp)
    surv = gammainc(float(K), mu)
    xh, wh = leggauss(int(qh))
    out = dict(u=u, w=w, Pmat=Pmat, surv=surv, xh=xh, wh=wh,
               sv_old=gammainc(float(K) - kk, np.full_like(kk, sigma * h)))
    _KERN_CACHE[key] = out
    return out


def kernel_rowsum_dev(kernel):
    """max |sum_k P_k(mu_j) + gamma_inc(K, mu_j) - 1| ; must be ~1e-16."""
    return float(np.max(np.abs(kernel["Pmat"].sum(1) + kernel["surv"] - 1.0)))


def poisson_T(sigma, h, K=K_BINS):
    """Exact propagator of the linear aging chain over h, T[k,m] = p_{k-m}, p_n>=0."""
    key = ("T", float(sigma), float(h), int(K))
    hit = _T_CACHE.get(key)
    if hit is not None:
        return hit
    nh = np.arange(K)
    p = (np.exp(-sigma * h) * (sigma * h) ** nh
         / np.array([math.factorial(int(x)) for x in nh], dtype=DTYPE))
    T = np.zeros((K, K), dtype=DTYPE)
    for k in range(K):
        T[k, :k + 1] = p[np.arange(k, -1, -1)]
    _T_CACHE[key] = T
    return T


# =============================================================================
# 4. M3: Lambda Taylor coefficients from the current state only
# =============================================================================
def _L_of(st, W, N, phis, beta_sp):
    return beta_sp * (W @ ((st[1] @ phis) / N))


def _B_of(st, K):
    bI = np.zeros_like(st[1])
    bI[:, 0] = st[0]
    return (-st[0], bI, np.zeros_like(st[0]))


def _A_of(st, sigma, K):
    zS = np.zeros_like(st[0])
    dI = -sigma * st[1]
    dI[:, 1:] += sigma * st[1][:, :-1]
    return (zS, dI, sigma * st[1][:, K - 1])


def lam_derivs(S, I, R, W, N, phis, sigma, beta_sp, p=M3_ORDER, K=K_BINS):
    """lambda_j = d^j Lambda/dt^j at the current state, j = 0..p.

    y' = A y + (L y)(B y) is quadratic, hence for m >= 0
        y^{(m+1)} = A y^{(m)} + sum_{j=0}^{m} C(m,j) (L y^{(j)})(B y^{(m-j)}).
    Everything is evaluated at the current state; no future value is used.
    """
    a = [(S.copy(), I.copy(), R.copy())]
    Lc = [_L_of(a[0], W, N, phis, beta_sp)]
    lam = [Lc[0].copy()]
    for m in range(p):
        acc = _A_of(a[m], sigma, K)
        for j in range(m + 1):
            c = math.comb(m, j) * Lc[j]
            bv = _B_of(a[m - j], K)
            acc = (acc[0] + c * bv[0],
                   acc[1] + c[:, None] * bv[1],
                   acc[2] + c * bv[2])
        a.append(acc)
        Lc.append(_L_of(acc, W, N, phis, beta_sp))
        lam.append(Lc[-1].copy())
    return lam


def m3_square_coeffs(S, I, R, W, N, phis, sigma, beta_sp, p=M3_ORDER):
    """(l0, d1, d2, d3, n_nonpos) of Lambda(s) = l0 (1 + d1 s + d2 s^2 + d3 s^3)^2."""
    lam = lam_derivs(S, I, R, W, N, phis, sigma, beta_sp, p=p)
    l0 = lam[0]
    pos = l0 > 0
    l0safe = np.where(pos, l0, 0.0)
    inv = np.where(pos, 1.0 / np.where(pos, l0, 1.0), 0.0)
    a1 = lam[1] * inv
    a2 = 0.5 * lam[2] * inv if p >= 2 else np.zeros_like(l0)
    a3 = (lam[3] / 6.0) * inv if p >= 3 else np.zeros_like(l0)
    d1 = a1 / 2.0
    d2 = (a2 - d1 ** 2) / 2.0
    d3 = a3 / 2.0 - d1 * d2
    return l0safe, d1, d2, d3, int((~pos).sum())


def hazard_and_lambda(l0, d1, d2, d3, kernel, h):
    """H(u_j), H(h) and Lambda(u_j) by POSITIVE-weight Gauss quadrature.

    Exact for polynomials of degree <= 2p in the integrand, and non-negative by
    construction: weights > 0, Lambda = l0 * P^2 with l0 >= 0.
    """
    u = kernel["u"]
    xh, wh = kernel["xh"], kernel["wh"]
    V = 0.5 * u[:, None] * (xh[None, :] + 1.0)
    Cw = 0.5 * u[:, None] * wh[None, :]
    PV = (1.0 + d1[:, None, None] * V[None, :, :]
          + d2[:, None, None] * V[None, :, :] ** 2
          + d3[:, None, None] * V[None, :, :] ** 3)
    lamV = l0[:, None, None] * PV ** 2
    H = np.einsum("jl,njl->nj", Cw, lamV)
    Vh = 0.5 * h * (xh + 1.0)
    Cwh = 0.5 * h * wh
    PVh = (1.0 + d1[:, None] * Vh[None, :]
           + d2[:, None] * Vh[None, :] ** 2
           + d3[:, None] * Vh[None, :] ** 3)
    lamVh = l0[:, None] * PVh ** 2
    Hh = (Cwh[None, :] * lamVh).sum(1)
    Pu = (1.0 + d1[:, None] * u[None, :]
          + d2[:, None] * u[None, :] ** 2
          + d3[:, None] * u[None, :] ** 3)
    lam_u = l0[:, None] * Pu ** 2
    return dict(H=H, Hh=Hh, lam_u=lam_u, Pu=Pu)


# =============================================================================
# 5. M3 production host step  (one macro step, no limiter / CFL / subcycling)
# =============================================================================
def host_step(S, Ibins, R, W, N, phis, sigma, beta_sp, h,
              p=M3_ORDER, kernel=None, T=None):
    """One M3 host-level macro step. Public signature unchanged.

    Returns (S_new, I_new, R_new, stat).  `stat` keeps the historical keys
    (`Lambda`, `theta`, `C`, `newinf`, `frac_active`, `max_C`, `min_S`,
    `mass_gain_R`) plus M3 diagnostics; `theta`/`frac_active` are 0/1-valued
    because there is no limiter in this step.
    """
    K = Ibins.shape[1]
    if kernel is None:
        kernel = static_kernel(h, sigma, K=K)
    if T is None:
        T = poisson_T(sigma, h, K=K)

    l0, d1, d2, d3, n_nonpos = m3_square_coeffs(
        S, Ibins, R, W, N, phis, sigma, beta_sp, p=p)
    hz = hazard_and_lambda(l0, d1, d2, d3, kernel, h)
    H, Hh, lam_u = hz["H"], hz["Hh"], hz["lam_u"]

    # incidence-time distribution: log-domain, no state clipping
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
                min_pj=float(np.min(pj)), min_toR=float(np.min(toR)),
                n_subcycle=1, n_lam_eval=p + 1, n_nonpos_l0=n_nonpos,
                n_Zzero=int((Z <= 0).sum()))
    return S_new, I_new, R_new, stat


# =============================================================================
# 6. Frozen legacy limiter step (reference only; NOT used by the M3 path)
# =============================================================================
def host_mpp_limiter(S, I_total, R, dS, dI, dR, lo=0.0):
    """Zhang-Shu single-factor limiter on the 3-vector [S, I_total, R].

    RETAINED AS A BASELINE REFERENCE ONLY: `p0b_experiment.py` reproduces this
    limiter analytically, and `host_step_limiter_reference` below is the frozen
    pre-M3 coupled step kept so the baseline can still be reproduced.  The M3
    production step (`host_step`) does NOT call this function.

        theta_i = min(1, S_i / X_i)   for X_i > 0,
        theta_i = 1                   for X_i = 0.
    """
    state = np.stack([S, I_total, R], axis=-1)
    delta = np.stack([dS, dI, dR], axis=-1)
    neg = delta < 0
    safe = np.where(neg, delta, -1.0)
    ratio = np.where(neg, (lo - state) / np.where(safe < 0, safe, -1.0), np.inf)
    theta_raw = np.min(ratio, axis=-1, keepdims=True)
    theta = np.clip(theta_raw, 0.0, 1.0)
    return theta[..., 0], theta_raw[..., 0]


def host_step_limiter_reference(S, Ibins, R, W, N, phis, sigma, beta_sp, h):
    """FROZEN pre-M3 coupled step: limiter-guarded transfer + exact Poisson aging.

    Kept verbatim so that the baseline comparison (`tab:bare`, `fig:stress`) and
    `p0b_experiment.validate_ours_matches_mainchain()` remain reproducible.
    This function is NOT on the production path.
    """
    K = Ibins.shape[1]
    Phi = Ibins @ phis
    Lambda = beta_sp * (W @ (Phi / N))
    C = h * Lambda
    a = Lambda * S
    I_total = Ibins.sum(axis=1)
    dS = -h * a
    dI = h * a
    dR = np.zeros_like(S)
    theta, _ = host_mpp_limiter(S, I_total, R, dS, dI, dR)
    newinf = theta * (h * a)
    S_new = S - newinf

    T = poisson_T(sigma, h, K=K)
    I_new = Ibins @ T.T
    removed = Ibins.sum(axis=1) - I_new.sum(axis=1)
    I_new[:, 0] += newinf
    R_new = R + removed

    stat = dict(Lambda=Lambda, theta=theta, C=C, newinf=newinf,
                frac_active=float(np.mean(theta < 1.0 - 1e-12)),
                max_C=float(C.max()), min_S=float(S_new.min()),
                mass_gain_R=float(removed.sum()), n_subcycle=1,
                n_lam_eval=1)
    return S_new, I_new, R_new, stat


# =============================================================================
# 7. Opponent set (second-order / clipping baselines), all Lambda-driven
# =============================================================================
def lam_of(S, Ibins, W, N, phis, beta_sp):
    return beta_sp * (W @ ((Ibins @ phis) / N))


def fresh_closed(h, S, lam, sigma, K=K_BINS):
    """Exact new-infection age distribution for constant Lambda (closed form)."""
    a = sigma - lam
    if np.any(a <= 0):
        raise FloatingPointError(
            "exponential-midpoint closed form has no continuation at "
            "min(sigma-Lam)=%g" % float(np.min(a)))
    kk = np.arange(K, dtype=DTYPE)
    ratio_sa = (sigma / a)[:, None] ** kk[None, :] / a[:, None]
    G = gammainc(kk[None, :] + 1.0, (a * h)[:, None])
    return (lam * S * np.exp(-lam * h))[:, None] * ratio_sa * G


def w_uniform(h, sigma, K=K_BINS):
    """One static age-weight vector (order-2 ceiling)."""
    kk = np.arange(K, dtype=DTYPE)
    return gammainc(kk + 1.0, sigma * h) / (sigma * h)


def exp2_step(S, Ibins, R, h, lam, sigma, mode="closed", w=None, T=None,
              K=K_BINS):
    """Second-order exponential step with Lambda FROZEN over the step."""
    if T is None:
        T = poisson_T(sigma, h, K=K)
    Iold = Ibins @ T.T
    removed = Ibins.sum(axis=1) - Iold.sum(axis=1)
    S_new = S * np.exp(-lam * h)
    delivered = S - S_new
    if mode == "closed":
        If = fresh_closed(h, S, lam, sigma, K=K)
    elif mode == "static":
        If = delivered[:, None] * w_uniform(h, sigma, K=K)[None, :]
    elif mode == "fix":
        If = delivered[:, None] * np.asarray(w)[None, :]
    elif mode == "strang":
        T2 = poisson_T(sigma, 0.5 * h, K=K)
        b1 = Ibins @ T2.T
        rm1 = Ibins.sum(axis=1) - b1.sum(axis=1)
        b1[:, 0] += delivered
        b2 = b1 @ T2.T
        rm2 = b1.sum(axis=1) - b2.sum(axis=1)
        return S_new, b2, R + rm1 + rm2
    else:
        raise ValueError(mode)
    toR = delivered - If.sum(axis=1)
    return S_new, Iold + If, R + removed + toR


def midpoint_lambda(S, Ibins, R, h, W, N, phis, sigma, beta_sp,
                    mode="closed", w=None):
    """Half-step predictor -> Lambda at the midpoint (no future truth).

    The half step uses the ARM'S OWN allocation rule (`mode`).  Only
    ``mode='closed'`` needs sigma > Lambda; the static and Strang arms are
    therefore defined for every coupling strength, and only the closed-form
    exponential arm can be genuinely undefined.
    """
    lam0 = lam_of(S, Ibins, W, N, phis, beta_sp)
    _, b_half, _ = exp2_step(S, Ibins, R, 0.5 * h, lam0, sigma,
                             mode=mode, w=w)
    return lam_of(S, b_half, W, N, phis, beta_sp)


def clip_step(S, Ibins, R, h, sigma, beta_sp, W, N, phis, measure=True):
    """Forward-Euler transfer with a post-hoc non-negativity CLIP (baseline)."""
    Phi = Ibins @ phis
    Lam = beta_sp * (W @ (Phi / N))
    X = h * Lam * S
    S_raw = S - X
    E_clip = float(np.maximum(-S_raw, 0.0).sum()) if measure else 0.0
    S_new = np.maximum(S_raw, 0.0)
    T = poisson_T(sigma, h, K=Ibins.shape[1])
    I_new = Ibins @ T.T
    removed = Ibins.sum(axis=1) - I_new.sum(axis=1)
    I_new[:, 0] += X
    return S_new, I_new, R + removed, E_clip, Lam


def adaptive_subcycle(S, Ibins, R, W, N, phis, sigma, beta_sp, h,
                      tol=ADAPT_TOL):
    """Coupling-resolution controller for the baseline arms (mirrors
    measles_solvers_pytorch.adaptive_coupling_subcycle).

        nu = h * beta_sp * max_i sum_j w_ij rho_j ,   rho_j = Phi_j / N_j
        K  = max(1, ceil(nu / tol))

    The baselines are given this controller so that they are never locked into
    a fixed-step degenerate mode.  M3 itself needs no controller (it is
    unconditionally stable and positive at the macro step).
    """
    Phi = Ibins @ phis
    rho = Phi / N
    nu = h * beta_sp * float(np.max(rho @ W.T))
    return max(1, int(math.ceil(nu / tol)))


def baseline_step(arm, S, Ibins, R, W, N, phis, sigma, beta_sp, h,
                  substeps=None, tol=ADAPT_TOL):
    """One macro step of an opponent arm on the SAME macro grid.

    arm:
      'EXP2-midpoint'        frozen-Lambda midpoint, adaptive subcycling
      'static_w+midpoint'    one static age-weight vector, adaptive subcycling
      'Strang+midpoint'      literal Strang splitting, adaptive subcycling
      'clip'                 post-hoc clip, adaptive subcycling
      'clip+2sub'/'clip+40sub' post-hoc clip, FIXED substep count (cost axis)
      'M3-p2'                M3 with a degree-2 Lambda predictor (ablation)
    """
    K = Ibins.shape[1]
    fixed = None
    if substeps is not None:
        fixed = max(1, int(substeps))
    elif arm in ("clip+2sub",):
        fixed = 2
    elif arm in ("clip+40sub",):
        fixed = 40
    elif arm == "M3-p2":
        fixed = 1

    nsub = fixed if fixed is not None else adaptive_subcycle(
        S, Ibins, R, W, N, phis, sigma, beta_sp, h, tol=tol)

    hs = h / nsub
    T = poisson_T(sigma, hs, K=K)
    E_clip = 0.0
    nlam = 0
    for _ in range(nsub):
        if arm == "M3-p2":
            S, Ibins, R, st = host_step(S, Ibins, R, W, N, phis, sigma,
                                        beta_sp, hs, p=2)
            nlam += 3
        elif arm == "clip":
            S, Ibins, R, e, _ = clip_step(S, Ibins, R, hs, sigma, beta_sp,
                                          W, N, phis, measure=True)
            E_clip += e
            nlam += 1
        elif arm == "clip+2sub" or arm == "clip+40sub":
            S, Ibins, R, e, _ = clip_step(S, Ibins, R, hs, sigma, beta_sp,
                                          W, N, phis, measure=True)
            E_clip += e
            nlam += 1
        else:
            if arm == "static_w+midpoint":
                mode = "static"
            elif arm == "Strang+midpoint":
                mode = "strang"
            elif arm == "EXP2-midpoint":
                mode = "closed"
            else:
                raise ValueError("unknown arm %r" % (arm,))
            lam = midpoint_lambda(S, Ibins, R, hs, W, N, phis, sigma,
                                  beta_sp, mode=mode)
            nlam += 2
            S, Ibins, R = exp2_step(S, Ibins, R, hs, lam, sigma,
                                    mode=mode, T=T, K=K)

    Itot = Ibins.sum(axis=1)
    stat = dict(n_subcycle=nsub, n_lam_eval=nlam,
                E_clip=E_clip,
                cons_resid=float(np.max(np.abs(S + Itot + R - N))),
                min_S=float(S.min()),
                neg_S=int((S < -1e-12).sum()),
                neg_I=int((Itot < -1e-12).sum()),
                all_finite=bool(np.all(np.isfinite(S))
                                and np.all(np.isfinite(Ibins))
                                and np.all(np.isfinite(R))))
    return S, Ibins, R, stat


# =============================================================================
# 8. Full simulation over a time grid
# =============================================================================
def simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp, h, t_grid,
             record_every=1, arm="M3-p3"):
    """Advance the coupled host-level system on `t_grid`.

    `arm='M3-p3'` (default) is the production path: one M3 macro step per grid
    interval, advanced with the ACTUAL increment t[k+1]-t[k] so that event
    times that fall on step boundaries are integrated exactly there.
    """
    S = S0.copy()
    Ibins = bins0.copy()
    R = R0.copy()
    nstep = len(t_grid) - 1
    S_rec = np.zeros((nstep // record_every + 1, S.shape[0]), dtype=DTYPE)
    I_rec = np.zeros_like(S_rec)
    R_rec = np.zeros_like(S_rec)
    t_rec = []
    S_rec[0] = S / N
    I_rec[0] = Ibins.sum(axis=1) / N
    R_rec[0] = R / N
    t_rec.append(t_grid[0])
    ri = 1
    maxC = 0.0
    minS = float("inf")
    cons = 0.0
    nsub_tot = 0
    nlam_tot = 0
    for k in range(nstep):
        dt = float(t_grid[k + 1] - t_grid[k])
        if arm in ("M3-p3", "M3-p2"):
            S, Ibins, R, st = host_step(S, Ibins, R, W, N, phis, sigma,
                                        beta_sp, dt,
                                        p=(2 if arm == "M3-p2" else M3_ORDER))
        else:
            S, Ibins, R, st = baseline_step(arm, S, Ibins, R, W, N, phis,
                                            sigma, beta_sp, dt)
        maxC = max(maxC, float(st.get("max_C", 0.0)))
        minS = min(minS, float(st.get("min_S", np.inf)))
        cons = max(cons, float(st.get("cons_resid", 0.0)))
        nsub_tot += int(st.get("n_subcycle", 1))
        nlam_tot += int(st.get("n_lam_eval", 0))
        if (k + 1) % record_every == 0:
            S_rec[ri] = S / N
            I_rec[ri] = Ibins.sum(axis=1) / N
            R_rec[ri] = R / N
            t_rec.append(t_grid[k + 1])
            ri += 1
    summ = dict(mp_activation_frac=0.0,
                max_C=float(maxC),
                max_cons_resid=float(cons),
                global_min_S=float(minS),
                substeps_total=int(nsub_tot),
                lam_evals_total=int(nlam_tot),
                final_R_frac=(R / N))
    return dict(t=np.array(t_rec), S=S_rec, I=I_rec, R=R_rec, stats=summ)


# =============================================================================
# 9. Demo driver + reporting
# =============================================================================
def run_demo(out_csv=OUT_CSV, out_txt=OUT_TXT):
    ker = build_infectious_kernel()
    phis = ker['phis']
    sigma = ker['sigma']
    states, rho = load_rho(RHO_CSV)
    W, N, beta_sp = build_gravity(states)
    n = len(states)

    S0 = (1.0 - rho) * N
    I0 = rho * N
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K_BINS, (1, K_BINS))

    t_grid = np.round(np.arange(T_SPAN[0], T_SPAN[1] + H / 2, H), 8)

    coupled = simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp, H, t_grid)
    decoup = simulate(S0, bins0, R0, W, N, phis, sigma, 0.0, H, t_grid)

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'state', 'S_frac', 'I_frac', 'R_frac'])
        for j, st in enumerate(states):
            for r in range(coupled['t'].shape[0]):
                w.writerow([f"{coupled['t'][r]:.4f}", st,
                            f"{coupled['S'][r, j]:.8e}",
                            f"{coupled['I'][r, j]:.8e}",
                            f"{coupled['R'][r, j]:.8e}"])

    final_c = coupled['stats']['final_R_frac']
    final_d = decoup['stats']['final_R_frac']

    def rank(x):
        return np.argsort(np.argsort(x)).astype(float)
    rc = np.corrcoef(rank(final_c), rank(final_d))[0, 1]

    lines = []
    L = lines.append
    L("=" * 78)
    L("PATH-C HOST-LEVEL + NESTED WITHIN-HOST DEMO  (M3 macro step)")
    L("=" * 78)
    L("")
    L("[0] Within-host infectious kernel (from the canonical Fig.1 trajectory)")
    L("    source             : figures/fig3_withinhost_traj.csv")
    L("    IC                 : y0=[3800, 0, 1, 0, 1e-4, 0, 0]")
    L("    V peak             : %.4e at tau=%.2f d" % (ker['V_peak'], ker['t_peak']))
    L("    tau* (V<=eps_v)    : %.3f d   ->  sigma = K/tau* = %.4f /d" %
      (ker['tau_star'], sigma))
    L("    K bins, dtau       : %d, %.4f d" % (K_BINS, ker['dtau']))
    L("    sum_k phi_k*dtau   : %.4f   (target = Tg = %.1f d; R0=beta_sp*Tg)" %
      (ker['sum_phi_dtau'], TG))
    L("    beta_sp (calib)    : %.4f  ->  R0 = %.3f" % (beta_sp, beta_sp * TG))
    L("")
    L("[1] Host macro step : M3 (p=%d Taylor force of infection, positive-square"
      % M3_ORDER)
    L("    parametrisation, exact phase-type flow).  No limiter, no clipping,")
    L("    no CFL condition, no subcycling: q=%d incidence / q'=%d hazard nodes,"
      % (Q_INCIDENCE, Q_HAZARD))
    L("    kernel matrices rebuilt once per (h, sigma) and shared by all patches.")
    L("    macro step H=%.2f d, horizon %.0f d, %d states."
      % (H, T_SPAN[1], n))
    L("")
    peak_I = coupled['I'].max(axis=0)
    t_peak_I = coupled['t'][np.argmax(coupled['I'], axis=0)]
    L("[a] Epidemic wave (coupled)")
    L("    max prevalence I^h_i/N_i over run: min=%.4e max=%.4e" %
      (peak_I.min(), peak_I.max()))
    order = np.argsort(-peak_I)[:5]
    L("    time-of-peak prevalence (top 5 by peak):")
    for j in order:
        L("        %-4s  peakI=%.4f at t=%.1f d   (init rho=%.3f)" %
          (states[j], peak_I[j], t_peak_I[j], rho[j]))
    sample = ['TX', 'CA', 'OH', 'NV', 'WI']
    L("    rise->peak->decline sample (prevalence at t=peak, 30 d, 60 d):")
    for st in sample:
        j = states.index(st)
        I = coupled['I'][:, j]
        t = coupled['t']
        tp = t[np.argmax(I)]
        ip = I.max()
        i30 = I[int(np.argmin(np.abs(t - 30.0)))]
        i60 = I[-1]
        L("        %-4s peak=%.3f@%.1fd  I(30d)=%.3f  I(60d)=%.3f" %
          (st, ip, tp, i30, i60))
    L("    positivity:  min S^h=%.3e | S<0 cells=%d | I<0 cells=%d | NaN=%d"
      % (coupled['stats']['global_min_S'],
         int((coupled['S'] < -1e-9).sum()),
         int((coupled['I'] < -1e-9).sum()),
         int(np.isnan(coupled['S']).sum() + np.isnan(coupled['I']).sum())))
    L("    conservation: max |S+I+R-N| over run = %.3e"
      % coupled['stats']['max_cons_resid'])
    L("    coupling number max_i(h*Lambda_i) = %.4e  (informational only:"
      % coupled['stats']['max_C'])
    L("    M3 imposes no CFL restriction of the form h*Lambda <= 1).")
    L("    (数值待用户数值确认)")
    L("")
    L("[b] Coupling vs no-coupling (beta_sp=0)")
    L("    coupled   final R^h/N : min=%.4f max=%.4f mean=%.4f" %
      (final_c.min(), final_c.max(), final_c.mean()))
    L("    decoupled final R^h/N : min=%.4f max=%.4f mean=%.4f" %
      (final_d.min(), final_d.max(), final_d.mean()))
    L("    Spearman(coupled final R, decoupled final R=rho0) = %.4f" % rc)
    L("    attacked states (coupled final R > init rho + 1e-3): %d / %d" %
      (int(np.sum(final_c > rho + 1e-3)), n))
    top = np.argsort(-final_c)[:8]
    L("    highest final attack (coupled): " +
      ", ".join("%s=%.3f" % (states[j], final_c[j]) for j in top))
    L("    (数值待用户数值确认)")
    L("")
    L("[c] M3 step budget")
    L("    Lambda (W@) evaluations total : %d" % coupled['stats']['lam_evals_total'])
    L("    sub-steps total               : %d  (= number of macro steps)"
      % coupled['stats']['substeps_total'])
    L("    (数值待用户数值确认)")
    L("")
    L("Outputs: %s  (coupled time series) ; this report." % out_csv)
    text = "\n".join(lines)
    with open(out_txt, 'w') as f:
        f.write(text)
    print(text)
    return dict(ker=ker, coupled=coupled, decoup=decoup, report=text,
                out_csv=out_csv, out_txt=out_txt)


if __name__ == "__main__":
    run_demo()
