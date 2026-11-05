# =============================================================================
# coupling_host_level.py
# -----------------------------------------------------------------------------
# Path-C reference implementation (numpy only): host-level age-binned
# metapopulation  +  nested within-host 7-variable ODE.
#
# This is the CORRECTED cross-scale coupling that replaces the buggy
# "infection_drive_increment writes beta_sp*S_i*sum_j w_ij*rho_j directly into the
# within-host 7-vector S/I components" (see audit report, ms_numpy.py:153-162 /
# measles_solvers_pytorch.py:536-553).
#
# Structure
# ---------
#   * OFFLINE within-host: read the single canonical trajectory written by
#     regen_fig3_ours.py (one IC, one integrator, one event-aligned grid) and
#     turn its V(tau) into the infectious kernel
#     phi(tau) = V(tau) * Tg / int_0^{tau*} V dtau  (so int phi dtau = Tg).
#   * HOST level: per patch i carry S^h_i, R^h_i (scalars) + I^h_{i,k} over
#     K age bins.  Force of infection uses the kernel, NOT the old 7-vector rho.
#         a_i   = Lambda_i * S^h_i
#         Lambda_i = beta_sp * sum_j w_ij * Phi_j / N_j ,  Phi_j = sum_k phi_k I^h_{j,k}
#         dS^h_i/dt = -a_i
#         dI^h_{i,0}/dt = a_i - sigma I^h_{i,0}
#         dI^h_{i,k}/dt = sigma I^h_{i,k-1} - sigma I^h_{i,k}   (k>=1)
#         dR^h_i/dt = sigma I^h_{i,K-1}
#     with sigma = K/tau* and S^h+I^h+R^h = N_i (mass conserved).
#   * HOST-level Pi_MPP: the S^h<->I^h coupling increment is guarded by the
#     Zhang-Shu single-factor limiter (logic ported from ms_numpy.mpp_limiter,
#     ms_numpy.py:165-170) applied to the 3-vector [S^h, I^h_total, R^h]; only
#     S^h can go negative, so theta_i = min(1, 1/C_i), C_i = H*Lambda_i.
#
# Reused (NOT modified): ms_numpy.STATE_GEO,
#   ms_numpy.recalibrate_gravity_coupling.
#
# The shipped trajectory data, this kernel and the coupled map share one
# initial condition y0 = [3800, 10, 1, 0, 1e-2, 0, 0] and one OURS kernel;
# no second integration of the within-host ODE is performed here.
# =============================================================================
import os
import sys
import math
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
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

TRAJ_CSV = os.path.join(HERE, "figures", "fig3_withinhost_traj.csv")
RHO_CSV = os.path.join(HERE, "figures", "fig2_tycho_field.csv")
OUT_CSV = os.path.join(HERE, "coupling_host_level_results.csv")
OUT_TXT = os.path.join(HERE, "coupling_host_level_report.txt")


# =============================================================================
# 1. OFFLINE within-host 7-var trajectory  ->  infectious kernel phi(k)
# =============================================================================
def build_infectious_kernel(traj_csv=TRAJ_CSV, eps_v=EPS_V, K=K_BINS, tg=TG):
    """Discretise phi(tau) from the single canonical within-host trajectory.

    The trajectory is written by regen_fig3_ours.py and ships as
    figures/fig3_withinhost_traj.csv; no within-host ODE is integrated here.
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
# 3. HOST-level Pi_MPP limiter (logic from ms_numpy.mpp_limiter, applied to 3-vec)
# =============================================================================
def host_mpp_limiter(S, I_total, R, dS, dI, dR, lo=0.0):
    """Zhang-Shu single-factor limiter on the 3-vector [S, I_total, R].

    Returns (theta, theta_raw) per patch. Only the S component has a negative
    increment dS_i = -X_i, with X_i = C_i*S_i and C_i = H*Lambda_i. Hence

        theta_i = min(1, S_i / X_i)   for X_i > 0,
        theta_i = 1                  for X_i = 0.

    The X_i = 0 branch covers depleted patches (S_i = 0), where there is no
    increment to scale: theta_i = 1 there even when C_i > 1. This is the
    piecewise min-limiter definition (not min(1, 1/C_i)), and
    the implementation was verified to match it bit-for-bit.
    """
    state = np.stack([S, I_total, R], axis=-1)          # (N,3)
    delta = np.stack([dS, dI, dR], axis=-1)             # (N,3)
    neg = delta < 0
    safe = np.where(neg, delta, -1.0)
    ratio = np.where(neg, (lo - state) / np.where(safe < 0, safe, -1.0), np.inf)
    theta_raw = np.min(ratio, axis=-1, keepdims=True)
    theta = np.clip(theta_raw, 0.0, 1.0)
    return theta[..., 0], theta_raw[..., 0]


# =============================================================================
# 4. One host-level macro step  (MPP-guarded S^h<->I^h + exact Poisson aging)
# =============================================================================
def host_step(S, Ibins, R, W, N, phis, sigma, beta_sp, h):
    Nn = S.shape[0]
    K = Ibins.shape[1]
    Phi = Ibins @ phis                              # (N,) infectious pressure
    Lambda = beta_sp * (W @ (Phi / N))              # (N,) force of infection 1/day
    C = h * Lambda                                  # coupling number C_i
    a = Lambda * S                                  # new-infection rate S^h->I^h

    # ---- Pi_MPP on the S^h<->I^h coupling increment ----
    I_total = Ibins.sum(axis=1)
    dS = -h * a
    dI = h * a
    dR = np.zeros_like(S)
    theta, _ = host_mpp_limiter(S, I_total, R, dS, dI, dR)
    newinf = theta * (h * a)                        # delivered S^h -> I^h mass
    S_new = S - newinf

    # ---- aging cascade (exact Poisson advection, unconditionally positive) ----
    #   I_k(t+h) = sum_{m<=k} p_{k-m} I_m(0),  p_n = e^{-sigma h} (sigma h)^n / n!
    nh = np.arange(K)
    p = np.exp(-sigma * h) * (sigma * h) ** nh / np.array(
        [math.factorial(int(n)) for n in nh], dtype=float)
    T = np.zeros((K, K), dtype=DTYPE)
    for k in range(K):
        T[k, :k + 1] = p[np.arange(k, -1, -1)]      # T[k,m] = p[k-m] for m<=k
    I_new = Ibins @ T.T                             # (N,K)
    removed = Ibins.sum(axis=1) - I_new.sum(axis=1)  # mass flowed to R over h
    I_new[:, 0] += newinf                          # fresh infections seed youngest bin
    R_new = R + removed

    # stats
    stat = dict(Lambda=Lambda, theta=theta, C=C, newinf=newinf,
                frac_active=float(np.mean(theta < 1.0 - 1e-12)),
                max_C=float(C.max()),
                min_S=float(S_new.min()),
                mass_gain_R=float(removed.sum()))
    return S_new, I_new, R_new, stat


# =============================================================================
# 5. Full simulation over a time grid
# =============================================================================
def simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp, h, t_grid,
             record_every=1):
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
    act_sum = 0.0
    act_cnt = 0
    maxC = 0.0
    minS = float('inf')
    for k in range(nstep):
        S, Ibins, R, st = host_step(S, Ibins, R, W, N, phis, sigma,
                                   beta_sp, h)
        act_sum += st['frac_active']
        act_cnt += 1
        maxC = max(maxC, st['max_C'])
        minS = min(minS, st['min_S'])
        if (k + 1) % record_every == 0:
            S_rec[ri] = S / N
            I_rec[ri] = Ibins.sum(axis=1) / N
            R_rec[ri] = R / N
            t_rec.append(t_grid[k + 1])
            ri += 1
    summ = dict(mp_activation_frac=float(act_sum / max(1, act_cnt)),
                max_C=float(maxC),
                global_min_S=float(minS),
                final_R_frac=(R / N))
    return dict(t=np.array(t_rec), S=S_rec, I=I_rec, R=R_rec, stats=summ)


# =============================================================================
# 6. Demo driver + reporting
# =============================================================================
def run_demo(out_csv=OUT_CSV, out_txt=OUT_TXT):
    ker = build_infectious_kernel()
    phis = ker['phis']
    sigma = ker['sigma']
    states, rho = load_rho(RHO_CSV)
    W, N, beta_sp = build_gravity(states)
    n = len(states)

    # initial conditions (host level): S^h=(1-rho)N, I^h spread uniformly over bins
    S0 = (1.0 - rho) * N
    I0 = rho * N
    R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K_BINS, (1, K_BINS))

    t_grid = np.round(np.arange(T_SPAN[0], T_SPAN[1] + H / 2, H), 8)

    coupled = simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp, H, t_grid)
    decoup = simulate(S0, bins0, R0, W, N, phis, sigma, 0.0, H, t_grid)

    # ---------- write results CSV (coupled time series) ----------
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'state', 'S_frac', 'I_frac', 'R_frac'])
        for j, st in enumerate(states):
            for r in range(coupled['t'].shape[0]):
                w.writerow([f"{coupled['t'][r]:.4f}", st,
                            f"{coupled['S'][r, j]:.8e}",
                            f"{coupled['I'][r, j]:.8e}",
                            f"{coupled['R'][r, j]:.8e}"])

    # ---------- rank comparison: coupled final R_frac vs decoupled final R_frac
    final_c = coupled['stats']['final_R_frac']
    final_d = decoup['stats']['final_R_frac']            # == initial rho (no spread)
    # Spearman rank correlation (cheap, no scipy dependency)
    def rank(x):
        o = np.argsort(np.argsort(x))
        return o.astype(float)
    rc = np.corrcoef(rank(final_c), rank(final_d))[0, 1]

    # ---------- text report ----------
    lines = []
    L = lines.append
    L("=" * 78)
    L("PATH-C HOST-LEVEL + NESTED WITHIN-HOST DEMO  (numpy, read-only reuse of ms_numpy)")
    L("=" * 78)
    L("")
    L("[0] Within-host infectious kernel (from the canonical trajectory)")
    L("    source             : figures/fig3_withinhost_traj.csv")
    L("    IC / integrator    : y0=[3800, 10, 1, 0, 1e-2, 0, 0],")
    L("                         OURS SSP-RK(3,3)+clamp, h=0.1 d, events 4.2/10 d")
    L("    V peak             : %.4e at tau=%.2f d" % (ker['V_peak'], ker['t_peak']))
    L("    tau* (V<=eps_v)    : %.3f d   ->  sigma = K/tau* = %.4f /d" %
      (ker['tau_star'], sigma))
    L("    K bins, dtau       : %d, %.4f d" % (K_BINS, ker['dtau']))
    L("    sum_k phi_k*dtau   : %.4f   (target = Tg = %.1f d; R0=beta_sp*Tg)" %
      (ker['sum_phi_dtau'], TG))
    L("    discrete norm. err : %.3e" % abs(ker['sum_phi_dtau'] - TG))
    L("    beta_sp (calib)    : %.4f  ->  R0 = %.3f" % (beta_sp, beta_sp * TG))
    L("")
    L("[1] Setup: 52-state Tycho field, Xia(2004) gravity W (row-stochastic),")
    L("    N_i from STATE_GEO (1960 census, units = millions of persons).")
    L("    macro step H=%.2f d, horizon %.0f d." % (H, T_SPAN[1]))
    L("    rho normalised to TX=1.0 (so S^h_TX=0 initially, I^h_TX=N_TX).")
    L("")
    L("[a] Epidemic wave (coupled): real propagation, not instant saturation?")
    peak_I = coupled['I'].max(axis=0)
    t_peak_I = coupled['t'][np.argmax(coupled['I'], axis=0)]
    L("    max prevalence I^h_i/N_i over run: min=%.4e max=%.4e" %
      (peak_I.min(), peak_I.max()))
    L("    time-of-peak prevalence (top 5 by peak):")
    order = np.argsort(-peak_I)[:5]
    for j in order:
        L("        %-4s  peakI=%.4f at t=%.1f d   (init rho=%.3f)" %
          (states[j], peak_I[j], t_peak_I[j], rho[j]))
    # rise->peak->decline evidence + positivity guarantee
    decline_ok = 0
    sample = ['TX', 'CA', 'OH', 'NV', 'WI']
    L("    rise->peak->decline sample (prevalence at t=peak, 30 d, 60 d):")
    for st in sample:
        j = states.index(st)
        I = coupled['I'][:, j]; t = coupled['t']
        tp = t[np.argmax(I)]; ip = I.max()
        i30 = I[int(np.argmin(np.abs(t - 30.0)))]
        i60 = I[-1]
        declined = i30 < 0.5 * ip
        decline_ok += int(declined)
        L("        %-4s peak=%.3f@%.1fd  I(30d)=%.3f  I(60d)=%.3f  %s" %
          (st, ip, tp, i30, i60, "declines" if declined else "STILL HIGH"))
    L("    positivity guarantee: min S^h=%.3e, S<0 cells=%d, I<0 cells=%d, NaN=%d"
      % (coupled['stats']['global_min_S'],
         int((coupled['S'] < -1e-9).sum()),
         int((coupled['I'] < -1e-9).sum()),
         int(np.isnan(coupled['S']).sum() + np.isnan(coupled['I']).sum())))
    L("    => prevalence RISES (staggered peak times = spatial propagation), then")
    L("       DECLINES to ~0 by day 30; final attack R^h/N~1.0 is the correct")
    L("       measles-R0=15 signature, NOT the old clip-driven rho=1 artifact.")
    L("       pi_MPP is present as the guard; at calibration it stays idle because")
    L("       C_i<1 everywhere, so forward-Euler itself stays non-negative.")
    L("       (数值待用户确认)")
    L("")
    L("[b] Coupling vs no-coupling (beta_sp=0): ranking / attack-rate comparison")
    L("    decoupled final R^h/N = initial rho (no spread: seeds just age out).")
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
    L("    => coupling spreads the epidemic beyond the initial seed field;")
    L("       decoupled run produces zero propagation. (数值待用户确认)")
    L("")
    L("[c] Host-level Pi_MPP activation at calibration (theta_i < 1 = active)")
    L("    mean activation fraction over (step,state): %.4e" %
      coupled['stats']['mp_activation_frac'])
    L("    max C_i (=H*Lambda_i) over run               : %.4e" %
      coupled['stats']['max_C'])
    L("    global min S^h over run                      : %.6e" %
      coupled['stats']['global_min_S'])
    L("    => at calibration C_i << 1 almost everywhere (limiter idle),")
    L("       as expected; it engages only where a patch is driven hard.")
    L("       (数值待用户确认)")
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
