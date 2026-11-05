"""torch-free (numpy) port of measles_solvers_pytorch + validate_positivity kernels.

Purpose: a faithful numpy re-implementation of the coupling-ablation pipeline,
torch-free. Every routine below is a line-by-line port of the torch original;
the port is validated by reproducing figures/fig3_withinhost_traj.csv to
round-off.

Ported from
  measles_solvers_pytorch.py : Params/default_params, measles_rhs,
                               ssp_rk33_pp_step, infection_drive_increment,
                               mpp_limiter, prolongation, build_aligned_grid
  validate_positivity.py     : STATE_GEO, haversine_km,
                               recalibrate_gravity_coupling
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Params (measles_solvers_pytorch.py:101-131)
# ---------------------------------------------------------------------------
PARAMS = dict(
    beta=0.040, kI=0.017, p=0.089, q=0.99, qs=0.028, r=0.058, s=0.0104,
    c=3.0, d=0.025, delta=0.5, td=4.2,
    rhoR=0.08, muR=0.0138629, etaR=2.0e-3, tAb=10.0,
    rhoAb=0.015, muAb=0.0083333, thetaAb=1.0e-3,
    t17=35.0, alpha_t=0.15, alpha_R=0.60, kappa17=0.020, uCTL=0.0,
)


class P:
    def __init__(self, **kw):
        self.__dict__.update(PARAMS)
        self.__dict__.update(kw)


def default_params():
    return P()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# measles_rhs (measles_solvers_pytorch.py:140-195). y: [N,7]
# ---------------------------------------------------------------------------
def measles_rhs(y, t, p):
    S = y[..., 0]
    I = y[..., 1]
    Ag = y[..., 2]
    A17 = y[..., 3]
    V = y[..., 4]
    R = y[..., 5]
    Ab = y[..., 6]

    Atot = Ag + A17
    fV = V / (p.s + V)

    delta_hat = 1.0 if t < p.td else 0.0
    H_ab = 1.0 if t >= p.tAb else 0.0
    sig = _sig(p.alpha_t * (t - p.t17) + p.alpha_R * np.log1p(R))
    gR = np.log1p(R)

    dS = -p.beta * S * V + p.qs * delta_hat * S + p.r * (1.0 - fV) * Atot
    dI = p.beta * (S + Atot) * V - p.delta * I - p.kI * Atot
    dAg = (p.q * fV * Atot * (1.0 - sig) - p.beta * Ag * V
           - (1.0 - fV) * (p.d + p.r) * Ag - p.kappa17 * Ag + p.uCTL)
    dA17 = (p.q * fV * Atot * sig + p.kappa17 * Ag - p.beta * A17 * V
            - (1.0 - fV) * (p.d + p.r) * A17)
    dV = p.p * I - p.c * V - p.thetaAb * Ab * V
    dR = p.rhoR * I - p.muR * R - p.etaR * Ab * R
    dAb = p.rhoAb * H_ab * gR - p.muAb * Ab

    return np.stack([dS, dI, dAg, dA17, dV, dR, dAb], axis=-1)


# ---------------------------------------------------------------------------
# ssp_rk33_pp_step (measles_solvers_pytorch.py:251-272)
#   NOTE: the torch original evaluates L1 and L2 both at t+dt (not t+dt/2);
#   the port keeps that behaviour verbatim.
# ---------------------------------------------------------------------------
def ssp_rk33_pp_step(y, t, dt, p):
    L0 = measles_rhs(y, t, p)
    y1 = np.maximum(y + dt * L0, 0.0)
    L1 = measles_rhs(y1, t + dt, p)
    y2 = np.maximum(0.75 * y + 0.25 * y1 + 0.25 * dt * L1, 0.0)
    L2 = measles_rhs(y2, t + dt, p)
    y3 = np.maximum((1.0 / 3.0) * y + (2.0 / 3.0) * y2 + (2.0 / 3.0) * dt * L2, 0.0)
    return y3


# ---------------------------------------------------------------------------
# coupling kernels (measles_solvers_pytorch.py:535-576)
# ---------------------------------------------------------------------------
def infection_drive_increment(y, W, beta_sp, dt, i_idx=1, s_idx=0):
    S = y[..., s_idx]
    I = y[..., i_idx]
    denom = S + I + 1e-12
    pressure = I / denom                      # [N]
    drive = (pressure @ W.T) * beta_sp        # Sigma_j W_ij pressure_j * beta_sp
    inc = np.zeros_like(y)
    inc[..., s_idx] = -drive * S * dt
    inc[..., i_idx] = drive * S * dt
    return inc


def mpp_limiter(y, delta_sp, lo=0.0):
    neg = delta_sp < 0
    ratio = np.where(neg, (lo - y) / np.where(neg, delta_sp, -1.0), np.inf)
    theta = np.min(ratio, axis=-1, keepdims=True)
    theta = np.clip(theta, 0.0, 1.0)
    return y + theta * delta_sp


def prolongation(rho, total=3800.0, seed_ag=1.0, seed_a17=0.0,
                 seed_v=1e-4, seed_ab=0.0):
    r = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0)
    y = np.zeros(r.shape + (7,), dtype=float)
    y[..., 0] = (1.0 - r) * total
    y[..., 1] = r * total
    y[..., 2] = seed_ag
    y[..., 3] = seed_a17
    y[..., 4] = r * seed_v
    y[..., 5] = 0.0
    y[..., 6] = seed_ab
    return y


def build_aligned_grid(t0, t1, dt, event_ts):
    grid = [t0]
    cur = t0
    eps = 1e-9
    for ev in sorted(event_ts):
        if t0 < ev < t1:
            while cur + dt < ev - eps:
                cur += dt
                grid.append(cur)
            if abs(cur - ev) > eps:
                cur = ev
                grid.append(cur)
    while cur < t1 - eps:
        nxt = cur + dt
        if nxt > t1:
            nxt = t1
        cur = nxt
        grid.append(cur)
    return grid


# ---------------------------------------------------------------------------
# geography / gravity coupling (validate_positivity.py:254-410)
# ---------------------------------------------------------------------------
STATE_GEO = {
    'AL': (32.7794, -86.8287, 3.266740),  'AK': (64.0685, -152.2782, 0.226167),
    'AZ': (34.2744, -111.6602, 1.302161), 'AR': (34.8938, -92.4426, 1.786272),
    'CA': (37.1841, -119.4696, 15.717204),'CO': (38.9972, -105.5478, 1.753947),
    'CT': (41.6219, -72.7273, 2.535234), 'DE': (38.9896, -75.5050, 0.446292),
    'DC': (38.9101, -77.0147, 0.763956),  'FL': (28.6305, -82.4497, 4.951560),
    'GA': (32.6415, -83.4426, 3.943116), 'HI': (20.2927, -156.3737, 0.632772),
    'ID': (44.3509, -114.6130, 0.667191), 'IL': (40.0417, -89.1965, 10.081158),
    'IN': (39.8942, -86.2816, 4.662498),  'IA': (42.0751, -93.4960, 2.757537),
    'KS': (38.49389, -98.37722, 2.178611),'KY': (37.5347, -85.3021, 3.038156),
    'LA': (31.0689, -91.9968, 3.257022),  'ME': (45.3695, -69.2428, 0.969265),
    'MD': (39.0550, -76.7909, 3.100689),  'MA': (42.2596, -71.8083, 5.148578),
    'MI': (44.3467, -85.4102, 7.823194),  'MN': (46.2807, -94.3053, 3.413864),
    'MS': (32.7364, -89.6678, 2.178141),  'MO': (38.3566, -92.4580, 4.319813),
    'MT': (47.0527, -109.6333, 0.674767), 'NE': (41.5378, -99.7951, 1.411330),
    'NV': (39.3289, -116.6312, 0.285278), 'NH': (43.6805, -71.5811, 0.606921),
    'NJ': (40.1907, -74.6728, 6.066782),  'NM': (34.4071, -106.1126, 0.951023),
    'NY': (42.9538, -75.5268, 16.782304), 'NC': (35.5557, -79.3877, 4.556155),
    'ND': (47.4501, -100.4659, 0.632446), 'OH': (40.2862, -82.7937, 9.706397),
    'OK': (35.5889, -97.4943, 2.328284),  'OR': (43.9336, -120.5583, 1.768687),
    'PA': (40.8781, -77.7996, 11.319366), 'RI': (41.6762, -71.5562, 0.859488),
    'SC': (33.9169, -80.8964, 2.382594),  'SD': (44.4443, -100.2263, 0.680514),
    'TN': (35.8580, -86.3505, 3.567089),  'TX': (31.4757, -99.3312, 9.579677),
    'UT': (39.3055, -111.6703, 0.890627), 'VT': (44.0687, -72.6658, 0.389881),
    'VA': (37.5215, -78.8537, 3.966949),  'WA': (47.3826, -120.4472, 2.853214),
    'WV': (38.6409, -80.6227, 1.860421),  'WI': (44.6243, -89.9941, 3.951777),
    'WY': (42.9957, -107.5512, 0.330066), 'PR': (18.2806, -66.3464, 2.349544),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2)
    return 2.0 * R * math.asin(min(1.0, math.sqrt(a)))


def recalibrate_gravity_coupling(state_names, geo, R0_target=15.0, Tg=13.0,
                                 theta_per_biweek=4.54e-9, pop_exp=0.5,
                                 dist_exp=1.0):
    N = len(state_names)
    Pop = np.zeros(N)
    coords = []
    for i, s in enumerate(state_names):
        if s not in geo:
            raise KeyError('state %r not in geo' % s)
        lat, lon, pop = geo[s]
        Pop[i] = float(pop)
        coords.append((lat, lon))
    G = np.zeros((N, N))
    raw = np.zeros((N, N))
    halflife = 14.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d < 1e-3:
                d = 1e-3
            g = (Pop[i] * Pop[j]) ** pop_exp / (d ** dist_exp)
            G[i, j] = g
            raw[i, j] = theta_per_biweek * g / halflife
    W = G / G.sum(axis=1, keepdims=True)

    beta_sp_cal = R0_target / Tg
    R0_check = beta_sp_cal * Tg
    lambda_max_W = float(np.abs(np.linalg.eigvals(W)).max())
    Gsym = 0.5 * (G + G.T)
    lambda_max_G = float(np.linalg.eigvalsh(Gsym).max())
    kappa_for_trueG = (R0_target / (Tg * lambda_max_G)) if lambda_max_G > 0 else float('nan')
    mean_theta_per_day = float(raw.sum(axis=1).mean())
    flat = [(i, j, float(G[i, j])) for i in range(N) for j in range(N) if i != j]
    flat.sort(key=lambda x: -x[2])
    top_pairs = [(state_names[i], state_names[j], g) for (i, j, g) in flat[:8]]
    return {
        'W': W, 'G': G, 'raw_theta': raw,
        'beta_sp_cal': beta_sp_cal, 'R0_target': R0_target, 'Tg': Tg,
        'R0_check': R0_check, 'lambda_max_W': lambda_max_W,
        'lambda_max_G': lambda_max_G, 'kappa_for_trueG': kappa_for_trueG,
        'mean_theta_per_day': mean_theta_per_day, 'top_pairs': top_pairs,
    }
