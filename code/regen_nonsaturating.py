# -*- coding: utf-8 -*-
# =============================================================================
# regen_nonsaturating.py  --  R8 / route B / experiment line 2
# -----------------------------------------------------------------------------
# "Non-saturating coupling scan + positive biological capability evidence".
#
# This script is SELF-CONTAINED in the sense that it only imports the existing
# project modules (coupling_host_level, ms_numpy) plus numpy / scipy / pandas /
# matplotlib and the Python standard library.  It does NOT edit any existing
# file.  It creates only:
#
#   submission/code/regen_nonsaturating.py            (this file)
#   figures/nonsaturating_scan.csv        + submission/figures/nonsaturating_scan.csv
#   figures/kernel_sensitivity.csv       + submission/figures/kernel_sensitivity.csv
#   figures/large_step_endpoint.csv      + submission/figures/large_step_endpoint.csv
#   figures/nonsaturating_attack_rate.png + submission/figures/nonsaturating_attack_rate.png
#   _review_new_tmp_lead/r8_nonsat_findings.md        (written by hand afterwards)
#
# The two copies of each CSV/PNG are byte-identical (single in-memory payload
# written to both paths, md5 verified at run time).
#
# HARD RULE: every number written below is produced by an actual run at run
# time.  Nothing is assigned from memory / from the manuscript.  All numbers
# are labelled "待用户数值确认" in the findings report.
#
# No random stream is used anywhere; all arithmetic is float64.
# =============================================================================
import os
import sys
import csv
import io
import math
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammainc
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, HERE)

import ms_numpy as msn                 # noqa: E402  (existing module, read-only)
import coupling_host_level as chl     # noqa: E402  (existing module, read-only)

DTYPE = np.float64

# ---------------------------------------------------------------------------
# Experiment settings (all explicit; chl defaults are NOT modified)
# ---------------------------------------------------------------------------
H_SCAN = 0.1           # host macro step (days) -- same as chl.H, used for 2A/2B
T_END_2A = 240.0       # 2A horizon (days). chl.T_SPAN=(0,60) is NOT enough:
                       # at R0<=1.5 the epidemic tail decays over ~(3-5)*tau*,
                       # and at R0=0.8 I_end(60 d) is still O(1e-3). 240 d is
                       # chosen so I_end < 1e-8 for every R0 (verified below).
T_END_2B = 180.0       # 2B canonical-IC horizon (days)
T_END_2B_PROBE = 300.0 # 2B low-seed diagnostic horizon (days); a small seed
                       # with R0=2..3 needs ~10 generations to burn out.
T_END_2A_PROBE = 300.0 # 2A low-seed supplementary horizon (days); usable for
                       # R0>=2 (R0<2 epidemics are too slow to finish at 300 d,
                       # hence those R0 values are NOT run in the probe group).
R0_LIST_PROBE = [2.0, 3.0, 5.0, 8.0, 12.0, 15.0]
T_END_2C = 60.0        # 2C horizon = chl.T_SPAN[1] (matches m3_extreme_cfl.csv)
H_LARGE = [9.0, 10.0, 20.0, 30.0, 60.0]

R0_LIST = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0]
R0_SENS = [2.0, 3.0, 5.0]
TG_LIST = [8.0, 10.0, 13.0, 16.0]
EPS_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
SHIFT_LIST = [0.0, 4.0, 6.0]
KV_SAT = 0.1

PREV_THRESHOLD = 1e-4  # epidemic-duration threshold on population prevalence

# ---------------------------------------------------------------------------
# Canonical shared setup (identical to chl / p0b_experiment / m3_experiment)
# ---------------------------------------------------------------------------
KER = chl.build_infectious_kernel()
PHIS0, SIGMA0 = KER["phis"], KER["sigma"]
STATES, RHO = chl.load_rho(chl.RHO_CSV)
W, N, BETA_CAL = chl.build_gravity(STATES)
K = chl.K_BINS
NST = len(STATES)
KK = np.arange(K, dtype=DTYPE)

S0_CAN = (1.0 - RHO) * N
I0_CAN = RHO * N
R0_CAN = np.zeros_like(N, dtype=DTYPE)
BINS0_CAN = np.tile(I0_CAN[:, None] / K, (1, K))

TRAJ = np.loadtxt(chl.TRAJ_CSV)
TS, VS = TRAJ[:, 0], TRAJ[:, 5]        # t,S,I,Agamma,A17,V,R,Ab

GEO = pd.read_csv(SUB / "state_geo_population.csv")
GEO_LAT = dict(zip(GEO["state"], GEO["lat"].astype(float)))
GEO_LON = dict(zip(GEO["state"], GEO["lon"].astype(float)))
ITX = STATES.index("TX")
DIST_TX_KM = np.array(
    [msn.haversine_km(GEO_LAT["TX"], GEO_LON["TX"], GEO_LAT[s], GEO_LON[s])
     for s in STATES], dtype=DTYPE)

# low-seed diagnostic IC: 1e-4 of TX population, all in age bin 0.
# THIS IS NOT the manuscript IC; it is an explicitly labelled controlled probe.
I0_PROBE = np.zeros_like(N)
I0_PROBE[ITX] = 1e-4 * N[ITX]
S0_PROBE = N - I0_PROBE
R0_PROBE = np.zeros_like(N)
BINS0_PROBE = np.zeros((NST, K), dtype=DTYPE)
BINS0_PROBE[:, 0] = I0_PROBE

TSTART = time.perf_counter()
PERF = {}


# =============================================================================
# 0. small utilities
# =============================================================================
def fmt(v):
    """Deterministic cell formatter -> identical bytes on both destinations."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.12e" % float(v)
    if v is None:
        return ""
    return str(v)


def write_double(name, header, rows):
    """Write byte-identical CSV to <repo>/figures and submission/figures."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow([fmt(v) for v in row])
    payload = buf.getvalue().encode("utf-8")
    dirs = [ROOT / "figures", SUB / "figures"]
    digests = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / name, "wb") as f:
            f.write(payload)
        digests.append(hashlib.md5((d / name).read_bytes()).hexdigest())
    if len(set(digests)) != 1:
        raise RuntimeError("md5 mismatch for %s: %s" % (name, digests))
    print("[write] %-30s rows=%-4d cols=%-3d md5=%s" %
          (name, len(rows), len(header), digests[0]))
    return digests[0]


def kendall_tau_b(x, y):
    """Kendall tau-b, same definition as regen_fig7_honest.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    C = D = Tx = Ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                Tx += 1
            elif dy == 0:
                Ty += 1
            elif dx * dy > 0:
                C += 1
            else:
                D += 1
    den = math.sqrt((C + D + Tx) * (C + D + Ty))
    tau = (C - D) / den if den > 0 else 0.0
    return float(tau), C, D, Tx, Ty


def final_size_analytic(R0, s0=1.0):
    """Non-trivial root of z = 1 - s0*exp(-R0*z) on (0,1)."""
    if R0 <= 1.0 and s0 >= 1.0 - 1e-15:
        return 0.0
    lo, hi = 1e-12, 1.0 - 1e-12
    def f(z):
        return z - 1.0 + s0 * math.exp(-R0 * z)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_grid(t_end, h):
    return np.round(np.arange(0.0, t_end + h / 2.0, h), 8)


def run_sim(beta_sp, phis, sigma, t_end, h, S0, bins0, R0v, arm="M3-p3"):
    grid = make_grid(t_end, h)
    out = chl.simulate(S0, bins0, R0v, W, N, phis, sigma, beta_sp, h, grid,
                       arm=arm)
    return out, grid


def weekly_incidence(out, h):
    """Population-weighted weekly NEW infections (absolute, in units of N)."""
    t, S = out["t"], out["S"]              # S = susceptible fraction S_i/N_i
    spw = int(round(7.0 / h))
    idx = np.arange(0, len(t) - 1, spw)
    inc = np.array([(S[a] - S[min(a + spw, len(t) - 1)]) * N for a in idx],
                   dtype=DTYPE)
    return inc, np.arange(1, len(inc) + 1)


def epidemic_endpoints(out, h):
    """All time/attack endpoints used in 2A and 2B, computed from the recorded
    S/I/R series only (no external reference solver)."""
    t = out["t"]
    S, I, R = out["S"], out["I"], out["R"]
    prev = (I * N[None, :]).sum(axis=1) / N.sum()          # population-weighted
    ever = (1.0 - S) * N[None, :]
    ever_tot = ever.sum(axis=1) / N.sum()
    newinf = (S[0] - S) * N[None, :]
    newinf_tot = newinf.sum(axis=1) / N.sum()

    tpk = float(t[int(np.argmax(prev))])
    peak_prev_val = float(prev.max())

    def first_cross(arr, level):
        ok = np.flatnonzero(arr >= level)
        if ok.size == 0:
            return float("nan")
        j = int(ok[0])
        if j == 0:
            return float(t[0])
        # linear interpolation for sub-step resolution
        y0, y1 = arr[j - 1], arr[j]
        if y1 == y0:
            return float(t[j])
        return float(t[j - 1] + (level - y0) / (y1 - y0) * (t[j] - t[j - 1]))

    t50 = first_cross(newinf_tot, 0.5 * newinf_tot[-1])
    t99 = first_cross(newinf_tot, 0.99 * newinf_tot[-1])

    below = prev < PREV_THRESHOLD
    ipk = int(np.argmax(prev))
    dur, reached = float(t[-1]), False
    for j in range(ipk + 1, len(t)):
        if bool(np.all(below[j:])):
            dur, reached = float(t[j]), True
            break

    inc, weeks = weekly_incidence(out, h)
    tot_inc = inc.sum(axis=1)
    global_peak_week = int(weeks[int(np.argmax(tot_inc))])
    state_peak_week = weeks[np.argmax(inc, axis=0)]
    state_peak_prev_day = t[np.argmax(I, axis=0)]

    # growth rate from the early increasing part of weekly incidence
    pw = int(np.argmax(tot_inc))
    sel = (tot_inc > 1e-12) & (weeks <= weeks[pw]) & (weeks >= 2)
    growth = float("nan")
    nfit = 0
    if int(sel.sum()) >= 3:
        growth = float(np.polyfit(weeks[sel] * 7.0, np.log(tot_inc[sel]), 1)[0])
        nfit = int(sel.sum())

    return dict(t=t, prev=prev, ever_tot=ever_tot, newinf_tot=newinf_tot,
                tpk=tpk, peak_prev_val=peak_prev_val, t50=t50, t99=t99,
                dur=dur, reached=reached, Iend=float(I[-1].max()),
                global_peak_week=global_peak_week,
                mean_state_peak_week=float(state_peak_week.mean()),
                min_state_peak_week=int(state_peak_week.min()),
                max_state_peak_week=int(state_peak_week.max()),
                min_state_peak_prev_day=float(state_peak_prev_day.min()),
                max_state_peak_prev_day=float(state_peak_prev_day.max()),
                state_peak_prev_day=state_peak_prev_day,
                growth=growth, nfit=nfit,
                ever_last=float(np.mean(1.0 - S[-1])))


def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    d = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc / d) if d > 0 else float("nan")


def safe_spearman(x, y):
    """spearmanr with constant-input warnings silenced (returns nan)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            return float(spearmanr(x, y).statistic)


# =============================================================================
# 1. EXPERIMENT 2A : non-saturating beta_sp scan
# =============================================================================
def experiment_2a():
    print("\n[2A] non-saturating beta_sp scan, R0 = beta_sp * T_g")
    t0 = time.perf_counter()
    rows = []
    raw = {}          # canonical group, keyed by R0 (used by the chart)
    rawp = {}         # low-seed supplementary group, keyed by R0
    for R0v in R0_LIST:
        beta_sp = R0v / chl.TG
        tc = time.perf_counter()
        out, grid = run_sim(beta_sp, PHIS0, SIGMA0, T_END_2A, H_SCAN,
                            S0_CAN, BINS0_CAN, R0_CAN)
        dec, _ = run_sim(0.0, PHIS0, SIGMA0, T_END_2A, H_SCAN,
                         S0_CAN, BINS0_CAN, R0_CAN)
        ep = epidemic_endpoints(out, H_SCAN)

        fR = out["stats"]["final_R_frac"]
        fD = dec["stats"]["final_R_frac"]
        ever_min = float((1.0 - out["S"][-1]).min())
        ever_max = float((1.0 - out["S"][-1]).max())

        z_cl = final_size_analytic(R0v, 1.0)
        rho_mean = float(RHO.mean())
        rho_pw = float((RHO * N).sum() / N.sum())
        z_rho = final_size_analytic(R0v, 1.0 - rho_mean)
        dev_abs = float(fR.mean() - z_cl)
        dev_rel = dev_abs / z_cl if z_cl > 0 else float("nan")

        tau_c, Cc, Dc, Txc, Tyc = kendall_tau_b(fR, RHO)
        tau_d, Cd, Dd, Txd, Tyd = kendall_tau_b(fD, RHO)
        newinf = fR - RHO
        tau_n, _, _, _, _ = kendall_tau_b(newinf, RHO)

        tp_i = ep["state_peak_prev_day"]
        pr_all = pearson_r(DIST_TX_KM, tp_i)
        sr_all = safe_spearman(DIST_TX_KM, tp_i)
        keep = np.arange(NST) != ITX
        pr_noTX = pearson_r(DIST_TX_KM[keep], tp_i[keep])
        sr_noTX = safe_spearman(DIST_TX_KM[keep], tp_i[keep])
        slope = float(np.polyfit(DIST_TX_KM[keep], tp_i[keep], 1)[0] * 1000.0)

        minS = float(min(out["S"].min(), dec["S"].min()))
        nnegS = int((out["S"] < -1e-12).sum() + (dec["S"] < -1e-12).sum())
        nnegI = int((out["I"] < -1e-12).sum() + (dec["I"] < -1e-12).sum())
        cons = max(float(out["stats"]["max_cons_resid"]),
                   float(dec["stats"]["max_cons_resid"]))

        r = dict(R0=R0v, beta_sp=beta_sp,
                 herd=(1.0 - 1.0 / R0v) if R0v > 0 else float("nan"),
                 attack_min=float(fR.min()), attack_mean=float(fR.mean()),
                 attack_max=float(fR.max()),
                 attack_spread=float(fR.max() - fR.min()),
                 attack_ever_min=ever_min, attack_ever_mean=ep["ever_last"],
                 attack_ever_max=ever_max,
                 z_cl=z_cl, dev_abs=dev_abs, dev_rel=dev_rel,
                 z_rho=z_rho, dev_rho=float(fR.mean() - z_rho),
                 rho_mean=rho_mean, rho_pw=rho_pw,
                 Iend=ep["Iend"], I0_pw=rho_pw,
                 pw=ep["global_peak_week"], tpk=ep["tpk"],
                 t50=ep["t50"], t99=ep["t99"], dur=ep["dur"],
                 reached=ep["reached"],
                 mean_pw=ep["mean_state_peak_week"],
                 min_pw=ep["min_state_peak_week"], max_pw=ep["max_state_peak_week"],
                 min_tpi=ep["min_state_peak_prev_day"],
                 max_tpi=ep["max_state_peak_prev_day"],
                 pr=pr_all, sr=sr_all, pr_noTX=pr_noTX, sr_noTX=sr_noTX,
                 slope=slope,
                 tau_c=tau_c, tau_d=tau_d, tau_n=tau_n,
                 Cc=Cc, Dc=Dc, Txc=Txc, Tyc=Tyc,
                 dec_mean=float(fD.mean()), dec_spread=float(fD.max() - fD.min()),
                 cons=cons, minS=minS, nnegS=nnegS, nnegI=nnegI,
                 n_steps=int(len(grid) - 1),
                 runtime=float(time.perf_counter() - tc))
        raw[R0v] = r
        print("  R0=%5.1f beta=%.6f attack=%.6f..%.6f (spread %.3e) "
              "peakwk=%2d tau_c=%.6f tau_d=%.6f  %.2fs"
              % (R0v, beta_sp, r["attack_min"], r["attack_max"],
                 r["attack_spread"], r["pw"], tau_c, tau_d, r["runtime"]))

    # ---- supplementary low-seed group (explicitly NOT the manuscript IC).
    # A tiny TX-only seed isolates the classical well-mixed final-size
    # relation and makes the distance-from-seed / peak-timing diagnostic
    # non-degenerate.  Only R0>=2 is run: at R0<2 the epidemic has not
    # finished by 300 d, so those cells are deliberately left empty.
    for R0v in R0_LIST_PROBE:
        beta_sp = R0v / chl.TG
        tc = time.perf_counter()
        out, grid = run_sim(beta_sp, PHIS0, SIGMA0, T_END_2A_PROBE, H_SCAN,
                            S0_PROBE, BINS0_PROBE, R0_PROBE)
        ep = epidemic_endpoints(out, H_SCAN)
        fR = out["stats"]["final_R_frac"]
        z_cl = final_size_analytic(R0v, 1.0)
        tau_c, Cc, Dc, Txc, Tyc = kendall_tau_b(fR, RHO)
        tp_i = ep["state_peak_prev_day"]
        pr_all = pearson_r(DIST_TX_KM, tp_i)
        sr_all = safe_spearman(DIST_TX_KM, tp_i)
        keep = np.arange(NST) != ITX
        pr_noTX = pearson_r(DIST_TX_KM[keep], tp_i[keep])
        sr_noTX = safe_spearman(DIST_TX_KM[keep], tp_i[keep])
        slope = float(np.polyfit(DIST_TX_KM[keep], tp_i[keep], 1)[0] * 1000.0)
        r = dict(R0=R0v, beta_sp=beta_sp,
                 herd=(1.0 - 1.0 / R0v),
                 attack_min=float(fR.min()), attack_mean=float(fR.mean()),
                 attack_max=float(fR.max()),
                 attack_spread=float(fR.max() - fR.min()),
                 attack_ever_min=float((1.0 - out["S"][-1]).min()),
                 attack_ever_mean=ep["ever_last"],
                 attack_ever_max=float((1.0 - out["S"][-1]).max()),
                 z_cl=z_cl, dev_abs=float(fR.mean() - z_cl),
                 dev_rel=float(fR.mean() - z_cl) / z_cl if z_cl > 0 else float("nan"),
                 z_rho=float("nan"), dev_rho=float("nan"),
                 rho_mean=float("nan"), rho_pw=float("nan"),
                 Iend=ep["Iend"], I0_pw=float("nan"),
                 pw=ep["global_peak_week"], tpk=ep["tpk"],
                 t50=ep["t50"], t99=ep["t99"], dur=ep["dur"],
                 reached=ep["reached"],
                 mean_pw=ep["mean_state_peak_week"],
                 min_pw=ep["min_state_peak_week"], max_pw=ep["max_state_peak_week"],
                 min_tpi=ep["min_state_peak_prev_day"],
                 max_tpi=ep["max_state_peak_prev_day"],
                 pr=pr_all, sr=sr_all, pr_noTX=pr_noTX, sr_noTX=sr_noTX,
                 slope=slope,
                 tau_c=tau_c, tau_d=float("nan"), tau_n=float("nan"),
                 Cc=Cc, Dc=Dc, Txc=Txc, Tyc=Tyc,
                 dec_mean=float("nan"), dec_spread=float("nan"),
                 cons=float(out["stats"]["max_cons_resid"]),
                 minS=float(out["S"].min()),
                 nnegS=int((out["S"] < -1e-12).sum()),
                 nnegI=int((out["I"] < -1e-12).sum()),
                 n_steps=int(len(grid) - 1),
                 runtime=float(time.perf_counter() - tc))
        rawp[R0v] = r
        print("  PROBE R0=%5.1f attack=%.8f  analytic_z=%.8f  dev=%+.3e  "
              "r(dist,tpeak)=%.4f  %.2fs"
              % (R0v, r["attack_mean"], z_cl, r["dev_abs"], pr_noTX,
                 r["runtime"]))

    spread15 = raw[15.0]["attack_spread"]
    spread15p = rawp[15.0]["attack_spread"]
    header = [
        "R0", "seed_mode", "beta_sp_per_day", "herd_immunity_threshold", "horizon_days",
        "h_days", "n_steps", "attack_min", "attack_mean", "attack_max",
        "attack_spread", "attack_spread_over_R0eq15",
        "attack_ever_min", "attack_ever_mean", "attack_ever_max",
        "analytic_z_closedSIR", "dev_abs_mean_vs_closedSIR",
        "dev_rel_mean_vs_closedSIR", "analytic_z_rho_corrected",
        "dev_abs_mean_vs_rho_corrected", "rho_mean_unweighted",
        "rho_mean_popweighted", "I_end_max", "I0_popweighted",
        "global_peak_week", "t_peak_global_prev_day", "t50_newinf_day",
        "t99_newinf_day", "epidemic_duration_day", "duration_reached",
        "mean_state_peak_week", "min_state_peak_week", "max_state_peak_week",
        "min_state_peak_prev_day", "max_state_peak_prev_day",
        "pearson_r_peakday_vs_distTX", "spearman_r_peakday_vs_distTX",
        "pearson_r_peakday_vs_distTX_exclTX",
        "spearman_r_peakday_vs_distTX_exclTX", "slope_day_per_1000km_exclTX",
        "tau_b_coupled_finalR", "tau_b_decoupled_finalR",
        "tau_b_coupled_newinf", "tau_b_coupled_C", "tau_b_coupled_D",
        "tau_b_coupled_Tx", "tau_b_coupled_Ty",
        "decoupled_attack_mean", "decoupled_attack_spread",
        "max_cons_resid", "min_S", "n_neg_S", "n_neg_I", "runtime_s"]
    def make_row(r, mode, horizon, refspread):
        return [
            r["R0"], mode, r["beta_sp"], r["herd"], horizon, H_SCAN, r["n_steps"],
            r["attack_min"], r["attack_mean"], r["attack_max"], r["attack_spread"],
            r["attack_spread"] / refspread,
            r["attack_ever_min"], r["attack_ever_mean"], r["attack_ever_max"],
            r["z_cl"], r["dev_abs"], r["dev_rel"], r["z_rho"], r["dev_rho"],
            r["rho_mean"], r["rho_pw"], r["Iend"], r["I0_pw"],
            r["pw"], r["tpk"], r["t50"], r["t99"], r["dur"], r["reached"],
            r["mean_pw"], r["min_pw"], r["max_pw"], r["min_tpi"], r["max_tpi"],
            r["pr"], r["sr"], r["pr_noTX"], r["sr_noTX"], r["slope"],
            r["tau_c"], r["tau_d"], r["tau_n"],
            r["Cc"], r["Dc"], r["Txc"], r["Tyc"],
            r["dec_mean"], r["dec_spread"],
            r["cons"], r["minS"], r["nnegS"], r["nnegI"], r["runtime"]]

    rows = []
    for R0v in R0_LIST:
        rows.append(make_row(raw[R0v], "canonical", T_END_2A, spread15))
    for R0v in R0_LIST_PROBE:
        rows.append(make_row(rawp[R0v], "low_seed_probe", T_END_2A_PROBE,
                             spread15p))
    md5 = write_double("nonsaturating_scan.csv", header, rows)
    PERF["2a_s"] = time.perf_counter() - t0

    # ---- grid convergence cross-check at R0=3 (H=0.1 vs H=0.05) ----
    conv = {}
    for hh in (0.1, 0.05):
        beta_sp = 3.0 / chl.TG
        out, grid = run_sim(beta_sp, PHIS0, SIGMA0, T_END_2A, hh,
                            S0_CAN, BINS0_CAN, R0_CAN)
        ep = epidemic_endpoints(out, hh)
        conv[hh] = dict(attack=float(out["stats"]["final_R_frac"].mean()),
                        tpk=ep["tpk"], t50=ep["t50"], pw=ep["global_peak_week"],
                        cons=float(out["stats"]["max_cons_resid"]),
                        n_steps=int(len(grid) - 1))
    PERF["2a_gridconv"] = conv
    return raw, md5, conv, rawp


# =============================================================================
# 2. EXPERIMENT 2B : within-host kernel sensitivity
# =============================================================================
def make_kernel(tg, eps_v, shape, shift, kv=KV_SAT):
    """Build an infectiousness kernel from the SAME V(tau) table.

    tau*  : first tau > t_peak with V(tau) <= eps_v (same rule as chl)
    shape : 'linear'  -> phi ∝ V                 (manuscript's linear kernel)
            'saturated' -> phi ∝ V/(V+K_V)      (source-model beta_eff, K_V=0.1)
    shift : right-translate phi by m = round(shift/dtau) bins on the FIXED
            [0,tau*] support; the front is zero-padded and the tail is dropped.
            The whole kernel is renormalised so sum(phi)*dtau == tg EXACTLY,
            hence beta_sp = R0/tg gives R0_eff = beta_sp * int(phi) = R0
            exactly for every configuration (no R0 drift confound).
    """
    t_peak = float(TS[int(np.argmax(VS))])
    sel = TS > t_peak
    below = TS[sel][VS[sel] <= eps_v]
    tau_star = float(below[0]) if below.size > 0 else float(TS[-1])
    dtau = tau_star / K
    tc = (np.arange(K, dtype=DTYPE) + 0.5) * dtau
    V = np.interp(tc, TS, VS)
    shp = V.copy() if shape == "linear" else V / (V + kv)
    p = shp.copy()
    m = int(round(shift / dtau))
    if m > 0:
        p = np.concatenate([np.zeros(m, dtype=DTYPE), p[:K - m]])
    p = p * (tg / (p.sum() * dtau))
    sigma = K / tau_star
    int_phi = float(p.sum() * dtau)
    GI = float((tc * p).sum() * dtau / int_phi)
    return dict(phis=p, sigma=sigma, tau_star=tau_star, dtau=dtau,
                int_phi=int_phi, GI=GI, shift_bins=m,
                shift_realized=m * dtau, peak_age=float(tc[int(np.argmax(p))]))


def sens_configs():
    """One-factor-at-a-time list around (tg=13, eps=1e-3, linear, shift=0)."""
    cfgs = []
    for tg in TG_LIST:
        cfgs.append(dict(tag="Tg", tg=tg, eps=1e-3, shape="linear", shift=0.0))
    for ev in EPS_LIST:
        cfgs.append(dict(tag="epsV", tg=13.0, eps=ev, shape="linear", shift=0.0))
    cfgs.append(dict(tag="shape", tg=13.0, eps=1e-3, shape="saturated", shift=0.0))
    for sh in (4.0, 6.0):
        cfgs.append(dict(tag="shift", tg=13.0, eps=1e-3, shape="linear", shift=sh))
    return cfgs


def experiment_2b():
    print("\n[2B] within-host kernel sensitivity (canonical IC + low-seed probe)")
    t0 = time.perf_counter()
    cfgs = sens_configs()
    rows = []
    rec = {}

    for R0v in R0_SENS:
        group = []
        for i, c in enumerate(cfgs):
            ker = make_kernel(c["tg"], c["eps"], c["shape"], c["shift"])
            beta_sp = R0v / c["tg"]
            tc = time.perf_counter()
            out, grid = run_sim(beta_sp, ker["phis"], ker["sigma"], T_END_2B,
                                H_SCAN, S0_CAN, BINS0_CAN, R0_CAN)
            ep = epidemic_endpoints(out, H_SCAN)
            fR = out["stats"]["final_R_frac"]
            d = dict(
                config_id="R%g_%02d_%s" % (R0v, i, c["tag"]), R0=R0v,
                seed_mode="canonical", tag=c["tag"], tg=c["tg"], eps=c["eps"],
                shape=c["shape"], kv=KV_SAT if c["shape"] == "saturated" else 0.0,
                shift_req=c["shift"], shift_real=ker["shift_realized"],
                tau_star=ker["tau_star"], sigma=ker["sigma"], dtau=ker["dtau"],
                int_phi=ker["int_phi"], GI=ker["GI"], peak_age=ker["peak_age"],
                beta_sp=beta_sp, R0_eff=beta_sp * ker["int_phi"],
                horizon=T_END_2B, h=H_SCAN,
                tpk=ep["tpk"], pw=ep["global_peak_week"], t50=ep["t50"],
                t99=ep["t99"],
                a_min=float(fR.min()), a_mean=float(fR.mean()),
                a_max=float(fR.max()), a_spread=float(fR.max() - fR.min()),
                growth=ep["growth"], nfit=ep["nfit"],
                cons=float(out["stats"]["max_cons_resid"]),
                minS=float(out["S"].min()),
                nneg=int((out["S"] < -1e-12).sum() + (out["I"] < -1e-12).sum()),
                Iend=ep["Iend"], runtime=float(time.perf_counter() - tc))
            group.append(d)
        base = next(g for g in group if g["tag"] == "Tg" and g["tg"] == 13.0
                    and g["eps"] == 1e-3 and g["shape"] == "linear"
                    and g["shift_req"] == 0.0)
        for d in group:
            d["d_attack_pct"] = 100.0 * (d["a_mean"] - base["a_mean"]) / base["a_mean"]
            d["d_GI_pct"] = 100.0 * (d["GI"] - base["GI"]) / base["GI"]
            d["d_tpk"] = d["tpk"] - base["tpk"]
            d["d_t50_pct"] = 100.0 * (d["t50"] - base["t50"]) / base["t50"]
            d["d_pw"] = d["pw"] - base["pw"]
        rec[R0v] = group
        rows.extend(group)
        print("  R0=%.1f : %d configs, baseline attack=%.6f GI=%.4f peakwk=%d"
              % (R0v, len(group), base["a_mean"], base["GI"], base["pw"]))

    # ---- supplementary low-seed diagnostic (explicitly NOT the manuscript IC) ----
    probe_cfgs = [
        dict(tag="baseline", shape="linear", shift=0.0),
        dict(tag="shape", shape="saturated", shift=0.0),
        dict(tag="shift", shape="linear", shift=4.0),
        dict(tag="shift", shape="linear", shift=6.0),
    ]
    R0p = 3.0
    for i, c in enumerate(probe_cfgs):
        ker = make_kernel(13.0, 1e-3, c["shape"], c["shift"])
        beta_sp = R0p / 13.0
        tc = time.perf_counter()
        out, grid = run_sim(beta_sp, ker["phis"], ker["sigma"], T_END_2B_PROBE,
                            H_SCAN, S0_PROBE, BINS0_PROBE, R0_PROBE)
        ep = epidemic_endpoints(out, H_SCAN)
        fR = out["stats"]["final_R_frac"]
        rows.append(dict(
            config_id="PROBE_R%g_%02d_%s" % (R0p, i, c["tag"]), R0=R0p,
            seed_mode="low_seed_probe", tag=c["tag"], tg=13.0, eps=1e-3,
            shape=c["shape"], kv=KV_SAT if c["shape"] == "saturated" else 0.0,
            shift_req=c["shift"], shift_real=ker["shift_realized"],
            tau_star=ker["tau_star"], sigma=ker["sigma"], dtau=ker["dtau"],
            int_phi=ker["int_phi"], GI=ker["GI"], peak_age=ker["peak_age"],
            beta_sp=beta_sp, R0_eff=beta_sp * ker["int_phi"],
            horizon=T_END_2B_PROBE, h=H_SCAN,
            tpk=ep["tpk"], pw=ep["global_peak_week"], t50=ep["t50"],
            t99=ep["t99"],
            a_min=float(fR.min()), a_mean=float(fR.mean()),
            a_max=float(fR.max()), a_spread=float(fR.max() - fR.min()),
            growth=ep["growth"], nfit=ep["nfit"],
            cons=float(out["stats"]["max_cons_resid"]),
            minS=float(out["S"].min()),
            nneg=int((out["S"] < -1e-12).sum() + (out["I"] < -1e-12).sum()),
            Iend=ep["Iend"], runtime=float(time.perf_counter() - tc),
            d_attack_pct=float("nan"), d_GI_pct=float("nan"),
            d_tpk=float("nan"), d_t50_pct=float("nan"), d_pw=float("nan")))
        print("  PROBE %-9s shift=%g GI=%.4f growth_r=%.6f peakwk=%2d attack=%.6f"
              % (c["shape"], c["shift"], ker["GI"], rows[-1]["growth"],
                 rows[-1]["pw"], rows[-1]["a_mean"]))
    # probe % changes relative to its own baseline
    pb = rows[-len(probe_cfgs)]
    for d in rows[-len(probe_cfgs):]:
        d["d_attack_pct"] = 100.0 * (d["a_mean"] - pb["a_mean"]) / pb["a_mean"]
        d["d_GI_pct"] = 100.0 * (d["GI"] - pb["GI"]) / pb["GI"]
        d["d_tpk"] = d["tpk"] - pb["tpk"]
        d["d_t50_pct"] = 100.0 * (d["t50"] - pb["t50"]) / pb["t50"]
        d["d_pw"] = d["pw"] - pb["pw"]

    header = [
        "config_id", "R0", "seed_mode", "factor", "tg_days", "eps_V",
        "kernel_shape", "K_V", "latent_shift_requested_d",
        "latent_shift_realized_d", "tau_star_d", "sigma_per_d", "dtau_d",
        "int_phi_d", "GI_d", "phi_peak_age_d", "beta_sp_per_d", "R0_eff",
        "horizon_d", "h_d", "t_peak_global_prev_d", "global_peak_week",
        "t50_newinf_d", "t99_newinf_d", "attack_min", "attack_mean",
        "attack_max", "attack_spread", "d_attack_mean_pct", "d_GI_pct",
        "d_tpeak_d", "d_t50_pct", "d_peakweek", "early_growth_rate_per_d",
        "early_growth_fit_weeks", "max_cons_resid", "min_S", "n_neg",
        "I_end_max", "runtime_s"]
    outrows = []
    for d in rows:
        outrows.append([
            d["config_id"], d["R0"], d["seed_mode"], d["tag"], d["tg"], d["eps"],
            d["shape"], d["kv"], d["shift_req"], d["shift_real"],
            d["tau_star"], d["sigma"], d["dtau"], d["int_phi"], d["GI"],
            d["peak_age"], d["beta_sp"], d["R0_eff"], d["horizon"], d["h"],
            d["tpk"], d["pw"], d["t50"], d["t99"],
            d["a_min"], d["a_mean"], d["a_max"], d["a_spread"],
            d["d_attack_pct"], d["d_GI_pct"], d["d_tpk"], d["d_t50_pct"],
            d["d_pw"], d["growth"], d["nfit"], d["cons"], d["minS"],
            d["nneg"], d["Iend"], d["runtime"]])
    md5 = write_double("kernel_sensitivity.csv", header, outrows)
    PERF["2b_s"] = time.perf_counter() - t0
    return rows, md5


# =============================================================================
# 3. EXPERIMENT 2C : large-step (h >= 9 d) epidemiological endpoint
# =============================================================================
# Local re-implementations of the published baseline arms, using chl building
# blocks (poisson_T, static_kernel, lam_of).  These mirror m3_experiment.py
# exactly; they are needed because chl.baseline_step does not expose the
# Richardson / explicit-RK / raw-Lagrange attacks.
def lam_of_local(S, I):
    return chl.lam_of(S, I, W, N, _2C["phis"], _2C["beta_sp"])


def fresh_closed_local(h, S, lam, sigma):
    a = sigma - lam
    with np.errstate(all="ignore"):
        asafe = np.where(a > 0, a, np.nan)
        ratio = (sigma / asafe)[:, None] ** KK[None, :] / asafe[:, None]
        G = gammainc(KK[None, :] + 1.0, (a * h)[:, None])
        return (lam * S * np.exp(-lam * h))[:, None] * ratio * G


def exp2_closed_step(S, I, R, h, lam, sigma, T):
    Iold = I @ T.T
    removed = I.sum(1) - Iold.sum(1)
    S_new = S * np.exp(-lam * h)
    delivered = S - S_new
    If = fresh_closed_local(h, S, lam, sigma)
    toR = delivered - If.sum(1)
    return S_new, Iold + If, R + removed + toR


def midpoint_lam(S, I, R, h, sigma):
    lam0 = lam_of_local(S, I)
    T2 = chl.poisson_T(sigma, 0.5 * h, K=K)
    _, bh, _ = exp2_closed_step(S, I, R, 0.5 * h, lam0, sigma, T2)
    return lam_of_local(S, bh)


def rhs_flat(y, sigma):
    S = y[:NST]
    I = y[NST:NST + NST * K].reshape(NST, K)
    lam = chl.lam_of(S, I, W, N, _2C["phis"], _2C["beta_sp"])
    dS = -lam * S
    dI = -sigma * I
    dI[:, 1:] += sigma * I[:, :-1]
    dI[:, 0] += lam * S
    dR = sigma * I[:, K - 1]
    return np.concatenate([dS, dI.ravel(), dR])


def series_from_raw(S, I, R, t):
    """Append one post-step state to the running diagnostic accumulators.

    Same convention as m3_experiment.run_explicit_rk: a post-step state that
    is not finite is flagged (all_finite=False) and NOT included in the
    conservation / negativity / peak accumulators.
    """
    It = I.sum(1)
    Sabs, Iabs, Rabs = S, It, R
    d = _2C["acc"]
    ok = bool(np.all(np.isfinite(Sabs)) and np.all(np.isfinite(Iabs))
              and np.all(np.isfinite(Rabs)))
    if not ok:
        d["all_finite"] = False
        return
    cons = float(np.max(np.abs(Sabs + Iabs + Rabs - N)))
    d["cons"] = max(d["cons"], cons)
    d["cons_rel"] = max(d["cons_rel"], cons / float(N.min()))
    d["neg_S"] += int((Sabs < -1e-9).sum())
    d["neg_I"] += int((Iabs < -1e-9).sum())
    d["neg_R"] += int((Rabs < -1e-9).sum())
    d["min_S"] = min(d["min_S"], float(Sabs.min()))
    d["min_I"] = min(d["min_I"], float(Iabs.min()))
    prev = float(Iabs.sum() / N.sum())
    if prev > d["peak_prev_value"]:
        d["peak_prev_value"] = prev
        d["peak_prev_day"] = float(t)
    d["S_series"].append(Sabs.copy())
    d["I_series"].append(Iabs.copy())
    d["R_series"].append(Rabs.copy())
    d["t_series"].append(float(t))


def run_arm_2c(arm, h, phis, sigma, beta_sp, kernel_label):
    """Return a diagnostics dict for one (arm, h) run on the 60-day grid."""
    _2C.update(dict(phis=phis, beta_sp=beta_sp))
    # grid ending EXACTLY at 60 d; for h=9 the last interval is a partial step
    grid = np.round(np.append(np.arange(0.0, T_END_2C, h), T_END_2C), 8)
    # de-duplicate in case 60 falls exactly on the grid
    grid = np.unique(grid)
    nstep = len(grid) - 1

    S = S0_CAN.copy()
    I = BINS0_CAN.copy()
    R = R0_CAN.copy()

    _2C["acc"] = dict(cons=0.0, cons_rel=0.0, neg_S=0, neg_I=0, neg_R=0,
                      min_S=float("inf"), min_I=float("inf"),
                      all_finite=True, peak_prev_value=-float("inf"),
                      peak_prev_day=float("nan"),
                      S_series=[S.copy()], I_series=[I.sum(1).copy()],
                      R_series=[R.copy()], t_series=[float(grid[0])])
    # initial-state peak
    prev0 = float((I.sum(1)).sum() / N.sum())
    _2C["acc"]["peak_prev_value"] = prev0
    _2C["acc"]["peak_prev_day"] = float(grid[0])

    crashed = ""
    if arm == "M3-p3":
        out = chl.simulate(S0_CAN, BINS0_CAN, R0_CAN, W, N, phis, sigma,
                           beta_sp, h, grid, arm="M3-p3")
        # rebuild diagnostics from the recorded series (same definitions)
        acc = dict(cons=0.0, cons_rel=0.0, neg_S=0, neg_I=0, neg_R=0,
                   min_S=float("inf"), min_I=float("inf"), all_finite=True,
                   peak_prev_value=-float("inf"), peak_prev_day=float("nan"))
        Sabs = out["S"] * N[None, :]
        Iabs = out["I"] * N[None, :]
        Rabs = out["R"] * N[None, :]
        for r in range(len(out["t"])):
            cons = float(np.max(np.abs(Sabs[r] + Iabs[r] + Rabs[r] - N)))
            acc["cons"] = max(acc["cons"], cons)
            acc["cons_rel"] = max(acc["cons_rel"], cons / float(N.min()))
            if r > 0:
                acc["neg_S"] += int((Sabs[r] < -1e-9).sum())
                acc["neg_I"] += int((Iabs[r] < -1e-9).sum())
                acc["neg_R"] += int((Rabs[r] < -1e-9).sum())
            acc["min_S"] = min(acc["min_S"], float(Sabs[r].min()))
            acc["min_I"] = min(acc["min_I"], float(Iabs[r].min()))
            if not (np.all(np.isfinite(Sabs[r])) and np.all(np.isfinite(Iabs[r]))
                    and np.all(np.isfinite(Rabs[r]))):
                acc["all_finite"] = False
            pv = float(Iabs[r].sum() / N.sum())
            if pv > acc["peak_prev_value"]:
                acc["peak_prev_value"] = pv
                acc["peak_prev_day"] = float(out["t"][r])
        S, I, R = Sabs[-1], Iabs[-1], Rabs[-1]
        _2C["acc"] = acc
    else:
        for k in range(nstep):
            dt = float(grid[k + 1] - grid[k])
            try:
                with np.errstate(all="ignore"):
                    if arm == "EXP2-midpoint":
                        lam = midpoint_lam(S, I, R, dt, sigma)
                        T = chl.poisson_T(sigma, dt, K=K)
                        S, I, R = exp2_closed_step(S, I, R, dt, lam, sigma, T)
                    elif arm == "Richardson(EXP2-midpoint)":
                        lam0 = lam_of_local(S, I)
                        T2 = chl.poisson_T(sigma, 0.5 * dt, K=K)
                        _, bh, _ = exp2_closed_step(S, I, R, 0.5 * dt, lam0,
                                                    sigma, T2)
                        lm = lam_of_local(S, bh)
                        Tf = chl.poisson_T(sigma, dt, K=K)
                        Sh, Ih, Rh = exp2_closed_step(S, I, R, dt, lm, sigma, Tf)
                        S2, I2, R2 = S.copy(), I.copy(), R.copy()
                        T4 = chl.poisson_T(sigma, 0.25 * dt, K=K)
                        for _ in range(2):
                            l0 = lam_of_local(S2, I2)
                            _, bq, _ = exp2_closed_step(S2, I2, R2, 0.25 * dt,
                                                        l0, sigma, T4)
                            lq = lam_of_local(S2, bq)
                            S2, I2, R2 = exp2_closed_step(S2, I2, R2, 0.5 * dt,
                                                          lq, sigma, T2)
                        S = (4.0 * S2 - Sh) / 3.0
                        I = (4.0 * I2 - Ih) / 3.0
                        R = (4.0 * R2 - Rh) / 3.0
                    elif arm == "RK4-explicit":
                        y = np.concatenate([S, I.ravel(), R])
                        k1 = rhs_flat(y, sigma)
                        k2 = rhs_flat(y + 0.5 * dt * k1, sigma)
                        k3 = rhs_flat(y + 0.5 * dt * k2, sigma)
                        k4 = rhs_flat(y + dt * k3, sigma)
                        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                        S = y[:NST].copy()
                        I = y[NST:NST + NST * K].reshape(NST, K).copy()
                        R = y[NST + NST * K:].copy()
                    elif arm == "Lagrange3-predictor":
                        kern = chl.static_kernel(dt, sigma, K=K)
                        T = chl.poisson_T(sigma, dt, K=K)
                        T2 = chl.poisson_T(sigma, 0.5 * dt, K=K)
                        lam0 = lam_of_local(S, I)
                        _, bh, _ = exp2_closed_step(S, I, R, 0.5 * dt, lam0,
                                                    sigma, T2)
                        _, bf, _ = exp2_closed_step(S, I, R, dt, lam0, sigma, T)
                        lm = lam_of_local(S, bh)
                        le = lam_of_local(S, bf)
                        A = lam0
                        B = (-3 * lam0 + 4 * lm - le) / dt
                        C = (2 * lam0 - 4 * lm + 2 * le) / (dt * dt)
                        poly = np.stack([A, B, C], axis=1)       # (n,3)
                        u = kern["u"]
                        upow = u[None, :] ** np.arange(3)[:, None]
                        lam_u = poly @ upow
                        Vint = (u[None, :] ** np.arange(1, 4)[:, None]
                                / np.arange(1, 4)[:, None])
                        H = poly @ Vint
                        Hh = poly @ (dt ** np.arange(1, 4) / np.arange(1, 4))
                        g = lam_u * S[:, None] * np.exp(-H)
                        wg = kern["w"][None, :] * g
                        Z = wg.sum(1)
                        den = np.where(np.abs(Z) > 0, Z, 1.0)
                        pj = np.where(np.abs(Z[:, None]) > 0, wg / den[:, None],
                                      kern["w"][None, :] / kern["w"].sum())
                        S_new = S * np.exp(-Hh)
                        delivered = S - S_new
                        Ifresh = delivered[:, None] * (pj @ kern["Pmat"])
                        toR = delivered * (pj @ kern["surv"])
                        Iold = I @ T.T
                        removed = I @ kern["sv_old"]
                        I = Iold + Ifresh
                        R = R + removed + toR
                        S = S_new
                    else:
                        raise ValueError(arm)
            except Exception as exc:                 # noqa: BLE001
                crashed = "%s: %s" % (type(exc).__name__, exc)
                S = np.full_like(S, np.nan)
                I = np.full_like(I, np.nan)
                R = np.full_like(R, np.nan)
                _2C["acc"]["all_finite"] = False
            series_from_raw(S, I, R, grid[k + 1])
            if not _2C["acc"]["all_finite"]:
                # same convention as m3_experiment.run_explicit_rk: break early
                break

    acc = _2C["acc"]
    # final cumulative attack = 1 - S(T)/N ; S+I+R=N makes this "ever infected"
    final_attack = float((N - S).sum() / N.sum())
    bounded = bool(np.isfinite(final_attack) and -1e-12 <= final_attack <= 1 + 1e-12)
    finite = bool(acc["all_finite"] and np.all(np.isfinite([final_attack,
                                                          acc["peak_prev_value"]])))
    physical = bool(finite and acc["neg_S"] == 0 and acc["neg_I"] == 0
                    and acc["neg_R"] == 0 and bounded and acc["min_S"] >= -1e-9)
    return dict(h=h, arm=arm, kernel_label=kernel_label, n_steps=nstep,
                all_finite=finite, neg_S=acc["neg_S"], neg_I=acc["neg_I"],
                neg_R=acc["neg_R"], min_S=acc["min_S"], min_I=acc["min_I"],
                cons=acc["cons"], cons_rel=acc["cons_rel"],
                final_attack=final_attack, bounded=bounded,
                peak_prev_day=acc["peak_prev_day"],
                peak_prev_value=acc["peak_prev_value"],
                physical=physical, crashed=crashed)


_2C = {}


def experiment_2c():
    print("\n[2C] large-step (h>=9 d) epidemiological endpoint")
    t0 = time.perf_counter()
    arms = ["M3-p3", "Richardson(EXP2-midpoint)", "RK4-explicit",
            "Lagrange3-predictor", "EXP2-midpoint"]
    order_class = {
        "M3-p3": "fourth_order_positivity_preserving",
        "Richardson(EXP2-midpoint)": "fourth_order_sign_indefinite",
        "RK4-explicit": "fourth_order_sign_indefinite",
        "Lagrange3-predictor": "fourth_order_sign_indefinite",
        "EXP2-midpoint": "second_order_positivity_preserving",
    }
    sign_guarantee = {
        "M3-p3": "yes_proposition_pos",
        "Richardson(EXP2-midpoint)": "no",
        "RK4-explicit": "no",
        "Lagrange3-predictor": "no",
        "EXP2-midpoint": "yes_exponential_form",
    }
    rows = []
    for h in H_LARGE:
        for arm in arms:
            tc = time.perf_counter()
            d = run_arm_2c(arm, h, PHIS0, SIGMA0, BETA_CAL, "eps1e-3")
            d.update(order_class=order_class[arm],
                     sign_guarantee=sign_guarantee[arm],
                     eps=1e-3, tau_star=KER["tau_star"], sigma=SIGMA0,
                     runtime=float(time.perf_counter() - tc))
            rows.append(d)
            print("  h=%4.0f %-26s finite=%-5s negS=%-4d minS=%11.3e "
                  "attack=%12.6e physical=%-5s"
                  % (h, arm, d["all_finite"], d["neg_S"], d["min_S"],
                     d["final_attack"], d["physical"]))

    # tau* -> endpoint sensitivity under M3-p3 and the two sign-indefinite arms
    for ev in (1e-2, 1e-5):
        ker = chl.build_infectious_kernel(eps_v=ev)
        for arm in ("M3-p3", "Richardson(EXP2-midpoint)", "RK4-explicit"):
            tc = time.perf_counter()
            d = run_arm_2c(arm, 10.0, ker["phis"], ker["sigma"], BETA_CAL,
                           "eps%g" % ev)
            d.update(order_class=order_class[arm],
                     sign_guarantee=sign_guarantee[arm],
                     eps=ev, tau_star=ker["tau_star"], sigma=ker["sigma"],
                     runtime=float(time.perf_counter() - tc))
            rows.append(d)
            print("  eps=%-6g h=10 %-26s tau*=%.1f attack=%.6e physical=%s"
                  % (ev, arm, ker["tau_star"], d["final_attack"], d["physical"]))

    # relative endpoint error vs M3-p3 on the SAME (h, kernel) cell
    ref = {}
    for d in rows:
        if d["arm"] == "M3-p3":
            ref[(d["h"], d["kernel_label"])] = d["final_attack"]
    for d in rows:
        a = ref.get((d["h"], d["kernel_label"]), float("nan"))
        if np.isfinite(a) and a != 0 and np.isfinite(d["final_attack"]):
            d["abs_err"] = abs(d["final_attack"] - a)
            d["rel_err"] = abs(d["final_attack"] - a) / abs(a)
        else:
            d["abs_err"] = float("nan")
            d["rel_err"] = float("nan")

    header = [
        "h_days", "arm", "order_class", "sign_guarantee", "kernel_variant",
        "eps_V", "tau_star_d", "sigma_per_d", "n_steps", "horizon_days",
        "all_finite", "neg_S_cells", "neg_I_cells", "neg_R_cells", "min_S",
        "min_I", "max_cons_resid_abs", "max_cons_resid_rel_to_minpop",
        "final_attack_rate", "final_attack_bounded", "peak_prev_day",
        "peak_prev_value", "attack_abs_err_vs_M3p3", "attack_rel_err_vs_M3p3",
        "endpoint_finite_and_physical", "crashed", "runtime_s"]
    outrows = []
    for d in rows:
        outrows.append([
            d["h"], d["arm"], d["order_class"], d["sign_guarantee"],
            d["kernel_label"], d["eps"], d["tau_star"], d["sigma"],
            d["n_steps"], T_END_2C, d["all_finite"], d["neg_S"], d["neg_I"],
            d["neg_R"], d["min_S"], d["min_I"], d["cons"], d["cons_rel"],
            d["final_attack"], d["bounded"], d["peak_prev_day"],
            d["peak_prev_value"], d["abs_err"], d["rel_err"], d["physical"],
            d["crashed"], d["runtime"]])
    md5 = write_double("large_step_endpoint.csv", header, outrows)
    PERF["2c_s"] = time.perf_counter() - t0
    return rows, md5


# =============================================================================
# 4. R0 -> attack-rate chart (matplotlib, Agg, no display)
# =============================================================================
def make_chart(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io as _io
    R0s = np.array(R0_LIST, dtype=float)
    mean = np.array([raw[r]["attack_mean"] for r in R0_LIST])
    lo = np.array([raw[r]["attack_min"] for r in R0_LIST])
    hi = np.array([raw[r]["attack_max"] for r in R0_LIST])
    z = np.array([final_size_analytic(r, 1.0) for r in R0_LIST])
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=160)
    ax.fill_between(R0s, lo, hi, color="#8ecae6", alpha=0.55,
                    label="52-entity min/max attack rate")
    ax.plot(R0s, mean, "o-", color="#023047", lw=1.8, ms=5,
            label="mean final attack rate (model)")
    ax.plot(R0s, z, "s--", color="#fb8500", lw=1.5, ms=4,
            label=r"closed SIR $z=1-e^{-R_0 z}$")
    ax.axvline(1.0, color="0.5", lw=0.9, ls=":")
    ax.set_xlabel(r"$R_0=\beta_{sp}T_g$")
    ax.set_ylabel("final attack rate")
    ax.set_title("Non-saturating coupling scan (measured)\n"
                 "All values pending user numerical confirmation", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    payload = buf.getvalue()
    dirs = [ROOT / "figures", SUB / "figures"]
    dig = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "nonsaturating_attack_rate.png", "wb") as f:
            f.write(payload)
        dig.append(hashlib.md5((d / "nonsaturating_attack_rate.png").read_bytes()).hexdigest())
    if len(set(dig)) != 1:
        raise RuntimeError("png md5 mismatch %s" % dig)
    print("[write] %-30s bytes=%-7d md5=%s"
          % ("nonsaturating_attack_rate.png", len(payload), dig[0]))
    return dig[0]


# =============================================================================
# 5. main
# =============================================================================
def main():
    print("=" * 78)
    print("R8 / route B / experiment line 2 : non-saturating scan")
    print("python=%s numpy=%s" % (sys.version.split()[0], np.__version__))
    print("kernel: tau*=%.4f d sigma=%.9f /d int_phi=%.9f d V_peak=%.7f @%.1f d"
          % (KER["tau_star"], SIGMA0, KER["sum_phi_dtau"],
             KER["V_peak"], KER["t_peak"]))
    print("states=%d K=%d beta_sp_cal=%.9f R0_cal=%.4f"
          % (NST, K, BETA_CAL, BETA_CAL * chl.TG))
    print("=" * 78)

    raw2a, md5_2a, conv, raw2ap = experiment_2a()
    rows2b, md5_2b = experiment_2b()
    rows2c, md5_2c = experiment_2c()
    md5_png = make_chart(raw2a)

    # ---- self-tests / verification, all measured ----
    z3 = final_size_analytic(3.0, 1.0)
    probe_base = next(d for d in rows2b
                      if d["config_id"].startswith("PROBE_R3_00"))
    tg_rows = [d for d in rows2b if d["seed_mode"] == "canonical"
               and d["tag"] == "Tg" and d["R0"] == 3.0]
    tg_spread = max(d["a_mean"] for d in tg_rows) - min(d["a_mean"] for d in tg_rows)
    all2a = list(raw2a.values()) + list(raw2ap.values())
    verify = dict(
        tg_invariance_max_attack_diff=tg_spread,
        probe_R03_attack_minus_analytic=raw2ap[3.0]["attack_mean"] - z3,
        low_seed_attack=probe_base["a_mean"],
        analytic_z_R0_3=z3,
        low_seed_attack_minus_analytic=probe_base["a_mean"] - z3,
        conv_h01_attack=conv[0.1]["attack"],
        conv_h005_attack=conv[0.05]["attack"],
        conv_attack_abs_diff=abs(conv[0.1]["attack"] - conv[0.05]["attack"]),
        conv_tpk_diff=abs(conv[0.1]["tpk"] - conv[0.05]["tpk"]),
        max_cons_2a=max(r["cons"] for r in all2a),
        min_S_2a=min(r["minS"] for r in all2a),
        max_cons_2b=max(d["cons"] for d in rows2b),
        min_S_2b=min(d["minS"] for d in rows2b),
        n_neg_2a=sum(r["nnegS"] + r["nnegI"] for r in all2a),
        n_neg_2b=sum(d["nneg"] for d in rows2b),
        Iend_max_2a=max(r["Iend"] for r in all2a),
        Iend_max_2b=max(d["Iend"] for d in rows2b),
        total_s=time.perf_counter() - TSTART,
    )
    print("\n[verify] (all 数值待用户数值确认)")
    for k, v in verify.items():
        print("  %-32s = %s" % (k, fmt(v)))
    print("\n[md5] scan=%s kernel=%s large_step=%s png=%s"
          % (md5_2a, md5_2b, md5_2c, md5_png))

    # persist verification payload for the findings report (no extra CSV in
    # the delivery contract; printed to stdout and transcribed into the .md)
    print("[done] total %.2f s" % verify["total_s"])
    return verify


if __name__ == "__main__":
    main()
