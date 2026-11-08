"""Regenerate fig7_rank.csv under the CORRECTED nested coupling, with M3.

Honest content (replaces the wrong-coupling fig7):
  columns: state, tycho_norm (= initial rho, TX normalised to 1),
           coupled_R  (= final attack fraction R^h/N under calibrated beta_sp),
           decoupled_R (= final R^h/N with beta_sp=0, equals initial rho)
All three columns are already fractions in [0,1]; the Tycho target used for the
rank statistic is tycho_norm.

The host-level step in ``chl.simulate`` is M3 (p = 3), so this file is an M3
product.  Written to BOTH ``submission/figures/`` and ``<repo>/figures/``; the
two copies are byte-identical by construction (single deterministic run).

Also prints the concordant / discordant pair counts, Kendall tau_b, Spearman
rho and Pearson r for the four candidate rank statistics, so that no number in
the manuscript has to be quoted without its pair counts.
"""
import os, sys, csv, math
import numpy as np
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ms_numpy as msn
import coupling_host_level as chl

SUB = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [SUB / "figures"]
if (ROOT / "figures").is_dir():
    OUT_DIRS.append(ROOT / "figures")


def kendall_tau_b(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    C = D = Tx = Ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
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
    denom = math.sqrt((C + D + Tx) * (C + D + Ty))
    tau = (C - D) / denom if denom > 0 else 0.0
    return tau, C, D, Tx, Ty


def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc) / den if den > 0 else float("nan")


def spearman_rho(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r
    return pearson_r(rank(x), rank(y))


def main():
    ker = chl.build_infectious_kernel()
    phis, sigma = ker['phis'], ker['sigma']
    states, rho = chl.load_rho(chl.RHO_CSV)
    W, N, beta_sp = chl.build_gravity(states)
    K = chl.K_BINS
    S0 = (1.0 - rho) * N; I0 = rho * N; R0 = np.zeros_like(S0)
    bins0 = np.tile(I0[:, None] / K, (1, K))
    t_grid = np.round(np.arange(chl.T_SPAN[0], chl.T_SPAN[1] + chl.H / 2, chl.H), 8)

    coupled = chl.simulate(S0, bins0, R0, W, N, phis, sigma, beta_sp, chl.H, t_grid)
    decoup = chl.simulate(S0, bins0, R0, W, N, phis, sigma, 0.0, chl.H, t_grid)

    fR_c = coupled['stats']['final_R_frac']      # ~ universal ~1.0
    fR_d = decoup['stats']['final_R_frac']      # == initial rho

    # tycho target = initial rho (already normalised TX=1 in fig2 field)
    ty = rho.copy()

    tb_c, Cc, Dc, Txc, Tyc = kendall_tau_b(fR_c, ty)
    tb_d, Cd, Dd, Txd, Tyd = kendall_tau_b(fR_d, ty)
    cumI_c = np.trapezoid(coupled['I'], coupled['t'], axis=0)
    cumI_d = np.trapezoid(decoup['I'], decoup['t'], axis=0)
    tb_cI, CcI, DcI, _, _ = kendall_tau_b(cumI_c, ty)
    tb_dI, CdI, DdI, _, _ = kendall_tau_b(cumI_d, ty)

    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "fig7_rank.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["state", "tycho_norm", "coupled_R", "decoupled_R"])
            for j, st in enumerate(states):
                w.writerow([st, "%.8e" % ty[j], "%.8e" % fR_c[j],
                            "%.8e" % fR_d[j]])

    print("Wrote fig7_rank.csv to:")
    for d in OUT_DIRS:
        print("  %s" % (Path(d) / "fig7_rank.csv"))
    print("coupled final R_frac : min=%.6e max=%.6e mean=%.6e"
          % (fR_c.min(), fR_c.max(), fR_c.mean()))
    print("decoupled final R_frac: min=%.6e max=%.6e mean=%.6e (= rho0)"
          % (fR_d.min(), fR_d.max(), fR_d.mean()))
    pairs = len(states) * (len(states) - 1) // 2
    print("rank statistics, STATE-only target (n=%d, %d pairs):"
          % (len(states), pairs))
    print("  coupled attack  vs Tycho: tau_b=%.10f  C=%d D=%d Tx=%d Ty=%d  "
          "spearman=%.10f  pearson=%.10f"
          % (tb_c, Cc, Dc, Txc, Tyc, spearman_rho(fR_c, ty),
             pearson_r(fR_c, ty)))
    print("  decoupled attack vs Tycho: tau_b=%.10f  C=%d D=%d Tx=%d Ty=%d  "
          "spearman=%.10f  pearson=%.10f"
          % (tb_d, Cd, Dd, Txd, Tyd, spearman_rho(fR_d, ty),
             pearson_r(fR_d, ty)))
    print("  coupled cumI    vs Tycho: tau_b=%.10f  C=%d D=%d  spearman=%.10f  "
          "pearson=%.10f"
          % (tb_cI, CcI, DcI, spearman_rho(cumI_c, ty), pearson_r(cumI_c, ty)))
    print("  decoupled cumI  vs Tycho: tau_b=%.10f  C=%d D=%d  spearman=%.10f  "
          "pearson=%.10f"
          % (tb_dI, CdI, DdI, spearman_rho(cumI_d, ty), pearson_r(cumI_d, ty)))
    print("  residual spread of coupled attack rate: %.6e"
          % (fR_c.max() - fR_c.min()))


if __name__ == "__main__":
    main()
