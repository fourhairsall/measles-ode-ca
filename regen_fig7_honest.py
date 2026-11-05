"""Regenerate figures/fig7_rank.csv under the CORRECTED nested coupling.

Honest content (replaces the wrong-coupling fig7):
  columns: state, tycho_norm (= initial rho, TX normalised to 1),
           coupled_R  (= final attack fraction R^h/N under calibrated beta_sp),
           decoupled_R (= final R^h/N with beta_sp=0, equals initial rho)
All three columns are already fractions in [0,1]; the Tycho target used for the
rank statistic is tycho_norm.

Also prints the Kendall tau_b of the coupled run.
"""
import os, sys, csv, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ms_numpy as msn
import coupling_host_level as chl

FIG7 = os.path.join(HERE, "figures", "fig7_rank.csv")


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
    return (C - D) / denom if denom > 0 else 0.0


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

    tb_c = kendall_tau_b(fR_c, ty)
    tb_d = kendall_tau_b(fR_d, ty)
    cumI_c = np.trapezoid(coupled['I'], coupled['t'], axis=0)
    cumI_d = np.trapezoid(decoup['I'], decoup['t'], axis=0)
    tb_cI = kendall_tau_b(cumI_c, ty)
    tb_dI = kendall_tau_b(cumI_d, ty)

    with open(FIG7, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "tycho_norm", "coupled_R", "decoupled_R"])
        for j, st in enumerate(states):
            w.writerow([st, "%.8e" % ty[j], "%.8e" % fR_c[j], "%.8e" % fR_d[j]])

    print("Wrote", FIG7)
    print("coupled final R_frac : min=%.6e max=%.6e mean=%.6e"
          % (fR_c.min(), fR_c.max(), fR_c.mean()))
    print("decoupled final R_frac: min=%.6e max=%.6e mean=%.6e (= rho0)"
          % (fR_d.min(), fR_d.max(), fR_d.mean()))
    print("Kendall tau_b (coupled attack  vs Tycho) = %.4f" % tb_c)
    print("Kendall tau_b (decoupled attack vs Tycho) = %.4f" % tb_d)
    print("Kendall tau_b (coupled cumI    vs Tycho) = %.4f" % tb_cI)
    print("Kendall tau_b (decoupled cumI  vs Tycho) = %.4f" % tb_dI)


if __name__ == "__main__":
    main()
