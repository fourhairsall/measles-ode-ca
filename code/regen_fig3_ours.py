"""Canonical generator for the within-host trajectory (compiled Fig.2).

One initial condition, one integrator, one event-aligned grid. The CSV produced
here is the sole source of ``figures/fig3_withinhost_traj.csv`` and of the
infectiousness kernel ``phi(tau)`` consumed by ``coupling_host_level.py``.

    state    : (S, I, A_gamma, A17, V, R, Ab)
    IC       : (3800, 0, 1, 0, 1e-4, 0, 0)  [source-paper initial condition]
    kernel   : OURS = SSP-RK(3,3) with a per-stage non-negative clamp
    grid     : h = 0.1 d, exact restarts at t = 4.2 and t = 10 d, horizon 50 d

Run:  python regen_fig3_ours.py
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

import ms_numpy as ms
from pathlib import Path

# 输出目录：<SUB>/figures；若 <repo>/figures 存在则双份同写（字节一致），
# 从任意 cwd 运行都更新随包 CSV，不产生嵌套目录
FIG = Path(__file__).resolve().parents[1] / 'figures'
OUT_FILES = [
    FIG / 'fig3_withinhost_traj.csv',
]
ROOT_FIG = Path(__file__).resolve().parents[2] / 'figures'
if ROOT_FIG.is_dir() and ROOT_FIG.resolve() != FIG.resolve():
    OUT_FILES.append(ROOT_FIG / 'fig3_withinhost_traj.csv')

Y0 = np.array([3800.0, 0.0, 1.0, 0.0, 1e-4, 0.0, 0.0])
EVENTS = [4.2, 10.0]
DT = 0.1
T_END = 50.0
T_REF = 10.0


def trajectory(y0=Y0, dt=DT, t_end=T_END):
    """Integrate the within-host ODE with the OURS kernel on an event grid."""
    p = ms.default_params()
    ts = np.array(ms.build_aligned_grid(0.0, t_end, dt, EVENTS), dtype=float)
    Y = np.zeros((len(ts), 7))
    Y[0] = y0
    y = np.asarray(y0, dtype=float).copy()
    for k in range(1, len(ts)):
        y = ms.ssp_rk33_pp_step(y, ts[k - 1], ts[k] - ts[k - 1], p)
        Y[k] = y
    return ts, Y


def headline(ts, Y):
    iI = int(np.argmax(Y[:, 1]))
    iV = int(np.argmax(Y[:, 4]))
    iS = int(np.argmin(Y[:, 0]))
    return dict(I_peak=Y[iI, 1], t_I=ts[iI],
                V_peak=Y[iV, 4], t_V=ts[iV],
                S_min=Y[iS, 0], t_S=ts[iS],
                S_50=Y[-1, 0], S_01=Y[1, 0],
                min_all=Y.min())


def stiff_reference(y0=Y0, t_ref=T_REF, dt=DT):
    """ode15s-like reference on the dynamical phase, restarted at each event."""
    p = ms.default_params()
    segs = [s for s in [0.0] + sorted(EVENTS) if s < t_ref] + [t_ref]
    ref_t, ref_y = [], []
    cur = np.asarray(y0, dtype=float).copy()
    for a, b in zip(segs[:-1], segs[1:]):
        # Hold both switches at their left-hand value on each segment, exactly
        # as ssp_rk33_pp_step does, so the reference and OURS see the same
        # discontinuous right-hand side.
        sw = (1.0 if a < p.td else 0.0, 1.0 if a >= p.tAb else 0.0)
        teval = np.arange(a, b + 1e-12, dt)
        teval = teval[teval <= b + 1e-12]
        sol = None
        for meth in ('LSODA', 'Radau', 'BDF'):
            s = solve_ivp(lambda t, z: ms.measles_rhs(z, t, p, switches=sw),
                          (a, b), cur, method=meth, rtol=1e-10, atol=1e-12,
                          dense_output=True)
            if s.success:
                sol = s
                break
        if sol is None:
            raise RuntimeError('no stiff solver converged on [%g, %g]' % (a, b))
        ref_t.append(teval)
        ref_y.append(sol.sol(teval).T)
        cur = sol.sol(b).copy()
    return np.concatenate(ref_t), np.vstack(ref_y)


def compare(ts, Y):
    ref_t, ref_y = stiff_reference()
    common = np.intersect1d(np.round(ts, 10), np.round(ref_t, 10))
    A = Y[np.searchsorted(np.round(ts, 10), common)]
    B = ref_y[np.searchsorted(np.round(ref_t, 10), common)]
    rel = np.abs(A - B) / np.maximum(np.abs(B), 1.0)
    no_ab = common != EVENTS[1]
    return dict(n=len(common), max_rel=float(rel.max()),
                median_rel=float(np.median(rel)),
                max_rel_no_ab=float(rel[no_ab].max()))


def write_csv(ts, Y, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        for k in range(len(ts)):
            f.write('%.8e ' % ts[k])
            f.write(' '.join('%.8e' % v for v in Y[k]))
            f.write('\n')


def main():
    ts, Y = trajectory()
    h = headline(ts, Y)
    print('OURS (SSP-RK(3,3) + per-stage clamp), IC = %s'
          % np.array2string(Y0, precision=4))
    print('  rows=%d, t in [%.2f, %.2f], min over all components = %.3e'
          % (len(ts), ts[0], ts[-1], h['min_all']))
    print('  S(0.1)   = %.8f' % h['S_01'])
    print('  I peak   = %.4e at t = %.2f d' % (h['I_peak'], h['t_I']))
    print('  V peak   = %.4f   at t = %.2f d' % (h['V_peak'], h['t_V']))
    print('  S min    = %.4e at t = %.2f d' % (h['S_min'], h['t_S']))
    print('  S(50)    = %.4f' % h['S_50'])

    _, Y_fine = trajectory(dt=0.001)
    print('  S(50) at h=0.001 d = %.4f  (relative change %.4f%%)'
          % (Y_fine[-1, 0], abs(Y_fine[-1, 0] - h['S_50']) / Y_fine[-1, 0] * 100))

    c = compare(ts, Y)
    print('  vs stiff reference (0-%g d, %d nodes): max %.3e, median %.3e, '
          'max excluding t=10 d node %.3e'
          % (T_REF, c['n'], c['max_rel'], c['median_rel'], c['max_rel_no_ab']))

    for path in OUT_FILES:
        write_csv(ts, Y, path)
        print('  wrote %s (%d rows)' % (path, len(ts)))


if __name__ == '__main__':
    main()
