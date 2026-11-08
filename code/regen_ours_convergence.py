"""Canonical, torch-free generator for the OURS temporal-convergence evidence.

Writes, into every ``figures/`` directory reachable from this file's location:

    ours_order_verify.csv   test,h,error,observed_order,min_val  (master record)
    ours_conv_measles.csv  h error   (two space-separated columns, plotted)
    ours_conv_scalar.csv   h error   (two space-separated columns, plotted)
    ours_conv_coupled.csv  h error   (coupled host-level step, M3, plotted)

Two independent records share the master file, which is why the ``test`` column
exists:

* ``measles`` and ``scalar`` are integrated HERE: one initial condition (source
  paper Eq. (7) + Table 1), one integrator (SSP-RK(3,3) with the per-stage
  non-negative clamp), one reference (scipy Radau restarted at each
  discontinuity).  The scalar test y'=-y is independent of the within-host ODE
  and is included so that both plotted files come from a single reproducible
  run.

ERROR METRIC (measles arm).  The Radau reference is kept as its piecewise
dense-output polynomials and evaluated AT THE DISCRETE SOLUTION'S OWN GRID
points.  The discrete trajectory is never resampled: an earlier version used
``np.interp`` to place it on a uniform 2001-point reference grid, which injected
an O(h^2) interpolation error at coarse h and polluted the observed orders
(2.60 / 2.35 / 4.31 on the three coarsest refinements).  With dense evaluation
the measured order is monotone in h and approaches 3 from below.  Peak scales
are still taken on the same fine grid, so the normalisation is unchanged.
* ``coupled-M3-p3`` and ``coupled-EXP2-midpoint`` are a pure FORMAT projection
  of ``figures/m3_convergence.csv``, the authoritative host-level measurement
  of the coupled macro-step (M3-p3 is the production arm, EXP2-midpoint the
  second-order reference).  Nothing is re-integrated here and no value is
  recomputed: the projection only re-expresses the recorded (h, L2_rel) pairs in
  this file's five-column layout and recomputes the adjacent-ratio order from
  those recorded errors.  If the source file is absent the two coupled blocks
  are skipped with a printed note; they are never synthesised.

Run:  python regen_ours_convergence.py
"""
import os
import math
import csv

import numpy as np
from scipy.integrate import solve_ivp

import ms_numpy as ms

from pathlib import Path

# Location-independent: this file is shipped both at <repo>/submission/code/
# and at <repo>/, with byte-identical content.  Both placements resolve the
# same <SUB>/figures and <repo>/figures, so the dual write is identical.
_HERE = Path(__file__).resolve().parent
if _HERE.name == 'code':
    SUB, ROOT = _HERE.parent, _HERE.parent.parent
else:
    ROOT, SUB = _HERE, _HERE / 'submission'
FIG = SUB / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
OUT_DIRS = [FIG]
ROOT_FIG = ROOT / 'figures'
if ROOT_FIG.is_dir() and ROOT_FIG.resolve() != FIG.resolve():
    OUT_DIRS.append(ROOT_FIG)

Y0 = np.array([3800.0, 0.0, 1.0, 0.0, 1e-4, 0.0, 0.0], dtype=float)
EVENTS = [4.2, 10.0]
T_END = 20.0
# Stop at h=1e-3 d: below this the seven-component error reaches round-off
# (~1e-11) and the measured order is no longer meaningful.
HS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
N_REF = 2001


def radau_reference(y0=Y0, t_end=T_END, n=N_REF):
    """Piecewise DENSE Radau reference, restarted at each event.

    Returns ``(eval_dense, peak)``.  ``eval_dense(ts)`` evaluates the Radau
    interpolation polynomial of the segment containing each t in ``ts``; it is
    defined at arbitrary times and never resamples a discrete solution.
    ``peak[k]`` is the peak |component k| on a uniform fine grid, used only as
    the normalisation scale of the relative L2 norm.
    """
    p = ms.default_params()
    t_fine = np.linspace(0.0, t_end, n)
    segs = [0.0] + [e for e in EVENTS if 0.0 < e < t_end] + [t_end]
    cur = np.asarray(y0, dtype=float).copy()
    sols = []
    for a, b in zip(segs[:-1], segs[1:]):
        # Same left-hand switch freezing as ssp_rk33_pp_step.
        sw = (1.0 if a < p.td else 0.0, 1.0 if a >= p.tAb else 0.0)
        sol = solve_ivp(lambda t, z: ms.measles_rhs(z, t, p, switches=sw),
                         (a, b), cur, method='Radau', rtol=1e-10, atol=1e-13,
                         max_step=0.2, dense_output=True)
        if not sol.success:
            raise RuntimeError('Radau failed on [%g, %g]: %s' % (a, b, sol.message))
        sols.append((a, b, sol))
        cur = sol.sol(b).copy()

    def eval_dense(ts):
        ts = np.asarray(ts, dtype=float)
        out = np.zeros((len(ts), 7), dtype=float)
        filled = np.zeros(len(ts), dtype=bool)
        for a, b, sol in sols:
            idx = np.where((ts >= a - 1e-12) & (ts <= b + 1e-12))[0]
            if len(idx):
                out[idx] = sol.sol(ts[idx]).T
                filled[idx] = True
        if not filled.all():
            raise RuntimeError('dense reference undefined at %d grid points'
                               % int((~filled).sum()))
        return out

    peak = np.max(np.abs(eval_dense(t_fine)), axis=0)
    return eval_dense, peak


def integrate(dt, t_end=T_END, y0=Y0):
    """OURS trajectory on the event-aligned grid."""
    p = ms.default_params()
    ts = np.array(ms.build_aligned_grid(0.0, t_end, dt, EVENTS), dtype=float)
    Y = np.zeros((len(ts), 7), dtype=float)
    Y[0] = y0
    y = np.asarray(y0, dtype=float).copy()
    for k in range(1, len(ts)):
        y = ms.ssp_rk33_pp_step(y, ts[k - 1], ts[k] - ts[k - 1], p)
        Y[k] = y
    return Y, ts, float(Y.min())


def l2_dense(Y, ts, eval_dense, peak):
    """Peak-normalised relative L2 error against the DENSE reference.

    The reference polynomial is evaluated at the numerical solution's own grid
    ``ts``; the discrete trajectory is never interpolated.  The norm is the RMS
    over all seven components of the componentwise peak-normalised L2 error.
    """
    ref = eval_dense(ts)
    err2 = 0.0
    for k in range(7):
        scale = max(float(peak[k]), 1e-12)
        err2 += float(np.mean((Y[:, k] - ref[:, k]) ** 2)) / (scale ** 2)
    return math.sqrt(err2 / 7.0)


def scalar_step(rhs, y, dt):
    """One SSP-RK(3,3) + per-stage clamp step on a scalar/vector problem."""
    L0 = rhs(y)
    y1 = np.maximum(y + dt * L0, 0.0)
    L1 = rhs(y1)
    y2 = np.maximum(0.75 * y + 0.25 * y1 + 0.25 * dt * L1, 0.0)
    L2 = rhs(y2)
    return np.maximum((1.0 / 3.0) * y + (2.0 / 3.0) * y2
                      + (2.0 / 3.0) * dt * L2, 0.0)


def project_coupled(src):
    """Format projection of m3_convergence.csv -> [(arm, h, L2, order, min_S)].

    No re-integration, no smoothing, no re-fitting: the errors are the recorded
    ones and the order is the adjacent-ratio estimate log(e_{k-1}/e_k) /
    log(h_{k-1}/h_k) evaluated on the recorded pairs in decreasing-h order.
    """
    if not src.is_file():
        print('  [skip] %s absent: coupled blocks not written' % src)
        return {}
    rec = {}
    with open(src, newline='', encoding='utf-8') as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) < 13 or row[0] != 'order':
                continue
            arm = row[1]
            if arm not in ('M3-p3', 'EXP2-midpoint'):
                continue
            try:
                h = float(row[2])
                e = float(row[3])
                min_s = float(row[8])
            except ValueError:
                continue
            rec.setdefault(arm, []).append((h, e, min_s))
    out = {}
    for arm, rows in rec.items():
        rows.sort(key=lambda r: -r[0])
        seq, prev = [], None
        for h, e, min_s in rows:
            order = (math.log(prev[1] / e) / math.log(prev[0] / h)
                     if prev and e > 0 else float('nan'))
            seq.append((h, e, order, min_s))
            prev = (h, e)
        out[arm] = seq
    return out


def main():
    if not OUT_DIRS:
        raise RuntimeError('no figures/ directory reachable from %s' % __file__)

    eval_dense, peak = radau_reference()
    rows = []
    measles = []
    prev = None
    for h in HS:
        Y, ts, minv = integrate(h)
        l2 = l2_dense(Y, ts, eval_dense, peak)
        order = (math.log(prev[1] / l2) / math.log(prev[0] / h)
                 if prev else float('nan'))
        rows.append(['measles', '%.6f' % h, '%.6e' % l2,
                     '' if prev is None else '%.3f' % order, '%.3e' % minv])
        measles.append((h, l2))
        prev = (h, l2)

    scalar = []
    prev = None
    # Stop at h=2^-11: beyond this point the error reaches round-off and the
    # measured order is no longer meaningful.
    for k in range(4, 12):
        h = 2.0 ** (-k)
        n = int(round(1.0 / h))
        y = np.array([1.0])
        for _ in range(n):
            y = scalar_step(lambda yy: -1.0 * yy, y, h)
        err = abs(y[0] - math.exp(-1.0))
        order = (math.log(prev[1] / err) / math.log(prev[0] / h)
                 if prev else float('nan'))
        rows.append(['scalar', '%.6e' % h, '%.6e' % err,
                     '' if prev is None else '%.3f' % order, ''])
        scalar.append((h, err))
        prev = (h, err)

    # ---- coupled host-level record: pure projection of the authoritative file
    proj = project_coupled(FIG / 'm3_convergence.csv')
    coupled = []
    arms = {'M3': [], 'EXP2': []}
    for arm in ('M3-p3', 'EXP2-midpoint'):
        key = 'M3' if arm == 'M3-p3' else 'EXP2'
        for k, (h, e, order, min_s) in enumerate(proj.get(arm, [])):
            rows.append(['coupled-' + arm, '%.6f' % h, '%.6e' % e,
                         '' if k == 0 else '%.3f' % order, '%.3e' % min_s])
            arms[key].append((h, e))
            if arm == 'M3-p3':
                coupled.append((h, e))
    # Two-column extract used by panel (c) of fig:ours_convergence. Written here
    # so the plot data is produced by this generator rather than by hand.
    arms_h = [h for h, _ in arms['M3']]
    if len(arms['EXP2']) != len(arms_h):
        raise RuntimeError('coupled arms have different h grids')

    for d in OUT_DIRS:
        with open(os.path.join(d, 'ours_order_verify.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['test', 'h', 'error', 'observed_order', 'min_val'])
            w.writerows(rows)
        for name, data in (('ours_conv_measles.csv', measles),
                           ('ours_conv_scalar.csv', scalar),
                           ('ours_conv_coupled.csv', coupled)):
            with open(os.path.join(d, name), 'w', newline='',
                      encoding='utf-8') as f:
                f.write('h error\n')
                for h, e in data:
                    f.write('%.6e %.6e\n' % (h, e))
        with open(os.path.join(d, 'ours_conv_coupled_arms.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            f.write('h,M3,EXP2\n')
            for (h, e3), (_, e2) in zip(arms['M3'], arms['EXP2']):
                f.write('%.6f,%.6e,%.6e\n' % (h, e3, e2))
        print('wrote 5 convergence files to %s' % d)

    print('measles error at h=1e-2 d: %.6e' % measles[3][1])
    print('scalar last order: %.3f' % (
        math.log(scalar[-2][1] / scalar[-1][1])
        / math.log(scalar[-2][0] / scalar[-1][0])))
    for arm, seq in proj.items():
        print('coupled %s: %d recorded h levels, L2_rel %.6e (h=%.4g) -> '
              '%.6e (h=%.4g)'
              % (arm, len(seq), seq[0][1], seq[0][0], seq[-1][1], seq[-1][0]))


if __name__ == '__main__':
    main()
