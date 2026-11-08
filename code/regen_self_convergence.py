# -*- coding: utf-8 -*-
"""Reference-free self-convergence of the coupled M3-p3 macro step.

No reference solution, no interpolation and no fitted parameter enter the
primary measurement.  The production M3-p3 arm is advanced on a nested
sequence of macro-step grids h, h/2, ... and each coarse trajectory is compared
with the EVEN-INDEXED points of the next finer one.  Because
``p0b_experiment.t_grid(h/2)[::2] == t_grid(h)`` holds exactly, the comparison
is pointwise on identical times; resampling a discrete solution (which would
inject an O(h^2) interpolation error) is never performed.

Two relative L2 norms are reported for every (h, h/2) pair:

  d_full      over all times and all S / I / R components (primary evidence);
  d_terminal  restricted to the final time t = T1 (the end-state definition).

Under e_h = C h^4 the difference of two nested solutions satisfies
d_h = e_h - e_{h/2} = (1 - 2^-4) e_h = (15/16) e_h, so e_h = (16/15) d_h.
The column ``e_h_full`` is this extrapolated error; ``L2_ref`` is the shipped
DOP853-based error of the SAME arm at the SAME h (read from
``m3_convergence.csv``), and ``ratio_e_over_ref`` divides one by the other.
Their agreement shows the fourth-order estimate is not an artefact of the
reference solver's own accuracy floor.

``d_terminal`` gives the same order on the coarse levels but reaches
double-precision round-off once the difference approaches 1e-13; its fitted
order is therefore computed only on the levels above that floor.  This
limitation is stated in the CSV and in the manuscript rather than hidden.

Run:  C:/Python314/python.exe submission/code/regen_self_convergence.py
Writes byte-identical ``ours_self_convergence.csv`` to <repo>/figures/ and
<repo>/submission/figures/.  No other file is created or modified.
"""
import csv
import math
from pathlib import Path

import numpy as np

import m3_experiment as M
import p0b_experiment as P

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

HS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
VARS = ('S', 'I', 'R')
FLOOR = 1e-13
COLS = ['row_type', 'h_coarse', 'h_fine', 'd_full', 'order_full',
        'd_terminal', 'order_terminal', 'e_h_full', 'L2_ref',
        'ratio_e_over_ref', 'nested_ok', 'fit_order', 'fit_R2', 'n_points']


def rel_l2(a, b):
    """Euclidean norm of the concatenated S/I/R difference over that of b."""
    num = np.concatenate([(a[v] - b[v]).ravel() for v in VARS])
    den = np.concatenate([b[v].ravel() for v in VARS])
    return float(np.linalg.norm(num) / np.linalg.norm(den))


def fit_order(hs, ds):
    x, y = np.log(np.asarray(hs)), np.log(np.asarray(ds))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(1.0 - ss_res / ss_tot), len(hs)


def published_l2():
    """Shipped DOP853-based errors of the same arm, keyed by macro-step h."""
    out = {}
    src = FIG / 'm3_convergence.csv'
    if not src.is_file():
        return out
    with src.open(newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 4 and row[0] == 'order' and row[1] == 'M3-p3':
                try:
                    out[round(float(row[2]), 9)] = float(row[3])
                except ValueError:
                    pass
    return out


def main():
    runs = {}
    for h in HS:
        d = M.run_attack('M3-p3', h)
        grid = np.asarray(d['grid'], dtype=float)
        if not np.allclose(grid, P.t_grid(h), atol=1e-9):
            raise RuntimeError('unexpected output grid at h=%g' % h)
        runs[h] = (grid, {v: np.asarray(d['series'][v], dtype=float).copy()
                          for v in VARS})

    ref = published_l2()
    rows = []
    for hc, hf in zip(HS[:-1], HS[1:]):
        tc, sc = runs[hc]
        tf, sf = runs[hf]
        even = tf[::2]
        if not (len(even) == len(tc) and np.allclose(even, tc, atol=1e-9)):
            raise RuntimeError('grids are not nested at h=%g / %g' % (hc, hf))
        se = {v: sf[v][::2] for v in VARS}
        d_full = rel_l2(sc, se)
        d_term = rel_l2({v: sc[v][-1:] for v in VARS},
                        {v: se[v][-1:] for v in VARS})
        e_full = d_full * 16.0 / 15.0
        l2_ref = ref.get(round(hc, 9), float('nan'))
        rows.append(dict(h_coarse=hc, h_fine=hf, d_full=d_full,
                         d_terminal=d_term, e_h_full=e_full,
                         L2_ref=l2_ref,
                         ratio=(e_full / l2_ref if l2_ref == l2_ref
                                else float('nan')),
                         nested_ok=True))

    for i, r in enumerate(rows):
        if i == 0:
            r['order_full'] = float('nan')
            r['order_terminal'] = float('nan')
        else:
            denom = math.log(rows[i - 1]['h_coarse'] / r['h_coarse'])
            r['order_full'] = math.log(rows[i - 1]['d_full'] / r['d_full']) / denom
            r['order_terminal'] = (math.log(rows[i - 1]['d_terminal']
                                            / r['d_terminal']) / denom)

    p_full, r2_full, n_full = fit_order([r['h_coarse'] for r in rows],
                                        [r['d_full'] for r in rows])
    usable = [r for r in rows if r['d_terminal'] > FLOOR]
    p_term, r2_term, n_term = fit_order([r['h_coarse'] for r in usable],
                                        [r['d_terminal'] for r in usable])

    out = []
    for r in rows:
        out.append({
            'row_type': 'delta',
            'h_coarse': '%.8f' % r['h_coarse'], 'h_fine': '%.8f' % r['h_fine'],
            'd_full': '%.6e' % r['d_full'],
            'order_full': '' if math.isnan(r['order_full'])
                          else '%.4f' % r['order_full'],
            'd_terminal': '%.6e' % r['d_terminal'],
            'order_terminal': '' if math.isnan(r['order_terminal'])
                              else '%.4f' % r['order_terminal'],
            'e_h_full': '%.6e' % r['e_h_full'],
            'L2_ref': '' if math.isnan(r['L2_ref']) else '%.6e' % r['L2_ref'],
            'ratio_e_over_ref': '' if math.isnan(r['ratio'])
                                else '%.6f' % r['ratio'],
            'nested_ok': 'true', 'fit_order': '', 'fit_R2': '', 'n_points': ''})
    for tag, (p, r2, n) in (('fit_full', (p_full, r2_full, n_full)),
                            ('fit_terminal', (p_term, r2_term, n_term))):
        rec = {c: '' for c in COLS}
        rec.update(row_type=tag, fit_order='%.6f' % p,
                   fit_R2='%.8f' % r2, n_points=str(n))
        out.append(rec)

    for d in OUT_DIRS:
        with (d / 'ours_self_convergence.csv').open('w', newline='',
                                                    encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
            w.writeheader()
            w.writerows(out)
        print('wrote ours_self_convergence.csv to %s' % d)

    print('nested-grid differences (no reference, no interpolation):')
    print('%-8s %-8s %-13s %-8s %-13s %-8s %-13s %-9s'
          % ('h', 'h/2', 'd_full', 'p_full', 'd_terminal', 'p_term',
             'e_h=16d/15', 'L2_ref'))
    for r in rows:
        print('%-8g %-8g %-13.6e %-8s %-13.6e %-8s %-13.6e %-9s'
              % (r['h_coarse'], r['h_fine'], r['d_full'],
                 '' if math.isnan(r['order_full']) else '%.4f' % r['order_full'],
                 r['d_terminal'],
                 '' if math.isnan(r['order_terminal'])
                 else '%.4f' % r['order_terminal'],
                 r['e_h_full'],
                 '' if math.isnan(r['L2_ref']) else '%.4e' % r['L2_ref']))
    print('fit full     : order %.6f  R2 %.8f  n=%d' % (p_full, r2_full, n_full))
    print('fit terminal : order %.6f  R2 %.8f  n=%d (levels above %.0e)'
          % (p_term, r2_term, n_term, FLOOR))
    ratios = [r['ratio'] for r in rows if r['ratio'] == r['ratio']]
    if ratios:
        print('e_h/L2_ref   : %.5f .. %.5f over %d shared h levels'
              % (min(ratios), max(ratios), len(ratios)))


if __name__ == '__main__':
    main()
