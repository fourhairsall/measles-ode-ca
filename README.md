# measles-ode-ca

Numerical code and data for a multiscale measles modeling study. A
seven-variable within-host ODE system is solved once, off line, and its
viral-load trajectory supplies a fixed infectiousness kernel phi(tau) and a
clearance age tau*. These drive a host-level metapopulation of 52 U.S.
reporting entities: an S -> I -> R map with K=40 Erlang infection-age bins,
a row-stochastic gravity mixing matrix W with zero diagonal, and a
state-dependent force of infection Lambda = beta_sp W (I phi(tau)/N).

The population step is advanced by the M3 macro-step integrator. M3 writes
the host-level field as a quadratic vector field, differentiates Lambda along
the exact flow by a Taylor recursion from the current state only, and
represents the predicted Lambda(s) as a non-negative square. The resulting
non-autonomous linear Markov subsystem is advanced in closed form with the
exact Erlang propagator. The one-step map is positive and conservative by
construction for any step size; no clipping, limiter, sub-cycling or
post-hoc correction is used.

This repository contains code and derived data only. It contains no
manuscript, no LaTeX source and no compiled PDF. The `submission/` directory
of the journal package is the authoritative source of every number quoted in
the paper, and this repository is its deterministic mirror (see below).

## Layout

```
.
|-- run_all.py                    # one-shot regeneration of all figure/table CSVs
|-- code/
|   |-- ms_numpy.py               # seven-variable within-host core (NumPy, float64)
|   |-- coupling_host_level.py    # gravity matrix, kernel, host-level step
|   |-- p0b_experiment.py         # shared setup: K, sigma, W, phi, DOP853 reference
|   |-- m3_experiment.py          # M3 accuracy / order / large-step / cost experiments
|   |-- regen_fig3_ours.py        # within-host trajectory + kernel (Fig.1 data)
|   |-- regen_ours_convergence.py # within-host + smooth-scalar convergence (Fig.2 data)
|   |-- regen_self_convergence.py # reference-free self-convergence (Table data)
|   |-- regen_fig4_and_bare.py    # coupling-stress figures and tables
|   |-- regen_fig7_honest.py      # STATE-only real-data ranks
|   |-- gen_fig7_model_rho.py, gen_figdata.py, prep_figures.py, gen_fig8_leading.py
|   |-- regen_m3_midstep.py, regen_m3_mprk.py, regen_nonsaturating.py,
|   |   regen_second_system.py, regen_wavefront.py  # standalone R8 generators
|   |-- data/                     # small bundled inputs
|   |-- W_gravity_52x52_rowstochastic.csv
|   `-- LICENSE
|-- figures/                      # 42 generated figure/table CSVs + 1 PNG
|-- data/                         # bundled Tycho 1945-1962 STATE subset
|-- reproducibility/              # environment lock, figure-script map, seed/dtype note
`-- LICENSE                       # MIT
```

## Quick start

Python 3.14 with NumPy 2.4.x, SciPy 1.17.x and pandas 3.0.x. No GPU, no
PyTorch, no network access and no external data are needed.

```bash
python run_all.py
```

`run_all.py` resolves every path from its own location and writes all outputs
into `figures/`. Verified end to end in a clean copy of this repository:
9 generators succeed, 22 output files are produced, exit code 0. The M3
numerical experiments are a separate step because they rewrite
machine-dependent timing columns:

```bash
python code/m3_experiment.py      # ~50 s; writes 8 m3_*.csv and m3_results.md
```

## Additional standalone generators

Five further generators support the extended comparison set. They are not part
of `run_all.py`: three write machine-dependent timing columns, and all five were
executed in the journal-package layout, whose project root contains a
`submission/` directory. In that layout, from `submission/`:

```bash
python code/regen_m3_midstep.py      # 34 intermediate step sizes
python code/regen_nonsaturating.py   # ~90 s; R0 sweep, kernel sensitivity, endpoints
python code/regen_m3_mprk.py         # ~58 s; modified-Patankar arms, RK4 attribution
python code/regen_second_system.py   # 8-stage / 12-patch model-class instance
python code/regen_wavefront.py       # ~693 s; lambda_0=0 branch and self-loop sweep
```

Each writes one in-memory payload to both `figures/` and `submission/figures/`
and asserts the two md5 values agree. In this repository mirror the scripts are
shipped for provenance only: they resolve their output directories relative to
the parent of `code/`, so running them here would write outside the repository.
The ten generated artifacts are provided in `figures/`, with md5 values recorded
in `reproducibility/figure_script_map.csv`. The main chain `python run_all.py`
remains fully self-contained in a clean copy of this repository.

## Convergence evidence

- `figures/ours_order_verify.csv` is the master record. The measles arm
  compares SSP-RK(3,3) with a piecewise Radau reference whose dense-output
  polynomial is evaluated at the numerical grid points; the numerical
  trajectory is never interpolated. The observed order rises monotonically
  from 2.913 to 2.997.
- `figures/ours_self_convergence.csv` confirms the fourth order of the
  coupled M3 step without any reference solution, by comparing nested grids
  at coincident times: fitted order 3.995723 (R^2 = 0.99999973, n = 7). The
  end-state-only norm is reported alongside and reaches double precision
  round-off below h = 0.0625 d.
- `figures/m3_convergence.csv` holds the eight-step fits of all 15 arms; the
  production arm M3-p3 fits order 3.991842.

## Data

`data/tycho_measles_1945_1962_minimal.csv` is a bundled subset of Project
Tycho Level 2 v1.1.0: all `loc_type=STATE` measles rows with report years
1945-1962, written out unchanged. It is enough to regenerate every real-data
figure and table. The full 448 MB Level 2 file and the 39 MB measles extract
are not redistributed here; Project Tycho data are subject to the terms of
use of the University of Pittsburgh (van Panhuis et al., 2013).

`code/W_gravity_52x52_rowstochastic.csv` is row-stochastic with a zero
diagonal, so the force of infection on a patch excludes that patch's own
infected stock.

## Relation to the submission package

| submission package | this repository |
|---|---|
| `submission/code/` | `code/` |
| `submission/figures/` | `figures/` |
| `submission/data/` | `data/` |
| `submission/reproducibility/` | `reproducibility/` |
| `submission/run_all.py` | `run_all.py` |

The reproducibility documents shipped here are path-adapted to the repository
layout; the versions in the submission package describe the same data and the
same commands. Files under `figures/` and `code/` are byte-identical in both.
Manuscript sources, the compiled PDF, the cover letter and the bibliography
are deliberately absent (see `.gitignore`).

## License

MIT (see `LICENSE`).
