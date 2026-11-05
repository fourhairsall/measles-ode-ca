# measles-ode-ca

Numerical code and data for a multiscale measles modeling study: a
seven-variable within-host ODE system (target cells, infected cells,
IFN-γ and IL-17 branches, infectious virus, persistent RNA, neutralising
antibody) coupled to a 52-entity host-level map through a pre-computed
infectiousness kernel φ(τ) and clearance age τ*.

The pipeline covers, end to end:

- within-host integration with a third-order SSP-RK(3,3) scheme plus a
  per-stage non-negative clamp, with event-aware restarts at the
  interferon switch (t = 4.2 d) and antibody onset (t = 10 d);
- the pre-computed infectiousness kernel extracted once, off line, from
  the canonical within-host trajectory;
- the coupled host-level map (cohort age bins, row-stochastic gravity
  coupling W with zero diagonal, β_sp = 15/13 d⁻¹, R0 = 15) whose
  bridging increment is rescaled by a mass-conserving min-limiter;
- positivity-stress experiments (β_sp up to 120 d⁻¹), grid-independence
  sweeps (N = 8 … 52), and a feasibility run against Project Tycho
  Level 2 U.S. measles surveillance (records 1906–2001; aggregation
  window 1945–1962, pre-vaccination).

## Layout

```
.
├── ms_numpy.py                  # within-host solver core (NumPy, float64)
├── coupling_host_level.py       # coupled 52-entity host-level system
├── regen_fig3_ours.py           # within-host trajectory + kernel (fig:traj data)
├── regen_fig4_and_bare.py       # coupling-stress figures + tables (fig:stress,
│                                #   fig:crossscale, tab:sota_coupled, tab:bare data)
├── regen_fig7_honest.py         # real-data feasibility ranks (fig:realdataval)
├── gen_fig8_leading.py          # observed 1945–1962 time series (fig:realdatats)
├── prep_figures.py / gen_figdata.py  # two-step chain for fig:field
├── W_gravity_52x52_rowstochastic.csv  # coupling matrix W (rows sum to 1, diag = 0)
├── data/                        # small bundled inputs (see Data below)
├── figures/                     # generated figure/table data (CSV)
├── reproducibility/             # environment lock, figure↔script map,
│                                #   seed/dtype statement, runtime tool
└── LICENSE                      # MIT
```

## Quick start

```bash
conda env create -f reproducibility/environment.yml   # or: pip install -r reproducibility/requirements.txt
python regen_fig3_ours.py        # within-host trajectory + phi(tau)
python regen_fig4_and_bare.py    # coupling-stress figures + both tables
python regen_fig7_honest.py      # real-data feasibility ranks
```

All main-chain scripts read and write `figures/` and `data/` relative to the
repository root and run as-is. `gen_fig8_leading.py` additionally needs `tycho_measles_cases.csv`
(≈40 MB, the measles extraction of Project Tycho Level 2; regenerate it
from the public Project Tycho source — it is not bundled). Its compiled
outputs `fig8_leading_annual.csv` / `fig8_leading_weekly.csv` are included
under `figures/`.

All reported results are deterministic single runs: the coupled map has no
random source, and the pipeline uses float64 throughout. See
`reproducibility/seeds_and_dtype.md` and `reproducibility/figure_script_map.csv`
for the full provenance of every figure and table.

## Data

- `data/tycho_ca_initial_field_state.csv` — 52-entity initial seeding-weight
  field derived from Project Tycho Level 2 state-level measles reports
  (1945–1962 aggregation).
- `data/state_geo_population.csv` — state centroids (lat/lon) with 1960
  Census populations and the seeding weights.
- Project Tycho Level 2 data are publicly available from the University of
  Pittsburgh (van Panhuis et al., 2013) and are subject to their terms of
  use; the raw and extracted case files are therefore not redistributed
  here.

## License

MIT (see `LICENSE`).
