# Reproducibility notes (repository layout)

The pipeline description is `README.md` in the repository root; the
journal-package version of the full verification record ships with the
submission. This directory holds the machine-readable provenance records:

- `environment.yml`, `requirements.txt` -- torch-free environment lock
  (Python 3.14, NumPy 2.4.x, SciPy 1.17.x, pandas 3.0.x).
- `figure_script_map.csv` -- one row per compiled figure/table and per
  generated CSV, with its generator and the measured values quoted in the
  paper.
- `seeds_and_dtype.md` -- float64, no random stream, true CSV column names,
  and the caveats on machine-dependent timing columns and on the intrinsic
  accuracy floor of the DOP853 reference.
- `measure_runtime.py` -- wall-clock / memory helper.

`figure_script_map.csv` and `seeds_and_dtype.md` are path-adapted copies of
the submission-package versions: `submission/figures/` maps to `figures/`,
`submission/code/` to `code/`, and `submission/run_all.py` to `run_all.py`.
No numerical value and no column name differs.

Regenerate everything with `python run_all.py` from the repository root. The
M3 numerical tables come from `python code/m3_experiment.py`; its timing
columns are machine-dependent and are not byte-reproducible.

## Route-B extension (2026-11-10)

Eleven additional rows in `figure_script_map.csv` and the closing section of
`seeds_and_dtype.md` record an extended comparison round: modified-Patankar
baselines, a non-saturating R0 sweep, large-step endpoints, a second illustrative
phase-type instance, and the lambda_0=0 / self-loop experiment. Ten new files in
`figures/` are byte-identical to the journal-package copies:

```
nonsaturating_scan.csv         d4b9fe22989eb564ed0892c167951e4b
kernel_sensitivity.csv         2f87408f069399dcb5183f6c1b9a6bbd
large_step_endpoint.csv        60939e3e2e0aff40c2fe61892f055330
nonsaturating_attack_rate.png  64032e74edc298e4ae4a3432e81abb19
m3_mprk.csv                    88f73fb41d9d52e43ef90b7d7fc7f7b8
m3_rk4_stability.csv           94cbd8929e21a36ae35f6b5e1d3e7212
m3_midstep.csv                 5fa9cc6ee72bcb4d565d7c2636a305af
second_system.csv              de2b4ff996d99d1c2093c4c878b9915a
wavefront_fallback.csv         39559098d27810172d9afc07f5dd1a2a
coupling_diag_seed.csv         b271406311ab02fade31f59fd2f4c83a
```

All five new generators use float64 and no random stream. Only `m3_mprk.csv`
and `m3_midstep.csv` carry machine-dependent timing columns. The generators were
executed in the journal-package layout and are shipped here for provenance; see
"Additional standalone generators" in the repository root README for the output
path caveat. No numerical value and no column name differs between the
submission-package records and the path-adapted records in this directory.
