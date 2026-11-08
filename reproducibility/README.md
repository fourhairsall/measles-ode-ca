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
