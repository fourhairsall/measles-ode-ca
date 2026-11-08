# -*- coding: utf-8 -*-
"""Generate figure data read by pgfplots (Tycho field / closed-loop stress).

The within-host trajectory (compiled Fig.2, ``fig3_withinhost_traj.csv``) is
generated exclusively by ``regen_fig3_ours.py``; this script never writes it.
"""
import os, csv
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]     # submission/
CODE = Path(__file__).resolve().parent        # submission/code/
FIG = SUB / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = CODE / "data"                          # 随包小数据（初值场 / 旧闭环结果）

# ---------------------------------------------------------------------------
# Tycho 52-state real rho initial field (descending)
# ---------------------------------------------------------------------------
tycho_csv = os.path.join(DATA, "tycho_ca_initial_field_state.csv")
rows = []
with open(tycho_csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append((row["state"], float(row["rho_norm"])))
rows.sort(key=lambda x: -x[1])
with open(os.path.join(FIG, "fig2_tycho_field.csv"), "w", newline="") as f:
    w = csv.writer(f, delimiter=" ", lineterminator="\n")
    for i, (st, rho) in enumerate(rows, 1):
        w.writerow([i, st, f"{rho:.6f}"])
print(f"[fig2] {len(rows)} states, top={rows[0][0]} rho={rows[0][1]:.4f}")

# ---------------------------------------------------------------------------
# Closed-loop robustness (from closed_loop_results.csv)
# ---------------------------------------------------------------------------
cl_csv = os.path.join(DATA, "closed_loop_results.csv")
sel = []
with open(cl_csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        sel.append(row)
want = {
    "row_stochastic,52": "calibrated",
    "true_rate,52": "resolved_N52",
    "true_rate,7": "resolved_N7",
    "true_rate_stress_x15": "stress_x15",
    "true_rate_stress_x30": "stress_x30",
    "true_rate_stress_x60": "stress_x60",
}
out = []
for row in sel:
    key = f"{row['mode']},{row['N']}"
    if key in want or row["mode"].startswith("true_rate_stress"):
        label = want.get(key, row["mode"])
        out.append((
            label,
            float(row["beta_sp"]),
            float(row["E_clip_CLIP_rel"]),
            0.0,
            float(row["macro_bias"]),
            row["cfl_ok"],
        ))
with open(os.path.join(FIG, "fig4_closedloop.csv"), "w", newline="") as f:
    w = csv.writer(f, delimiter=" ", lineterminator="\n")
    for o in out:
        w.writerow(o)
print(f"[fig4] {len(out)} points; CLIP rel at stress_x60 = {out[-1][2]:.3e}")
print("DONE")
