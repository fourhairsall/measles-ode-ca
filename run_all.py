# -*- coding: utf-8 -*-
"""One-shot end-to-end regeneration of every figure/table CSV in this submission.

Usage (works from ANY working directory):

    C:/Python314/python.exe <SUB>/run_all.py

or

    cd <SUB> && C:/Python314/python.exe run_all.py

Every generator resolves its paths from ``Path(__file__).resolve().parents[1]``
(the ``submission/`` directory), so all outputs always land in
``submission/figures/`` and no nested ``code/figures/`` or
``code/submission/figures/`` directories are ever created.

Environment (torch-free, no install / no network needed):
    Python 3.14, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6

Data: ZERO dependencies outside this ``submission/`` directory. The
1945-1962 Tycho STATE rows are shipped as
``submission/data/tycho_measles_1945_1962_minimal.csv``; the 39 MB full
extract at ``../data/tycho_measles_cases.csv`` is used only if present
(optional, produces byte-identical aggregates).
"""
import subprocess
import sys
from pathlib import Path

SUB = Path(__file__).resolve().parent
CODE = SUB / "code"
FIG = SUB / "figures"

# (script, [expected output CSV basenames]) in strict dependency order:
#  1 fig3 traj  -> kernel phi(tau) + compiled Fig.1
#  2 prep       -> fig2_tycho_field.csv (initial rho field)
#  3 fig7_rho   -> fig7_model_rho.csv (needs 1 + 2)
#  4 gen_figdata-> state_geo.csv (needs 2 + 3)
#  5..9 coupled / rank / Tycho / convergence / self-convergence
#  (need 1 + 2 or ship their own data)
#
# Step 5 is the sole writer of the coupling stress data, and it records EVERY
# arm in both coupling-resolution modes (one-step and auto-substepped), so the
# two regimes are never confounded.  Step 8 projects the authoritative
# host-level convergence record (figures/m3_convergence.csv, shipped) into
# ours_order_verify.csv; it re-integrates nothing and invents nothing.  Step 9
# is the reference-free self-convergence check: it re-runs M3-p3 on nested
# grids and compares on coincident times, using no reference solution at all.
STEPS = [
    ("regen_fig3_ours.py", [
        "fig3_withinhost_traj.csv"]),
    ("prep_figures.py", [
        "fig2_tycho_field.csv", "fig4_closedloop.csv"]),
    ("gen_fig7_model_rho.py", [
        "fig7_model_rho.csv"]),
    ("gen_figdata.py", [
        "ca_heatmap_long.csv", "state_geo.csv"]),
    ("regen_fig4_and_bare.py", [
        "fig4_closedloop_dense.csv", "fig4_stress_wide.csv",
        "tab_sota_coupled.csv", "fig4_symmetric_metrics.csv", "tab_bare.csv",
        "fig6_band_dense.csv", "percell_error_N_dense.csv"]),
    ("regen_fig7_honest.py", [
        "fig7_rank.csv"]),
    ("gen_fig8_leading.py", [
        "fig8_leading_annual.csv", "fig8_leading_weekly.csv"]),
    ("regen_ours_convergence.py", [
        "ours_order_verify.csv", "ours_conv_measles.csv",
        "ours_conv_scalar.csv", "ours_conv_coupled.csv",
        "ours_conv_coupled_arms.csv"]),
    ("regen_self_convergence.py", [
        "ours_self_convergence.csv"]),
]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("run_all.py   SUB = %s" % SUB)
    print("python       = %s" % sys.executable)
    print("=" * 78)

    failed = []
    for i, (script, outs) in enumerate(STEPS, 1):
        path = CODE / script
        print("\n[%d/%d] %s" % (i, len(STEPS), script))
        if not path.is_file():
            print("  FAIL  missing script: %s" % path)
            failed.append(script)
            continue
        proc = subprocess.run([sys.executable, str(path)], cwd=str(SUB))
        if proc.returncode != 0:
            print("  FAIL  exit code %d" % proc.returncode)
            failed.append(script)
            continue
        print("  OK")
        for name in outs:
            p = FIG / name
            if p.is_file():
                print("    -> %s  (%d bytes)" % (p, p.stat().st_size))
            else:
                print("    !! EXPECTED OUTPUT MISSING: %s" % p)
                failed.append("%s:%s" % (script, name))

    print("\n" + "=" * 78)
    if failed:
        print("RESULT: FAIL  %d problem(s): %s"
              % (len(failed), ", ".join(failed)))
        return 1
    print("RESULT: PASS  all %d generators succeeded, %d output files verified"
          % (len(STEPS), sum(len(o) for _, o in STEPS)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
