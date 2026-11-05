# -*- coding: utf-8 -*-
"""
measure_runtime.py — 实测每个图生成脚本的 wall-clock 与峰值内存，
用于在可复现性包中回填"实测 wall-clock/内存"。

用法：
    python reproducibility/measure_runtime.py            # 测全部（耗时较长）
    python reproducibility/measure_runtime.py regen_fig3_ours.py verify_ours_order.py

说明：
- 通过 subprocess 调用目标脚本，测量 wall-clock（time.perf_counter）与峰值 RSS
  （resource.getrusage 在 POSIX；Windows 用 psutil 或回退到 tracemalloc 近似）。
- 需要在 Anaconda (torch) 环境下运行，且已设置 KMP_DUPLICATE_LIB_OK=TRUE。
- 输出 reproducibility/runtime_measurements.csv（script, wall_s, peak_mem_MB）。

注意：本机测量值仅作格式示范；请在目标机器上复测后回填 README §10。
"""
import os
import sys
import time
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# 图生成脚本清单（与 figure_script_map.csv 对应）
SCRIPTS = [
    "regen_fig3_ours.py",
    "validate_theory.py",
    "prep_figures.py",
    "gen_figdata.py",
    "run_closed_loop.py",
    "densify_figs.py",
    "regen_wh_conv_sota.py",
    "verify_ours_order.py",
    "densify_crossscale_only.py",
    "per_cell_error_sweep_v2.py",
    "rerun_fig7.py",
    "gen_fig8_leading.py",
]


def peak_mem_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # macOS 返回 bytes，Linux 返回 KB；Windows 走 psutil
        if sys.platform == "darwin":
            return r / 1e6
        return r / 1024.0
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return float("nan")


def run_one(script):
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        return script, float("nan"), float("nan"), "missing"
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, path],
            cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=1800,
        )
        dt = time.perf_counter() - t0
        mem = peak_mem_mb()
        status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
        return script, dt, mem, status
    except subprocess.TimeoutExpired:
        return script, float("nan"), float("nan"), "timeout>1800s"
    except Exception as e:  # noqa
        return script, float("nan"), float("nan"), f"err:{e}"


def main():
    targets = sys.argv[1:] or SCRIPTS
    rows = []
    for s in targets:
        print(f"[measure] {s} ...", flush=True)
        rows.append(run_one(s))
    out = os.path.join(HERE, "runtime_measurements.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["script", "wall_s", "peak_mem_MB", "status"])
        for r in rows:
            w.writerow(r)
            print(f"  {r[0]:32s} wall={r[1]:.1f}s  mem={r[2]:.1f}MB  {r[3]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
