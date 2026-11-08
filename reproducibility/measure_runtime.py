# -*- coding: utf-8 -*-
"""
measure_runtime.py — 实测每个图生成脚本的 wall-clock 与峰值内存，
用于在可复现性包 (P1-33) 中回填"实测 wall-clock/内存"。

用法：
    python reproducibility/measure_runtime.py            # 测全部（耗时较长）
    python reproducibility/measure_runtime.py regen_fig3_ours.py regen_ours_convergence.py

说明：
- 通过 subprocess 调用目标脚本，测量 wall-clock（time.perf_counter）与峰值 RSS
  （resource.getrusage 在 POSIX；Windows 用 psutil 或回退到 tracemalloc 近似）。
- 全链 torch-free：Python 3.14 + NumPy 2.4.6 / SciPy 1.17.1 / pandas 3.0.6；无需 GPU、CUDA 或任何环境变量。
- 输出 reproducibility/runtime_measurements.csv（script, wall_s, peak_mem_MB）。

注意：本机测量值仅作格式示范；请在提交/评审机器上复测后回填 README §9。
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
    "regen_ours_convergence.py",
    "prep_figures.py",
    "gen_figdata.py",
    "regen_fig7_honest.py",
    "gen_fig8_leading.py",
    "regen_fig4_and_bare.py",
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
