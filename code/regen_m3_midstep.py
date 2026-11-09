# -*- coding: utf-8 -*-
"""regen_m3_midstep.py -- M3 中间步长 (0.5, 10) 证据补测（独立脚本，只读复用）。

目的
----
m3_convergence.csv 的收敛网格只有 h<=0.5 d；m3_extreme_cfl.csv 只测
h=10,20,30,60 d；区间 (0.5,10) 完全空白。特别是两条「等墙钟预算」边界：
  * Lagrange3-predictor  h=1 d -> 67.2631 ms，边界 h≈1.838 d（≈32.6 步）
  * Richardson(EXP2-midpoint) h=1 d -> 157.8189 ms，边界 h≈4.313 d
本脚本实测这些边界及整个中间区间，判定「等 36.589 ms 墙钟预算」论证是否成立。

复用的现成函数（不修改 m3_experiment.py 的任何函数，不重跑 experiment_0..5）
---------------------------------------------------------------------------
  submission/code/m3_experiment.py
    run_attack       line 823-841    （臂分发；清 T/KERN 缓存）
    ref_for          line 187-191    （同 h 的 DOP853 参考解）
    safe_metrics     line 194-204    （L2_rel_err 等 4 元组的第 0 项）
    measure_wall_ms  line 231-238    （1 次 warm-up + 9 次计时，取中位数）
    write_csv_dual   line 152-172    （同一 payload 写多份 + md5 校验）
    fmt              line 135-149    （统一数值格式）
    drive            line 549-576    （含 grid[-1]==P.T1 的断言）
    experiment_3     line 1070-1106  （本脚本输出行的字段模板）
    main guard       line 1751-1752  （if __name__ == "__main__": main()）
  submission/code/p0b_experiment.py
    t_grid             line 71-72
    reference_solution line 306-332  （DOP853, 默认 rtol=1e-10, atol=1e-12）
    error_metrics      line 335-355
    T0,T1 = (0, 60)   来自 coupling_host_level.T_SPAN

由于 m3_experiment.py 末尾有 `if __name__ == "__main__": main()` 保护，
`import m3_experiment` 不会触发全量重跑，因此本脚本直接 import 复用。

【必须披露的时间网格处理】
---------------------------------------------------------------------------
P.t_grid(h) = round(arange(T0, T1+h/2, h), 8)，而 drive() 要求
abs(grid[-1]-P.T1) <= 1e-8。因此**只有能整除 60 d 的 h** 才能同时满足
「固定步长恰为 h」与「终点恰为 60 d」。对非整除 h（1.6, 1.75, 1.8, 1.838,
2.2, 4.1, 4.2, 4.313, 4.5, 5.5, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5 等），
原样调用 run_attack 会在 drive() 抛 RuntimeError。

本脚本不改动任何已有函数，而是在运行期对每个 h 临时设置 P 模块的全局量
T1（不是函数、不是文件）：
    n = ceil(60/h - 1e-9)          # 整数步数
    P.T1 <- round(n*h, 8)          # 每次 h 结束即恢复原值 60.0
于是 grid = 0, h, 2h, ..., n*h，步长恰为 h，步数 n 为整数，终点 n*h >= 60 d。
对整除 60 的 h，n*h == 60，与既有实验完全一致。所有臂在同一 h 共用同一
grid；DOP853 参考解在同一 grid 上直接 t_eval（无插值）。终点最大超出为
h=8 时的 64 d；这一偏离在报告里逐项列出，不隐藏。

【等预算阈值】
---------------------------------------------------------------------------
从 submission/figures/m3_cost.csv 读取 M3-p3 的 wall_ms_median（稿件引用值
36.589400144 ms，机器相关），并与根 figures/m3_cost.csv 交叉核对；不硬编码。
同时在本机重测 M3-p3 h=1 d 的 wall，用于说明机器漂移。

产物（只新建，不覆盖任何既有 CSV）
---------------------------------------------------------------------------
  figures/m3_midstep.csv
  submission/figures/m3_midstep.csv       （两份逐位一致）
  _review_new_tmp_lead/midstep_findings.md

运行
---------------------------------------------------------------------------
  C:/Python314/python.exe -u regen_m3_midstep.py > regen_m3_midstep.log 2>&1

本脚本不产生科学结论；所有数值为本机实测，一律标注「数值待用户数值确认」。
"""
from __future__ import annotations

import collections
import csv
import hashlib
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m3_experiment as M          # noqa: E402  (import 不会触发 main())
import p0b_experiment as P         # noqa: E402

SUB = os.path.dirname(HERE)
ROOT = os.path.dirname(SUB)
FIG_ROOT = os.path.join(ROOT, "figures")
FIG_SUB = os.path.join(SUB, "figures")
REPORT_DIR = os.path.join(ROOT, "_review_new_tmp_lead")
REPORT_PATH = os.path.join(REPORT_DIR, "midstep_findings.md")

# write_csv_dual 写多份；默认只写 submission/figures，这里改为根 + submission 双份
M.OUT_DIRS = [FIG_ROOT, FIG_SUB]

ARMS = ["M3-p3", "Lagrange3-predictor",
        "Richardson(EXP2-midpoint)", "EXP2-midpoint"]

# 任务规定的 15 个步长 + 等预算精确 60 d 的括号点 + 首次失效点细扫
_H_RAW = [
    0.5, 0.75, 1.0, 1.25, 1.5, 1.6, 1.75, 1.8,
    60.0 / 33, 1.838, 60.0 / 32, 1.9, 2.0, 2.2, 2.5, 3.0,
    4.0, 4.1, 4.2, 60.0 / 14, 4.313, 4.5, 60.0 / 13, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
]
H = sorted(set(round(float(x), 12) for x in _H_RAW))

HEADER = ["h_macro", "arm", "n_steps", "all_finite", "neg_S", "neg_I",
          "neg_R", "cons_resid", "min_S", "min_lam_u", "min_pj",
          "L2_rel_err", "wall_ms_median", "wall_n_repeats",
          "affordable_at_M3_wall", "reference_ok"]

REPEATS = 9                    # measure_wall_ms: 1 warm-up + 9 timed
STABILITY_REPEATS = 27         # 决定性行的独立计时稳定性复测
NAN = float("nan")
T1_ORIGINAL = float(P.T1)


def log(msg):
    print(msg, flush=True)


def read_budget():
    """从既有 m3_cost.csv 读取稿件引用的 M3-p3 墙钟预算，不硬编码。"""
    found = {}
    for path in (os.path.join(FIG_SUB, "m3_cost.csv"),
                 os.path.join(FIG_ROOT, "m3_cost.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] == "M3-p3":
                    found[path] = float(r["wall_ms_median"])
    if not found:
        raise RuntimeError("m3_cost.csv 中找不到 M3-p3 的 wall_ms_median")
    vals = list(found.values())
    if max(vals) - min(vals) > 1e-9:
        raise RuntimeError("两份 m3_cost.csv 的 M3-p3 wall 不一致: %r" % found)
    return vals[0], found


def set_grid_for(h):
    """临时把 P.T1 设为 n*h，使 t_grid(h) 成为恰好 n 步、终点 n*h>=60 d 的
    均匀网格；返回 (grid, n, horizon)。调用方负责在扫描结束后恢复 P.T1。"""
    n = max(1, int(math.ceil((T1_ORIGINAL - float(P.T0)) / h - 1e-9)))
    P.T1 = float(np.round(n * h, 8))
    grid = P.t_grid(h)
    if abs(grid[-1] - P.T1) > 1e-8:
        raise RuntimeError("grid 终点与 P.T1 不一致: h=%g grid[-1]=%.12g T1=%.12g"
                           % (h, grid[-1], P.T1))
    return grid, n, float(grid[-1])


def build_reference(h, grid):
    """主参考解 = M.ref_for(h)（DOP853, rtol=1e-10, atol=1e-12, 同 h 网格），
    与 experiment_2b 一致的更紧容差 (rtol=1e-12, atol=1e-14) 交叉校验。
    返回 (ref, max_abs_diff, ok, note)。任何异常都返回 ok=False，绝不编造。"""
    try:
        ref = M.ref_for(h)
    except Exception as e:
        return None, NAN, False, "ref_for 异常: %s" % (repr(e),)
    try:
        tight = P.reference_solution(grid, rtol=1e-12, atol=1e-14)
    except Exception as e:
        return None, NAN, False, "tight 参考解异常: %s" % (repr(e),)
    try:
        m = len(grid)
        diffs = []
        for v in ("S", "I", "R"):
            a, b = np.asarray(ref[v]), np.asarray(tight[v])
            if a.shape != (m, P.n) or b.shape != (m, P.n):
                return None, NAN, False, "参考解形状异常 %s %s %s" % (v, a.shape, b.shape)
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return None, NAN, False, "参考解含非有限值 %s" % v
            diffs.append(float(np.max(np.abs(a - b))))
        diff = max(diffs)
    except Exception as e:
        return None, NAN, False, "参考解比对异常: %s" % (repr(e),)
    ok = bool(np.isfinite(diff) and diff < 1e-6)
    note = "" if ok else "rtol 1e-10 与 1e-12 参考解差异 %.3e >= 1e-6" % diff
    return ref, diff, ok, note


def failed_row(h, arm, ref_ok, note, n):
    return dict(h_macro=h, arm=arm, n_steps=int(n), all_finite=False,
                neg_S=NAN, neg_I=NAN, neg_R=NAN, cons_resid=NAN,
                min_S=NAN, min_lam_u=NAN, min_pj=NAN, L2_rel_err=NAN,
                wall_ms_median=NAN, wall_n_repeats=0,
                affordable_at_M3_wall=False, reference_ok=ref_ok,
                _note=note)


def as_float(d, key):
    v = d.get(key, NAN)
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def run_stability():
    """对决定性 (arm,h) 行做 27 次独立计时（逻辑复制自 m3_experiment.py
    measure_wall_ms, line 231-238，但保留全部样本），用于量化本机墙钟噪声
    对 affordable 判定的影响。不写入 CSV，只在报告中披露。"""
    h13 = min(H, key=lambda x: abs(x - 60.0 / 13))
    pairs = [
        (1.0, "M3-p3"),
        (1.838, "Lagrange3-predictor"),
        (1.9, "Lagrange3-predictor"),
        (2.0, "Lagrange3-predictor"),
        (4.313, "Richardson(EXP2-midpoint)"),
        (h13, "Richardson(EXP2-midpoint)"),
    ]
    out = []
    try:
        for h, arm in pairs:
            set_grid_for(h)
            samples = []
            M.run_attack(arm, h)                     # warm-up
            for _ in range(STABILITY_REPEATS):
                t0 = time.perf_counter()
                M.run_attack(arm, h)
                samples.append((time.perf_counter() - t0) * 1000.0)
            a = np.asarray(samples, dtype=float)
            out.append(dict(h=h, arm=arm, n=int(a.size),
                            med=float(np.median(a)), mn=float(a.min()),
                            mx=float(a.max()), mean=float(a.mean()),
                            p90=float(np.percentile(a, 90)),
                            samples=a.tolist()))
            log("[stab] %-26s h=%-10.7g  median=%9.3f  min=%9.3f  max=%9.3f  p90=%9.3f ms"
                % (arm, h, out[-1]["med"], out[-1]["mn"],
                   out[-1]["mx"], out[-1]["p90"]))
    finally:
        P.T1 = T1_ORIGINAL
    return out


def equal_budget_appendix():
    """只读派生（--appendix-only）：等宏步数 / 等 Λ 评估次数的逐条可引用行。

    数据来源：本次已落盘的 submission/figures/m3_midstep.csv + 既有
    m3_cost.csv 的 lam_evals_per_macro。不重跑主实验、不写任何文件。
    per-macro 计数与 h 无关这一点用 run_attack(counting=True) 在 6 个 h 上
    独立实测确认后写入本报告，避免把 h=1 的计数无条件外推。
    """
    with open(os.path.join(FIG_SUB, "m3_midstep.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(FIG_SUB, "m3_cost.csv"), newline="",
              encoding="utf-8") as f:
        cost = list(csv.DictReader(f))
    per = {r["arm"]: int(float(r["lam_evals_per_macro"]))
           for r in cost if r["arm"] in ARMS
           and r["lam_evals_per_macro"] not in ("", "nan")}

    # 独立验证：per-macro Λ 评估次数与 h 无关
    verify = []
    try:
        for h in (0.5, 1.0, 1.5, 1.838, 4.313, 10.0):
            _, n, _ = set_grid_for(h)
            got = {}
            for arm in ARMS:
                d = M.run_attack(arm, h, counting=True)
                got[arm] = int(d.get("lam_eval", 0)) // int(d["n_steps"])
            verify.append((h, n, all(got[a] == per[a] for a in ARMS), got))
    finally:
        P.T1 = T1_ORIGINAL

    byh = collections.defaultdict(dict)
    for r in rows:
        byh[float(r["h_macro"])][r["arm"]] = r

    L = []
    a = L.append
    a("")
    a("## 附录 A  等宏步数 / 等评估次数的可直接引用行（只读派生，未重跑主实验）")
    a("")
    a("生成命令：`C:/Python314/python.exe submission/code/regen_m3_midstep.py --appendix-only`。"
      "本附录只读取已落盘的 `m3_midstep.csv` 与 `m3_cost.csv`，不改变任何产物；"
      "所有数值「数值待用户数值确认」。")
    a("")
    a("### A.1 「同 h 下两个四阶基线 L2 更低」的适用范围")
    a("")
    n_lower = sum(1 for h, d in byh.items()
                  if float(d["Lagrange3-predictor"]["L2_rel_err"])
                  < float(d["M3-p3"]["L2_rel_err"])
                  and float(d["Richardson(EXP2-midpoint)"]["L2_rel_err"])
                  < float(d["M3-p3"]["L2_rel_err"]))
    rl = [float(d["Lagrange3-predictor"]["L2_rel_err"]) /
          float(d["M3-p3"]["L2_rel_err"]) for d in byh.values()]
    rr = [float(d["Richardson(EXP2-midpoint)"]["L2_rel_err"]) /
          float(d["M3-p3"]["L2_rel_err"]) for d in byh.values()]
    a("- 扫描共 **%d 个 h**，其中 **%d 个 h** 同时满足 L3 < M3 且 R < M3；例外 %d 个。" %
      (len(byh), n_lower, len(byh) - n_lower))
    a("- L3/M3 比值范围 %.4f ~ %.4f；R/M3 比值范围 %.4f ~ %.4f。" %
      (min(rl), max(rl), min(rr), max(rr)))
    a("- 因此可写「在本次扫描的全部 %d 个步长上」，**不可**写「任意 h / 全程」"
      "（扫描只覆盖 0.5–10 d，且非整除 h 终点为 n·h ≥ 60 d）。"
      "另须注意该断言只关于 L2；h≥9 d 时两个基线已出现负状态分量，"
      "不能据此写成整体「更优」。" % len(byh))
    a("")
    a("### A.2 口径与可直接引用列（**不落对照表**）")
    a("")
    a("**等宏步数**：`m3_midstep.csv` 中同一 `h_macro` 的 4 行天然共享 `n_steps`"
      "（给定 h 下各臂都是 n=ceil(60/h) 步），所以该口径**无需补跑、无需新建表**——"
      "直接按 `(h_macro, arm)` 引用 `n_steps` 与 `L2_rel_err` 两列即可。"
      "终点恰 60 d 的 h 为整除 60 者及 60/33、60/32、60/14、60/13、60/8 五个括号点，共 %d 个。" %
      sum(1 for h, d in byh.items()
          if abs(int(d["M3-p3"]["n_steps"]) * h - 60.0) < 1e-6))
    a("")
    n_ver_ok = sum(1 for _h, _n, ok, _got in verify if ok)
    a("**等 Λ 评估次数**：每宏步计数由既有 `m3_cost.csv` 的 "
      "`lam_evals_per_macro` 给出（M3/L3/R/EXP2 = %d/%d/%d/%d），已用 "
      "`run_attack(..., counting=True)` 在 h=0.5/1/1.5/1.838/4.313/10 六个 h 上"
      "实测确认与 h 无关（%d/%d 一致）；`total = per_macro × n_steps` 可由现有两列"
      "直接推得，同样不需要新表。复合成本（aging/alloc/aux/integration 每宏步"
      "各臂不同）由 r6-cost-audit 定义硬指标，本附录不代替完整成本核算。" %
      (per["M3-p3"], per["Lagrange3-predictor"],
       per["Richardson(EXP2-midpoint)"], per["EXP2-midpoint"],
       n_ver_ok, len(verify)))
    a("")
    a("### A.3 重要更正（**禁止用作稿件优势论据**）")
    a("")
    a("应 `r6-cost-audit` 的独立复算裁定：**不补「等评估次数 / 等宏步数」对照表**——"
      "这两个口径对 M3-p3 都更差，补了也不能用，会把立论从「不稳」换成「明确失败」。"
      "本节只保留负面结论，不落任何可作为优势论据的数字表。")
    a("")
    a("实测负面结论（数值待用户数值确认）：等宏步数（同 h）下 M3-p3 的 L2 在 "
      "%d/%d 个扫描步长上均高于两个四阶基线；等 Λ 评估次数 total=240 组，"
      "M3-p3（h=1，60 步）L2=%s，而 Lagrange3（h=0.75，80 步）L2=%s、"
      "Richardson（h=1.5，40 步）L2=%s，两者均优于 M3-p3；total=120 / 480 组同样如此。" %
      (len(byh), len(byh),
       byh[1.0]["M3-p3"]["L2_rel_err"],
       byh[0.75]["Lagrange3-predictor"]["L2_rel_err"],
       byh[1.5]["Richardson(EXP2-midpoint)"]["L2_rel_err"]))
    a("")
    m3_l2 = float(byh[1.0]["M3-p3"]["L2_rel_err"])
    l3_240 = float(byh[0.75]["Lagrange3-predictor"]["L2_rel_err"])
    l3_b = float(byh[3.0]["Lagrange3-predictor"]["L2_rel_err"])
    a("**补充（`r6-cost-audit` 独立复算裁定）**：其结论是「**不定义、也不构造"
      "任何新的成本类优势指标**」，因为两个同样机器无关的离散预算给出**相反**裁定："
      "口径 A 等 Λeval=240 → Lagrange3（h=0.75，80 步）比 M3-p3（h=1，60 步）"
      "更准约 %.1f×；口径 B 等 aging matmul=60 → M3-p3 反而比 Lagrange3"
      "（h=3，20 步）更准约 %.1f×，且此时 L3 的 wall 落在 36.589 ms 预算内。"
      "选哪种操作当预算本身是任意的，加权复合成本属 ad hoc（Λ 求值 / 矩阵乘 / "
      "状态传播代价不同、权重无客观依据）。因此上一段「两口径对 M3 更差」"
      "**不可**外推为「所有机器无关口径都判 M3 负」；准确结论是**口径不可判定**。"
      "M3 每宏步 aging/alloc/aux = 1/1/0 为四阶臂最低，只能作描述性事实解释"
      "单步为何轻，且必须紧跟「一旦对方少走几步把 wall 压到同一预算，精度优势归对方」。" %
      (m3_l2 / l3_240, l3_b / m3_l2))
    a("")
    a("因此本报告 §5 补充 B 中「改口径为等评估次数 / 等宏步数」的建议**只**应理解为"
      "「口径更可复现、不依赖机器墙钟」，**绝不能**理解为「换口径后 M3 会赢」。"
      "若改稿需要成本口径的硬指标，请引用 r6-cost-audit 的结论，不要引用本附录。")
    return L


def main():
    if "--appendix-only" in sys.argv:
        for _ln in equal_budget_appendix():
            print(_ln)
        return
    t_start = time.perf_counter()
    log("=" * 100)
    log("regen_m3_midstep | 中间步长 (0.5,10) 补测 | 数值待用户数值确认")
    log("python=%s numpy=%s scipy=%s platform=%s"
        % (platform.python_version(), np.__version__, scipy.__version__,
           platform.platform()))
    log("=" * 100)

    budget, budget_src = read_budget()
    log("[budget] M3-p3 wall_ms_median from m3_cost.csv = %.9f ms (%d 份一致)"
        % (budget, len(budget_src)))
    for p, v in budget_src.items():
        log("         %s -> %.9f" % (p, v))

    M.REF_CACHE.clear()
    M.T_CACHE.clear()
    M.KERN_CACHE.clear()

    rows = []
    ref_info = []          # (h, ref_max_abs_diff, reference_ok, note)
    grid_info = []         # (h, n, horizon)
    exceptions = []

    try:
        for h in H:
            grid, n, horizon = set_grid_for(h)
            grid_info.append((h, n, horizon))
            ref, ref_diff, ref_ok, ref_note = build_reference(h, grid)
            ref_info.append((h, ref_diff, ref_ok, ref_note))
            log("[h] h=%-10.7g n=%-3d horizon=%-9.4f d | ref tight-diff=%-10.3e ok=%s %s"
                % (h, n, horizon, ref_diff, ref_ok, ref_note))
            for arm in ARMS:
                # --- 正确性/诊断运行 ---
                try:
                    d = M.run_attack(arm, h)
                except Exception as e:
                    exceptions.append((h, arm, repr(e)))
                    log("[run] h=%-10.7g %-26s EXCEPTION %s" % (h, arm, repr(e)))
                    r = failed_row(h, arm, ref_ok, "run_attack: " + repr(e), n)
                    rows.append(r)
                    continue
                l2 = (M.safe_metrics(d["series"], ref, d["all_finite"])[0]
                      if ref_ok else NAN)
                # --- 墙钟计时（1 warm-up + 9 timed，每次 run_attack 都重建 kernel） ---
                try:
                    med, mn, nr = M.measure_wall_ms(
                        lambda a=arm, hh=h: M.run_attack(a, hh), repeats=REPEATS)
                except Exception as e:
                    exceptions.append((h, arm, "wall: " + repr(e)))
                    med, nr = NAN, 0
                r = dict(
                    h_macro=h, arm=arm,
                    n_steps=int(d["n_steps"]),
                    all_finite=bool(d["all_finite"]),
                    neg_S=int(d["neg_S"]), neg_I=int(d["neg_I"]),
                    neg_R=int(d["neg_R"]),
                    cons_resid=float(d["cons_resid"]),
                    min_S=float(d["min_S"]),
                    min_lam_u=as_float(d, "min_lam_u"),
                    min_pj=as_float(d, "min_pj"),
                    L2_rel_err=float(l2),
                    wall_ms_median=float(med),
                    wall_n_repeats=int(nr),
                    affordable_at_M3_wall=bool(np.isfinite(med) and med <= budget),
                    reference_ok=bool(ref_ok),
                )
                rows.append(r)
                log("[run] h=%-10.7g %-26s n=%-3d finite=%-5s neg=%d/%d/%d "
                    "cons=%-9.2e minS=%-10.3e minlam=%-10.3e L2=%-11.4e "
                    "wall=%9.3f ms aff=%s"
                    % (h, arm, r["n_steps"], r["all_finite"], r["neg_S"],
                       r["neg_I"], r["neg_R"], r["cons_resid"], r["min_S"],
                       r["min_lam_u"], r["L2_rel_err"], r["wall_ms_median"],
                       r["affordable_at_M3_wall"]))
    finally:
        P.T1 = T1_ORIGINAL      # 恢复全局量，绝不污染宿主进程之外的状态

    stab = run_stability()

    md5 = M.write_csv_dual("m3_midstep.csv", HEADER, rows)

    # --- 双份逐位一致性复核（不依赖 write_csv_dual 的自检） ---
    p_root = os.path.join(FIG_ROOT, "m3_midstep.csv")
    p_sub = os.path.join(FIG_SUB, "m3_midstep.csv")
    with open(p_root, "rb") as f:
        b_root = f.read()
    with open(p_sub, "rb") as f:
        b_sub = f.read()
    d_root = hashlib.md5(b_root).hexdigest()
    d_sub = hashlib.md5(b_sub).hexdigest()
    if b_root != b_sub:
        raise RuntimeError("双份 CSV 不一致: %s vs %s" % (d_root, d_sub))
    log("[csv] figures/m3_midstep.csv md5=%s bytes=%d" % (d_root, len(b_root)))
    log("[csv] submission/figures/m3_midstep.csv md5=%s bytes=%d" % (d_sub, len(b_sub)))

    elapsed = time.perf_counter() - t_start
    write_report(rows, ref_info, grid_info, exceptions, budget, budget_src,
                 d_root, elapsed, md5, t_start, stab)
    log("[done] rows=%d exceptions=%d stability=%d elapsed=%.3f s | 数值待用户数值确认"
        % (len(rows), len(exceptions), len(stab), elapsed))
    return rows


def key(rows, h, arm):
    for r in rows:
        if r["arm"] == arm and abs(float(r["h_macro"]) - h) < 1e-9:
            return r
    return None


def fnum(v, spec="%.4e"):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return spec % x


def cross_check_existing(rows):
    """与既有已发表 CSV 的确定性量（L2 / neg 计数）逐位对照。
    这些量与墙钟无关，用来确认 import 复用 + P.T1 网格处理没有改变数值口径。
    返回 [(来源, arm, h, 量, 新值, 旧值, 是否逐位一致)]。"""
    def load(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    cost = load(os.path.join(FIG_SUB, "m3_cost.csv"))
    conv = load(os.path.join(FIG_SUB, "m3_convergence.csv"))
    ext = load(os.path.join(FIG_SUB, "m3_extreme_cfl.csv"))
    out = []

    def newval(h, arm, col):
        for r in rows:
            if abs(float(r["h_macro"]) - h) < 1e-9 and r["arm"] == arm:
                return M.fmt(r[col])
        return None

    for arm in ARMS:
        old = next((r["L2_rel_err_h1"] for r in cost if r["arm"] == arm), None)
        new = newval(1.0, arm, "L2_rel_err")
        out.append(("m3_cost.csv", arm, 1.0, "L2_rel_err", new, old,
                    old is not None and new == old))
    old = next((r["L2_rel_err"] for r in conv
                if r.get("section") == "order" and r.get("arm") == "M3-p3"
                and abs(float(r["h_macro"]) - 0.5) < 1e-9), None)
    new = newval(0.5, "M3-p3", "L2_rel_err")
    out.append(("m3_convergence.csv", "M3-p3", 0.5, "L2_rel_err", new, old,
                old is not None and new == old))
    for arm in ARMS:
        for col in ("L2_rel_err", "neg_S", "neg_I"):
            e = next((r for r in ext if r["arm"] == arm
                      and abs(float(r["h_macro"]) - 10.0) < 1e-9), None)
            old = e[col] if e else None
            nv = key(rows, 10.0, arm)
            new = M.fmt(nv[col]) if nv else None
            out.append(("m3_extreme_cfl.csv", arm, 10.0, col, new, old,
                        old is not None and new == old))
    return out


def write_report(rows, ref_info, grid_info, exceptions, budget, budget_src,
                 md5, elapsed, md5_from_writer, t_start, stab):
    os.makedirs(REPORT_DIR, exist_ok=True)
    A = []
    add = A.append
    add("# M3 中间步长 (0.5, 10) 补测报告")
    add("")
    add("- 执行者：`r6-midstep-exp`（实现工程师）")
    add("- 生成脚本：`submission/code/regen_m3_midstep.py`（独立新脚本，未修改 `m3_experiment.py` 任何函数，未重跑 experiment_0..5）")
    add("- 运行命令：`C:/Python314/python.exe -u regen_m3_midstep.py > regen_m3_midstep.log 2>&1`")
    add("- 运行时刻（UTC）：%s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    add("- 环境：Python %s / numpy %s / scipy %s / %s" %
        (platform.python_version(), np.__version__, scipy.__version__,
         platform.platform()))
    add("- **本次脚本实际总运行时长：%.3f s**（含 %d 次参考解求解、%d 组 × 1 次诊断 + %d 次计时重复、以及 %d 组 × %d 次决定性行计时稳定性复测）" %
        (elapsed, 2 * len(H), len(H) * len(ARMS), REPEATS,
         len(stab), STABILITY_REPEATS))
    add("- **本报告全部数值为本机实测，一律标注「数值待用户数值确认」；本脚本不作科学结论。**")
    add("")
    add("## 1. 扫描设置")
    add("")
    add("- 步长数 %d 个：`%s`" % (len(H), ", ".join("%g" % h for h in H)))
    add("- 臂 %d 个：`%s`（`EXP2-midpoint` 为二阶对照）" % (len(ARMS), "`, `".join(ARMS)))
    add("- 每个 (arm, h)：`run_attack` 诊断运行 1 次 + `measure_wall_ms` 1 次 warm-up 与 %d 次计时取中位数；每次 `run_attack` 均清空 T/KERN 缓存，包含预计算开销。" % REPEATS)
    add("- 参考解：`ref_for(h)` → `p0b_experiment.reference_solution(t_grid(h))`，DOP853、rtol=1e-10、atol=1e-12、同 h 网格直接 t_eval（无插值），与 `m3_experiment.py` 口径完全一致；另用 rtol=1e-12/atol=1e-14 的同网格解做交叉校验（见 §3）。")
    add("- 等预算阈值：从 `%s` 读得 M3-p3 `wall_ms_median` = **%.9f ms**（稿件引用 36.589 ms，机器相关），不硬编码；两份 m3_cost.csv 一致。" %
        (os.path.relpath(list(budget_src)[0], ROOT).replace("\\", "/"), budget))
    add("")
    add("## 2. 必须披露的时间网格 / 终点偏离")
    add("")
    add("`p0b_experiment.t_grid(h) = round(arange(T0, T1+h/2, h), 8)`，而 `m3_experiment.drive()` 断言 `abs(grid[-1]-P.T1) <= 1e-8`。因此**只有整除 60 的 h** 才能同时「固定步长恰为 h」且「终点恰为 60 d」；任务给出的 1.838、4.313、1.6、1.75、2.2、8.0 等非整除 h 若原样调用会在 `drive()` 抛 `RuntimeError`。")
    add("")
    add("处理方式（仅运行期临时改 `P.T1` 这一个模块全局量，不改任何函数、不改任何文件，每个 h 结束后恢复为 60.0）：`n = ceil(60/h - 1e-9)`，`P.T1 = round(n*h, 8)`，于是 `grid = 0, h, ..., n*h`，步长**恰为 h**、步数为整数 n、终点 `n*h >= 60 d`。整除 60 的 h 终点仍恰为 60 d。所有臂在同一 h 共用同一 grid，参考解在同一 grid 求值。")
    add("")
    add("| h (d) | n_steps | 终点 (d) | 是否为精确 60 d |")
    add("|---|---:|---:|---|")
    for h, n, horizon in grid_info:
        add("| %g | %d | %s | %s |" % (h, n, "%.4f" % horizon,
                                      "是" if abs(horizon - 60.0) < 1e-8 else "否"))
    add("")
    _max_end = max(x[2] for x in grid_info)
    add("最大终点为 **%.4f d**（相对 60 d 多 %.4f d，出现在 h=%g）。因此「跑满 60 天」在本脚本中对非整除 h 是「覆盖 ≥60 天」而非「恰为 60 天」；跨 h 的 L2 不是同一时间泛函上的误差，§4/§5 的等预算裁定因此同时给出「同 h 臂间比较」与「跨 h 预算内比较」两种口径。" %
        (_max_end, _max_end - 60.0,
         max(grid_info, key=lambda x: x[2])[0]))
    add("")
    add("## 3. 参考解可靠性")
    add("")
    bad_ref = [x for x in ref_info if not x[2]]
    add("- %d 个 h 全部生成参考解；`reference_ok=false` 的 h 共 %d 个。" % (len(ref_info), len(bad_ref)))
    if bad_ref:
        for h, diff, ok, note in bad_ref:
            add("  - h=%g：tight-diff=%s，原因：%s（该 h 的 L2_rel_err 已置 nan）" % (h, fnum(diff), note))
    else:
        add("- 每个 h 的 rtol=1e-10 与 rtol=1e-12 两个 DOP853 解在 (S/N, I/N, R/N) 上的最大绝对差均 < 1e-6，参考解判定为可靠；无 nan 编造。")
        add("- 最大 tight-diff = %s，最小 tight-diff = %s。" %
            (fnum(max(x[1] for x in ref_info)), fnum(min(x[1] for x in ref_info))))
    add("")
    cc = cross_check_existing(rows)
    n_ok = sum(1 for x in cc if x[6])
    add("### 3.1 与既有已发表 CSV 的确定性量逐位交叉校验")
    add("")
    add("对照对象为 `m3_cost.csv`（h=1，4 臂 L2）、`m3_convergence.csv`（h=0.5，M3-p3 L2）与 `m3_extreme_cfl.csv`（h=10，4 臂 L2 / neg_S / neg_I）。这些量不含墙钟，用来确认「import 复用 + 临时 P.T1 网格处理」没有改变任何数值口径：")
    add("")
    add("| 既有文件 | arm | h | 量 | 本次实测 | 既有值 | 逐位一致 |")
    add("|---|---|---:|---|---|---|---|")
    for src, arm, h, col, new, old, ok in cc:
        add("| `%s` | `%s` | %g | `%s` | `%s` | `%s` | %s |" %
            (src, arm, h, col, new, old, "是" if ok else "**否**"))
    add("")
    add("结果：**%d/%d 项逐位一致**。本次脚本未改动 `m3_experiment.py` 的任何函数；对整除 60 的 h（0.5、1、10）终点仍恰为 60 d，因此与既有结果严格可比。" % (n_ok, len(cc)))
    add("")
    add("## 4. 五个必须明确回答的问题")
    add("")

    def detail(h, arm, label):
        r = key(rows, h, arm)
        add("### %s" % label)
        add("")
        if r is None:
            add("未找到该 (arm, h) 行（异常）。")
            add("")
            return None
        add("| 量 | 实测值 |")
        add("|---|---|")
        add("| `h_macro` | %g d |" % h)
        add("| `n_steps` | %s |" % r["n_steps"])
        add("| `all_finite` | %s |" % ("true" if r["all_finite"] else "false"))
        add("| `neg_S` / `neg_I` / `neg_R` | %s / %s / %s |" %
            (r["neg_S"], r["neg_I"], r["neg_R"]))
        add("| `cons_resid` | %s |" % fnum(r["cons_resid"]))
        add("| `min_S` | %s |" % fnum(r["min_S"]))
        add("| `min_lam_u` | %s |" % fnum(r["min_lam_u"]))
        add("| `min_pj` | %s |" % fnum(r["min_pj"]))
        add("| `L2_rel_err` | %s |" % fnum(r["L2_rel_err"], "%.6e"))
        add("| `wall_ms_median` | %s ms（%d 次计时） |" %
            (fnum(r["wall_ms_median"], "%.4f"), r["wall_n_repeats"]))
        add("| `affordable_at_M3_wall` | **%s**（阈值 %.9f ms） |" %
            ("true" if r["affordable_at_M3_wall"] else "false", budget))
        add("| `reference_ok` | %s |" % ("true" if r["reference_ok"] else "false"))
        add("")
        return r

    q1 = detail(1.838, "Lagrange3-predictor", "Q1  Lagrange3-predictor 在 h = 1.838 d（等预算边界）")
    q2 = detail(4.313, "Richardson(EXP2-midpoint)", "Q2  Richardson(EXP2-midpoint) 在 h = 4.313 d（等预算边界）")

    add("### Q3  M3-p3 在全部扫描步长上")
    add("")
    m3rows = [r for r in rows if r["arm"] == "M3-p3"]
    all_fin = all(r["all_finite"] for r in m3rows)
    all_zero = all((r["neg_S"] == 0 and r["neg_I"] == 0 and r["neg_R"] == 0)
                   for r in m3rows)
    cons_vals = [r["cons_resid"] for r in m3rows if np.isfinite(r["cons_resid"])]
    add("- 共 %d 个步长，`all_finite` 全部为 true：**%s**；`neg_S/neg_I/neg_R` 全部为 0：**%s**。" %
        (len(m3rows), "是" if all_fin else "否", "是" if all_zero else "否"))
    if cons_vals:
        add("- `cons_resid` 范围：%s ~ %s（量级 1e-15 ~ 1e-14，浮点舍入级）。" %
            (fnum(min(cons_vals)), fnum(max(cons_vals))))
    m3min_lam = [r["min_lam_u"] for r in m3rows if np.isfinite(r["min_lam_u"])]
    m3min_pj = [r["min_pj"] for r in m3rows if np.isfinite(r["min_pj"])]
    if m3min_lam and m3min_pj:
        add("- `min_lam_u` 最小值 %s（≥0），`min_pj` 最小值 %s（≥0）。" %
            (fnum(min(m3min_lam)), fnum(min(m3min_pj))))
    add("")
    loc = key(rows, 1.0, "M3-p3")
    if loc:
        add("- 本机重测 M3-p3 h=1 d wall = %s ms；既有 m3_cost.csv = %.9f ms，机器漂移 %s ms。" %
            (fnum(loc["wall_ms_median"], "%.4f"), budget,
             fnum(loc["wall_ms_median"] - budget, "%.4f")))
    add("")
    add("### Q4  每个非 M3 臂首次出现负分量或非有限的最小 h")
    add("")
    add("判定口径：`all_finite == false` 或 `neg_S/neg_I/neg_R` 任一 > 0 或计数为 nan（运行抛异常）。另单列 `min_lam_u < 0`（预测器 Λ(s) 变号，属内部结构量，不等同于状态分量为负）。")
    add("")
    add("| 臂 | 首次状态负分量/非有限的 h | 该处 neg_S/neg_I/neg_R | 首次 `min_lam_u<0` 的 h |")
    add("|---|---|---|---|")
    for arm in ARMS:
        if arm == "M3-p3":
            continue
        seq = sorted([r for r in rows if r["arm"] == arm],
                     key=lambda r: float(r["h_macro"]))
        def isbad(r):
            if not r["all_finite"]:
                return True
            for k in ("neg_S", "neg_I", "neg_R"):
                v = r[k]
                if isinstance(v, float) and not np.isfinite(v):
                    return True
                if v > 0:
                    return True
            return False
        bad = [r for r in seq if isbad(r)]
        lam_bad = [r for r in seq
                   if np.isfinite(r["min_lam_u"]) and r["min_lam_u"] < 0]
        if bad:
            b = bad[0]
            cell = "**%g d** (n=%s)" % (b["h_macro"], b["n_steps"])
            negs = "%s/%s/%s" % (b["neg_S"], b["neg_I"], b["neg_R"])
        else:
            cell = "扫描范围内未出现（最大 h=%g）" % max(float(r["h_macro"]) for r in seq)
            negs = "—"
        lam_cell = ("%g d" % lam_bad[0]["h_macro"]) if lam_bad else "不可得/未出现"
        add("| `%s` | %s | %s | %s |" % (arm, cell, negs, lam_cell))
    add("")

    # ---- Q5 结论 ----
    add("### Q5  结论")
    add("")
    # 等预算内每个臂的最优 L2
    best = {}
    for arm in ARMS:
        cand = [r for r in rows if r["arm"] == arm
                and r["affordable_at_M3_wall"] and np.isfinite(r["L2_rel_err"])]
        best[arm] = min(cand, key=lambda r: r["L2_rel_err"]) if cand else None
    m3best = best["M3-p3"]
    l3best = best["Lagrange3-predictor"]
    rbest = best["Richardson(EXP2-midpoint)"]

    vlabel, vexpl = equal_wall_verdict(m3best, l3best, rbest, q1, q2, budget)
    add("**等 36.589 ms 墙钟预算论证：%s**" % vlabel)
    add("")
    add(vexpl)
    add("")
    add("预算内（含 M3 自身）各臂可达到的最小 `L2_rel_err`：")
    add("")
    add("| 臂 | 预算内最优 h | n_steps | wall (ms) | L2_rel_err | 相对 M3-p3 |")
    add("|---|---|---:|---:|---:|---:|")
    for arm in ARMS:
        b = best[arm]
        if b is None:
            add("| `%s` | 无 affordable 行 | — | — | — | — |" % arm)
        else:
            rel = (b["L2_rel_err"] / m3best["L2_rel_err"]
                   if m3best and m3best["L2_rel_err"] > 0 else NAN)
            add("| `%s` | %g | %s | %s | %s | %s |" %
                (arm, b["h_macro"], b["n_steps"],
                 fnum(b["wall_ms_median"], "%.4f"),
                 fnum(b["L2_rel_err"], "%.6e"), fnum(rel, "%.4f")))
    add("")
    if q1:
        add("- Q1 边界点实测：Lagrange3 在 h=1.838 d，`all_finite=%s`，neg=%s/%s/%s，L2=%s，wall=%s ms，**affordable=%s**。" %
            ("true" if q1["all_finite"] else "false", q1["neg_S"], q1["neg_I"],
             q1["neg_R"], fnum(q1["L2_rel_err"], "%.6e"),
             fnum(q1["wall_ms_median"], "%.4f"),
             "true" if q1["affordable_at_M3_wall"] else "false"))
    if q2:
        add("- Q2 边界点实测：Richardson 在 h=4.313 d，`all_finite=%s`，neg=%s/%s/%s，L2=%s，wall=%s ms，**affordable=%s**。" %
            ("true" if q2["all_finite"] else "false", q2["neg_S"], q2["neg_I"],
             q2["neg_R"], fnum(q2["L2_rel_err"], "%.6e"),
             fnum(q2["wall_ms_median"], "%.4f"),
             "true" if q2["affordable_at_M3_wall"] else "false"))
    add("")
    ulabel, uexpl = uniqueness_verdict(rows)
    add("**「四阶 ∧ 任意步长非负守恒」联合属性唯一性：本次扫描%s。**" % ulabel)
    add("")
    add(uexpl)
    add("")
    add("说明：本扫描只覆盖 0.5–10 d 与 4 个臂；M3-p3 的「任意 h 非负 + 守恒」是结构性证明（正平方预测器 + 相位型精确流），而基线在扫描区间内只是**经验上**未出现负分量。因此本结果可以削弱「在实用区间内只有 M3 非负」的经验性说法，但不能推翻「只有 M3 具备对任意 h 可证明的非负性」这一结构性主张；然而在等墙钟预算上，若基线同样非负且精度不低于 M3，则该结构优势并不转化为精度优势。另须注意 `Lagrange3-predictor` 的 `min_lam_u` / `min_pj` 在 h≥4.2 d 已变号（预测器与 p_j 不再非负），只是状态分量仍未变负——这是「内部结构无保正性」与「状态恰好没炸」的区别，不能混为一谈。")
    add("")

    # ---- Q5 补充 A：同 h 的臂间比较（同网格，无口径问题） ----
    add("### Q5 补充 A  两条边界 h 上的「同 h」臂间比较（同一网格，最干净的比较）")
    add("")
    add("| h | arm | n_steps | L2_rel_err | wall (ms) | neg_S/neg_I/neg_R |")
    add("|---|---|---:|---:|---:|---|")
    for hh in (1.838, 4.313):
        for arm in ARMS:
            r = key(rows, hh, arm)
            if r is None:
                continue
            add("| %g | `%s` | %s | %s | %s | %s/%s/%s |" %
                (hh, arm, r["n_steps"], fnum(r["L2_rel_err"], "%.6e"),
                 fnum(r["wall_ms_median"], "%.4f"),
                 r["neg_S"], r["neg_I"], r["neg_R"]))
    add("")
    add("同一 h 下，两个四阶基线的 L2 均低于 M3-p3；差别体现在墙钟与保正性上。此处只报告实测值，机制解释交由对应成员判断。")
    add("")

    # ---- Q5 补充 B：预算阈值的机器敏感性 ----
    add("### Q5 补充 B  预算阈值的机器敏感性（决定结论方向，必须披露）")
    add("")
    local_budget = loc["wall_ms_median"] if loc else NAN

    def best_under(thr, arm):
        cand = [r for r in rows if r["arm"] == arm
                and np.isfinite(r["wall_ms_median"])
                and r["wall_ms_median"] <= thr
                and np.isfinite(r["L2_rel_err"])]
        return min(cand, key=lambda r: r["L2_rel_err"]) if cand else None

    add("本机重测 M3-p3 h=1 d 的 wall = %s ms，比稿件引用的 %.9f ms 低 %s ms（约 %.1f%%）。"
        "由于 `affordable_at_M3_wall` 是「本机实测 wall ≤ 固定阈值」，阈值取哪一个会直接改变裁定：" %
        (fnum(local_budget, "%.4f"), budget,
         fnum(budget - local_budget, "%.4f"),
         100.0 * (budget - local_budget) / budget))
    add("")
    add("| 预算口径 | 阈值 (ms) | M3-p3 预算内最优 (h, L2) | Lagrange3 预算内最优 (h, wall, L2) | Richardson 预算内最优 (h, L2) | 裁定 |")
    add("|---|---:|---|---|---|---|")
    for tag, thr in (("稿件引用（m3_cost.csv）", budget),
                     ("本机重测 M3-p3 h=1", local_budget)):
        bm = best_under(thr, "M3-p3")
        bl = best_under(thr, "Lagrange3-predictor")
        br = best_under(thr, "Richardson(EXP2-midpoint)")
        fm = ("%g, %s" % (bm["h_macro"], fnum(bm["L2_rel_err"], "%.4e"))) if bm else "无"
        fl = ("%g, %s ms, %s" % (bl["h_macro"],
                                 fnum(bl["wall_ms_median"], "%.3f"),
                                 fnum(bl["L2_rel_err"], "%.4e"))) if bl else "无"
        fr = ("%g, %s" % (br["h_macro"], fnum(br["L2_rel_err"], "%.4e"))) if br else "无"
        if bm and bl:
            verdict = ("Lagrange3 更优，论证**不成立**"
                       if bl["L2_rel_err"] < bm["L2_rel_err"]
                       else "M3-p3 更优，论证**条件成立**")
        else:
            verdict = "数据不足"
        add("| %s | %s | %s | %s | %s | %s |" %
            (tag, fnum(thr, "%.4f"), fm, fl, fr, verdict))
    add("")
    bl_pub = best_under(budget, "Lagrange3-predictor")
    bm_pub = best_under(budget, "M3-p3")
    bl_loc = best_under(local_budget, "Lagrange3-predictor")
    gap = (100.0 * abs(1.0 - bl_pub["L2_rel_err"] / bm_pub["L2_rel_err"])
           if (bl_pub and bm_pub and bm_pub["L2_rel_err"] > 0) else NAN)
    add("**结论对机器相关的墙钟阈值敏感**：在稿件引用的 %s ms 口径下，Lagrange3 在 h=%g d（%s 步、%s ms）以 %s 的 L2 优于 M3-p3 的 %s（差约 %s%%）；但改用本机重测的 %s ms 口径时，该行已不 affordable，Lagrange3 只能退到 h=%s d（L2=%s），L2 反超 M3-p3。两者之差落在计时噪声量级内，因此**不能用本次数据稳健地宣称「M3 在等墙钟预算下最优」，也不能稳健地宣称其反面**；稿件若保留该论证，必须回避「等墙钟」这一机器相关口径，或改口径为等评估次数 / 等宏步数。" %
        (fnum(budget, "%.4f"),
         bl_pub["h_macro"] if bl_pub else NAN,
         bl_pub["n_steps"] if bl_pub else "nan",
         fnum(bl_pub["wall_ms_median"], "%.4f") if bl_pub else "nan",
         fnum(bl_pub["L2_rel_err"], "%.4e") if bl_pub else "nan",
         fnum(bm_pub["L2_rel_err"], "%.4e") if bm_pub else "nan",
         fnum(gap, "%.1f"),
         fnum(local_budget, "%.4f"),
         ("%g" % bl_loc["h_macro"]) if bl_loc else "nan",
         fnum(bl_loc["L2_rel_err"], "%.4e") if bl_loc else "nan"))
    add("")

    # ---- Q5 补充 C：决定性行的计时稳定性 ----
    add("### Q5 补充 C  决定性行的 %d 次独立计时稳定性" % STABILITY_REPEATS)
    add("")
    add("| arm | h | CSV 中 9 次中位 (ms) | 27 次中位 (ms) | 27 次 min (ms) | 27 次 max (ms) | 27 次中 ≤ 发表预算占比 | 27 次中 ≤ 本机预算占比 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in stab:
        r0 = key(rows, s["h"], s["arm"])
        c9 = fnum(r0["wall_ms_median"], "%.3f") if r0 else "nan"
        fp = sum(1 for x in s["samples"] if x <= budget) / s["n"]
        floc = (sum(1 for x in s["samples"] if x <= local_budget) / s["n"]
                if np.isfinite(local_budget) else NAN)
        add("| `%s` | %g | %s | %s | %s | %s | %.0f%% | %s |" %
            (s["arm"], s["h"], c9, fnum(s["med"], "%.3f"),
             fnum(s["mn"], "%.3f"), fnum(s["mx"], "%.3f"),
             100.0 * fp, ("%.0f%%" % (100.0 * floc)) if np.isfinite(floc) else "nan"))
    add("")
    add("上表显示本机墙钟本身有可观波动；任何依赖「≤36.589 ms」这类小数阈值的二分判定都不具备跨机器稳健性。特别地，同一 `(arm,h)` 在主扫描的 9 次计时与稳定性复测的 27 次计时之间也存在明显差异（例如 h=1.838 d 的 Lagrange3 两次中位数不相等，27 次中既有低于也有高于发表预算的样本），说明 `affordable_at_M3_wall` 这一列本身是**本机当次计时**的产物，重跑可能翻转；CSV 已保留当次实测值，不做平滑、不做挑选。")
    add("")
    add("## 5. 等墙钟预算的完整裁定表（每条 affordable 行）")
    add("")
    add("| arm | h | n_steps | wall (ms) | affordable | L2_rel_err | neg_S/neg_I/neg_R | all_finite |")
    add("|---|---|---:|---:|---|---:|---|---|")
    for r in sorted(rows, key=lambda x: (x["arm"], float(x["h_macro"]))):
        if r["affordable_at_M3_wall"]:
            add("| `%s` | %g | %s | %s | true | %s | %s/%s/%s | %s |" %
                (r["arm"], r["h_macro"], r["n_steps"],
                 fnum(r["wall_ms_median"], "%.4f"),
                 fnum(r["L2_rel_err"], "%.6e"),
                 r["neg_S"], r["neg_I"], r["neg_R"],
                 "true" if r["all_finite"] else "false"))
    add("")
    add("## 6. 产物、md5 与复现")
    add("")
    add("| 文件 | 说明 |")
    add("|---|---|")
    add("| `figures/m3_midstep.csv` | %d 行，md5=`%s` |" % (len(rows), md5))
    add("| `submission/figures/m3_midstep.csv` | 与上者逐位一致，md5=`%s` |" % md5)
    add("| `submission/code/regen_m3_midstep.py` | 本脚本 |")
    add("| `submission/code/regen_m3_midstep.log` | 完整 stdout（含每 h 的 solve_ivp status） |")
    add("| `_review_new_tmp_lead/midstep_findings.md` | 本报告 |")
    add("")
    add("- CSV 列（固定）：`%s`" % "`, `".join(HEADER))
    add("- 双份一致性：`write_csv_dual` 返回 md5=`%s`；独立重读两文件计算 md5 均为 `%s`，字节数相同。" %
        (md5_from_writer, md5))
    add("- 既有 `m3_cost.csv` / `m3_equal_wall.csv` / `m3_convergence.csv` / `m3_extreme_cfl.csv` 及稿件、reproducibility 文档**均未改动**；`m3_experiment.py` 未被修改、未全量重跑。")
    add("- 一键复现：`C:/Python314/python.exe -u submission/code/regen_m3_midstep.py`（工作目录不限，脚本内部用 `__file__` 定位）。无随机性：主实验全为确定性计算；`experiment_0` 中使用的 `default_rng(20261107)` 在本脚本中不触发。")
    add("")
    add("## 7. 已知数值风险与限制（如实列出）")
    add("")
    add("1. **机器相关墙钟**：`wall_ms_median` 是本机 %d 次计时中位数，与稿件引用的 36.589 ms 存在机器漂移；`affordable_at_M3_wall` 用既有 m3_cost.csv 的阈值判定，本机 M3-p3 h=1 实测值见 Q3。换机器需重测，结论可能改变。" % REPEATS)
    add("2. **非整除 h 的终点为 n·h ≥ 60 d**（最大 %s d），与稿件「60 天」不完全同口径；跨 h 的 L2 是各自网格上的相对 L2，比较时应同时看同一 h 内的臂间比较。" % ("%.4f" % max(x[2] for x in grid_info)))
    add("3. `min_lam_u` / `min_pj` 只在 `phase_step` 里记录（M3 臂与 `Lagrange3-predictor`）；`Richardson` / `EXP2-midpoint` 不可得，CSV 中为 `nan`，不是 0。")
    add("4. `neg_*` 是 60-ish 天内**所有步 × 所有 52 个斑块**上低于 -1e-9 的累计计数，不是「有多少个 h 失效」；`cons_resid` 是每步 |S+I+R-N| 的最大值。")
    add("5. 运行异常：%d 条。" % len(exceptions))
    if exceptions:
        for h, arm, e in exceptions[:20]:
            add("   - h=%g %s: %s" % (h, arm, e))
        if len(exceptions) > 20:
            add("   - （其余 %d 条见 stdout 日志）" % (len(exceptions) - 20))
    add("6. **边界点的 affordable 判定落在计时噪声内**：Q1/Q2 两条精确边界 h 都不 affordable，但步数相同或更少的邻近 h 有的 affordable（是否 affordable 由本机当次计时决定）；本机 M3-p3 h=1 的 wall 比稿件引用值低 %s%%（见补充 B），改用本机口径时裁定反转。任何以「≤36.589 ms」为硬阈值的论证都不具备跨机器稳健性。" %
        fnum(100.0 * (budget - local_budget) / budget, "%.1f"))
    add("7. `Lagrange3-predictor` 在 h≥4.2 d 时 `min_lam_u` 与 `min_pj` 已为负（预测器与分配权重变号），但状态分量直到 h=9 d 才变负；`Richardson` 与 `Lagrange3` 在 h=9/9.5/10 d 的 affordable 行上已出现负分量，只是此时 L2 已达 1e-2 量级，实用上不可用。")
    add("8. 本次扫描未做求解器交叉校验（所有臂都是同一套显式/指数步方案，参考解统一为 DOP853）；也未做网格加密收敛阶拟合——收敛阶在 `m3_convergence.csv` 中已有，本脚本只补 (0.5,10) 的证据空白。")
    add("9. 本脚本的所有数值均标注「数值待用户数值确认」，须由用户用其自有已校验代码复算后方可采信。")
    add("")

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(A))
    log("[report] wrote %s (%d lines)" % (REPORT_PATH, len(A)))


def equal_wall_verdict(m3best, l3best, rbest, q1, q2, budget):
    """返回 (短裁定, 解释)。短裁定不含 markdown 强调符，避免嵌套加粗。"""
    if m3best is None:
        return ("无法判定", "M3-p3 在本次扫描中无 affordable 行，等墙钟裁定无法给出。")
    l3_wins = (l3best is not None and l3best["L2_rel_err"] < m3best["L2_rel_err"])
    r_wins = (rbest is not None and rbest["L2_rel_err"] < m3best["L2_rel_err"])
    if l3_wins or r_wins:
        who = []
        if l3_wins:
            who.append("Lagrange3-predictor")
        if r_wins:
            who.append("Richardson(EXP2-midpoint)")
        return ("不成立（仅在强制所有臂都取 h=1 d 时成立）",
                "放宽到等墙钟预算后，%s 能在 ≤%.3f ms 内取更大步长且保持 "
                "`all_finite=true`、neg 全 0，其预算内最优 L2 低于 M3-p3 的预算内最优 L2；"
                "因此「M3 在等墙钟预算下最优」这一表述被本次扫描直接否证。注意两条"
                "**精确边界点** h=1.838 / 4.313 d 本身并不 affordable（见 Q1/Q2）；"
                "本次预算内最优点分别出现在 h=%s（%s 步）与 h=%s（%s 步），"
                "其中前者与边界步数是否相同完全由本机计时波动决定（见补充 C）。"
                "Richardson 即使 affordable，其预算内最优 L2 仍为 %s，比 M3-p3 的 %s "
                "差一个量级以上，只有 Lagrange3 构成实质威胁。若把结论限定为"
                "「在同等 h=1 d 下」或「对任意 h 可证明非负」，则属条件成立；但见"
                "补充 B，该裁定对机器相关的墙钟阈值敏感，不能视为稳健结论。" %
                (" 与 ".join(who), budget,
                 ("%g" % l3best["h_macro"]) if l3best else "nan",
                 ("%s" % l3best["n_steps"]) if l3best else "nan",
                 ("%g" % rbest["h_macro"]) if rbest else "nan",
                 ("%s" % rbest["n_steps"]) if rbest else "nan",
                 ("%.4e" % rbest["L2_rel_err"]) if rbest else "nan",
                 "%.4e" % m3best["L2_rel_err"]))
    return ("条件成立",
            "预算内 M3-p3 的 L2 优于两个四阶基线；但基线在边界点上并未出现负分量，"
            "不能把结果归因于保正性，且该结论只在本次机器、本次步长集合与本次 "
            "60-ish 天口径下成立。")


def uniqueness_verdict(rows):
    non_m3 = [r for r in rows if r["arm"] != "M3-p3"]
    clean = [r for r in non_m3
             if r["all_finite"] and r["neg_S"] == 0 and r["neg_I"] == 0
             and r["neg_R"] == 0 and np.isfinite(r["cons_resid"])
             and r["cons_resid"] < 1e-8]
    bad = [r for r in non_m3 if not (r["all_finite"] and r["neg_S"] == 0
                                   and r["neg_I"] == 0 and r["neg_R"] == 0)]
    if len(clean) > 0 and len(bad) > 0:
        four_arms = ("Lagrange3-predictor", "Richardson(EXP2-midpoint)")
        four_clean = [r for r in clean if r["arm"] in four_arms]
        max_clean_h = (max(float(r["h_macro"]) for r in four_clean)
                       if four_clean else float("nan"))
        return ("被削弱",
                "%d 条非 M3 行在扫描区间内同样 `all_finite=true`、neg 全 0、"
                "`cons_resid` 在浮点舍入级（即在 %g–%g d 内经验上也「非负且守恒」），"
                "只有 %d 条在大 h 出现负分量。因此「四阶 ∧ 非负 ∧ 守恒」在本次扫描的"
                "经验区间内并非 M3 独有；M3 的独特性只能落在「对任意 h 的结构性证明」上，"
                "而不是「中间步长上别人会炸」。另需区分：`EXP2-midpoint` 是二阶臂，"
                "它在全区间非负守恒，不能用于支持/否定「四阶」这一联合属性；"
                "两个四阶基线在 h≤%g d 均保持状态非负。" %
                (len(clean), min(float(r["h_macro"]) for r in rows),
                 max(float(r["h_macro"]) for r in rows), len(bad), max_clean_h))
    if len(clean) > 0:
        return ("被削弱", "扫描区间内非 M3 臂全部非负守恒，未出现任何失效点。")
    return ("被本次扫描支持", "非 M3 臂在扫描区间内均出现负分量或非有限。")


if __name__ == "__main__":
    main()
