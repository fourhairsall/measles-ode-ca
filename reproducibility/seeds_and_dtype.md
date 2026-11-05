# 随机种子表与数值精度 (Seeds & dtype)

## 数值精度 dtype

- 主链全程 **float64**（NumPy `np.float64` / SciPy float64），无混合精度、无半精度。
- 运行环境：Python 3.14 + NumPy 2.4.x + SciPy 1.17.x + pandas 3.0.x；**不需要 GPU，不需要 torch**。
- 宿主内核的 NumPy 端口为 `ms_numpy.py`；唯一权威轨迹生成器 `regen_fig3_ours.py`；唯一权威耦合压力数据生成器 `regen_fig4_and_bare.py`。
- torch 端口仅用于作者原始项目中的历史收敛图，不参与耦合压力数据链，也不随本仓库分发。

## 随机种子：主链不使用随机流

**耦合映射是确定性的。** 给定同一初值场、同一行随机 `W`、同一 `beta_sp` 与同一 `h`，两次运行逐位相同。因此：

- 没有 `NSEED`、没有 `SIGMA=0.2` 的对数正态扰动、没有 `BOOT_SEED` / `BOOT_N` bootstrap；
- 图数据不含误差带、置信区间或多种子阴影；
- 所有表值与图值均为**单次确定性运行**的实测输出；
- 早期多种子探索脚本不进入本仓库，其输出不用于任何权威数值。

## 唯一权威生成器与数据文件

`regen_fig4_and_bare.py` 一次运行写出以下文件到 `figures/`（逗号分隔、具名列、无伪带、无复制列）：

| 文件 | 用途 | 读取列 |
|------|------|--------|
| `fig4_closedloop_dense.csv` | fig:stress | `x=beta_sp`, `y=E_clip_clip` |
| `fig6_band_dense.csv` | fig:crossscale | `x=N`, `y=E_clip_clip` |
| `percell_error_N_dense.csv` | grid independence | `N`, `E_cell_L1`, `E_cell_L2` |
| `fig4_symmetric_metrics.csv` | 四个对称指标 | 守恒残差 / `theta_min` / `frac_theta_lt_1` / `frac_C_gt_1` / 负状态计数 / `max_abs_y` |
| `tab_sota_coupled.csv` | `tab:sota_coupled` 数据底表 | `E_clip_clip`, `E_clip_mpp_defcheck`, `cons_resid_*`, `max_abs_y_*` |
| `tab_bare.csv` | `tab:bare` 数据底表 | `bare_all_finite`, `bare_neg_S_stored` |

`E_clip_mpp_defcheck` **不是赋值的 0**：它在每个 patch、每个 step 对「已交付的 MPP 缩放增量」计算与 post hoc 臂相同的 pre-clip 赤字泛函并累加，是定义一致性检查（数值上恒为零是代数恒等式，实现层面逐步实测）。**零值本身不构成守恒、保正或有界性的证据**；这三项由 `fig4_symmetric_metrics.csv` 的四个对称指标独立支撑。

`setup_at_N()` 在每个分辨率把重力主子矩阵重新行归一化，保证 `W` 行随机、有效 `R0` 不随 `N` 变化。

静态 CFL 判据 `cfl_ok_at_t0`（只看 `t=0` 的 `C_i`）与动态判据 `cfl_ok_60d`（整段 60 天最大 `C_i`）是不同口径；阈值陈述一律使用动态判据。`C_i>1` 的计数与 `theta_i<1` 的激活也不同一：`S_i^h=0` 时 `X_i=0`，$\theta_i=1$（min-limiter 定义的边界分支）。

**列的轨迹归属（provenance）**：`fig4_closedloop_dense.csv` 同一行中，`cfl_ok_at_t0`、`cfl_ok_60d`、`max_C_60d`、`frac_C_gt_1_60d` 取自 **clip 臂**轨迹；`frac_theta_lt_1_60d` 取自 **OURS 臂**轨迹。两臂轨迹不同，不可把同一行读成单条轨迹的统计量。

**limiter 激活计数的口径**：`frac_theta_lt_1` 是严格计数（阈值 `theta < 1-1e-12`），对 ① `W` 变体（RAW `W_all` / 行重归一化）、② limiter 实现与求值结合顺序（`run_arm` 内联 `C=hΛ, X=C·S` vs `host_step` 的 `a=ΛS, dS=−h·a`）、③ 是否加容差均敏感。β=60 下不同组合给出 51/52/53/54/55/56 次，全部是 `1e-16` 量级浮点尾数（`max|W_ren−W_raw|=5.55e-17`）；加「激活时刻 `S_i>1e-12`」容差后，全部组合收敛到 **51 次 / 51 个不同 patch / 每 patch 恰一次**。因此权威数值只报告比例范围（10.0–24.4% 与 0.16–0.17%），不列激活计数；任何复现者若列计数，须同时注明 `W` 变体、limiter 求值路径与容差。守恒残差、`E_clip`、`C_i>1` 事件数在所有组合下不受影响。

## 可复现性核验

```bash
python regen_fig3_ours.py
python regen_fig4_and_bare.py
head figures/tab_sota_coupled.csv
```

全部六个耦合 CSV 由同一脚本在一次运行中写出，具名列可直接核对。

## 补充实测量（2026-11-05，NumPy 链）

以下三组实测由只读复用 `ms_numpy.py` 的独立脚本产出（torch-free；复现配方保留在作者本地数值报告）：

1. **清除相 $h_{\rm FE}$**：字面 $h_{\rm FE}=\min\{y_k/(-f_k)\}$ 在清除相 $(17,50]$ 全部 3301 个采样点为 0——I 自 $t\approx16.7$ d 起被夹制钉在边界且 $f_I<0$；剔除边界分量后 $h_{\rm FE}^+=3.3299\text{--}3.3326\times10^{-1}$ d（限制分量恒为 V，$\approx1/c$）。裸 SSP-RK(3,3)（无夹制，$h=0.1$）第一个 Euler stage 即把 I 推到 $-1.365$（$t\approx17.1$），三天内 $-10^5$、$t\approx19.9$ 起 NaN；夹制在 $(17,50]$ 的 3300/3300 步触发。校验：同一配方在 canonical dt=0.1 轨迹上重建的 $\phi_k$ 与 `build_infectious_kernel()` 逐位相同（max|Δ|=0，$\sum\phi\Delta\tau=13.00089404$）。
2. **$\varepsilon_V$ 敏感性**：$\tau^\ast$ 17.330→19.632 d（$\varepsilon_V$ 1e-2→1e-5），$\int V$ 仅变 0.0014%；$\phi_k$ 尾 bin 相对基准最大变 872%（1e-2）/ 90%（1e-4）/ 99%（1e-5）。权威声明仅覆盖归一化/∫V 稳健，不覆盖尾 bin 形状。
3. **线性 vs 饱和核 $K_V=0.1$**：归一化剖面最大差 2.079 d$^{-1}$（线性峰的 76.29%，在 V 峰）；bin 级 $(sat-lin)/lin\in[-0.763,+154.3]$；$\sum\phi_k\Delta\tau-13$ 线性 $+3.7\times10^{-4}$ vs 饱和 $+2.27\times10^{-2}$（58–71×，所有 $\varepsilon_V$ 一致）。

**dense vs canonical 轨迹差异**：$t=17$ 处最大分量差 $A_\gamma$ 1.257（0.53%）、S 0.0276（0.52%）。主表用 dt=1e-3 dense，canonical 参考行见作者本地数值报告。
