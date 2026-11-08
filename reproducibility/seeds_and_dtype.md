# 随机种子表与数值精度 (Seeds & dtype)

> 本文档描述**当前投稿版本**的数值精度、随机流使用与数据列口径。
> 当前主链为 M3（宿主级 phase-type 宏步 + 感染力 Taylor 正平方预测器）；
> 早期版本中的逐格通量限制器 / $\theta_i$ / $E_{\rm clip}$ 管线已整体移除，
> 其列与脚本不再属于投稿链，本文不再记录。

## 数值精度 dtype

- 投稿主链全程 **float64**（NumPy `np.float64` / SciPy float64），无混合精度、无半精度。
- 运行环境：`C:/Python314/python.exe`，Python 3.14 + NumPy 2.4.x + SciPy 1.17.x + pandas 3.0.x；**不需要 GPU，不需要 torch**。
- 宿主内核的 NumPy 端口为 `ms_numpy.py`；唯一权威宿主内轨迹生成器 `regen_fig3_ours.py`；
  唯一权威耦合压力数据生成器 `regen_fig4_and_bare.py`；M3 数值实验驱动 `m3_experiment.py`。
- 全链（含收敛证据 `fig:ours_convergence` 与 `tab:selfconv`）均为 torch-free；
  `regen_ours_convergence.py` 以 NumPy/SciPy 写出 `ours_order_verify.csv`、
  `ours_conv_measles.csv`、`ours_conv_scalar.csv`、`ours_conv_coupled.csv`、
  `ours_conv_coupled_arms.csv`（5 个）；`regen_self_convergence.py` 写出第 6 个
  `ours_self_convergence.csv`。
- 参考解为 `p0b_experiment.reference_solution`（DOP853，`rtol=1e-10`, `atol=1e-12`），
  在 `t_grid(h)` 的粗网格上返回 `S`, `I`, `R` 三个 `(M+1, 52)` 数组（已按 `N` 归一化）。
  误差度量 `p0b_experiment.error_metrics` 为 `‖d‖₂/‖ref‖₂`，两侧在**同一物理时点**上取值，
  不做插值。宿主内核收敛表另用**分段重启的 scipy Radau**（`rtol=1e-10`, `atol=1e-13`）：
  参考解保留为 `dense_output` 分段多项式，在 SSP-RK(3,3) 自身的离散网格点求值，
  峰值标度取自同一参考解的 2001 点细网格；离散轨迹全程不插值。

## 随机种子：投稿主链不使用随机流

**耦合映射是确定性的。** 给定同一初值场、同一行随机 `W`、同一 `beta_sp` 与同一 `h`，
两次运行逐位相同。因此：

- 没有 `NSEED`、没有 `SIGMA=0.2` 的对数正态扰动、没有 `BOOT_SEED` / `BOOT_N` bootstrap；
- 图表不含误差带、置信区间或多种子阴影；
- 稿件中所有表值与图值均为**单次确定性运行**的实测输出。

早期多种子探索脚本不进入投稿链，其输出不用于稿件任何数值。

## 唯一权威生成器与数据列（按当前 CSV 表头逐项核对）

### `m3_experiment.py` — 8 个 `m3_*.csv`

| 文件 | 表头（真实列名） |
|---|---|
| `m3_baselines.csv` | `h_macro`, `arm`, `L2_rel_err`, `max_abs_err`, `L2_ratio_vs_M3p3`, `M3p3_better_by`, `better_than_E2`, `cons_resid`, `neg_S`, … |
| `m3_convergence.csv` | `section`, `arm`, `h_macro`, `L2_rel_err`, `max_abs_err`, `cons_resid`, `neg_S`, `neg_I`, `min_S`, `all_finite`, `fit_order`, `fit_R2`, `n_h_points` |
| `m3_ref_floor.csv` | `h_macro`, `ref_max_abs_diff`, `M3_L2_vs_rtol1e10`, `M3_L2_vs_rtol1e12`, `rel_change` |
| `m3_extreme_cfl.csv` | `h_macro`, `arm`, `n_steps`, `all_finite`, `cons_resid`, `neg_S`, `neg_I`, `neg_R`, `min_S`, `n_Zzero`, `n_l0_nonpos`, `L2_rel_err` |
| `m3_cost.csv` | `arm`, `n_macro_steps`, `lam_evals_per_macro`, `aging_matmul_per_macro`, `alloc_matmul_per_macro`, `aux_state_prop_per_macro`, `integration_steps_per_macro`, `wall_ms_median`, `wall_ms_min`, `L2_rel_err_h1`, … |
| `m3_equal_wall.csv` | `arm`, `m_substeps`, `wall_ms_median`, `wall_ms_min`, `wall_n_repeats`, `L2_rel_err_h1`, `wall_ratio_vs_M3p3`, `affordable_at_M3_wall` |
| `m3_ablations.csv` | `section`, `variant`, `h_macro`, `q`, `L2_rel_err`, `cons_resid`, `neg_S`, `fit_order`, `fit_R2` |
| `m3_trivial_attacks.csv` | `arm`, `description`, `fit_order`, `fit_R2`, `h1_L2`, `h1_L2_vs_E2`, `h1_L2_vs_M3p3`, `h1_negS`, `h1_negI` |

`m3_convergence.csv` 含 **15 个臂**的 8 点阶数记录与 16 条 `section=fit` 拟合行；
稿件 `tab:conv_ours` 最后一块的拟合阶与 $R^2$ 取自该文件的 `fit` 段，
而表体行取自 `ours_order_verify.csv`。`tab:selfconv` 的全部数值取自
`ours_self_convergence.csv`（列 `row_type,h_coarse,h_fine,d_full,order_full,d_terminal,
order_terminal,e_h_full,L2_ref,ratio_e_over_ref,nested_ok,fit_order,fit_R2,n_points`；
7 条 `delta` 行 + 2 条 `fit` 行），其 `L2_ref` 列只读本文件的 `section=order` 行做交叉验证，
不参与 `d_full` / `d_terminal` 的计算。

### `regen_fig4_and_bare.py` — 7 个耦合 CSV

| 文件 | 表头（真实列名） |
|---|---|
| `fig4_closedloop_dense.csv` | `label`, `beta_sp`, `arm`, `substep_mode`, `substeps_per_macro`, `lam_evals_per_macro`, `L2_rel`, `final_attack_min`, `t_active_days`, `cons_resid_1e6`, `cons_resid_frac`, `min_S_1e6`, `min_S_frac`, `E_preclip_deficit_1e6`, `E_preclip_deficit_frac`, `neg_S`, `neg_I`, `all_finite`, `status`, `max_abs_y_1e6`, `max_C_t0`, `max_C_run` |
| `fig6_band_dense.csv` | `N`, `E_clip_clip_1e6`, `E_clip_per_cell_1e6`, `E_clip_frac`, `cons_resid_clip_1e6`, `cons_resid_clip_frac`, `neg_S_clip`, `all_finite_clip`, `cons_resid_M3_1e6`, `cons_resid_M3_frac`, `min_S_M3_1e6`, `neg_S_M3`, `neg_I_M3`, `all_finite_M3`, `E_cell_L1`, `E_cell_L2`, `rel_L2_vs_h001` |
| `percell_error_N_dense.csv` | `N`, `E_cell_L1`, `E_cell_L2` |
| `fig4_symmetric_metrics.csv` | `beta_sp`, `arm`, `substep_mode`, `E_preclip_deficit_1e6`, `cons_resid_1e6`, `E_preclip_deficit_frac`, `cons_resid_frac`, `max_abs_y_1e6`, `min_S_1e6`, `neg_S`, `neg_I`, `all_finite`, `substeps_per_macro` |
| `tab_sota_coupled.csv` | `beta_sp`, `arm`, `substep_mode`, `substeps_per_macro`, `L2_rel`, `cons_resid_frac`, `cons_resid_1e6`, `min_S_frac`, `neg_S`, `neg_I`, `all_finite`, `status`, `E_preclip_deficit_frac`, `E_preclip_deficit_1e6`, `wall_ms` |
| `tab_bare.csv` | `beta_sp`, `treatment`, `arm`, `substep_mode`, `substeps_per_macro`, `all_finite`, `neg_S`, `cons_resid_frac`, `cons_resid_1e6`, `min_S_frac`, `E_preclip_deficit_frac`, `E_preclip_deficit_1e6` |
| `fig4_stress_wide.csv` | `beta_sp`, `E_clip_clip`, `E_clip_clip_auto`, `cons_resid_clip`, `cons_resid_clip_auto`, `cons_resid_M3`, `substeps_clip_auto`, `substeps_clip_onestep`, `L2_M3`, `L2_clip_onestep`, `L2_clip_auto`, `max_coupling_number_M3`, `t_active_days_M3` |

## 口径与机器相关性说明

- **墙钟列是机器相关的。** `m3_cost.csv` 的 `wall_ms_median`/`wall_ms_min`、
  `m3_equal_wall.csv` 的对应列与 `affordable_at_M3_wall`、以及
  `tab_sota_coupled.csv` 的 `wall_ms` 每次运行都会变。
  `affordable_at_M3_wall = (clip_wall <= M3_wall)` 的判定阈值依赖本机实测值，
  因此等预算结论**不应**被复现为固定区间；稿件改用「所有被测裁剪臂的精度」这一
  与墙钟无关的陈述。除这三组计时列外，其余列在干净拷贝重跑中逐位一致。
- **参考解有内禀精度地板。** `m3_ref_floor.csv` 的 `rel_change` 只衡量「换参考解后
  M3 误差的相对变化」，不能度量参考解自身的误差。实测参考解在
  `rtol=1e-10` 与 `1e-12` 之间的差与 $h$ 无关地饱和在 $L^2\sim10^{-11}$；
  把步长细化到 $h<0.0625$ d 以下会触及该地板，观测阶的下降不是格式失稳。
- **`n_l0_nonpos` 与 `n_Zzero` 是两个不同分支。** `m3_extreme_cfl.csv` 中
  `n_l0_nonpos` 在全部 64 行均为 0（该分支从未触发）；`n_Zzero` 在
  $h=10,20,30,60$ d 分别为 6, 3, 2, 1（该分支确实触发）。
  重力矩阵 $W$ 的对角为零（见 `W_gravity_52x52_rowstochastic.csv`），
  故 $\Lambda_i$ 不含 patch $i$ 自身感染者的贡献。
- **`tab:m3large` 的 $L^2_{\rm rel}$ 列不是精度度量。** 大步下仅在少数时点采样，
  $h=60$ d 的小值是采样伪影；稿件只主张有限性、守恒与非负性。

## 可复现性核验

```bash
python code/m3_experiment.py            # 8 个 m3_*.csv；不在 run_all.py 内
python code/regen_ours_convergence.py   # 5 个 ours_*.csv
python code/regen_self_convergence.py   # ours_self_convergence.csv（无参考解自收敛）
python code/regen_fig4_and_bare.py      # 7 个耦合 CSV
python code/regen_fig3_ours.py          # fig3_withinhost_traj.csv
python run_all.py                       # 其余图表数据与一致性检查
md5sum figures/*.csv figures/*.csv
```

`figures/` 与 `figures/` 由各生成器在一次运行中双写，不需要事后 `cp` 同步；
实测 31 组同名文件逐位一致。`figures/` 与 `figures/` 各自另有少量
仅单侧存在的文件（不参与稿件绘图），详见 `README.md`。
