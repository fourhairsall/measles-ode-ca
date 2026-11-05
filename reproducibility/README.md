# 可复现性包 (Reproducibility Package) — measles-ode-ca

> 本复现包是仓库 `fourhairsall/measles-ode-ca` 的组成部分（数据 + 代码；仓库不含任何文档手稿类文件）。
> 内容：环境锁定、种子表、dtype、实测 wall-clock/内存、图⇄脚本映射表。

---

## 0. 要素清单（自核对）

| 要素 | 状态 | 位置 |
|------|------|------|
| 仓库 URL + commit | ✅ | §1 |
| 环境锁定 | ✅ | `environment.yml` / `requirements.txt` / §2 |
| 种子表（无随机流声明） | ✅ | `seeds_and_dtype.md` / §3 |
| dtype | ✅ | §2.3 / `seeds_and_dtype.md` |
| 实测 wall-clock / 内存 | 🟡 主链已实测，torch 图待目标机复测 | §9 / `measure_runtime.py` |
| 图⇄脚本映射表 | ✅ | `figure_script_map.csv` / §5 |

---

## 1. 仓库与版本 (Repository & commit)

- **公开仓库**：`https://github.com/fourhairsall/measles-ode-ca`。
- 本仓库即代码权威版本；作者本地快照与本仓库内容同源（2026-11-05）。
- 如需长期存档，可另建 Zenodo/Software Heritage 镜像并在此补一行 DOI。

---

## 2. 运行环境 (Environment lock)

### 2.1 版本（实测，开发用解释器 `C:/Python314/python.exe`）

| 组件 | 版本 |
|------|------|
| Python | 3.14 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.6 |
| torch | 本仓库主链不需要 |

### 2.2 环境变量

主链（NumPy/SciPy）**不需要任何环境变量**。

### 2.3 数值精度 dtype

- 主链全程 **float64**（NumPy `np.float64` / SciPy float64），无混合/半精度。
- 所有耦合压力图数据与表值均在该精度下产出（见 `seeds_and_dtype.md`）。

### 2.4 说明

- **主链（fig:traj / fig:stress / fig:crossscale 与两张耦合表，含 percell 数据）**：Python 3.14 + NumPy/SciPy 即可，脚本 `regen_fig3_ours.py` 与 `regen_fig4_and_bare.py`；后者一次运行写出六个 CSV 到 `figures/`。
- 本仓库不含 torch 链脚本与其图数据（作者原始项目中的历史 torch 图不随仓库分发）。

---

## 3. 随机种子表 (Seed table)

详见 `seeds_and_dtype.md`。要点：

- **主链不使用任何随机流。** 耦合映射是确定性的：给定同一初值场、同一行随机 `W`、同一 `beta_sp` 与同一 `h`，两次运行逐位相同。
- 没有 `NSEED`、没有 `SIGMA=0.2` 扰动、没有 `BOOT_SEED` / `BOOT_N` bootstrap；图数据不含误差带或置信区间。
- 所有表值与图值均为**单次确定性运行**的实测输出。
- `beta_sp_cal = R0/Tg = 15/13 = 1.1538 d^-1`；stress 序列 `beta_sp ∈ {15, 30, 60, 120}`；`h = 0.1 d`；`N_list = [8,13,20,26,35,45,52]`，每个 `N` 的重力主子矩阵重新行归一化。
- 宿主内核 `ssp_rk33_pp`（SSP-RK(3,3) + 逐 stage 非负锥投影）为确定性。

---

## 4. 数据依赖 (Data dependencies)

| 文件 | 角色 | 来源 |
|------|------|------|
| `data/fig2_tycho_field.csv` | 初始播种权重场 ρ（52 实体，TX=1） | 随仓库提供 |
| `data/tycho_ca_initial_field_state.csv` | CA 层初值场（52 州 rho 场） | 随仓库提供 |
| `data/closed_loop_results.csv` | 闭环结果小数据 | 随仓库提供 |
| `data/state_geo_population.csv` | 州质心 + 1960 Census 人口 + ρ | 随仓库提供 |
| `W_gravity_52x52_rowstochastic.csv` | 耦合矩阵 W 导出备查 | 由 `STATE_GEO`（`ms_numpy.py` 内硬编码经纬度）+ `recalibrate_gravity_coupling` 实时计算；**非运行时必需输入** |
| `figures/*.csv` | 图/表数据 **（随仓库提供，已生成）** | 各生成脚本（见 §5） |
| `tycho_measles_cases.csv`（≈40 MB，**不随仓库分发**） | `gen_fig8_leading.py` 的输入 | Project Tycho Level 2 公开数据的 measles 抽取 |

**数据策略**：原始 Tycho 数据受使用条款约束且体量大，不随仓库分发；其派生小数据（ρ 场、聚合表）随仓库提供，保证图可离线核对与复现。

**关键路径提示**：全部主链脚本的输入输出路径相对仓库根（`figures/`、`data/`），在本仓库内开箱即跑。两个权威脚本 `regen_fig3_ours.py` 与 `regen_fig4_and_bare.py` 一次运行写出全部六个权威 CSV。

---

## 5. 图 ⇄ 脚本映射表 (Figure ↔ script map)

完整表见 `figure_script_map.csv`（按 `\label` 定位，含数据文件、唯一生成脚本、命令、备注）。摘要：

| 图/表 | 数据文件 | 生成脚本 | 备注 |
|----|----------|----------|------|
| fig:traj | fig3_withinhost_traj.csv | regen_fig3_ours.py | 唯一 IC `y0=(3800,10,1,0,1e-2,0,0)`；同时是 `phi(tau)` 唯一数据源 |
| fig:field | fig2_tycho_field.csv + state_geo.csv | prep_figures.py → gen_figdata.py | 两步链，顺序不可调换 |
| fig:stress | fig4_closedloop_dense.csv | **regen_fig4_and_bare.py** | 单次确定性运行，`x=beta_sp, y=E_clip_clip` |
| fig:crossscale | fig6_band_dense.csv | **regen_fig4_and_bare.py** | `W` 逐 `N` 重行归一化；`x=N, y=E_clip_clip`；无种子/带 |
| fig:realdataval | fig7_rank.csv (+fig7_ablation.csv) | regen_fig7_honest.py | 列 `state,tycho_norm,coupled_R,decoupled_R` |
| fig:realdatats | fig8_leading_annual.csv + fig8_leading_weekly.csv | gen_fig8_leading.py | 1945–1962 观测时序；需外部 40 MB cases 文件 |
| fig:percell | percell_error_N_dense.csv | **regen_fig4_and_bare.py** | L1 从 5.54e-5(N=8) 到 3.26e-6(N=52) |
| tab:sota_coupled | tab_sota_coupled.csv + fig4_symmetric_metrics.csv | **regen_fig4_and_bare.py** | 四对称指标，单位 `10^6 persons` |
| tab:bare | tab_bare.csv | **regen_fig4_and_bare.py** | bare 臂实测：β=60 非有限、550 个负状态 |

**唯一化**：fig:stress / fig:crossscale、grid independence 数据、四对称指标、`tab:sota_coupled`、`tab:bare` 全部由 `regen_fig4_and_bare.py` 一次运行写出，逗号分隔、具名列；不存在第二个写同名 CSV 的脚本。

---

## 6. 复现步骤 (Quick start)

```bash
# 1) 环境
conda env create -f reproducibility/environment.yml
conda activate measles-ode-ca

# 2) 主链（顺序已验证）
python regen_fig3_ours.py          # fig:traj + phi(tau) 唯一来源
python regen_fig4_and_bare.py      # fig:stress / fig:crossscale + 两张耦合表
python regen_fig7_honest.py        # fig:realdataval
# gen_fig8_leading.py 需先取得 tycho_measles_cases.csv（见 §4）

# 3) 逐表核对（图数据已随仓库提供）
head figures/tab_sota_coupled.csv
head figures/fig7_rank.csv
```

---

## 7. 一致性验证状态 (Verification status)

- **唯一 IC / 唯一积分器**：fig:traj 由 `regen_fig3_ours.py` 独占生成，`y0=(3800,10,1,0,1e-2,0,0)`，SSP-RK(3,3) + 逐 stage 非负锥投影，`h=0.1 d`，事件重启 `t=4.2`/`10 d`。
- **唯一耦合数据生成器**：`regen_fig4_and_bare.py` 独占写出 fig:stress / fig:crossscale、grid independence、四对称指标、`tab:sota_coupled`、`tab:bare`；无硬编码 0、无伪带、无复制列。
- **W 重行归一化**：`setup_at_N()` 在每个 `N` 重新行归一化重力主子矩阵；`E_clip(52)/E_clip(8)=0.995346`，log–log 斜率 `-0.0385`（$R^2\approx0.0154$）。
- **数值验证**：主链脚本在测试机无崩溃退出；图数据 CSV 已随仓库提供，可直接核对。

---

## 8. 已知复现缺口（诚实披露 / Honest disclosure）

1. **`gen_fig8_leading.py` 的 40 MB 输入不随仓库分发**（§4 数据策略）；其输出 `fig8_leading_annual/weekly.csv` 已在 `figures/` 内。
2. **wall-clock / 内存**：主链已在本测试机实测（§9），其他环境须复测回填。

---

## 9. 运行时间与内存（实测 / 待补）

测量工具：`reproducibility/measure_runtime.py`（`python reproducibility/measure_runtime.py [脚本名...]`）。

**本测试机实测（仅供参考，非承诺值；机器非目标机）：**

| 脚本 | wall-clock | 说明 |
|------|-----------|------|
| regen_fig3_ours.py | **7.5 s** | fig:traj + `phi(tau)` 唯一来源；含 0–10 d LSODA 参考解 |
| regen_fig4_and_bare.py | **12.1 s** | 六个耦合 CSV；含 `N`-sweep 与 `dt_ref=0.01` 参考解 |
| gen_fig8_leading.py | 1.6 s | 读 40 MB cases |

> 请在目标硬件运行 `measure_runtime.py` 后，将真实值回填本表。

---

## 10. 本包文件清单

```
reproducibility/
├── README.md                  # 本文件
├── figure_script_map.csv      # 图⇄脚本⇄数据 完整映射
├── seeds_and_dtype.md         # 随机种子表 + dtype 说明
├── environment.yml            # conda 环境锁定
├── requirements.txt           # pip 依赖锁定
└── measure_runtime.py         # wall-clock/内存 测量工具
```

---
*生成日期：2026-10-31；2026-11-05 转为公开仓库版（fourhairsall/measles-ode-ca，数据 + 代码）。*
