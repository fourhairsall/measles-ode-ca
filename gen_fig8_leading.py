# -*- coding: utf-8 -*-
"""
Generate the leading-state dynamic time-series comparison figure data (C1c)
from REAL Project Tycho Level 2 measles cases (state-level, 1945-1962).
No fabrication: every number is aggregated directly from tycho_measles_cases.csv.

Outputs (for pgfplots, comma-separated, header row):
  figures/fig8_leading_annual.csv : year, TX, NY, CA, PA, OH, IL
      (annual total measles cases of the six highest-burden states)
  figures/fig8_leading_weekly.csv : year_f, cases
      (weekly measles cases of the single leading state, fractional year on x)
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "tycho_measles_cases.csv")
ANN_OUT = os.path.join(HERE, "figures", "fig8_leading_annual.csv")
WK_OUT = os.path.join(HERE, "figures", "fig8_leading_weekly.csv")
YLO, YHI = 1945, 1962

annual = {}   # (state, year) -> sum
weekly = {}   # (state, year, week) -> sum   (only needed for the leading state)
with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        lt = (row.get("loc_type") or "").strip().upper()
        if lt != "STATE":
            continue
        try:
            ew = int(row["epi_week"])
        except (TypeError, ValueError):
            continue
        yr, wk = ew // 100, ew % 100
        if yr < YLO or yr > YHI:
            continue
        try:
            num = float(row.get("number") or 0)
        except (TypeError, ValueError):
            num = 0.0
        st = (row.get("state") or "").strip()
        annual[(st, yr)] = annual.get((st, yr), 0.0) + num
        weekly[(st, yr, wk)] = weekly.get((st, yr, wk), 0.0) + num

# top states by 1945-1962 total burden
tot = {}
for (st, yr), v in annual.items():
    tot[st] = tot.get(st, 0.0) + v
top = sorted(tot, key=tot.get, reverse=True)[:6]
print("[top states] " + ", ".join(f"{s}={int(tot[s]):,}" for s in top))

years = list(range(YLO, YHI + 1))
with open(ANN_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year"] + top)
    for yr in years:
        w.writerow([yr] + [int(round(annual.get((st, yr), 0.0))) for st in top])
print(f"[annual] saved {ANN_OUT}  ({len(years)} years x {len(top)} states)")

lead = top[0]
wk = sorted(((yr, w_), v) for (st, yr, w_), v in weekly.items() if st == lead)
with open(WK_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["year_f", "cases"])
    for (yr, wk_i), v in wk:
        w.writerow([round(yr + (wk_i - 1) / 53.0, 4), int(round(v))])
print(f"[weekly] leading state={lead}, rows={len(wk)}, saved {WK_OUT}")
