"""Generate data files for the two new spatial figures.
- figures/ca_heatmap.csv : 52 (state, Tycho-rank order) x 73 (time, ~5 d sampling)
  matrix of infected fraction rho_i(t), space-separated, no header (for pgfplots matrix plot).
- figures/state_geo.csv  : state,lat,lon,rho  (comma, header) for a geographic scatter.
"""
import csv, numpy as np
from pathlib import Path

# 绝对路径 <SUB>/figures；从任意 cwd 运行都更新随包 CSV，不产生嵌套目录
HERE = Path(__file__).resolve().parents[1] / "figures"
# 52 state abbreviations in the paper's canonical order (config state_names)
STATES = ["TX","NY","CA","WI","NJ","PA","MI","MA","OH","IL","WA","VA","CT","TN","KY",
          "IN","IA","CO","OR","WV","MD","AZ","UT","AL","NC","KS","FL","ME","AR","MO",
          "SC","GA","MN","MT","OK","VT","NM","ND","RI","ID","HI","NE","MS","DC","PR",
          "LA","NH","WY","DE","AK","SD","NV"]
# approximate geographic centers (lat, lon) for the 52 entities
GEO = {
 "TX":(31.0,-99.0),"NY":(42.9,-75.5),"CA":(36.7,-119.7),"WI":(44.6,-89.9),"NJ":(40.2,-74.5),
 "PA":(40.9,-77.7),"MI":(43.3,-84.6),"MA":(42.2,-71.5),"OH":(40.3,-82.8),"IL":(40.0,-89.0),
 "WA":(47.4,-120.0),"VA":(37.5,-78.8),"CT":(41.6,-72.7),"TN":(35.9,-86.4),"KY":(37.5,-85.3),
 "IN":(39.9,-86.3),"IA":(42.0,-93.5),"CO":(38.9,-105.3),"OR":(43.8,-120.5),"WV":(38.6,-80.6),
 "MD":(39.0,-76.7),"AZ":(34.2,-111.7),"UT":(39.3,-111.7),"AL":(32.8,-86.8),"NC":(35.6,-79.4),
 "KS":(38.5,-98.4),"FL":(27.8,-81.7),"ME":(45.4,-69.2),"AR":(34.8,-92.4),"MO":(38.4,-92.4),
 "SC":(33.9,-80.9),"GA":(32.6,-83.4),"MN":(46.0,-94.3),"MT":(46.9,-110.4),"OK":(35.6,-97.5),
 "VT":(44.0,-72.7),"NM":(34.4,-106.1),"ND":(47.5,-100.5),"RI":(41.6,-71.5),"ID":(44.4,-114.6),
 "HI":(20.8,-156.3),"NE":(41.5,-99.8),"MS":(32.7,-89.7),"DC":(38.9,-77.0),"PR":(18.2,-66.5),
 "LA":(31.0,-91.8),"NH":(43.7,-71.6),"WY":(43.0,-107.5),"DE":(39.0,-75.5),"AK":(64.0,-152.0),
 "SD":(44.4,-100.2),"NV":(39.3,-116.6),
}

# ---- rho per state from the existing Tycho field file (rank state rho) ----
rho_by_state = {}
with open(f"{HERE}/fig2_tycho_field.csv") as f:
    for row in csv.reader(f, delimiter=" "):
        if len(row) >= 3 and row[1] in GEO:
            rho_by_state[row[1]] = float(row[2])

# ---- heatmap matrix from fig7_model_rho.csv (already Tycho-rank ordered) ----
mat = []
with open(f"{HERE}/fig7_model_rho.csv") as f:
    r = csv.reader(f)
    header = next(r)
    for line in r:
        mat.append([float(x) for x in line[1:]])  # drop time_days column
mat = np.array(mat)  # [T=146, N=52]
# instantaneous rho clears after the acute phase; visualize the *cumulative attack
# fraction* over time instead (monotone, in [0,1] per state), which shows the
# spatiotemporal attack ordering driven by the initial-field heterogeneity.
step = 2
h = 0.1 * step  # days per kept row
cum = np.cumsum(mat, axis=0) * h          # [T, N] cumulative attack (days)
cum_frac = cum / cum[-1:].clip(min=1e-9)   # fraction of final attack reached by time t
cf_d = cum_frac[::step].T                   # [N=52, 73]
# write long format (x=time index, y=state index, z=value) for a scatter heatmap
import csv as _csv
with open(f"{HERE}/ca_heatmap_long.csv", "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["x", "y", "z"])
    for j in range(cf_d.shape[0]):           # each state row
        for k in range(cf_d.shape[1]):        # each time column
            w.writerow([k, j, f"{cf_d[j, k]:.6f}"])
print("ca_heatmap_long.csv written:", cf_d.shape, "T span days ~",
      round((len(mat)-1)*0.1*step, 1))

# ---- geographic scatter ----
with open(f"{HERE}/state_geo.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["state","lat","lon","rho"])
    for st in STATES:
        if st in rho_by_state:
            lat, lon = GEO[st]
            w.writerow([st, lat, lon, f"{rho_by_state[st]:.6f}"])
        else:
            print("WARNING missing rho for", st)
print("state_geo.csv written:", len(STATES), "states")
