"""차종 전환(노후 경유 -> 전기/수소) 시 연간 온실가스·에너지비용 비교.
운행거리는 analysis.py 결과(figures/summary.json)를 사용. 계수는 아래 상수(가정·출처는 README)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(BASE, "figures", "summary.json"), encoding="utf-8"))
km_A = S["scenarios"]["A"]["vkm_day"]*365
km_B = [x for x in S["fleet_sensitivity"] if x["K"] == 12][0]["vkm_day"]*365
# 중형 셔틀 기준 가정값
DIESEL_KM_PER_L, DIESEL_CO2_PER_L, DIESEL_WON_PER_L = 5.0, 2.6, 1500
EV_KWH_PER_KM, GRID_CO2_PER_KWH, EV_WON_PER_KWH = 0.8, 0.4567, 300
H2_KM_PER_KG, H2_WON_PER_KG = 13.0, 10000
H2_CO2_PER_KG = {"추출(그레이)": 10.0, "그린": 0.0}
def calc(km):
    return {
        "경유": (km/DIESEL_KM_PER_L*DIESEL_CO2_PER_L/1000, km/DIESEL_KM_PER_L*DIESEL_WON_PER_L/1e6),
        "전기": (km*EV_KWH_PER_KM*GRID_CO2_PER_KWH/1000, km*EV_KWH_PER_KM*EV_WON_PER_KWH/1e6),
        "수소(추출)": (km/H2_KM_PER_KG*H2_CO2_PER_KG["추출(그레이)"]/1000, km/H2_KM_PER_KG*H2_WON_PER_KG/1e6),
        "수소(그린)": (0.0, km/H2_KM_PER_KG*H2_WON_PER_KG/1e6),
    }
out = {"km_year_A": round(km_A), "km_year_B12": round(km_B),
       "A": {k: [round(a, 1), round(b, 1)] for k, (a, b) in calc(km_A).items()},
       "B12": {k: [round(a, 1), round(b, 1)] for k, (a, b) in calc(km_B).items()}}
json.dump(out, open(os.path.join(BASE, "figures", "fuel_transition.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))
r = calc(km_B); labs = list(r); col = ["#8d99ae", "#2a6f97", "#e07a5f", "#3d9970"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (i, t) in enumerate([(0, "연간 온실가스 배출(tCO2)"), (1, "연간 에너지비용(백만 원)")]):
    v = [r[k][i] for k in labs]; ax[j].bar(labs, v, color=col)
    for x, y in enumerate(v): ax[j].text(x, y, f"{y:,.0f}", ha="center", va="bottom")
    ax[j].set_title(t)
fig.suptitle(f"호출형 자율주행 셔틀 12대(연 {km_B/1e4:,.0f}만 km) 차종별 비교")
fig.tight_layout(); fig.savefig(os.path.join(BASE, "figures", "fig5_fuel.png"), dpi=170)
