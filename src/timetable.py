"""보은군 농어촌버스운행표(보은군청, 공공데이터포털 15053571 연계) 노선별 일 운행횟수 분석"""
import os, json, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pd.read_csv(os.path.join(BASE, "data", "boeun_rural_bus_timetable_counts.csv")).sort_values("daily_departures_from_boeun")
n = len(t); le4 = int((t.daily_departures_from_boeun <= 4).sum()); le2 = int((t.daily_departures_from_boeun <= 2).sum())
res = dict(routes=n, le4=le4, le4_pct=round(le4/n*100, 1), le2=le2, median=float(t.daily_departures_from_boeun.median()),
           mean=round(float(t.daily_departures_from_boeun.mean()), 1), headway_le4_min=round(13*60/4))
json.dump(res, open(os.path.join(BASE, "figures", "timetable.json"), "w"), ensure_ascii=False, indent=2); print(res)
fig, ax = plt.subplots(figsize=(8, 7))
c = ["#e07a5f" if v <= 4 else "#2a6f97" for v in t.daily_departures_from_boeun]
ax.barh(t.route, t.daily_departures_from_boeun, color=c); ax.axvline(4, ls="--", c="gray")
ax.set_xlabel("보은 출발 일 운행횟수(회)"); ax.tick_params(axis="y", labelsize=7.5)
ax.set_title(f"보은군 농어촌버스 노선별 일 운행횟수 (주황: 4회 이하, {le4}/{n}개 노선)")
fig.tight_layout(); fig.savefig(os.path.join(BASE, "figures", "fig6_timetable.png"), dpi=170)
