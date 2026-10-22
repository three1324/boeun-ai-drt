"""
보은군 버스정류장 공공데이터 기반 AI 수요응답형 자율주행 셔틀 배차 시뮬레이션
- 입력: data/boeun_stop_clusters.csv  (국토교통부_전국 버스정류장 위치정보_20251031 에서
        보은군(정류장번호 BEB275*) 정류장을 약 1km 격자로 묶은 마을 단위 정류장 클러스터)
- 수요: 정류장 수를 마을 규모 대리변수로 사용한 합성 수요 (가정, README 참조)
- 비교: (A) 고정노선·고정시각  (B) 호출형 DRT  (C) 패턴예측(AI) 기반 DRT
"""
import math, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import GradientBoostingRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "figures"); os.makedirs(OUT, exist_ok=True)
for f in []:
    if "NotoSansCJK-Regular" in f or "NanumGothic" in f or "malgun" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(2026)

# ---------------- 1. 데이터 ----------------
st = pd.read_csv(os.path.join(BASE, "data", "boeun_stop_clusters.csv"))
HUB = (36.48320, 127.72190)  # 보은시외버스터미널(데이터 내 좌표)
def km(a, b):
    dy = (a[0]-b[0])*111.0; dx = (a[1]-b[1])*111.0*math.cos(math.radians(36.45))
    return math.hypot(dx, dy)*1.35  # 도로 우회계수 1.35
st["d_hub"] = [km((r.lat, r.lon), HUB) for r in st.itertuples()]
rural = st[st.d_hub > 2.5].reset_index(drop=True)  # 읍내 2.5km 밖 마을 = 분석대상
rural["ang"] = [math.atan2(r.lat-HUB[0], r.lon-HUB[1]) for r in rural.itertuples()]
NSEC = 8
rural["sector"] = (((rural.ang + math.pi) / (2*math.pi)) * NSEC).astype(int) % NSEC
summary = {
    "clusters_total": int(len(st)), "stops_total": int(st.n_stops.sum()),
    "rural_clusters": int(len(rural)), "rural_stops": int(rural.n_stops.sum()),
    "d_hub_mean": round(float(rural.d_hub.mean()), 1), "d_hub_max": round(float(rural.d_hub.max()), 1),
    "share_over10km": round(float((rural.d_hub > 10).mean()*100), 1),
}

# ---------------- 2. 합성 수요 (가정) ----------------
HOURS = np.arange(6, 21)
prof_in = {6:.3,7:.8,8:1.6,9:1.8,10:1.4,11:.8,12:.4,13:.3,14:.3,15:.2,16:.2,17:.1,18:.1,19:.05,20:.05}
prof_out= {6:.02,7:.05,8:.1,9:.2,10:.4,11:.8,12:1.3,13:1.5,14:1.4,15:1.1,16:.9,17:.7,18:.4,19:.2,20:.1}
DAILY_PER_STOP = 0.25   # 정류장 1개소당 일 통행(왕복의 편도 기준) 가정
def gen_day(day):
    market = (day % 5) in (0, 5) or ((day+1) % 10) in (1, 6)  # 보은장(1·6일) 근사
    rain = rng.random() < 0.2
    mult = (1.6 if market else 1.0) * (0.75 if rain else 1.0) * (0.6 if day % 7 == 6 else 1.0)
    reqs = []
    tot_in, tot_out = sum(prof_in.values()), sum(prof_out.values())
    for i, r in rural.iterrows():
        lam = r.n_stops * DAILY_PER_STOP * mult
        for h in HOURS:
            for kind, prof, tot in (("in", prof_in, tot_in), ("out", prof_out, tot_out)):
                n = rng.poisson(lam * prof[h] / tot)
                for _ in range(n):
                    reqs.append((h*60 + rng.uniform(0, 60), i, kind))
    reqs.sort()
    return reqs, market, rain, day % 7

N_TRAIN, N_TEST = 56, 28
days = [gen_day(d) for d in range(N_TRAIN + N_TEST)]

# ---------------- 3. 수요예측 모델 (패턴 분석) ----------------
rows = []
for d, (reqs, market, rain, dow) in enumerate(days):
    cnt = np.zeros((NSEC, len(HOURS)))
    for t, i, k in reqs:
        cnt[rural.sector[i], int(t//60) - 6] += 1
    for s in range(NSEC):
        for hi, h in enumerate(HOURS):
            rows.append(dict(day=d, sector=s, hour=h, dow=dow, market=int(market), rain=int(rain), y=cnt[s, hi]))
df = pd.DataFrame(rows)
tr, te = df[df.day < N_TRAIN], df[df.day >= N_TRAIN]
X = ["sector", "hour", "dow", "market", "rain"]
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, loss="squared_error", random_state=0)
gbr.fit(tr[X], tr.y)
te = te.copy(); te["pred_gbr"] = np.clip(gbr.predict(te[X]), 0, None)
mean_tab = tr.groupby(["sector", "hour"]).y.mean()
te["pred_mean"] = [mean_tab[(s, h)] for s, h in zip(te.sector, te.hour)]
mae = {"historical_mean": round(float((te.y - te.pred_mean).abs().mean()), 3),
       "gbr": round(float((te.y - te.pred_gbr).abs().mean()), 3)}
summary["forecast_mae"] = mae

# ---------------- 4. 운행 시뮬레이션 ----------------
SPEED = 35/60  # km/min
def nn_tour(start, pts):
    order, cur, rem, dist = [], start, list(pts), 0.0
    while rem:
        j = min(rem, key=lambda p: km(cur, p)); dist += km(cur, j); cur = j; rem.remove(j); order.append(j)
    return order, dist, cur
coords = [(r.lat, r.lon) for r in rural.itertuples()]

# (A) 고정노선: 섹터별 순환노선, 하루 4회(07,10,13,17시) 출발
FIX_DEP = [7*60, 10*60, 13*60, 17*60]
fix_routes, fix_km_day = {}, 0.0
for s in range(NSEC):
    idx = list(rural.index[rural.sector == s])
    if not idx: continue
    order, dist, last = nn_tour(HUB, [coords[i] for i in idx])
    dist += km(last, HUB)
    arr, cur, tt = {}, HUB, 0.0
    for p in order:
        tt += km(cur, p)/SPEED + 0.5; cur = p
        arr[coords.index(p)] = tt
    fix_routes[s] = arr; fix_km_day += dist*len(FIX_DEP)
def sim_fixed(reqs):
    waits, unserved = [], 0
    for t, i, k in reqs:
        s = rural.sector[i]; off = fix_routes[s][i] if k == "in" else 0.0
        cand = [dep + off for dep in FIX_DEP if dep + off >= t]
        if not cand: unserved += 1; continue
        waits.append(cand[0] - t)
    return waits, unserved, fix_km_day, len(fix_routes)*(20-7)

CAP, BATCH = 15, 30
def sim_drt(reqs, day_feat=None, predictive=False, K_MAX=8):
    """섹터 단위 통합 운행: 허브 출발 -> (하차·승차 마을 최근접 순회) -> 허브 복귀.
    B: 8대 전일 투입 / C: AI 수요예측으로 시간대별 투입 대수 결정(유휴 차량 감축)"""
    if predictive:
        f = pd.DataFrame([dict(sector=s, hour=h, **day_feat) for s in range(NSEC) for h in HOURS])
        f["p"] = np.clip(gbr.predict(f[X]), 0, None)
        ph = f.groupby("hour").p.sum()
        active = {h: int(min(K_MAX, max(2, math.ceil(ph[h] / 4.0)))) for h in HOURS}
    else:
        active = {h: K_MAX for h in HOURS}
    vhours = sum(active.values())
    veh = [6*60.0]*K_MAX
    waits, vkm = [], 0.0
    t0, qi = 6*60, 0
    while qi < len(reqs):
        batch = []
        while qi < len(reqs) and reqs[qi][0] < t0 + BATCH:
            batch.append(reqs[qi]); qi += 1
        t0 += BATCH
        h = min(20, max(6, int((t0-1)//60)))
        groups = {}
        for t, i, k in batch: groups.setdefault(rural.sector[i], []).append((t, i, k))
        for sct, g in sorted(groups.items(), key=lambda x: -len(x[1])):
            for c in range(0, len(g), CAP):
                chunk = g[c:c+CAP]
                j = min(range(active[h]), key=lambda j: veh[j])
                start = max(veh[j], t0)
                pts = list({coords[i] for _, i, _ in chunk})
                order, dist, last = nn_tour(HUB, pts)
                tt, cur, arr = start, HUB, {}
                for p_ in order: tt += km(cur, p_)/SPEED + 1.0; cur = p_; arr[p_] = tt
                for t, i, k in chunk: waits.append((arr[coords[i]] if k == "in" else start) - t)
                dist += km(last, HUB); tt += km(last, HUB)/SPEED
                veh[j] = tt; vkm += dist
    return waits, 0, vkm, vhours

res = {"A": [], "B": [], "C": []}
for d in range(N_TRAIN, N_TRAIN + N_TEST):
    reqs, market, rain, dow = days[d]
    res["A"].append(sim_fixed(reqs) + (len(reqs),))
    res["B"].append(sim_drt(reqs) + (len(reqs),))
    res["C"].append(sim_drt(reqs, dict(dow=dow, market=int(market), rain=int(rain)), predictive=True) + (len(reqs),))
def agg(lst):
    w = np.concatenate([np.array(x[0]) for x in lst]); n = sum(x[4] for x in lst)
    return dict(wait_mean=round(float(w.mean()), 1), wait_p90=round(float(np.percentile(w, 90)), 1),
                within30=round(float((w <= 30).mean()*100), 1), unserved_pct=round(sum(x[1] for x in lst)/n*100, 1),
                vkm_day=round(float(np.mean([x[2] for x in lst])), 1), vhours_day=round(float(np.mean([x[3] for x in lst])), 1),
                pax_day=round(n/len(lst), 1), km_per_pax=round(float(np.mean([x[2] for x in lst]))/(n/len(lst)), 2))
summary["scenarios"] = {k: agg(v) for k, v in res.items()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# ---------------- 5. 시각화 ----------------
C1, C2, C3, GR = "#2a6f97", "#e07a5f", "#3d9970", "#8d99ae"
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(st.lon, st.lat, s=st.n_stops*6, c=GR, alpha=.35, label="읍내(2.5km 이내) 정류장")
sc = ax.scatter(rural.lon, rural.lat, s=rural.n_stops*8, c=rural.d_hub, cmap="YlOrRd", edgecolor="k", lw=.3, label="농촌 마을 정류장")
ax.scatter(*HUB[::-1], marker="*", s=300, c=C1, label="보은시외버스터미널(허브)")
plt.colorbar(sc, ax=ax, label="허브까지 도로거리 추정(km)")
ax.set_title("보은군 정류장 클러스터와 읍내까지의 거리 (국토교통부 정류장 위치정보)")
ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_map.png"), dpi=170); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tday = te[te.day == te.day.unique()[3]].groupby("hour")[["y", "pred_gbr", "pred_mean"]].sum()
ax[0].bar(tday.index, tday.y, color=GR, label="실제(시뮬레이션)")
ax[0].plot(tday.index, tday.pred_gbr, "-o", c=C2, label="AI 예측(GBR)")
ax[0].plot(tday.index, tday.pred_mean, "--", c=C1, label="단순 평균")
ax[0].set_title("시간대별 호출 수요: 실제 vs 예측 (테스트일 예시)"); ax[0].set_xlabel("시"); ax[0].legend(fontsize=8)
ax[1].bar(["단순 평균", "AI 예측(GBR)"], [mae["historical_mean"], mae["gbr"]], color=[C1, C2])
for i, v in enumerate([mae["historical_mean"], mae["gbr"]]): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
ax[1].set_title("섹터·시간대 수요예측 오차(MAE, 건)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_forecast.png"), dpi=170); plt.close()

S = summary["scenarios"]; lab = ["A 고정노선", "B 호출형 DRT", "C AI 예측 DRT"]; col = [GR, C1, C2]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for j, (key, title) in enumerate([("wait_mean", "평균 대기시간(분)"), ("within30", "30분 이내 탑승 비율(%)"), ("vkm_day", "일 운행거리(km)")]):
    vals = [S[k][key] for k in "ABC"]; ax[j].bar(lab, vals, color=col)
    for i, v in enumerate(vals): ax[j].text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
    ax[j].set_title(title); ax[j].tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_scenarios.png"), dpi=170); plt.close()

# ---------------- 6. 차량 대수 민감도 (호출형 DRT) ----------------
sens = []
for K in [8, 12, 16, 20, 24]:
    lst = [sim_drt(days[d][0], K_MAX=K) + (len(days[d][0]),) for d in range(N_TRAIN, N_TRAIN + 7)]
    a = agg(lst); a["K"] = K; sens.append(a)
summary["fleet_sensitivity"] = sens
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(sens, ensure_ascii=False))
fig, ax = plt.subplots(figsize=(7, 4))
Ks = [x["K"] for x in sens]
ax.plot(Ks, [x["wait_mean"] for x in sens], "-o", c=C1, label="호출형 DRT 평균 대기(분)")
ax.plot(Ks, [x["within30"] for x in sens], "-s", c=C3, label="30분 이내 탑승 비율(%)")
ax.axhline(S["A"]["wait_mean"], ls="--", c=GR, label=f"고정노선 평균 대기 {S['A']['wait_mean']}분")
ax.set_xlabel("투입 셔틀 대수"); ax.set_title("셔틀 대수에 따른 호출형 DRT 서비스 수준"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_fleet.png"), dpi=170); plt.close()
