# boeun-ai-drt — 농촌 교통소외지역 AI 호출형 자율주행 셔틀 분석 (충북 보은군)

AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전 출품 코드.

## 데이터
- 국토교통부_전국 버스정류장 위치정보_20251031 (공공데이터포털, https://www.data.go.kr/data/15067528/fileData.do)
  - 원본 CSV(EUC-KR, 227,065건)에서 보은군 정류장(정류장번호 `BEB275*`, 702개소)을 추출하고, 약 1km 격자(위도 0.008°, 경도 0.01°)로 묶어 `data/boeun_stop_clusters.csv`(248개 클러스터)로 저장
- 마을별 시간대 호출 수요는 공개 데이터가 없어 **합성 수요(가정)**를 사용함: 정류장 1개소당 일 0.25통행, 오전 읍내행/오후 귀가 곡선, 5일장(1·6일) ×1.6, 강우 ×0.75, 일요일 ×0.6

- 충청북도 보은군_농어촌버스운행표 (공공데이터포털, https://www.data.go.kr/data/15053571/fileData.do → 보은군청 운행시간표, 2025.8.12 최종수정)
  - 29개 노선의 보은 출발 일 운행횟수를 `data/boeun_rural_bus_timetable_counts.csv`로 집계

## 실행
```
pip install pandas numpy scikit-learn matplotlib
python src/analysis.py
python src/timetable.py
python src/fuel_transition.py
```
결과: `figures/summary.json`, `figures/fig1~4.png`

## 시나리오
- A 고정노선: 권역별 순환노선 8개, 하루 4회
- B 호출형 자율주행 DRT: 30분 단위 합승 배차, 정원 15
- C AI 예측 배차: GradientBoosting 수요예측으로 시간대별 투입 대수 조정
- 차량 대수 민감도: 8~24대

## 주요 결과
| 시나리오 | 평균 대기(분) | 30분 내 탑승 | 미운행 | 일 운행거리(km) |
|---|---|---|---|---|
| A 고정노선(8대) | 100.5 | 15.8% | 8.8% | 2,188 |
| B 호출형(8대) | 141.5 | 7.3% | 0% | 4,083 |
| B 호출형(12대, 7일 평가) | 30.6 | 53.7% | 0% | 3,876 |
| C AI 예측 배차 | 179.7 | 2.6% | 0% | 4,083 (차량 운행시간 −32%) |

수요예측 MAE: 단순평균 1.415 → GBR 1.269 (−10.3%)

## AI 활용
분석 모델: scikit-learn GradientBoostingRegressor. 코드 작성·보고서 초안에 Anthropic Claude를 활용했고, 모든 결과는 코드를 직접 실행해 검증함.

## 추가 분석
- 운행표: 29개 노선 중 18개(62.1%)가 하루 4회 이하, 중앙값 4회 (`src/timetable.py`, `figures/fig6_timetable.png`)
- 차종 전환(셔틀 12대, 연 141만 km): 경유 736tCO2·4.24억 원 → 전기 517t·3.40억 원, 추출수소 1,088t·10.88억 원, 그린수소 0t·10.88억 원 (`src/fuel_transition.py`, 계수는 코드 상단 가정값)
