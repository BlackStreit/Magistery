"""
=========================================================
 OFF-LINE СППР: региональный и «координатный» прогноз
=========================================================
– Чтение каталога earthquakes.csv (поля: id, coordinate, date, class)
– Расчёт показателей по трём зонам (Запад, Центр, Восток)
– Прогноз макро-рисков (EarthquakeExpertSystem)
– Карта долгосрочной вероятности KDE + вывод «горячих точек»
  (CoordinateExpertSystem, TOP-N ячеек с наибольшей λ)
Зависимости: pandas, numpy, shapely, scipy, experta
"""
#%% Cell init
import pandas as pd
import numpy as np
from shapely import wkb
from scipy.stats import gaussian_kde
from experta import (KnowledgeEngine, Fact, Field,
                     Rule, AS, MATCH, TEST)

#%% Cell 1
# -------------------------------------------------
# 1. Загрузка и подготовка каталога
# -------------------------------------------------
CSV_FILE = r"C:\Users\Администратор\OneDrive\Рабочий стол\Ирниту\Мага диплом\data-1747556556317.csv"    # путь к каталогу
df = pd.read_csv(CSV_FILE)

# Дата → datetime; отбрасываем NaT
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Декодируем PostGIS-WKB в широту/долготу
df["geometry"] = df["coordinate"].apply(lambda x: wkb.loads(bytes.fromhex(x)))
df["lon"] = df["geometry"].apply(lambda p: p.x)
df["lat"] = df["geometry"].apply(lambda p: p.y)

# Назначаем макро-зону
def categorize_region(row):
    if row["lon"] < 100:
        return "Западная зона"
    elif row["lon"] < 114:
        return "Центральная зона"
    else:
        return "Восточная зона"

df["region"] = df.apply(categorize_region, axis=1)

#%% Cell 2
# -------------------------------------------------
# 2. Показатели зон для средне/долгосрочного прогноза
# -------------------------------------------------
end_date = df["date"].max()
recent_years = 3
previous_years = 5
start_recent = end_date - pd.DateOffset(years=recent_years)
start_prev = end_date - pd.DateOffset(years=(recent_years + previous_years))

recent_ev = df[(df["date"] >= start_recent) & (df["class"] >= 10)]
prev_ev   = df[(df["date"] >= start_prev) & (df["date"] < start_recent) & (df["class"] >= 10)]

region_stats = {}
for region, grp in df.groupby("region"):
    major = grp[grp["class"] >= 15]
    last_major_year = int(major["date"].dt.year.max()) if not major.empty else None
    gap = end_date.year - last_major_year if last_major_year else 0
    prev_rate   = prev_ev[prev_ev["region"] == region].shape[0] / previous_years
    recent_rate = recent_ev[recent_ev["region"] == region].shape[0] / recent_years
    region_stats[region] = {"gap": gap,
                            "prev_rate": prev_rate,
                            "recent_rate": recent_rate}

#%% Cell 2
# -------------------------------------------------
# 3. Экспертная система для макро-зон
# -------------------------------------------------
class RegionStatus(Fact):
    region = Field(str, mandatory=True)
    gap = Field(int, mandatory=True)
    prev_rate = Field(float, mandatory=True)
    recent_rate = Field(float, mandatory=True)
    mid_level = Field(str, default=None)
    long_level = Field(str, default=None)

class EarthquakeExpertSystem(KnowledgeEngine):
    @Rule(AS.f << RegionStatus(mid_level=None,
                               recent_rate=MATCH.rr,
                               prev_rate=MATCH.pr))
    def mid_term(self, f, rr, pr):
        if pr == 0:
            level = "Высокая" if rr > 0 else "Низкая"
        else:
            ratio = rr / pr
            if ratio >= 1.5 and (rr - pr) > 5:
                level = "Высокая"
            elif rr > 20 or pr > 20:
                level = "Средняя"
            else:
                level = "Низкая"
        self.modify(f, mid_level=level)

    @Rule(AS.f << RegionStatus(long_level=None, gap=MATCH.g))
    def long_term(self, f, g):
        if g >= 30:
            level = "Высокая"
        elif g >= 15:
            level = "Средняя"
        else:
            level = "Низкая"
        self.modify(f, long_level=level)

    @Rule(
        AS.f << RegionStatus(region=MATCH.r,
                             mid_level=MATCH.mid,
                             long_level=MATCH.long),
        TEST(lambda mid, long: mid is not None and long is not None)
    )
    def report(self, r, mid, long):
        expected_max = {"Западная зона": 18,
                        "Центральная зона": 17,
                        "Восточная зона": 17}
        if mid == "Высокая":
            max_cls = expected_max.get(r, 17)
            rng = f"{max_cls-2}–{max_cls}"
        elif mid == "Средняя":
            rng = "10–15"
        else:
            rng = "<10"
        print(f"Регион: {r} | Среднесрочный риск: {mid}, "
              f"Долгосрочный риск: {long}. Ожидаемый класс: {rng}.")


#%% Cell 4
# -------------------------------------------------
# 4. KDE-карта и «горячие точки»
# -------------------------------------------------
# 4.1 плотность по сильным событиям (K>=12)
class_thr = 12
coords = df[df["class"] >= class_thr][["lon", "lat"]].to_numpy().T
kde = gaussian_kde(coords, bw_method="scott")

# регулярная сетка
lon_grid = np.arange(df["lon"].min(), df["lon"].max(), 0.25)
lat_grid = np.arange(df["lat"].min(), df["lat"].max(), 0.25)
lon_m, lat_m = np.meshgrid(lon_grid, lat_grid)
prob = kde(np.vstack([lon_m.ravel(), lat_m.ravel()])).reshape(lat_m.shape)
prob /= prob.sum()             # нормировка

# 4.2 правила для ячеек
class CellStatus(Fact):
    lon = Field(float)
    lat = Field(float)
    prob = Field(float)
    risk_level = Field(str, default=None)

class CoordinateExpertSystem(KnowledgeEngine):
    @Rule(AS.f << CellStatus(prob=MATCH.p & TEST(lambda p: p > 0),  # всё, что >0
                             risk_level=None))
    def classify(self, f, p):
        lvl = ("Высокий" if p > 0.005       # 0.5 %
               else "Средний" if p > 0.001  # 0.1 %
               else "Низкий")
        self.modify(f, risk_level=lvl)

    @Rule(CellStatus(risk_level=MATCH.lvl & TEST(lambda lvl: lvl != "Низкий"),
                     lon=MATCH.x, lat=MATCH.y, prob=MATCH.p))
    def hotspot(self, x, y, p, lvl):
        print(f"[{lvl}] λ≈{p:.2%}  ({y:.2f}°N, {x:.2f}°E)")


# выбираем TOP-N ячеек по вероятности
TOP_N = 5
flat = prob.ravel()
idx = np.argsort(flat)[::-1][:TOP_N]
cells = [(lat_grid[i], lon_grid[j], flat[k])
         for k in idx
         for i, j in [np.unravel_index(k, prob.shape)]]

#%% Cell 5
# -------------------------------------------------
# 5. Запуск обоих движков
# -------------------------------------------------
print("\n========== ПРОГНОЗ ПО ЗОНАМ ==========")
zone_engine = EarthquakeExpertSystem()
zone_engine.reset()
for reg, s in region_stats.items():
    zone_engine.declare(RegionStatus(region=reg,
                                     gap=s["gap"],
                                     prev_rate=s["prev_rate"],
                                     recent_rate=s["recent_rate"]))
zone_engine.run()

print("\n========== ГОРЯЧИЕ ТОЧКИ (TOP-{}) ==========".format(TOP_N))
cell_engine = CoordinateExpertSystem()
cell_engine.reset()
for lat, lon, p in cells:
    print(f"λ≈{p:.2%}  ({lat:.2f}°N, {lon:.2f}°E)")
cell_engine.run()
