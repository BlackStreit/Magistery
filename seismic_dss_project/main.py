# main.py
# Точка входа: расчёт регионального прогноза по частям + глобальный прогноз и «горячие точки»

import os
from datetime import datetime
import yaml
import numpy as np

import data_loader
import zone_model
import etas_model
import exposure_model
import visualizer
import csep_eval

from zone_engine import RegionStatus, ZoneExpertSystem, CellStatus, CoordinateExpertSystem


def main() -> None:
    # 1. Загрузка конфигурации
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- НОВОЕ: вычисляем общие границы по всем зонам -----------------
    all_bounds = [r["bounds"] for r in cfg["regions"]]
    lon_min = min(b["lon_min"] for b in all_bounds)
    lon_max = max(b["lon_max"] for b in all_bounds)
    lat_min = min(b["lat_min"] for b in all_bounds)
    lat_max = max(b["lat_max"] for b in all_bounds)
    global_bounds = {"lon_min": lon_min, "lon_max": lon_max,
                     "lat_min": lat_min, "lat_max": lat_max}
    # ------------------------------------------------------------------

    # 2. Загрузка каталога
    catalog = data_loader.load_catalog(cfg["catalog_csv"])
    t_cut = datetime.fromisoformat(cfg["etas"]["training_end_date"])

    # 3. Глобальный прогноз (KDE + ETAS) на всей области
    base_map = zone_model.compute_long_term_kde(
        catalog,
        region_bounds   = global_bounds,          # ← заменили
        grid_resolution = cfg["grid_res"],
        bandwidth_km    = cfg["kde"]["bandwidth_km"],
    )
    recent_events = catalog[catalog["date"] >= t_cut]
    final_map = etas_model.etas_forecast(
        base_map,
        recent_events = recent_events,
        etas_params   = cfg["etas"],
    )

    # 4. Сбор статистики по регионам
    region_stats = []
    for reg in cfg["regions"]:
        name = reg["name"]
        b = reg["bounds"]
        sub = catalog[
            catalog["lat"].between(b["lat_min"], b["lat_max"]) &
            catalog["lon"].between(b["lon_min"], b["lon_max"])
        ]
        prev   = sub[sub["date"] <  t_cut]
        recent = sub[sub["date"] >= t_cut]

        # средняя годовая частота
        yrs_prev   = (t_cut - prev["date"].min()).days / 365.25 if not prev.empty   else 0
        yrs_recent = (sub["date"].max() - t_cut).days / 365.25   if len(recent) > 1 else 0
        prev_rate   = len(prev)   / yrs_prev   if yrs_prev   > 0 else 0
        recent_rate = len(recent) / yrs_recent if yrs_recent > 0 else 0
        gap_years   = int((t_cut - prev["date"].max()).days / 365.25) if not prev.empty else 0

        # плотность событий (events per 10k km²·yr)
        area_km2 = (b["lat_max"] - b["lat_min"]) * 111 \
                  * (b["lon_max"] - b["lon_min"]) * 75
        dens = recent_rate / area_km2 * 1e4

        # средняя магнитуда/класс
        if "magnitude" in sub.columns:
            mean_mag = recent["magnitude"].mean() if not recent.empty else 0
        else:
            mean_mag = recent["class"].mean() if "class" in recent.columns and not recent.empty else 0

        # суммарная вероятность в регионе
        arr = final_map.to_numpy()
        lat_idx = np.where((final_map._lat >= b["lat_min"]) &
                           (final_map._lat <= b["lat_max"]))[0]
        lon_idx = np.where((final_map._lon >= b["lon_min"]) &
                           (final_map._lon <= b["lon_max"]))[0]
        prob_sum = arr[np.ix_(lat_idx, lon_idx)].sum()

        region_stats.append({
            "name": name,
            "prev_rate": prev_rate,
            "recent_rate": recent_rate,
            "gap_years": gap_years,
            "ratio": recent_rate / prev_rate if prev_rate > 0 else 0,
            "dens": dens,
            "mag": mean_mag,
            "prob": prob_sum
        })

    # 5. Нормировка признаков и расчёт score
    def normalize(x: np.ndarray) -> np.ndarray:
        ptp = x.ptp()
        return (x - x.min()) / ptp if ptp > 0 else np.zeros_like(x)

    r = np.array([s["ratio"] for s in region_stats])
    d = np.array([s["dens"]  for s in region_stats])
    m = np.array([s["mag"]   for s in region_stats])
    p = np.array([s["prob"]  for s in region_stats])

    rn, dn, mn, pn = normalize(r), normalize(d), normalize(m), normalize(p)
    for s, ri, di, mi, pi in zip(region_stats, rn, dn, mn, pn):
        s["score"] = 0.4*ri + 0.3*di + 0.2*mi + 0.1*pi

    # 6. Ранжирование и присвоение mid_level
    sorted_stats = sorted(region_stats, key=lambda x: x["score"], reverse=True)
    n = len(sorted_stats)
    for idx, s in enumerate(sorted_stats):
        if idx < n/3:
            s["mid_level"] = "Высокая"
        elif idx < 2*n/3:
            s["mid_level"] = "Средняя"
        else:
            s["mid_level"] = "Низкая"

    # 7. Экспертная система RegionStatus с предустановленным mid_level
    print("\n=== Региональный прогноз по зонам ===")
    zes = ZoneExpertSystem()
    zes.reset()
    for s in sorted_stats:
        zes.declare(RegionStatus(
            region      = s["name"],
            gap         = s["gap_years"],
            prev_rate   = s["prev_rate"],
            recent_rate = s["recent_rate"],
            mid_level   = s["mid_level"]
        ))
    zes.run()

    # 8. «Горячие точки» — CoordinateExpertSystem
    ces = CoordinateExpertSystem()
    ces.reset()
    for i in range(final_map.shape[0]):
        for j in range(final_map.shape[1]):
            ces.declare(CellStatus(
                lon  = float(final_map._lon[j]),
                lat  = float(final_map._lat[i]),
                prob = float(final_map[i, j])
            ))
    #print("\n=== Top 10 «горячих точек» ===")
    ces.run()

     # 9. Визуализация карт
    bounds = (lon_min, lon_max, lat_min, lat_max)  # ← используем вычисленные границы
    out_dir = cfg["visualization"].get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    visualizer.plot_static_map(
        final_map.to_numpy(),
        region_bounds=bounds,
        outfile=os.path.join(out_dir, "prob_map.png")
    )
    visualizer.plot_interactive_map(
        final_map.to_numpy(),
        region_bounds=bounds,
        outfile=os.path.join(out_dir, "prob_map.html")
    )

    # 10. Расчёт потерь (если заданы слои)
    exp_cfg = cfg["exposure"]
    if exp_cfg.get("population_tiff") and exp_cfg.get("cost_tiff"):
        loss = exposure_model.compute_loss_map(
            final_map.to_numpy(),
            population_tiff=exp_cfg["population_tiff"],
            cost_tiff=exp_cfg["cost_tiff"],
        )
        visualizer.plot_static_map(
            loss,
            region_bounds=bounds,
            outfile=os.path.join(out_dir, "loss_map.png")
        )

    # 11. Оценка качества (CSEP) с пояснениями
    if cfg["evaluation"].get("enable", False):
        metrics = csep_eval.compute_metrics(
            forecast_map    = final_map,
            actual_events   = recent_events.itertuples(),
            baseline_map    = base_map if cfg["evaluation"].get("baseline_long_term", False) else None,
            alarm_threshold = cfg["evaluation"].get("alarm_threshold", 0.005),
        )

        explanations = {
            "log_likelihood":       "Сумма ln P для реальных событий (↑ лучше)",
            "log_likelihood_gain":  "Приращение к базовому прогнозу (↑ лучше)",
            "POD":                  "Доля обнаруженных событий (↑ лучше)",
            "FAR":                  "Доля ложных тревог (↓ лучше)",
            "Brier":                "MSE вероятностей (↓ лучше)"
        }

        print("\n=== METRICS ===")
        for name, val in metrics.items():
            note = explanations.get(name, "")
            # форматирование: 4 знака после запятой для дробей, целое для лог-правд
            if "likelihood" in name:
                print(f"{name:20s}: {val: .4f}    — {note}")
            else:
                print(f"{name:20s}: {val: .4%}    — {note}")


if __name__ == "__main__":
    main()
