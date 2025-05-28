# etas_model.py
"""
Краткосрочный прогноз методом ETAS:
коррекция базовой карты вероятностей с учётом афтершоковой активности.
"""

import numpy as np
from prob_map import ProbMap


def etas_forecast(
    base_map: ProbMap,
    recent_events,
    etas_params: dict
) -> ProbMap:
    """
    Добавляет к базовой ProbMap вклад от каждого события по модели ETAS.

    Параметры
    ---------
    base_map : ProbMap
        Нормированная карта long-term вероятностей.
    recent_events : DataFrame
        Содержит колонки 'lat', 'lon', 'date' и 'magnitude' или 'class'.
    etas_params : dict
        Параметры {'A','alpha','c','p','d','q','R',…}.

    Возвращает
    ---------
    ProbMap — нормированная итоговая карта.
    """
    # копируем массив и создаём новый ProbMap
    arr = base_map.to_numpy().copy()
    map_obj = ProbMap(arr, base_map._lon, base_map._lat)

    # текущее время для расчёта dt
    current_time = recent_events['date'].max()
    R = etas_params.get('R', 50.0)  # радиус влияния, км

    for _, ev in recent_events.iterrows():
        # берём магнитуду, если нет — используем класс события
        M = ev['magnitude'] if 'magnitude' in ev.index else ev['class']
        t = ev['date']
        K = etas_params['A'] * 10 ** (etas_params['alpha'] * M)
        dt_days = max((current_time - t).days, 0)
        time_factor = 1.0 / ((dt_days + etas_params['c']) ** etas_params['p'])

        # суммируем вклад по ячейкам в радиусе R
        for i, j in map_obj.grid_cells_within(ev['lat'], ev['lon'], radius_km=R):
            r = map_obj.distance_to(i, j, ev['lat'], ev['lon'])
            spatial_factor = 1.0 / ((r + etas_params['d']) ** etas_params['q'])
            map_obj[i, j] += K * time_factor * spatial_factor

    # нормируем итоговую карту
    result = map_obj.to_numpy()
    total = result.sum()
    if total > 0:
        result /= total

    return ProbMap(result, base_map._lon, base_map._lat)
