# zone_model.py
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from prob_map import ProbMap

def compute_long_term_kde(
    catalog: pd.DataFrame,
    region_bounds: dict,
    grid_resolution: float,
    bandwidth_km: float           # добавили этот параметр
) -> ProbMap:
    """
    Долгосрочная карта плотности на базе KDE.

    Параметры
    ---------
    catalog : DataFrame с колонками ['lon','lat']
    region_bounds : {lat_min, lat_max, lon_min, lon_max}
    grid_resolution : шаг сетки в градусах
    bandwidth_km : ширина ядра в километрах

    Возвращает
    ---------
    ProbMap — нормированная карта вероятностей
    """
    # 1. Линейная сетка в градусах
    lon_lin = np.arange(region_bounds['lon_min'],
                        region_bounds['lon_max'] + grid_resolution/2,
                        grid_resolution)
    lat_lin = np.arange(region_bounds['lat_min'],
                        region_bounds['lat_max'] + grid_resolution/2,
                        grid_resolution)
    LON, LAT = np.meshgrid(lon_lin, lat_lin)
    points = np.vstack([catalog['lon'], catalog['lat']]).T

    # 2. Перевод bandwidth из км в градусы (приближённо)
    km_per_deg = 111.0
    bandwidth_deg = bandwidth_km / km_per_deg

    # 3. Обучаем KDE
    kde = KernelDensity(bandwidth=bandwidth_deg, kernel='gaussian')
    kde.fit(points)

    # 4. Оцениваем плотность на сетке и нормируем
    grid_points = np.vstack([LON.ravel(), LAT.ravel()]).T
    log_dens = kde.score_samples(grid_points)
    dens = np.exp(log_dens).reshape(LON.shape)
    prob = dens / dens.sum()

    return ProbMap(prob, lon_lin, lat_lin)
