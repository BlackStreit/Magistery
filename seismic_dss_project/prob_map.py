# prob_map.py
import numpy as np
from typing import Tuple, Iterable

class ProbMap:
    """
    Контейнер над numpy-массивом вероятностей + геопривязка.
    Позволяет:
      • получить/изменить ячейку по индексу или [i,j];s
      • найти ячейку по координатам (lon, lat);
      • перебрать ячейки в радиусе R (км) от точки;
      • быстро вычислять расстояния «центр→точка» (в км, haversine).
    """
    __slots__ = ("_arr", "_lon", "_lat")

    def __init__(self, arr: np.ndarray,
                 lon_grid: np.ndarray,
                 lat_grid: np.ndarray):
        self._arr = arr.astype("float64", copy=False)
        self._lon = lon_grid      # 1-D, длина = nX
        self._lat = lat_grid      # 1-D, длина = nY

    # --- базовый интерфейс numpy-массива -----------------------------
    def __getitem__(self, idx):
        return self._arr[idx]

    def __setitem__(self, idx, value):
        self._arr[idx] = value

    @property
    def shape(self):
        return self._arr.shape

    def to_numpy(self) -> np.ndarray:
        """Возвращает внутренний массив (копию)."""
        return self._arr.copy()

    # --- гео-утилиты ---------------------------------------------------
    def get_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Возвращает (i,j) индексы ближайшей ячейки к точке."""
        j = int(round((lon - self._lon[0]) / (self._lon[1] - self._lon[0])))
        i = int(round((lat - self._lat[0]) / (self._lat[1] - self._lat[0])))
        i = max(0, min(i, self.shape[0] - 1))
        j = max(0, min(j, self.shape[1] - 1))
        return i, j

    # радиус берём в километрах по сфере Земли (R≈6371 км)
    def _haversine_km(self, lat1, lon1, lat2, lon2):
        import numpy as np, math
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat/2)**2 +
             math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(d_lon/2)**2)
        return 6371.0 * 2 * math.asin(min(1, math.sqrt(a)))

    def distance_to(self, i: int, j: int, lat: float, lon: float) -> float:
        """Расстояние (км) от центра ячейки (i,j) до точки lat,lon."""
        return self._haversine_km(
            self._lat[i], self._lon[j], lat, lon)

    def grid_cells_within(self,
                          lat: float, lon: float,
                          radius_km: float) -> Iterable[Tuple[int, int]]:
        """Генератор индексов ячеек внутри radius_km от точки."""
        # приблизительный диапазон индексов (по широте/долготе)
        km_per_deg = 111.0
        d_deg = radius_km / km_per_deg
        i0, j0 = self.get_cell(lat, lon)
        di = int(d_deg / (self._lat[1] - self._lat[0])) + 1
        dj = int(d_deg / (self._lon[1] - self._lon[0])) + 1
        for i in range(max(0, i0-di), min(self.shape[0], i0+di+1)):
            for j in range(max(0, j0-dj), min(self.shape[1], j0+dj+1)):
                if self.distance_to(i, j, lat, lon) <= radius_km:
                    yield (i, j)
