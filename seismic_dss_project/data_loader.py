# data_loader.py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from shapely import wkb 

def load_catalog(csv_path, min_class=None):
    """Загружает каталог из CSV, конвертирует координаты WKB в числовые столбцы."""
    df = pd.read_csv(csv_path)
    # --- новая строка: парсим даты в datetime ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # ------------------------------------------------
    # Преобразуем WKB (hex) -> координаты
    df['geometry'] = df['coordinate'].apply(_decode_hex)
    df = df.dropna(subset=['geometry', 'date'])
    df['lon'] = df['geometry'].apply(lambda p: p.x)
    df['lat'] = df['geometry'].apply(lambda p: p.y)
    if "magnitude" not in df.columns:
        if "mag" in df.columns:
            df["magnitude"] = df["mag"]
        else:
            # используем 'class' как proxy-магнитуду
            df["magnitude"] = df["class"]
    if min_class:
        df = df[df['class'] >= min_class]
    df.sort_values('date', inplace=True)
    return df

def _decode_hex(hexstr):
    try:
        return wkb.loads(hexstr, hex=True)
    except Exception:
        return None 

def load_population_grid(tiff_path):
    # Загружает GeoTIFF с плотностью населения, возвращает numpy 2D массив и профиль координат
    import rasterio
    with rasterio.open(tiff_path) as ds:
        pop_data = ds.read(1)  # первый канал
        transform = ds.transform
    return pop_data, transform

def load_strain_data(csv_path: str,
                     lon_grid: np.ndarray,
                     lat_grid: np.ndarray,
                     method: str = "linear") -> np.ndarray:
    """
    Читает CSV с колонками lon, lat, strain
    и интерполирует на регулярную сетку lon_grid × lat_grid.

    Возвращает 2D-массив strain_grid той же формы, что meshgrid(lon_grid, lat_grid).
    Значения нормируются 0…1.
    """
    df = pd.read_csv(csv_path)
    pts = df[['lon', 'lat']].values
    vals = df['strain'].values

    # создаём двумерную сетку координат
    XX, YY = np.meshgrid(lon_grid, lat_grid)
    grid = griddata(points=pts,
                    values=vals,
                    xi=(XX, YY),
                    method=method,
                    fill_value=0.0)

    # нормируем в диапазон 0–1
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid
