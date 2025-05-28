# exposure_model.py
"""
Расчёт ожидаемых потерь на основе карты вероятностей
и растровых слоёв населения / стоимости недвижимости.

Функция
-------
compute_loss_map(prob_arr, population_tiff, cost_tiff) -> np.ndarray
    prob_arr        – 2-D numpy-массив (P, суммируется к 1)
    population_tiff – путь к GeoTIFF с плотностью населения
    cost_tiff       – путь к GeoTIFF со стоимостью сооружений

Возвращает 2-D numpy-массив той же формы, где
loss[i,j] = P[i,j] · Pop[i,j] · Cost[i,j]
"""

from __future__ import annotations
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling


# ------------------------------------------------------------------
def _load_tiff(path: str) -> tuple[np.ndarray, rasterio.Affine]:
    """Чтение одного слоя GeoTIFF (маски → 0)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).filled(0).astype("float64")
        return arr, src.transform


def _resample_to_shape(
    src_arr: np.ndarray,
    src_transform: rasterio.Affine,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Билинейная ресэмплинг-подгонка растров до целевого разрешения."""
    scale_y = src_arr.shape[0] / target_shape[0]
    scale_x = src_arr.shape[1] / target_shape[1]

    kwargs = {
        "out_shape": target_shape,
        "resampling": Resampling.bilinear,
        "dst_transform": src_transform * src_transform.scale(scale_x, scale_y),
    }
    # rasterio.warp.reproject непосредственно в массив-приёмник
    dst_arr = np.empty(target_shape, dtype="float64")
    rasterio.warp.reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=kwargs["dst_transform"],
        dst_crs="EPSG:4326",
        resampling=kwargs["resampling"],
    )
    return dst_arr


# ------------------------------------------------------------------
def compute_loss_map(
    prob_arr: np.ndarray,
    population_tiff: str | None,
    cost_tiff: str | None,
) -> np.ndarray:
    """
    Возвращает карту ожидаемых потерь
    (та же форма, что и prob_arr).

    Исключения
    ----------
    ValueError – если любой из путей не указан
    """
    if population_tiff is None or cost_tiff is None:
        raise ValueError(
            "Нет слоёв населения/стоимости — exposure не рассчитывается."
        )

    # ---------- читаем слои ---------------------------------------
    pop_arr, pop_tr = _load_tiff(population_tiff)
    cost_arr, cost_tr = _load_tiff(cost_tiff)

    # ---------- подгоняем к размеру прогноза -----------------------
    if pop_arr.shape != prob_arr.shape:
        pop_arr = _resample_to_shape(pop_arr, pop_tr, prob_arr.shape)

    if cost_arr.shape != prob_arr.shape:
        cost_arr = _resample_to_shape(cost_arr, cost_tr, prob_arr.shape)

    # ---------- ожидаемые потери ----------------------------------
    loss = prob_arr * pop_arr * cost_arr
    return loss
