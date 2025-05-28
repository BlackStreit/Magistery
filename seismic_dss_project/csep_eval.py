# csep_eval.py
"""
Оценка качества прогнозов (CSEP-style).

forecast_map  – ProbMap (контейнер над numpy-массивом с методами get_cell(), to_numpy())  
actual_events – итерабельный объект с атрибутами .lat и .lon  
baseline_map  – (опц.) ProbMap для расчёта выигрыша лог-правдоподобия  
alarm_threshold – порог вероятности, выше которого считается «тревога»
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Iterable
from prob_map import ProbMap


def compute_metrics(
    forecast_map: ProbMap,
    actual_events: Iterable,
    baseline_map: ProbMap | None = None,
    alarm_threshold: float = 0.005
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # --- 1. Log-likelihood (и LL-gain) -------------------------------
    ll_model = 0.0
    ll_base = 0.0
    for ev in actual_events:
        i, j = forecast_map.get_cell(ev.lat, ev.lon)
        p = forecast_map[i, j]
        ll_model += np.log(max(p, 1e-12))
        if baseline_map is not None:
            p0 = baseline_map[i, j]
            ll_base += np.log(max(p0, 1e-12))

    metrics["log_likelihood"] = ll_model
    if baseline_map is not None:
        metrics["log_likelihood_gain"] = ll_model - ll_base

    # --- 2. POD и FAR (по «тревожному» порогу) ------------------------
    alarms = forecast_map.to_numpy() >= alarm_threshold
    observed = np.zeros_like(alarms, dtype=bool)
    for ev in actual_events:
        i, j = forecast_map.get_cell(ev.lat, ev.lon)
        observed[i, j] = True

    tp = np.logical_and(alarms, observed).sum()
    fp = np.logical_and(alarms, ~observed).sum()
    fn = np.logical_and(~alarms, observed).sum()

    metrics["POD"] = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["FAR"] = fp / (tp + fp) if (tp + fp) else 0.0

    # --- 3. Brier-score ---------------------------------------------
    prob_arr = forecast_map.to_numpy()
    metrics["Brier"] = float(np.mean((prob_arr - observed.astype(float)) ** 2))

    return metrics
