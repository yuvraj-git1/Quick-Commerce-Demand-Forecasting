"""Forecast accuracy metrics."""
import numpy as np
import pandas as pd


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum(y).

    Unlike MAPE it is defined when individual days have zero sales, and it weights
    errors by volume, which is what matters for stock and revenue.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum())


def bias(y, yhat) -> float:
    """Signed volume bias: sum(yhat - y) / sum(y). Positive means over-forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / y.sum())


def summarize(fc: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """WAPE and bias for each model, optionally grouped (for example by fold or department)."""
    def _one(g):
        row = {}
        for m in models:
            row[f"{m}_wape"] = wape(g["y"], g[m])
            row[f"{m}_bias"] = bias(g["y"], g[m])
        return pd.Series(row)

    if by is None:
        return _one(fc).to_frame("all").T
    return fc.groupby(by, observed=True).apply(_one, include_groups=False)


def weekly(fc: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Aggregate daily item forecasts to item-week totals (the grain replenishment runs at)."""
    fc = fc.copy()
    fc["week"] = (fc["d"] - fc["cutoff"] - 1) // 7
    return fc.groupby(["cutoff", "id", "week"], observed=True)[["y", *models]].sum().reset_index()
