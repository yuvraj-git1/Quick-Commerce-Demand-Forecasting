"""Rolling-origin backtest: seasonal naive, moving average, ETS and LightGBM.

Five folds, each forecasting the 28 days after its cutoff. The last fold ends on
d_1941 (2016-05-22), the final day of the public M5 data.

Run:  python -m src.backtest
Writes data/outputs/forecasts.parquet (one row per item, day and fold).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, WindowAverage

from .data import ROOT, load_panel
from .features import CATEGORICAL, HORIZON, add_features, feature_columns
from .metrics import summarize

OUT = ROOT / "data" / "outputs"
LAST_DAY = 1941
N_FOLDS = 5
CUTOFFS = [LAST_DAY - HORIZON * k for k in range(N_FOLDS, 0, -1)]  # 1801 ... 1913
TRAIN_START = 1000  # LightGBM trains on roughly the last 2.5 years before each cutoff
MODELS = ["SeasonalNaive", "MA28", "AutoETS", "LightGBM"]

LGB_PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=1.1,
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=0.1,
    n_estimators=800,
    verbose=-1,
    n_jobs=4,
    seed=42,
)


def run_baselines(panel: pd.DataFrame) -> pd.DataFrame:
    sf_df = panel[["id", "date", "sales"]].rename(columns={"id": "unique_id", "date": "ds", "sales": "y"})
    sf_df["unique_id"] = sf_df["unique_id"].astype(str)
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=7), WindowAverage(window_size=28, alias="MA28"),
                AutoETS(season_length=7)],
        freq="D",
        n_jobs=-1,
    )
    cv = sf.cross_validation(df=sf_df, h=HORIZON, n_windows=N_FOLDS, step_size=HORIZON)
    day0 = pd.Timestamp("2011-01-29")
    cv["d"] = ((cv["ds"] - day0).dt.days + 1).astype(np.int16)
    cv["cutoff"] = ((cv["cutoff"] - day0).dt.days + 1).astype(np.int16)
    cv = cv.rename(columns={"unique_id": "id"})
    for m in ["SeasonalNaive", "MA28", "AutoETS"]:
        cv[m] = cv[m].clip(lower=0)  # demand cannot be negative
    return cv[["id", "d", "cutoff", "y", "SeasonalNaive", "MA28", "AutoETS"]]


def run_lightgbm(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One model per forecast week and fold: week k (days 7k-6 .. 7k) uses lags >= 7k days."""
    preds, importances = [], []
    for week in range(1, HORIZON // 7 + 1):
        min_lag = 7 * week
        feat = add_features(panel, min_lag=min_lag)
        cols = feature_columns(feat)
        for cutoff in CUTOFFS:
            train = feat[(feat.d >= TRAIN_START) & (feat.d <= cutoff)]
            days = (feat.d > cutoff + min_lag - 7) & (feat.d <= cutoff + min_lag)
            test = feat[days]
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(train[cols], train["sales"], categorical_feature=[c for c in CATEGORICAL if c in cols])
            p = test[["id", "d"]].copy()
            p["cutoff"] = cutoff
            p["LightGBM"] = np.clip(model.predict(test[cols]), 0, None)
            preds.append(p)
            importances.append(pd.DataFrame({"feature": cols, "gain": model.booster_.feature_importance("gain"),
                                             "cutoff": cutoff, "week": week}))
            print(f"  LightGBM week {week} cutoff=d_{cutoff}: {len(train):,} train rows", flush=True)
    return pd.concat(preds), pd.concat(importances)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    panel = panel[panel.d <= LAST_DAY]
    # Keep items with at least a year of history before the first backtest cutoff,
    # so every model sees a full yearly cycle. New launches need a separate cold-start approach.
    first_d = panel.groupby("id", observed=True)["d"].transform("min")
    n_before = panel["id"].nunique()
    panel = panel[first_d <= CUTOFFS[0] - 364].copy()
    panel["id"] = panel["id"].cat.remove_unused_categories()
    print(f"{panel['id'].nunique()} of {n_before} items kept (>= 1 year of history)")

    t = time.time()
    base_path = OUT / "baseline_forecasts.parquet"
    if base_path.exists() and "--refit-baselines" not in sys.argv:
        base = pd.read_parquet(base_path)  # ETS takes ~8 minutes; reuse unless asked
    else:
        base = run_baselines(panel)
        base.to_parquet(base_path, index=False)
    print(f"baselines ready in {time.time() - t:.0f}s", flush=True)

    t = time.time()
    lgb_fc, imp = run_lightgbm(panel)
    print(f"LightGBM done in {time.time() - t:.0f}s")

    lgb_fc["id"] = lgb_fc["id"].astype(str)
    fc = base.merge(lgb_fc, on=["id", "d", "cutoff"], how="inner")
    assert len(fc) == len(base), (len(fc), len(base))
    dept = panel[["id", "dept_id"]].drop_duplicates().assign(id=lambda x: x["id"].astype(str))
    fc = fc.merge(dept, on="id")
    fc.to_parquet(OUT / "forecasts.parquet", index=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    print("\nOverall (daily, item level):")
    print(summarize(fc, MODELS).round(3).T)
    print("\nWAPE by fold:")
    print(summarize(fc, MODELS, by=["cutoff"])[[f"{m}_wape" for m in MODELS]].round(3))


if __name__ == "__main__":
    main()
