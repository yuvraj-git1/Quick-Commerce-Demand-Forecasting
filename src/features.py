"""Feature engineering for the LightGBM model.

Sales-based features are lagged by at least `min_lag` days. The backtest trains one model
per forecast week (min_lag = 7, 14, 21, 28): the model for days 1-7 after the cutoff can
use last week's sales, the model for days 22-28 can only use sales from 28+ days ago.
No model ever feeds its own predictions back in (a "direct" strategy, no recursion).
Prices and calendar fields are treated as known in advance: a retailer sets its own
prices and promotions ahead of time.
"""
import numpy as np
import pandas as pd

HORIZON = 28
N_WEEKLY_LAGS = 6
ROLL_WINDOWS = [7, 14, 28, 56, 112]

CATEGORICAL = ["item_id", "dept_id", "event_name_1", "event_type_1"]


def add_features(df: pd.DataFrame, min_lag: int = HORIZON) -> pd.DataFrame:
    """Build features using only sales at least `min_lag` days old (min_lag is a multiple of 7).

    Lag names are relative so every horizon model has the same columns:
    lag_w0 = the most recent same-weekday value available (min_lag days ago), lag_w1 a week before that, ...
    """
    df = df.sort_values(["id", "d"]).copy()
    g = df.groupby("id", observed=True)["sales"]

    # --- same-weekday lags ---
    for k in range(N_WEEKLY_LAGS):
        df[f"lag_w{k}"] = g.shift(min_lag + 7 * k).astype(np.float32)
    df["lag_364"] = g.shift(364).astype(np.float32)  # same day last year (always > 28)

    # --- rolling stats on the series shifted by min_lag ---
    shifted = g.shift(min_lag)
    sg = shifted.groupby(df["id"], observed=True)
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = sg.transform(lambda s: s.rolling(w, min_periods=1).mean()).astype(np.float32)
    df["rstd_28"] = sg.transform(lambda s: s.rolling(28, min_periods=7).std()).astype(np.float32)
    # Share of zero-sales days: tells the model how intermittent the item is.
    df["zero_share_56"] = sg.transform(lambda s: (s == 0).rolling(56, min_periods=7).mean()).astype(np.float32)
    # Same-weekday average over the four most recent available weeks.
    df["wday_mean_4"] = df[["lag_w0", "lag_w1", "lag_w2", "lag_w3"]].mean(axis=1).astype(np.float32)
    # Trend: short window versus long window.
    df["trend_28_112"] = (df["rmean_28"] / (df["rmean_112"] + 1e-3)).astype(np.float32)

    # --- price features ---
    pg = df.groupby("id", observed=True)["sell_price"]
    # Depth of discount vs the highest / average price seen so far (expanding, so no look-ahead).
    df["price_rel_max"] = (df["sell_price"] / pg.cummax()).astype(np.float32)
    df["price_rel_mean"] = (
        df["sell_price"] / pg.transform(lambda s: s.expanding().mean())
    ).astype(np.float32)
    df["price_change_7"] = (df["sell_price"] / pg.shift(7) - 1).astype(np.float32)  # change vs last week
    df["price_momentum_28"] = (
        df["sell_price"] / pg.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    ).astype(np.float32)
    dept_week_mean = df.groupby(["dept_id", "wm_yr_wk"], observed=True)["sell_price"].transform("mean")
    df["price_rel_dept"] = (df["sell_price"] / dept_week_mean).astype(np.float32)
    df["price_n_changes"] = pg.transform(lambda s: (s.diff().fillna(0) != 0).cumsum()).astype(np.float32)

    # --- calendar ---
    df["mday"] = df["date"].dt.day.astype(np.int8)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8).values
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(np.int8)  # M5: wday 1 = Saturday

    # Days since the item launched in this store.
    df["age"] = (df["d"] - df.groupby("id", observed=True)["d"].transform("min")).astype(np.int16)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"id", "d", "date", "sales", "wm_yr_wk", "year"}
    return [c for c in df.columns if c not in drop]
