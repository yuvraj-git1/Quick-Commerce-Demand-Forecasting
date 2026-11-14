"""Load the raw M5 files and build a long (item x day) panel for one store / category."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

STORE = "CA_1"
CATEGORY = "FOODS"


def load_panel(store: str = STORE, category: str = CATEGORY, cache: bool = True) -> pd.DataFrame:
    """Return a long dataframe with one row per item per day.

    Columns: id, item_id, dept_id, d (int day index), date, sales, sell_price,
    calendar fields and the SNAP flag for the store's state.
    Days before an item's first recorded price are dropped (the item was not on shelf).
    """
    out = PROCESSED / f"panel_{store}_{category}.parquet"
    if cache and out.exists():
        return pd.read_parquet(out)

    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[(sales.store_id == store) & (sales.cat_id == category)].copy()
    # The Kaggle file has an `id` column; some mirrors drop it, so rebuild it.
    sales["id"] = sales["item_id"] + "_" + sales["store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["id", "item_id", "dept_id"], value_vars=day_cols, var_name="d", value_name="sales"
    )
    long["d"] = long["d"].str[2:].astype(np.int16)
    long["sales"] = long["sales"].astype(np.float32)

    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    # d_1 is 2011-01-29. Derive the index from the date so mirrors without a `d` column work.
    cal["d"] = ((cal["date"] - pd.Timestamp("2011-01-29")).dt.days + 1).astype(np.int16)
    state = store.split("_")[0]
    cal["snap"] = cal[f"snap_{state}"].astype(np.int8)
    cal["is_event"] = cal["event_name_1"].notna().astype(np.int8)
    cal = cal[["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1", "is_event", "snap"]]

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.store_id == store].drop(columns="store_id")

    long = long.merge(cal, on="d", how="left").merge(prices, on=["item_id", "wm_yr_wk"], how="left")
    # An item with no price that week was not on sale; drop the leading pre-launch zeros.
    first_price_d = long.loc[long.sell_price.notna()].groupby("id")["d"].min()
    long = long[long["d"] >= long["id"].map(first_price_d)]
    long["sell_price"] = long.groupby("id")["sell_price"].ffill()
    long = long.sort_values(["id", "d"]).reset_index(drop=True)

    for c in ["id", "item_id", "dept_id", "event_name_1", "event_type_1"]:
        long[c] = long[c].astype("category")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, index=False)
    return long


if __name__ == "__main__":
    df = load_panel(cache=False)
    print(df.shape, df["id"].nunique(), "series")
    print(df.head())
