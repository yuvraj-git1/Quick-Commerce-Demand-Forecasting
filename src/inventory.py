"""Turn forecasts into a replenishment rule and test it in a simulation.

Setting: a dark store (quick commerce) or a forward warehouse that is replenished from a
mother hub. Each evening it reviews stock and places an order (review period R = 1 day);
the order arrives L days later. Unmet demand is lost: a quick-commerce customer who sees
"out of stock" orders from a competitor app rather than waiting.

Policy (periodic review, order-up-to):
    protection period  P = L + R days
    order-up-to level  S_t = forecast demand over the next P days + safety stock
    safety stock       SS  = z * sigma_P
    order              q_t = max(0, ceil(S_t - inventory position))

sigma_P is the standard deviation of the forecast error summed over P days, measured
out of sample on earlier backtest folds for each item. That is the key link between
forecast accuracy and inventory: a better forecast has a smaller sigma_P, so it needs
less safety stock for the same service level.

Run:  python -m src.inventory
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import CUTOFFS, MODELS, OUT
from .data import load_panel

LEAD_TIME = 2  # days from order to arrival at the dark store
REVIEW = 1  # order every day
P = LEAD_TIME + REVIEW
Z_GRID = [0.0, 0.5, 0.84, 1.28, 1.65, 2.05, 2.33]  # cycle service levels 50% .. 99%
ABC_Z = {"A": norm.ppf(0.98), "B": norm.ppf(0.95), "C": norm.ppf(0.90)}


def sigma_p(fc: pd.DataFrame, model: str, cutoffs: list[int]) -> pd.Series:
    """Per-item std of the P-day summed forecast error on the given (calibration) folds."""
    rows = fc[fc.cutoff.isin(cutoffs)].sort_values(["id", "cutoff", "d"])
    err = (rows["y"] - rows[model]).to_numpy()
    rows = rows.assign(err=err)
    roll = rows.groupby(["id", "cutoff"], observed=True)["err"].transform(
        lambda s: s.rolling(P).sum()
    )
    s = roll.groupby(rows["id"]).std()
    return s.fillna(s.median()).clip(lower=0.5)  # floor so slow movers keep some cover


def abc_class(panel: pd.DataFrame, before_d: int, days: int = 182) -> pd.Series:
    """ABC on revenue over the half-year before the simulation: A = top 80%, B = next 15%, C = rest."""
    hist = panel[(panel.d > before_d - days) & (panel.d <= before_d)]
    rev = (hist["sales"] * hist["sell_price"]).groupby(hist["id"].astype(str)).sum().sort_values(ascending=False)
    share = rev.cumsum() / rev.sum()
    return pd.Series(np.where(share <= 0.80, "A", np.where(share <= 0.95, "B", "C")), index=rev.index)


def simulate(demand: np.ndarray, fcst: np.ndarray, ss: np.ndarray) -> dict[str, np.ndarray]:
    """Simulate all items at once.

    demand, fcst: (items, days + P) arrays; the extra P days of forecast let the last
    order of the window be sized. Only the first `days` columns of demand are played.
    ss: (items,) safety stock.
    """
    n, total = demand.shape
    days = total - P
    # Start each item at its order-up-to level, with nothing in transit.
    on_hand = fcst[:, 1 : P + 1].sum(1) + ss
    pipeline = np.zeros((n, LEAD_TIME + 1))  # pipeline[:, k] arrives k days from now
    sold = np.zeros((n, days))
    stock = np.zeros((n, days))
    lost = np.zeros((n, days))
    for t in range(days):
        on_hand += pipeline[:, 0]
        pipeline = np.roll(pipeline, -1, axis=1)
        pipeline[:, -1] = 0
        s = np.minimum(on_hand, demand[:, t])
        sold[:, t], lost[:, t] = s, demand[:, t] - s
        on_hand -= s
        stock[:, t] = on_hand  # closing stock, what is held overnight
        position = on_hand + pipeline.sum(1)
        target = fcst[:, t + 1 : t + 1 + P].sum(1) + ss
        pipeline[:, LEAD_TIME - 1] += np.ceil(np.maximum(target - position, 0))
    return {"sold": sold, "stock": stock, "lost": lost}


def run(fc: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_cutoff = CUTOFFS[-1]
    calib = CUTOFFS[:-1]
    test = fc[fc.cutoff == sim_cutoff].sort_values(["id", "d"])
    ids = test["id"].unique()
    days = test["d"].nunique()

    demand = test.pivot(index="id", columns="d", values="y").loc[ids].to_numpy()
    demand = np.hstack([demand, np.zeros((len(ids), P))])
    price = (
        panel[panel.d == sim_cutoff].assign(id=lambda x: x["id"].astype(str)).set_index("id")["sell_price"]
        .reindex(ids).to_numpy()
    )
    abc = abc_class(panel, sim_cutoff).reindex(ids).fillna("C").to_numpy()

    results, policies = [], []
    for m in MODELS:
        f = test.pivot(index="id", columns="d", values=m).loc[ids].to_numpy()
        f = np.hstack([f, f[:, -7:][:, :P]])  # beyond the 28-day horizon, repeat last week's pattern
        sig = sigma_p(fc, m, calib).reindex(ids).to_numpy()
        z_options = [(f"z={z:.2f}", np.full(len(ids), z)) for z in Z_GRID]
        z_options.append(("ABC (98/95/90%)", np.array([ABC_Z[c] for c in abc])))
        for label, z in z_options:
            ss = z * sig
            r = simulate(demand, f, ss)
            d = demand[:, :days]
            results.append({
                "model": m,
                "policy": label,
                "z": float(np.mean(z)),
                "fill_rate": r["sold"].sum() / d.sum(),
                "stockout_item_days": float((r["lost"] > 0).mean()),
                "avg_stock_units": r["stock"].sum(1).mean() / days * len(ids),
                "avg_stock_value": float((r["stock"].mean(1) * price).sum()),
                "days_of_cover": r["stock"].sum() / d.sum(),
                "lost_revenue": float((r["lost"].sum(1) * price).sum()),
            })
            if label.startswith("ABC"):
                policies.append(pd.DataFrame({
                    "id": ids, "model": m, "abc": abc,
                    "fcst_daily": f[:, :days].mean(1),
                    "sigma_P": sig, "z": z, "safety_stock": ss,
                    "reorder_point": f[:, :LEAD_TIME].sum(1) + ss,
                    "order_up_to": f[:, :P].sum(1) + ss,
                    "fill_rate": r["sold"].sum(1) / np.maximum(d.sum(1), 1e-9),
                }))
    return pd.DataFrame(results), pd.concat(policies)


def main():
    fc = pd.read_parquet(OUT / "forecasts.parquet")
    panel = load_panel()
    res, pol = run(fc, panel)
    res.to_csv(OUT / "inventory_simulation.csv", index=False)
    pol.to_csv(OUT / "reorder_policy.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
