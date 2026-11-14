"""Charts and summary tables for the README.

Run:  python -m src.report   (after src.backtest and src.inventory)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import CUTOFFS, MODELS, OUT
from .data import ROOT, load_panel
from .features import HORIZON
from .metrics import summarize, weekly

FIG = ROOT / "reports" / "figures"
COLORS = {"SeasonalNaive": "#eb6834", "MA28": "#eda100", "AutoETS": "#1baf7a", "LightGBM": "#2a78d6"}
LABELS = {"SeasonalNaive": "Seasonal naive", "MA28": "28-day average", "AutoETS": "ETS", "LightGBM": "LightGBM"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False,
})


def fig_history(panel):
    s = panel.groupby("date")["sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s.index, s.values, color="#2a78d6", lw=0.7)
    for c in CUTOFFS:
        start = pd.Timestamp("2011-01-29") + pd.Timedelta(days=int(c))
        ax.axvspan(start, start + pd.Timedelta(days=HORIZON - 1), color=GRID, alpha=0.7, lw=0)
    ax.set_title("CA_1 FOODS: units sold per day (shaded = 5 backtest windows)")
    ax.set_ylabel("units")
    fig.savefig(FIG / "01_history.png")
    plt.close(fig)


def fig_wape_by_fold(fc):
    t = summarize(fc, MODELS, by=["cutoff"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(t))
    w = 0.2
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1.5) * w, t[f"{m}_wape"], width=w * 0.9, color=COLORS[m], label=LABELS[m])
    dates = [(pd.Timestamp("2011-01-29") + pd.Timedelta(days=int(c))).strftime("%d %b %Y") for c in t.index]
    ax.set_xticks(x, [f"from {d}" for d in dates])
    ax.set_ylabel("WAPE (item x day)")
    ax.set_ylim(0, t[[f"{m}_wape" for m in MODELS]].max().max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("WAPE by backtest window (lower is better)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / "02_wape_by_fold.png")
    plt.close(fig)


def fig_examples(fc):
    last = fc[fc.cutoff == CUTOFFS[-1]]
    vol = last.groupby("id")["y"].sum().sort_values()
    picks = [vol.index[int(len(vol) * q)] for q in (0.98, 0.75, 0.40)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, item in zip(axes, picks):
        g = last[last.id == item].sort_values("d")
        dates = pd.Timestamp("2011-01-29") + pd.to_timedelta(g["d"] - 1, unit="D")
        ax.plot(dates, g["y"], color=INK, lw=1.4, marker="o", ms=3, label="Actual")
        for m in ["SeasonalNaive", "AutoETS", "LightGBM"]:
            ax.plot(dates, g[m], color=COLORS[m], lw=1.6, label=LABELS[m])
        ax.set_title(f"{item.replace('_CA_1', '')}  ({g['y'].sum():.0f} units in 28 days)", fontsize=10)
        ax.set_ylabel("units/day")
    axes[0].legend(ncol=4, loc="upper right", fontsize=9)
    fig.suptitle("Last backtest window: actuals vs forecasts for a fast, medium and slow mover",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "03_example_forecasts.png")
    plt.close(fig)


def fig_importance():
    imp = pd.read_csv(OUT / "feature_importance.csv").groupby("feature")["gain"].mean()
    imp = (imp / imp.sum()).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color="#2a78d6", height=0.7)
    ax.set_xlabel("share of total gain")
    ax.grid(axis="y", visible=False)
    ax.set_title("LightGBM: top 20 features (mean gain over 5 folds)")
    fig.savefig(FIG / "04_feature_importance.png")
    plt.close(fig)


def fig_inventory(sim):
    curve = sim[~sim.policy.str.startswith("ABC")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in MODELS:
        g = curve[curve.model == m].sort_values("z")
        ax.plot(g["avg_stock_value"] / 1000, g["fill_rate"] * 100, color=COLORS[m], lw=2, marker="o", ms=5,
                label=LABELS[m])
    ax.axhline(95, color=MUTED, lw=0.8, ls="--")
    ax.text(ax.get_xlim()[0], 95.3, " 95% fill rate", color=MUTED, fontsize=9)
    ax.set_xlabel("average stock held overnight ($ thousands, at shelf price)")
    ax.set_ylabel("fill rate (% of demand served)")
    ax.set_title("Inventory vs service: each point is one safety-stock level (z)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "05_inventory_tradeoff.png")
    plt.close(fig)


def stock_needed_for(sim, target):
    """Interpolate the stock value each model needs to reach a target fill rate."""
    out = {}
    curve = sim[~sim.policy.str.startswith("ABC")]
    for m in MODELS:
        g = curve[curve.model == m].sort_values("fill_rate")
        out[m] = float(np.interp(target, g["fill_rate"], g["avg_stock_value"])) if g.fill_rate.max() >= target else np.nan
    return pd.Series(out)


def tables(fc, sim):
    lines = []
    overall = summarize(fc, MODELS).iloc[0]
    wk = weekly(fc, MODELS)
    wk_overall = summarize(wk, MODELS).iloc[0]
    dept = summarize(fc, MODELS, by=["dept_id"])

    lines.append("| Model | WAPE (item-day) | WAPE (item-week) | Bias |")
    lines.append("|---|---:|---:|---:|")
    for m in MODELS:
        lines.append(f"| {LABELS[m]} | {overall[f'{m}_wape']:.1%} | {wk_overall[f'{m}_wape']:.1%} | "
                     f"{overall[f'{m}_bias']:+.1%} |")
    lines.append("")
    lines.append("| Department | " + " | ".join(LABELS[m] for m in MODELS) + " |")
    lines.append("|---|" + "---:|" * len(MODELS))
    for d, r in dept.iterrows():
        lines.append(f"| {d} | " + " | ".join(f"{r[f'{m}_wape']:.1%}" for m in MODELS) + " |")
    lines.append("")
    for tgt in (0.95, 0.97):
        s = stock_needed_for(sim, tgt)
        lines.append(f"Stock value needed for {tgt:.0%} fill rate: " +
                     ", ".join(f"{LABELS[m]} ${v:,.0f}" for m, v in s.items()))
    lines.append("")
    abc = sim[sim.policy.str.startswith("ABC")]
    lines.append("| Model (ABC policy) | Fill rate | Stock-out item-days | Avg stock value | Days of cover | Lost revenue (28 days) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in abc.iterrows():
        lines.append(f"| {LABELS[r.model]} | {r.fill_rate:.1%} | {r.stockout_item_days:.1%} | "
                     f"${r.avg_stock_value:,.0f} | {r.days_of_cover:.2f} | ${r.lost_revenue:,.0f} |")
    text = "\n".join(lines)
    (ROOT / "reports" / "results.md").write_text(text + "\n")
    print(text)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    fc = pd.read_parquet(OUT / "forecasts.parquet")
    sim = pd.read_csv(OUT / "inventory_simulation.csv")
    fig_history(panel[panel.d <= 1941])
    fig_wape_by_fold(fc)
    fig_examples(fc)
    fig_importance()
    fig_inventory(sim)
    tables(fc, sim)


if __name__ == "__main__":
    main()
