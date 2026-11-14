"""Generate notebooks/m5_walkthrough.ipynb (run from the repo root, then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# M5 demand forecasting walkthrough: CA_1 FOODS\n\n"
       "This notebook reads the outputs of the pipeline (`python -m src.backtest`, `python -m src.inventory`) "
       "and walks through the data, the backtest and the inventory policy. It does not retrain anything, "
       "so it runs in seconds."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd, numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from src.data import load_panel\n"
         "from src.metrics import summarize, weekly, wape\n"
         "from src.backtest import MODELS, CUTOFFS, OUT\n"
         "pd.set_option('display.float_format', '{:.3f}'.format)\n"
         "panel = load_panel()\n"
         "fc = pd.read_parquet(OUT / 'forecasts.parquet')\n"
         "sim = pd.read_csv(OUT / 'inventory_simulation.csv')"),
    md("## 1. The data\n\nOne Walmart store in California, the FOODS category: three departments, "
       "roughly 1,400 items, daily unit sales from 2011 to May 2016, weekly shelf prices, and a calendar "
       "with holidays and SNAP (food-stamp) days."),
    code("print(panel.groupby('dept_id', observed=True)['id'].nunique())\n"
         "print(f\"share of item-days with zero sales: {(panel.sales == 0).mean():.1%}\")\n"
         "daily = panel[panel.d <= 1941].groupby('date')['sales'].sum()\n"
         "daily.plot(figsize=(11, 3), lw=0.6, title='Units sold per day'); plt.show()"),
    md("Demand is **intermittent**: almost half of all item-days have zero sales. That is typical of a "
       "dark store's long tail too, and it is why we use WAPE rather than MAPE (MAPE divides by zero)."),
    code("by_wday = panel[panel.d > 1941 - 365].groupby(['wday'])['sales'].mean()\n"
         "by_wday.index = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']\n"
         "snap = panel[panel.d > 1941 - 365].groupby('snap')['sales'].mean()\n"
         "print('Mean units per item-day by weekday:'); print(by_wday.round(2))\n"
         "print(f'SNAP-day uplift: {snap[1] / snap[0] - 1:.1%}')"),
    md("## 2. Backtest results\n\nFive rolling origins, each forecasting the next 28 days. "
       "WAPE = sum of absolute errors / sum of actual units."),
    code("summarize(fc, MODELS).T"),
    code("summarize(fc, MODELS, by=['cutoff'])[[f'{m}_wape' for m in MODELS]]"),
    md("Item-week WAPE is the more relevant number for replenishment: errors on individual days "
       "partly cancel within a week."),
    code("summarize(weekly(fc, MODELS), MODELS).T"),
    code("summarize(fc, MODELS, by=['dept_id'])[[f'{m}_wape' for m in MODELS]]"),
    md("### Where does LightGBM win?\nSplit items into volume tiers and compare WAPE per tier."),
    code("vol = fc.groupby('id')['y'].mean()\n"
         "fc['tier'] = fc['id'].map(pd.qcut(vol, [0, .5, .8, .95, 1], labels=['slow 50%', 'mid 30%', 'fast 15%', 'top 5%']))\n"
         "summarize(fc, MODELS, by=['tier'])[[f'{m}_wape' for m in MODELS]]"),
    md("### WAPE by days ahead\nThe week-1 model can use last week's sales; the week-4 model cannot."),
    code("fc['h'] = fc['d'] - fc['cutoff']\n"
         "h = fc.groupby('h').apply(lambda g: pd.Series({m: wape(g.y, g[m]) for m in MODELS}), include_groups=False)\n"
         "h.plot(figsize=(9, 3.5), title='WAPE by days ahead'); plt.show()"),
    code("imp = pd.read_csv(OUT / 'feature_importance.csv').groupby('feature')['gain'].mean()\n"
         "(imp / imp.sum()).sort_values(ascending=False).head(15)"),
    md("## 3. From forecast to reorder rule\n\n"
       "Daily review, 2-day lead time, lost sales. Order-up-to level S = forecast over the next 3 days "
       "+ z × σ₃, where σ₃ is each item's out-of-sample error over 3 days, measured on backtest folds 1 to 4. "
       "The policy is then simulated on fold 5, which none of the σ estimates saw."),
    code("sim.pivot_table(index='policy', columns='model', values='fill_rate')"),
    code("sim.pivot_table(index='policy', columns='model', values='avg_stock_value').round(0)"),
    code("pol = pd.read_csv(OUT / 'reorder_policy.csv')\n"
         "pol[pol.model == 'LightGBM'].sort_values('fcst_daily', ascending=False).head(10)"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/m5_walkthrough.ipynb")
print("wrote notebooks/m5_walkthrough.ipynb")
