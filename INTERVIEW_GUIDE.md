# M5 Demand Forecasting: Interview Guide

A complete explanation of the project: what was built, why each choice was made, the
results, and how to answer the questions an interviewer is likely to ask.

---

## 1. The 30-second pitch

> "I built a demand forecasting and replenishment system on Walmart's M5 dataset for about
> 1,400 grocery items in one store. I compared simple baselines (seasonal naive, moving
> average, ETS) against a global LightGBM model using rolling backtests and WAPE. LightGBM
> was the most accurate in every backtest window. Then I turned the forecasts into a daily
> reorder rule with safety stock and simulated it on unseen data: to serve 95% of demand,
> the LightGBM-driven policy needed 13% less inventory than the naive rule. I framed it
> around quick commerce dark stores, where stock-outs are lost orders and shelf space is tight."

## 1b. The whole project in plain English (start here if you're new to this)

Imagine you run a small grocery shop that delivers in 10 minutes. Every evening you
have to answer one question for each of the 1,400 products on your shelves:
**"How many should I order from the warehouse for the next few days?"**

- Order too few and you run out. A customer wanting Maggi finds it "out of stock" and
  buys from another app. You lose that sale.
- Order too many and your tiny shop fills up with stock that sits there, costs money,
  and might expire.

To decide well, you need a good guess of **how many will sell**. That guess is called a
**forecast**. This project does three things:

1. **Makes forecasts** of daily sales for each product, using several methods from very
   simple to more advanced.
2. **Checks which method guesses best**, using past data where we already know the true answer.
3. **Turns the forecast into an ordering rule** and tests it in a pretend shop, to show how
   much stock a better forecast saves.

### A tiny worked example (one product)

Say a packet of biscuits sold this many units over the last two weeks:

```
Mon Tue Wed Thu Fri Sat Sun | Mon Tue Wed Thu Fri Sat Sun
 3   2   2   1   3   5   6  |  2   3   1   2   3   6   5
```

- **Seasonal naive guess** for next Monday: "same as last Monday" = **2**.
- **Moving average guess**: average of recent days ≈ **3** every day.
- **A smarter model** notices weekends are busier, a discount starts on Saturday, and
  it's a festival week, so it guesses Saturday = **8**.

If next Saturday 7 actually sell, the smart model was off by 1, the naive one by 1 (it
said 6), and the average by 4. We measure these misses over all products and days to
see who's best overall.

### Every technical term, explained simply

| Term | What it means in plain words |
|---|---|
| **Time series** | A list of numbers in date order, such as daily sales of one product. |
| **Forecast** | A guess about future numbers in that list. |
| **Horizon** | How far ahead we guess. Here, 28 days. |
| **Baseline** | A simple, "dumb" method. If a fancy model can't beat it, the fancy model is useless. |
| **Seasonal naive** | "This Monday will be like last Monday." |
| **Moving average** | "Each day will be like the average of the last 28 days." |
| **ETS (exponential smoothing)** | A classic method that averages the past but gives more weight to recent days. It also learns the weekly up-and-down pattern. One small model per product. |
| **Machine learning model** | A program that learns patterns from examples instead of following fixed rules. |
| **LightGBM** | A popular machine learning method. It builds hundreds of small decision trees ("if it's a weekend AND on discount, then sales go up by…"), and each new tree fixes the mistakes of the previous ones. One model learns from all products at once. |
| **Feature** | A piece of information we give the model to help it guess, such as "sales same day last week", "is it a holiday?" or "is the price discounted?". |
| **Lag** | A past value of sales. "Lag 7" = sales 7 days ago. |
| **Rolling mean** | The average over a recent window, such as the last 14 days. It tells the model the product's "normal" level. |
| **Data leakage** | Accidentally letting the model peek at information it wouldn't have in real life, like using tomorrow's sales to predict tomorrow. It makes results look great on paper and fail in reality. |
| **Backtest** | Pretend we're back in the past, make forecasts using only data up to that day, then compare them with what really happened. |
| **Rolling-origin backtest** | Do that backtest 5 times, each time moving the "pretend today" forward by 28 days. More reliable than a single test. |
| **WAPE** | "Of all the units we sold, by how many units were our guesses off in total?" For example: 1,000 units sold, guesses off by 400 in total → WAPE = 40%. Lower is better. |
| **Bias** | Do we guess too high or too low on average? Negative = too low. |
| **Intermittent demand** | Products that often sell 0 on a day. Here, 47% of product-days had zero sales. |
| **Tweedie loss** | The "scoring rule" LightGBM learns with. It's designed for numbers that are often 0, never negative, and occasionally large, which is exactly what sales look like. |
| **Lead time** | Days between placing an order and it arriving. Here, 2 days. |
| **Safety stock** | Extra stock kept on top of the forecast, because forecasts are never perfect. Like leaving home 10 minutes early in case of traffic. |
| **σ (sigma)** | How big our forecast mistakes usually are for a product. Big mistakes → need more safety stock. |
| **z** | A "how careful do I want to be" dial. Higher z = more safety stock = fewer stock-outs. z = 1.65 aims to avoid stock-outs about 95% of the time. |
| **Order-up-to level** | The target amount to have (on shelf + on the way). Each day we order the gap between the target and what we have. |
| **Reorder point** | The stock level at which we must order so we don't run out before the delivery arrives. |
| **Fill rate** | Out of everything customers wanted, what % we could actually sell them. |
| **Days of cover** | How many days the stock on the shelf would last at normal sales. |
| **ABC classification** | Rank products by revenue. A = the few products that earn most of the money (be very careful never to run out), C = the many small sellers (accept a bit more risk). |
| **Simulation** | A pretend shop run day by day on a computer: deliveries arrive, real customers' demand is served, orders are placed using our rule, and we count stock-outs and leftover stock. |

### Why the steps happen in this order

1. **Get and clean the data:** you can't forecast without history.
2. **Build simple baselines first:** they tell you what "good enough" looks like.
3. **Build the smarter model:** see if extra information (prices, holidays, patterns
   across products) helps.
4. **Test fairly with backtests:** prove it works on data the model has never seen.
5. **Turn forecasts into orders:** a forecast alone doesn't save money; a decision does.
6. **Simulate:** show the saving in a language managers care about: less stock and fewer stock-outs.

### The one-line result, in plain words

> "My smarter forecast made smaller mistakes than the simple methods. When I used it to
> decide orders in a simulated shop, the shop needed **13% less stock** to keep **95% of
> customers happy**, compared with the simple 'same as last week' rule."

The sections below go into each step in more detail, and the jargon in them is explained
in the table above.

## 2. The business problem

A quick-commerce dark store (Blinkit, Zepto, Instamart style) or a grocery fulfilment
centre has to decide **every day, for every SKU, how much to order** from the mother hub.

- **Order too little:** the item goes out of stock. In quick commerce the customer doesn't
  wait, they open a competitor's app. The sale is **lost**, and so is some customer trust.
- **Order too much:** stock takes up scarce shelf space in a small dark store, ties up
  working capital, and perishables expire.

A forecast is only useful if it improves that decision. So the project doesn't stop at
"which model has the lowest error". It goes on to "**how much stock does each forecast need
to hit a service target?**"

## 3. The data

**M5 Forecasting - Accuracy** (Kaggle, 2020): real daily unit sales from Walmart.
- 3,049 products, 10 stores in California, Texas and Wisconsin, about 5.4 years
  (2011-01-29 to 2016-05-22, 1,941 days).
- `sales_train_evaluation.csv`: daily units per item per store.
- `sell_prices.csv`: weekly shelf price per item per store.
- `calendar.csv`: dates, holidays/events (Super Bowl, Thanksgiving, Ramadan start...),
  and **SNAP** days (US food-stamp payment days, when grocery sales rise).

**Scope chosen:** store **CA_1**, category **FOODS** (departments FOODS_1, FOODS_2, FOODS_3).
- Why one store: keeps compute manageable (about 2.2M item-day rows) while still being a
  real-size problem.
- Why FOODS: groceries are what quick commerce sells, and FOODS has the most items.
- 1,437 items. I kept the **1,406 with at least one year of history** before the first
  backtest, so every model sees a full yearly cycle. Brand-new items need a separate
  "cold start" approach, which I mention as a limitation.

**Key data facts (good to quote):**
- **47% of item-days have zero sales**: demand is *intermittent*.
- Median item sells about 1 unit/day; the top items sell 30 to 50/day.
- Strong weekly pattern: weekends (Sat/Sun about 2.5 units per item-day) vs midweek (about 1.7).
- SNAP days lift sales by about 8%.
- The store is closed on Christmas (zero sales across all items).

**Data preparation (`src/data.py`):**
1. Filter to CA_1 + FOODS.
2. "Melt" the wide table (one column per day) into a long table: one row per item per day.
3. Join the calendar (events, SNAP for California) and weekly prices.
4. Drop days before an item's first recorded price: the item wasn't on the shelf yet,
   so those zeros are not real "zero demand". Leaving them in would bias the models down.
5. Save as Parquet for fast reloads.

## 4. How I evaluated: rolling-origin backtesting

**The question:** "If I had used this model in the past, how accurate would it have been?"

A single train/test split is noisy: one unlucky month can flatter or punish a model.
So I used **rolling-origin backtesting** (also called time-series cross-validation):

```
  train on everything up to d_1801 → forecast d_1802 to d_1829   (fold 1)
  train on everything up to d_1829 → forecast d_1830 to d_1857   (fold 2)
  train on everything up to d_1857 → forecast d_1858 to d_1885   (fold 3)
  train on everything up to d_1885 → forecast d_1886 to d_1913   (fold 4)
  train on everything up to d_1913 → forecast d_1914 to d_1941   (fold 5)
```

- **Horizon = 28 days** (the M5 competition horizon, and a realistic planning window).
- **5 folds** cover February to May 2016.
- The model never sees the future: training always stops at the cutoff.
- Why not random K-fold? Shuffling time-series data leaks future information into
  training, which makes results look better than they would be in production.

## 5. The metric: WAPE (and bias)

**WAPE = Σ |actual − forecast| / Σ actual**

Why WAPE and not the alternatives:
- **MAPE** divides by each actual value. With 47% zeros, MAPE is undefined or explodes.
- **RMSE** is in units and hard to compare across items or explain to business people.
- **WAPE** is a percentage, handles zeros, and **weights errors by volume**. A 10-unit miss on
  a fast mover matters more than a 1-unit miss on a slow mover, which matches the impact
  on revenue and stock.

I report WAPE at two grains:
- **Item-day:** the strict view. It's high (about 66%) because daily sales of one item are
  very noisy. For example: a forecast of 0.8 and an actual of 0 or 2.
- **Item-week:** the replenishment view (about 37%). Daily errors partly cancel within a week.

**Bias = Σ (forecast − actual) / Σ actual.** It shows whether a model systematically
over-forecasts (+) or under-forecasts (−). All models were slightly negative (−5% to −7%),
partly because the backtest period was a growth period, and Tweedie/average models lag
upward trends.

**How to explain "66% WAPE" without it sounding bad:** daily SKU-level data for a single
store is about the noisiest forecasting level there is. Half of the items sell about one unit
a day, and a model cannot predict whether a given customer comes in on Tuesday or Wednesday.
What matters is the *relative* improvement over baselines and the impact on inventory.

## 6. The baselines (`src/backtest.py`)

Always start with simple baselines. They set the bar, and in retail they are hard to beat.
I fitted them per item using **statsforecast** (Nixtla), which is fast.

| Baseline | What it does | Why include it |
|---|---|---|
| **Seasonal naive** | Forecast = sales on the same weekday last week | The simplest "rule of thumb" a store manager might use |
| **28-day moving average** | Forecast = mean of the last 28 days, flat | Smooths noise; good for intermittent items |
| **ETS (AutoETS)** | Exponential smoothing with error/trend/season components, weekly season, best form picked per item by AIC | The classic statistical forecasting model |

Negative forecasts were clipped to 0 (demand can't be negative).

## 7. The main model: LightGBM (`src/features.py`, `src/backtest.py`)

### Why LightGBM
- **Global model:** one model learns from all 1,406 items at once, so slow items borrow
  patterns from similar items. Per-item models (like ETS) only see their own short, noisy history.
- Handles many features (price, events, calendar) naturally; ETS cannot use price or events.
- Fast on about 1.2M training rows; it was the winning model family in the M5 competition.

### Objective: Tweedie
Sales are non-negative counts with many zeros and a long right tail. The **Tweedie** loss
(variance power 1.1) is designed for this "zero-inflated, positive, skewed" data. Plain
squared error (L2) would be pulled around by spikes and could predict negatives.

### Avoiding leakage: the direct multi-horizon strategy
This is the most important design decision. Be ready to explain it.

When forecasting 28 days ahead, I don't know sales for days 1 to 27 of the forecast window.
So a feature like "yesterday's sales" is not available for day 15.

Two common options:
1. **Recursive:** predict day 1, feed it back as a feature for day 2, and so on. Errors compound.
2. **Direct (what I did):** train **4 separate models, one per forecast week**:
   - Week 1 (days 1 to 7) model: uses sales at least **7** days old
   - Week 2 (days 8 to 14) model: at least **14** days old
   - Week 3: at least **21** days old
   - Week 4: at least **28** days old

Each model only uses information that would actually exist at prediction time.
In total: 4 horizon models × 5 folds = **20 LightGBM models**.

My first version used one model with lag ≥ 28 for all days. Its WAPE was 67.3%, barely
better than ETS. Moving to per-week models brought it to **66.3%**, because the week-1
model can use last week's sales. Good story: *"I diagnosed that the model was starved of
recent information and fixed it without introducing leakage."*

### Features (35 in total)
All sales features are lagged by at least the horizon (`min_lag`).

**Lag features (same weekday):**
- `lag_w0 ... lag_w5`: sales on the same weekday for the 6 most recent available weeks
  (same weekday because of the strong weekly pattern)
- `lag_364`: same day last year (yearly seasonality)

**Rolling statistics:**
- `rmean_7/14/28/56/112`: mean sales over different windows (short- vs long-term level)
- `rstd_28`: volatility
- `zero_share_56`: share of zero days, which tells the model how intermittent the item is
- `wday_mean_4`: average of the last 4 same-weekday values
- `trend_28_112`: short-term mean ÷ long-term mean (is the item growing or declining?)

**Price features** (retailers know their own future prices, so these are legitimately known in advance):
- `sell_price`
- `price_rel_max`: price ÷ highest price so far (depth of discount)
- `price_rel_mean`: price ÷ average price so far
- `price_change_7`: change vs last week (has a promotion just started?)
- `price_momentum_28`: price vs its recent 28-day average
- `price_rel_dept`: price vs department average that week (relative cheapness)
- `price_n_changes`: how often the price has changed (promo-heavy items)

"So far" means *expanding* windows: the max/mean only use past prices, so no future leakage.

**Calendar:** weekday, day of month, month, week of year, weekend flag, event name/type,
event flag, SNAP flag.

**Identity:** `item_id`, `dept_id` as categorical features; `age` = days since launch.

**Hyperparameters:** learning rate 0.05, 800 trees, 127 leaves, min 100 rows per leaf,
70% feature sampling, 80% row bagging, small L2. Not tuned; reasonable defaults.

### Feature importance (what the model learned)
1. **Rolling means** (56, 14, 28, 112-day): together about 60% of gain. The item's recent level matters most.
2. **item_id** (about 15%): item-specific behaviour.
3. **Same-weekday averages and lags**: the weekly pattern.
4. Smaller but real: same day last year, events, price relative to average.

Price features have modest importance overall because most items rarely change price.
For promoted items they matter a lot.

## 8. Results

### Accuracy (5 backtest folds, 1,406 items)

| Model | WAPE item-day | WAPE item-week | Bias |
|---|---:|---:|---:|
| Seasonal naive | 80.4% | 40.8% | −5.7% |
| 28-day average | 69.1% | 37.2% | −5.1% |
| ETS | 67.7% | 38.9% | −4.7% |
| **LightGBM** | **66.3%** | **36.7%** | −6.8% |

**By fold (item-day WAPE):** LightGBM was the best in all 5 folds (0.680, 0.680, 0.665, 0.658, 0.636).

**By department:** best in all three (FOODS_3: 60.5% vs ETS 61.7%; FOODS_2: 78.0% vs 79.9%; FOODS_1: 85.6% vs 86.9%).

**By volume tier (important insight):**

| Tier | Seasonal naive | MA28 | ETS | LightGBM |
|---|---:|---:|---:|---:|
| Top 5% of items | 49.0% | 44.1% | 42.9% | **40.0%** |
| Fast 15% | 75.1% | 63.2% | 61.4% | **59.8%** |
| Mid 30% | 99.7% | 82.7% | 80.7% | **80.0%** |
| Slow 50% | 130% | 115% | **114.6%** | 116% |

**Interpretation:** LightGBM's advantage is biggest on the fast movers, which carry most of
the volume and revenue. On the slowest half of items, *nothing* forecasts well (WAPE above 100%):
they sell sporadically. The right response there is a **policy** decision (minimum stock,
lower service target), not a fancier model.

**Honest framing:** the overall gain over ETS is modest (1.4 points). Say this openly.
Interviewers trust candidates who don't oversell. The bigger story is the inventory impact.

## 9. From forecast to decision: the reorder rule (`src/inventory.py`)

### The setting (modelled on a dark store)
- **Review period R = 1 day:** an order is placed every evening.
- **Lead time L = 2 days:** the order arrives 2 days later from the mother hub.
- **Lost sales:** unmet demand disappears (no backorders).

### The policy: periodic review, order-up-to (also called (R, S))
```
Protection period   P  = L + R = 3 days
Safety stock        SS = z × σ_P
Order-up-to level   S  = forecast demand over the next 3 days + SS
Reorder point       ROP = forecast over the lead time (2 days) + SS
Order today         q  = max(0, ceil(S − on-hand − in-transit))
```

**Why protect over L + R?** An order placed today has to cover demand until the *next*
order arrives: 2 days of lead time plus 1 day until the next review.

### Safety stock: the key link between accuracy and money
- `σ_P` = standard deviation of **that model's own 3-day summed forecast error**, per item.
- It's measured **out of sample on folds 1 to 4**, then the policy is tested on **fold 5**,
  which none of the σ estimates saw. This avoids overfitting the safety stock.
- Summing errors over 3 days (instead of daily σ × √3) captures correlated errors,
  for example a model under-forecasting several days in a row.
- **A more accurate forecast → smaller σ_P → less safety stock for the same service.**
  That's the whole economic argument for better forecasting.
- `z` sets the service level: z = 1.65 ≈ 95% cycle service, 2.05 ≈ 98%.

### ABC segmentation (a differentiated policy)
Items are ranked by revenue over the last 6 months:
- **A** = top 80% of revenue → 98% service target (z = 2.05)
- **B** = next 15% → 95% (z = 1.65)
- **C** = last 5% → 90% (z = 1.28)

Safety stock goes where the money is. A standard supply-chain practice.

### The simulation
For each forecast model, I simulated all 1,406 items day by day over the 28 days of fold 5:
1. Start each item at its order-up-to level.
2. Each morning, receive orders that arrive today.
3. Serve demand (actual M5 sales) up to stock on hand; the rest is lost.
4. Each evening, compute the order using the policy above.

Measured: **fill rate** (% of demand served), stock-out item-days, average stock held
overnight (units and value at shelf price), days of cover, and lost revenue.

I swept z from 0 to 2.33 to trace each model's **inventory vs service trade-off curve**.

### Inventory results

**Stock needed to reach a 95% fill rate:**

| Forecast | Stock value | vs naive |
|---|---:|---:|
| Seasonal naive | $30,506 | |
| 28-day average | $28,466 | −7% |
| ETS | $27,641 | −9% |
| **LightGBM** | **$26,468** | **−13%** |

At 97% fill: LightGBM $36,411 vs naive $39,787 (−8.5%).

**ABC policy:** LightGBM reached 96.5% fill with $35.7K stock (3.9 days of cover) vs
seasonal naive's 97.0% with $43.7K (4.75 days). About 18% less stock for half a point of fill rate.

**Interview line:** *"Same service level, 13% less inventory. In a dark store, that's shelf
space for more SKUs and less working capital stuck in stock."*

## 10. The Indian e-commerce / quick commerce angle

How M5 concepts map to India:

| M5 (Walmart, US) | Indian quick commerce equivalent |
|---|---|
| Store CA_1 | A dark store serving a few km radius |
| Distribution centre | Mother hub / city warehouse |
| SNAP payment days | Salary/payday weekends, month start |
| US holidays (Thanksgiving, Super Bowl) | Diwali, Holi, Raksha Bandhan, IPL match days, Big Billion Days-type sales |
| Weekly price changes | App-wide discounts, flash deals, bank offers |
| Lost sale | Customer switches to a competitor app in seconds |

Key points to make:
1. **Lost sales are the default:** that's why the simulation uses lost sales, not backorders.
2. **Dark stores are small:** they hold days, not weeks, of stock. The policy runs at about
   4 days of cover, which is realistic. Saving 13% of stock means room for more SKUs.
3. **Long-tail SKUs are intermittent:** like M5, many dark-store SKUs sell less than one unit
   a day. Use ABC tiers and simple rules for them, and focus modelling effort on the A items.
4. **Promotions and festivals drive spikes:** the same price and event features carry over directly.
5. **Perishables** (milk, bread, produce) need a waste/expiry term. This is an extension.

## 11. Limitations (say these before the interviewer does)

1. **Censored demand:** M5 records *sales*, not *demand*. If an item was out of stock,
   history shows 0 even though people wanted it. Production systems detect stock-out days
   and adjust (impute demand or drop those days).
2. **Single forecast origin per 28 days in the simulation:** a real system would
   re-forecast daily. This would favour LightGBM even more, since its week-1 model uses recent data.
3. **Normal-approximation safety stock:** for slow movers, demand is not normal. Better:
   quantile forecasts (LightGBM quantile loss) or conformal prediction intervals.
4. **No operational constraints:** case packs, minimum order quantities, shelf capacity,
   shelf life and hub capacity are not modelled.
5. **No hyperparameter tuning:** sensible defaults only; Optuna on early folds is a cheap win.
6. **One store:** cross-store features (an item's sales in other stores) would help.
7. **Cold start:** 31 newly launched items were excluded. New SKUs need similarity-based
   forecasts (borrow from similar items).

## 12. Likely interview questions and answers

**Q: Why not use deep learning (LSTM, Transformers, N-BEATS)?**
A: On tabular retail data with many features, gradient boosting usually matches or beats
deep learning at a fraction of the cost. It won M5. It also trains in minutes, is easier to
debug and has feature importance. I'd try a neural model only after the GBM was tuned.

**Q: How do you make sure there's no data leakage?**
A: Three ways. (1) Rolling-origin backtests: training always stops at the cutoff. (2) Every
sales feature is lagged by at least the forecast horizon for that model (7/14/21/28 days).
(3) Price normalisations use expanding windows (only past prices). Prices and calendar are
known in advance because the retailer sets them.

**Q: Why Tweedie loss?**
A: Sales are non-negative, zero-heavy and right-skewed. Tweedie is a compound Poisson-Gamma
distribution that fits exactly that shape. MSE would be distorted by spikes and could predict negatives.

**Q: Your WAPE is 66%. Isn't that bad?**
A: At item-day level for one store it's normal; half the items sell about 1 unit a day. At
item-week level it's 37%. The meaningful measures are the improvement over baselines and the
inventory saving: 13% less stock at the same service.

**Q: Why is LightGBM only slightly better than ETS?**
A: Overall the gap is small because half the items are so sparse that no model does well,
and they dilute the average. On the top 5% of items, LightGBM's WAPE is 40.0% vs 42.9%, and
those items carry most of the volume. The inventory simulation shows the gain is real in money terms.

**Q: How did you choose the safety stock?**
A: Safety stock = z × σ_P, where σ_P is each item's out-of-sample standard deviation of the
3-day summed forecast error (lead time + review period). I estimated it on earlier folds and
tested on a later unseen fold. Service targets vary by ABC class: 98/95/90%.

**Q: What's the difference between cycle service level and fill rate?**
A: Cycle service level = probability of no stock-out in a replenishment cycle (what z targets).
Fill rate = share of demand actually served. Fill rate is what customers feel, so I report it
from the simulation.

**Q: What would you do next / how would you deploy this?**
A: Daily batch pipeline: pull sales, rebuild features, retrain weekly, forecast daily, compute
order quantities, push to the warehouse system. Monitor WAPE and bias per category with alerts
for drift. Then add quantile forecasts, stock-out correction, and perishability.

**Q: Why did you choose WAPE over MAPE or RMSE?**
A: MAPE breaks with zeros (47% of rows). RMSE isn't scale-free. WAPE is a volume-weighted
percentage that business teams understand and that lines up with revenue impact.

**Q: How long does it take to run?**
A: Data prep about 1 minute; ETS baselines about 8 minutes; 20 LightGBM models about 12 minutes
on 4 CPU cores; the simulation takes seconds.

## 13. Repo map (what each file does)

```
src/data.py        Load M5, filter CA_1 FOODS, melt to item-day rows, join calendar and prices
src/features.py    Lag, rolling, price, calendar features (parameterised by min_lag)
src/backtest.py    5-fold rolling backtest: statsforecast baselines + 4 horizon LightGBM models
src/metrics.py     WAPE, bias, weekly aggregation
src/inventory.py   Safety stock, reorder point, order-up-to simulation, ABC tiers
src/report.py      Charts and result tables
notebooks/m5_walkthrough.ipynb   Executed walkthrough of data, results and policy
reports/figures/   5 charts used in the README
scripts/download_data.sh         Fetch data (Kaggle CLI or public mirror)
```

## 14. Key numbers to memorise

- **1,406 items**, store CA_1, FOODS, **5 folds × 28 days**
- **47%** zero-sales days
- WAPE item-day: naive **80.4%** → ETS **67.7%** → LightGBM **66.3%**
- WAPE item-week: LightGBM **36.7%**
- Top 5% items: LightGBM **40.0%** vs ETS **42.9%**
- **13% less stock** than seasonal naive (4% less than ETS) at a **95% fill rate**
- Policy: daily review, **2-day lead time**, about **4 days of cover**, ABC **98/95/90%**
