| Model | WAPE (item-day) | WAPE (item-week) | Bias |
|---|---:|---:|---:|
| Seasonal naive | 80.4% | 40.8% | -5.7% |
| 28-day average | 69.1% | 37.2% | -5.1% |
| ETS | 67.7% | 38.9% | -4.7% |
| LightGBM | 66.3% | 36.7% | -6.8% |

| Department | Seasonal naive | 28-day average | ETS | LightGBM |
|---|---:|---:|---:|---:|
| FOODS_1 | 102.9% | 86.9% | 86.9% | 85.6% |
| FOODS_2 | 95.1% | 80.9% | 79.9% | 78.0% |
| FOODS_3 | 73.3% | 63.5% | 61.7% | 60.5% |

Stock value needed for 95% fill rate: Seasonal naive $30,506, 28-day average $28,466, ETS $27,641, LightGBM $26,468
Stock value needed for 97% fill rate: Seasonal naive $39,787, 28-day average $38,778, ETS $37,893, LightGBM $36,411

| Model (ABC policy) | Fill rate | Stock-out item-days | Avg stock value | Days of cover | Lost revenue (28 days) |
|---|---:|---:|---:|---:|---:|
| Seasonal naive | 97.0% | 2.1% | $43,723 | 4.75 | $7,679 |
| 28-day average | 96.1% | 3.0% | $36,715 | 4.07 | $9,723 |
| ETS | 96.6% | 2.4% | $37,748 | 4.14 | $8,529 |
| LightGBM | 96.5% | 2.3% | $35,708 | 3.88 | $8,521 |
