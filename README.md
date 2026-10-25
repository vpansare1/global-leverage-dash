# Leverage Environment dashboard

A daily page that shows, for SPY, EFA and EEM, how favorable conditions are for
holding leverage relative to each ETF's own history. Which ETF to hold is left to a
separate momentum system; this page only addresses how much leverage conditions support.

The core measure is **effective volatility**: the risk a position held for a month
actually carries.

    effective volatility = volatility from daily returns x sqrt(VR21)

- **Volatility from daily returns** is the root-mean-square of the 21- and 63-day
  annualized readings.
- **VR21** is the Lo-MacKinlay bias-corrected variance ratio at a 21-day horizon,
  estimated over two years. Below 1, daily moves have tended to reverse within the
  month, so monthly risk is lower than daily volatility implies; above 1 it is higher.

**Favorability** = 100 minus the point-in-time percentile of effective volatility
against the ETF's own history, shown as five equal-frequency levels (very favorable
to very unfavorable).

The page also shows the leverage a volatility budget would support:
implied leverage = budget / effective volatility, for budgets of 15%, 30% and 45%
and caps of 2x, 2.5x and 3x (set by `VOL_BUDGETS` and `LEV_CAPS` in `dashboard.py`).

Each run produces two pages:

- **`site/index.html`, the dashboard:** a card per ETF, favorability against
  drawdowns, the volatility-budget leverage grid and charts, the core readings
  against history, and a point-in-time check of favorability against later outcomes.
- **`site/report.html`, the detail report:** every reading with its history,
  momentum and breadth for context (not part of favorability), a chart explorer,
  breadth by sector and country, the leverage simulation over 3 months to 5 years,
  favorability levels, bands and each metric checked against later returns,
  recent level changes, and how far back every data series goes.

**The full methodology and every caveat are written into both pages**, under
"Methodology and caveats", using the actual numbers from each run. Read that
section before relying on anything the page shows. It is a monitoring tool, not
investment advice.

## Set it up on GitHub

1. Create a repository (public, unless you have a paid plan) and add these files,
   keeping the folder layout (`.github/workflows/daily.yml` must stay at that path).
2. In the repository, open **Settings > Pages** and set **Source** to
   **GitHub Actions**.
3. Open the **Actions** tab, choose **Daily leverage dashboard**, and click
   **Run workflow** to build the first page. The page address appears on the
   finished run and under Settings > Pages.

After that it runs by itself at about 1am Pacific time, Tuesday to Saturday, so each
morning's page reflects the previous trading day's close. The workflow holds that
local time across daylight saving changes.

Things to know:

- On a free GitHub account, Pages only works for public repositories, so both the
  code and the page are public. A private repository needs a paid plan, and the
  published page is still reachable by anyone with the address.
- Each run appends that day's readings to `data/history.csv` and saves both pages
  in the `site` folder, then commits them. This builds a true point-in-time
  record, and the regular commits also stop GitHub from pausing the schedule,
  which it does after 60 days without repository activity.
- GitHub shows an HTML file in the repository as source code. To see a saved page
  rendered, download it and open it in a browser, or use the published site.
- If Yahoo Finance cannot supply the three core ETFs, or the newest data is more
  than a week old, the run fails and the previous page stays up. Missing
  secondary data (backfill funds, a country ETF, the T-bill rate) does not stop
  the run; it is listed in a "Data notes" box at the top of the page.
- Scheduled runs can start several minutes late when GitHub is busy.

## Run it yourself

```
pip install -r requirements.txt
python dashboard.py --out site               # live data
python dashboard.py --out site --synthetic   # made-up data, works offline
python -m pytest -q tests                    # check the calculations
```

Open `site/index.html` or `site/report.html` in a browser. The synthetic mode
stamps a banner on both pages and never writes to the readings log. Use a
different `--out` folder for test runs so the saved pages in `site` are not overwritten. Charts load the Plotly library from a
public CDN, so they need an internet connection; the tables do not.

## Files

| File | Purpose |
|---|---|
| `dashboard.py` | Everything: data fetch, calculations, and the page template |
| `tests/test_metrics.py` | Checks each calculation against a case with a known answer |
| `.github/workflows/daily.yml` | The daily schedule and publishing steps |
| `data/history.csv` | One row per ETF per day, appended on each run |
| `site/index.html` | The latest dashboard page |
| `site/report.html` | The latest detail report |
| `site/latest.json` | Today's readings in machine-readable form |

## Changing things

All settings are at the top of `dashboard.py`: window lengths, the variance-ratio
horizon and window (`VR_Q`, `VR_WINDOW`), volatility budgets and caps, the level
cut points, the leverage levels used in simulations, financing spread and fee, and
the breadth component lists.

**Breadth weights** (`ASSETS[...]["breadth"]`) are approximate, fixed, hand-set
index weights. Update them from the fund providers' fact sheets if you use the
index-weighted breadth figure.

## Design choices worth knowing about

- **No forecasting in the core measure.** Momentum and breadth are context only.
- **The variance ratio is a risk adjustment, not a score in its own right.** Above 1
  also means trends tend to persist, which helps a trend-following position; that is
  a statement about returns, so it is shown but not scored.
- **SPY breadth uses the eleven sector ETFs, not individual stocks.** Rebuilding
  stock-level breadth from today's index members would leave out companies that
  failed or were removed, which biases the historical percentile.
- **History before the ETFs existed is backfilled** from Vanguard index mutual
  funds: a fitted Europe + Pacific blend (VEURX, VPACX) for EFA before August 2001,
  and VEIEX for EEM before April 2003. Stale pricing in that era inflates serial
  correlation, so the variance ratio, and therefore effective volatility and
  favorability, start only once two years of ETF prices exist.
- **Percentiles are point-in-time** everywhere, so today's reading and the
  historical check use the same calculation.
