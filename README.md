# Leverage Environment dashboard

Daily pages that show how favorable conditions are for holding leverage in each
asset, relative to that asset's own history:

- **Equities:** SPY, EFA and EEM
- **Gold, Treasuries and Bitcoin:** GLD, IEF, TLT and BTC-USD

Which asset to hold is left to a separate momentum system; these pages only address
how much leverage conditions support.

The core measure is **effective volatility**: the risk a position held for a month
actually carries.

    effective volatility = volatility from daily returns x sqrt(VR21)

- **Volatility from daily returns** is the root-mean-square of the 1-month and
  3-month annualized readings (21 and 63 trading days).
- **VR21** is the Lo-MacKinlay bias-corrected variance ratio at a 1-month horizon,
  estimated over two years. Below 1, daily moves have tended to reverse within the
  month, so monthly risk is lower than daily volatility implies; above 1 it is higher.
  A 3-month variance ratio is shown alongside it for context only.

**Bitcoin trades every day, so its windows count calendar days** (30 and 91-day
volatility, a 30-day variance ratio over 730 days) and annualize with 365 days. The
other assets use trading days and 252. Both cover the same spans of time.

**Favorability** = 100 minus the point-in-time percentile of effective volatility
against the asset's own history, shown as five equal-frequency levels (very favorable
to very unfavorable). "Own history" is an expanding window: everything from 1998 (or
the asset's first data) up to each date. A lasting shift in an asset's normal
volatility, such as bitcoin's long decline or Treasuries' rise since 2022, is not
absorbed; the pages carry a caution about this.

The page also shows the leverage a volatility budget would support:
implied leverage = budget / effective volatility, for budgets of 15%, 30% and 45%
and caps of 2x, 2.5x and 3x (set by `VOL_BUDGETS` and `LEV_CAPS` in `dashboard.py`),
plus the leverage that would bring each asset to its own median effective volatility.

Each run produces two dashboards, each with its own detail report:

- **`site/index.html` (equities) and `site/other.html` (gold, Treasuries, bitcoin):**
  a sortable summary table, favorability against drawdowns, the volatility-budget
  leverage grid and charts, volatility drag, the core readings against history,
  history charts, and a point-in-time check of favorability against later outcomes.
- **`site/report.html` and `site/other-report.html`, the detail reports:** every
  reading with its history, momentum (and breadth, for equities) as context, a chart
  explorer, the leverage simulation over 3 months to 5 years, favorability levels,
  bands and each metric checked against later returns, recent level changes, and
  how far back every data series goes.

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
- If Yahoo Finance cannot supply the three equity ETFs, or the newest data is more
  than a week old, the run fails and the previous pages stay up. A missing or stale
  gold, Treasury or bitcoin series is left off its page with a note. Missing
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

Open any of the pages in `site` in a browser. The synthetic mode
stamps a banner on both pages and never writes to the readings log. Use a
different `--out` folder for test runs so the saved pages in `site` are not overwritten. Charts load the Plotly library from a
public CDN, so they need an internet connection; the tables do not.

## Files

| File | Purpose |
|---|---|
| `dashboard.py` | Everything: data fetch, calculations, and the page template |
| `tests/test_metrics.py` | Checks each calculation against a case with a known answer |
| `.github/workflows/daily.yml` | The daily schedule and publishing steps |
| `data/history.csv` | One row per asset per reading date, appended on each run |
| `site/index.html`, `site/report.html` | Equities dashboard and detail report |
| `site/other.html`, `site/other-report.html` | Gold, Treasuries and bitcoin dashboard and detail report |
| `site/latest.json` | Today's readings in machine-readable form |

## Changing things

All settings are at the top of `dashboard.py`: window lengths, the variance-ratio
horizons and window (`VR_Q`, `VR_Q_LONG`, `VR_WINDOW`), volatility budgets and caps,
the level cut points, the leverage levels used in simulations, financing spread and
fee, the assets and their backfill funds (`ASSETS`), which page each asset is on
(`GROUPS`), and the breadth component lists. Windows are set in trading days and
converted for calendar-day assets.

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
  favorability, start only once two years of ETF prices exist. IEF and TLT are
  backfilled before July 2002 from Vanguard Treasury funds (VFITX, VUSTX), with
  returns scaled to match each ETF's interest-rate sensitivity; Treasury funds are
  not stale-priced, so their variance ratio is used. GLD and bitcoin are not
  backfilled.
- **Percentiles are point-in-time** everywhere, so today's reading and the
  historical check use the same calculation.
