# Leverage Environment dashboard

A daily page that asks one question for SPY, EFA and EEM: are current conditions
friendly or hostile to holding daily-reset 2x or 3x exposure?

It tracks four things for each ETF and shows every reading with its percentile
against that ETF's own history since 1998:

| Metric | What is measured |
|---|---|
| Momentum | 3-month and 12-month total return (dividend- and split-adjusted) |
| Volatility | 21- and 63-day realized volatility, plus the yearly drag implied at 2x and 3x |
| Serial correlation | Variance ratios over 3-month and 1-year windows, and 1-year lag-1 autocorrelation |
| Breadth | Share of sector ETFs (SPY) or country ETFs (EFA, EEM) with a positive 3- and 12-month return |

It also simulates how a daily-reset 2x and 3x fund actually compounded over the
past year, and checks the composite label against what followed historically.

Each run produces two pages:

- **`site/index.html`, the dashboard:** today's readings, the comparison grid, a chart of each
  ETF's composite score against its drawdown, and history charts for every metric.
- **`site/report.html`, the detail report:** every reading with its history, a
  chart explorer for any series (raw value or percentile), breadth by individual
  sector and country, the leverage simulation over 3 months to 5 years, the
  label, score bands and each metric checked against later returns at several
  horizons, recent label changes, and how far back every data series goes.

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

After that it runs by itself at 22:17 UTC on weekdays, after the US close.

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

All settings are at the top of `dashboard.py`: window lengths, the leverage
levels, financing spread and fee, the score weights and thresholds, and the
breadth component lists.

Two settings deserve attention before you rely on them:

- **Breadth weights** (`ASSETS[...]["breadth"]`) are approximate, fixed, hand-set
  index weights. Update them from the fund providers' fact sheets if you use the
  index-weighted breadth figure.
- **Score weights** (`SCORE_WEIGHTS`, `FAVORABLE_AT`, `HOSTILE_AT`) are a
  judgment call, not a fitted model. The page's point-in-time check shows how the
  resulting label has lined up with later leveraged returns.

## Design choices worth knowing about

- **No momentum gate.** Momentum is reported as raw values and percentiles only.
- **Breadth uses the same 3- and 12-month windows as momentum**, with no moving-average rule.
- **SPY breadth uses the eleven sector ETFs, not individual stocks.** Rebuilding
  stock-level breadth from today's index members would leave out companies that
  failed or were removed, which biases the historical percentile.
- **History before the ETFs existed is backfilled** from Vanguard index mutual
  funds: a fitted Europe + Pacific blend (VEURX, VPACX) for EFA before August 2001,
  and VEIEX for EEM before April 2003. That era's daily data suffers from stale
  pricing, so serial-correlation readings touching it are excluded from ranking.
- **Percentiles are point-in-time** everywhere, so today's reading and the
  historical check use the same calculation.
