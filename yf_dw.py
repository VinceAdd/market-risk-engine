import os
import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inbound")

FFILL_LIMIT = 5
NAN_THRESHOLD = 0.8 # drop tickers missing more than 80 % of rows
RISKY_FILL_PCT = 5.0 # warn when > 5 % of a ticker's history was filled


def _download_and_ffill(tickers, period, interval):
    """Download closing prices for *tickers*, forward-fill gaps, and print a
    diagnostic summary.  Returns a wide DataFrame (dates × tickers)."""

    data = yf.download(tickers, period=period, interval=interval, group_by="ticker")

    # Keep only 'Close' and flatten the MultiIndex
    close_prices = data.loc[:, pd.IndexSlice[:, "Close"]]
    close_prices.columns = close_prices.columns.droplevel(1)

    # Drop tickers that are missing too many rows
    min_rows = int(len(close_prices) * NAN_THRESHOLD)
    before_cols = set(close_prices.columns)
    close_prices = close_prices.dropna(thresh=min_rows, axis=1)
    dropped = before_cols - set(close_prices.columns)
    print(f"\nDropped tickers ({len(dropped)}): {sorted(dropped)}")

    # Forward-fill (limit to avoid propagating stale prices too far)
    ffilled = close_prices.ffill(limit=FFILL_LIMIT)

    # Diagnostics
    nans_before = close_prices.isna().sum()
    nans_after = ffilled.isna().sum()
    filled_count = nans_before - nans_after
    filled_pct = (filled_count / len(close_prices)).round(4)

    summary = (
        pd.DataFrame({
            "Filled_Count": filled_count,
            "Filled_%": filled_pct,
            "Remaining_NaN": nans_after,
        })
        .loc[lambda df: df["Filled_Count"] > 0]
        .sort_values("Filled_%", ascending=False)
    )

    print("Forward-fill impact per ticker (sorted by % filled):")
    print("=" * 50)
    print(summary)
    print("\n" + "=" * 50)
    print(f"Total trading days in sample : {len(close_prices):,}")
    print(f"Total missing prices before ffill : {nans_before.sum():,}")
    print(f"Total prices filled by ffill : {filled_count.sum():,} : "
          f"{filled_pct.sum():.2%} of all cells")
    print(f"Total missing prices after ffill : {nans_after.sum():,}")
    print(f"Tickers still having NaNs : {(nans_after > 0).sum()}")
    print(f"\nTickers with >{RISKY_FILL_PCT}% of data filled "
          f"(consider reviewing or dropping):")
    print(summary.loc[summary["Filled_%"] > RISKY_FILL_PCT / 100,
                      ["Filled_Count", "Filled_%"]])

    return ffilled


def main(tickers, period = "5y", interval = "1d", out_dir = OUT_DIR):
    """Download equity prices, enrich with currency metadata, and save to CSV."""

    ffilled = _download_and_ffill(tickers, period, interval)

    # Melt to long format
    formatted_ffilled = (
        ffilled
        .reset_index()
        .melt(id_vars=["Date"], var_name="Security", value_name="Price")
    )

    # Attach currency for each ticker
    currencies = {}
    for ticker in tickers:
        try:
            currencies[ticker] = yf.Ticker(ticker).info.get("currency", "Unknown")
        except Exception:
            currencies[ticker] = "Unknown"

    formatted_ffilled["Price_Currency"] = formatted_ffilled["Security"].map(currencies)

    formatted_ffilled = formatted_ffilled.sort_values(["Date", "Security"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, "prices.csv")
    formatted_ffilled.to_csv(out_path, index=False)
    print(f"\nSaved equity prices {out_path}")


def main_fx(tickers, period = "5y", interval= "1d", out_dir = OUT_DIR):
    """Download FX rates and save to CSV."""

    ffilled = _download_and_ffill(tickers, period, interval)

    # Invert tickers where EUR is the base (ticker starts with 'EUR')
    for ticker in tickers:
        if ticker.startswith("EUR"):
            ffilled[ticker] = 1 / ffilled[ticker]

    EUR_BASED = {
        "EURUSD=X": "USD",
        "EURJPY=X": "JPY",
    }

    ffilled.rename(columns=EUR_BASED, inplace=True)
    ffilled["EUR"] = 1.0

    formated_ffilled = (
        ffilled
        .reset_index()
        .melt(id_vars=["Date"], var_name="Currency", value_name="FX_to_Report_Ccy")
        .sort_values(["Date", "Currency"])
        .reset_index(drop=True)
    )

    out_path = os.path.join(out_dir, "fx_rates.csv")
    formated_ffilled.to_csv(out_path, index=False)
    print(f"\nSaved FX prices {out_path}")


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "BNP.PA", "ASML.AS", "SHELL.AS", "7203.T"]
    FX = ["EURUSD=X", "EURJPY=X"]
    PERIOD = "5y"
    INTERVAL = "1d"

    main(TICKERS, period=PERIOD, interval=INTERVAL, out_dir=OUT_DIR)
    main_fx(FX, period=PERIOD, interval=INTERVAL, out_dir=OUT_DIR)