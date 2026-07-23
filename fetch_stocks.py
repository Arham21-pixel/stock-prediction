"""
fetch_stocks.py
===============
Downloads the complete NSE equity list and saves it as nse_stocks.json.
Run this once manually, or it is called automatically by app.py on startup.

Source: NSE India public CSV
"""

import requests, json, io, os
import pandas as pd

NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
OUT_FILE = "nse_stocks.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_and_save():
    print("Downloading NSE equity list...")
    try:
        resp = requests.get(NSE_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        # Columns are: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, ...
        # Keep only EQ series (equity, not ETF / debt etc.)
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        stocks = []
        sym_col  = "SYMBOL"
        name_col = " NAME OF COMPANY" if " NAME OF COMPANY" in df.columns else "NAME OF COMPANY"

        for _, row in df.iterrows():
            sym  = str(row[sym_col]).strip()
            name = str(row[name_col]).strip()
            if sym and name:
                stocks.append({
                    "symbol": sym + ".NS",
                    "name"  : name,
                    "exchange": "NSE",
                })

        # Sort by symbol
        stocks.sort(key=lambda x: x["symbol"])

        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(stocks, f, ensure_ascii=False)

        print(f"Saved {len(stocks)} NSE stocks to {OUT_FILE}")
        return stocks

    except Exception as e:
        print(f"WARNING: Could not fetch NSE stock list: {e}")
        print("Falling back to built-in popular stocks list.")
        return []

def load_stocks():
    """Load from cache, or fetch fresh if cache missing."""
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} stocks from {OUT_FILE}")
        return data
    return fetch_and_save()

if __name__ == "__main__":
    stocks = fetch_and_save()
    print(f"Done. Example: {stocks[:3]}")
