import yfinance as yf
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

tickers = [
    "^NSEI",
    "^NSEBANK",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS"
]

all_data = []

for ticker in tickers:

    print("Downloading:", ticker)

    df = yf.Ticker(ticker).history(start="2000-01-01")

    df = df.reset_index()

    # keep only needed columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df["ticker"] = ticker

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

data.to_csv("data/raw/market_data.csv", index=False)

print("Market data downloaded successfully")