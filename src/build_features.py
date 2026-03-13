import pandas as pd
import pandas_ta as ta
import os

def build_features():

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/raw/market_data.csv", low_memory=False)

    # remove rows with missing ticker
    df = df.dropna(subset=["ticker"])

    all_features = []

    tickers = df["ticker"].unique()

    for ticker in tickers:

        print("Processing:", ticker)

        temp = df[df["ticker"] == ticker].copy()

        temp.sort_values("Date", inplace=True)

        # convert price columns to numeric
        cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in cols:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

        # technical indicators
        temp["RSI"] = ta.rsi(temp["Close"], length=14)
        temp["EMA20"] = ta.ema(temp["Close"], length=20)
        temp["EMA50"] = ta.ema(temp["Close"], length=50)

        temp.ta.bbands(close="Close", length=20, append=True)

        temp["volatility_10"] = temp["Close"].pct_change().rolling(10).std()

        # crash label
        temp["future_return"] = temp["Close"].shift(-5) / temp["Close"] - 1
        temp["crash"] = (temp["future_return"] < -0.03).astype(int)

        temp.dropna(inplace=True)

        all_features.append(temp)

    final_df = pd.concat(all_features)

    final_df.to_csv("data/processed/features.csv", index=False)

    print("Features created successfully")


if __name__ == "__main__":
    build_features()