import os
import requests
import pandas as pd
import streamlit as st

# API keys - set these in Streamlit secrets or environment variables
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

symbol = "EUR/USD"
twelve_interval = "15min"
alpha_symbol_from = "EUR"
alpha_symbol_to = "USD"

@st.cache_data(ttl=600)
def fetch_twelve_data():
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={twelve_interval}&apikey={TWELVE_API_KEY}&format=JSON&outputsize=500"
    )
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df["close"] = pd.to_numeric(df["close"])
    return df

@st.cache_data(ttl=600)
def fetch_alpha_daily_data():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={alpha_symbol_from}&to_symbol={alpha_symbol_to}"
        f"&outputsize=compact&apikey={ALPHA_API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get("Time Series FX (Daily)", {})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={"4. close": "close"}, inplace=True)
    return df

def compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period):
    df = df.copy()
    df["SMA_fast"] = df["close"].rolling(sma_fast_period).mean()
    df["SMA_slow"] = df["close"].rolling(sma_slow_period).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))

    # Signal: 1 = SMA_fast crossed above SMA_slow (buy), -1 = crossed below (sell), 0 = no signal
    df["signal"] = 0
    df.loc[(df["SMA_fast"] > df["SMA_slow"]) & (df["SMA_fast"].shift(1) <= df["SMA_slow"].shift(1)), "signal"] = 1
    df.loc[(df["SMA_fast"] < df["SMA_slow"]) & (df["SMA_fast"].shift(1) >= df["SMA_slow"].shift(1)), "signal"] = -1

    return df

st.title("📈 Forex Dashboard: 15-min Intraday with Alpha Vantage Daily Fallback")

if not (TWELVE_API_KEY and ALPHA_API_KEY):
    st.warning("⚠️ Please set both TWELVE_API_KEY and ALPHAVANTAGE_API_KEY in secrets or env variables.")
else:
    df = fetch_twelve_data()
    source_used = "Twelve Data (15min Intraday)"

    if df.empty:
        st.info("No intraday data from Twelve Data, falling back to Alpha Vantage daily.")
        df = fetch_alpha_daily_data()
        source_used = "Alpha Vantage Daily"

    if df.empty:
        st.error("No data available from either source.")
    else:
        # Sidebar sliders
        sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
        sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

        df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

        st.subheader(f"Data Source: {source_used}")

        st.subheader("Price + SMA with Signals")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10,5))
        plt.plot(df.index, df["close"], label="Close")
        plt.plot(df.index, df["SMA_fast"], label=f"SMA {sma_fast_period}")
        plt.plot(df.index, df["SMA_slow"], label=f"SMA {sma_slow_period}")

        # Plot buy
        
