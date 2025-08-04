import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# API keys from secrets or env
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

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
        f"&outputsize=compact&apikey={ALPHAVANTAGE_API_KEY}"
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
    df["signal"] = 0
    df.loc[(df["SMA_fast"] > df["SMA_slow"]) & (df["SMA_fast"].shift(1) <= df["SMA_slow"].shift(1)), "signal"] = 1
    df.loc[(df["SMA_fast"] < df["SMA_slow"]) & (df["SMA_fast"].shift(1) >= df["SMA_slow"].shift(1)), "signal"] = -1
    return df

st.title("📈 Forex Dashboard: 15-min Intraday with Alpha Vantage Daily Fallback")

if not (TWELVE_API_KEY and ALPHAVANTAGE_API_KEY):
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

        # Debug info
        st.write("📊 Data snapshot:")
        st.write(df.tail(10))
        st.write(f"✅ Buy signals: {len(df[df['signal'] == 1])} | 🚫 Sell signals: {len(df[df['signal'] == -1])}")

        # Streamlit native line chart preview
        st.subheader("Quick Price + SMA preview (Streamlit chart)")
        st.line_chart(df[["close", "SMA_fast", "SMA_slow"]].dropna())

        # Matplotlib chart with buy/sell markers
        st.subheader("Detailed Price + SMA with Buy/Sell Signals (Matplotlib)")
        plt.figure(figsize=(10,5))
        plt.plot(df.index, df["close"].dropna(), label="Close")
        plt.plot(df.index, df["SMA_fast"].dropna(), label=f"SMA {sma_fast_period}")
        plt.plot(df.index, df["SMA_slow"].dropna(), label=f"SMA {sma_slow_period}")

        buys = df[df["signal"] == 1]
        sells = df[df["signal"] == -1]
        plt.scatter(buys.index, buys["close"], marker="^", color="green", label="Buy Signal", s=100)
        plt.scatter(sells.index, sells["close"], marker="v", color="red", label="Sell Signal", s=100)

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"{symbol} Close Price and SMA Signals")
        plt.grid(True)
        plt.tight_layout()

        st.pyplot(plt)

        st.subheader("RSI")
        st.line_chart(df["RSI"].dropna())
    
