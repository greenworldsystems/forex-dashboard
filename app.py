import os
import requests
import pandas as pd
import streamlit as st

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")
symbol, target = "EUR", "USD"
interval = "15min"

@st.cache_data(ttl=600)
def fetch_intraday_data():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY&from_symbol={symbol}&to_symbol={target}"
        f"&interval={interval}&outputsize=compact&apikey={API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get(f"Time Series FX ({interval})", {})
    if not data:
        st.warning("⚠️ No intraday data returned. Check API key / limits.")
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

st.title("📈 Forex Dashboard: EUR/USD Intraday 15min + Signals")

if API_KEY:
    df = fetch_intraday_data()
    if not df.empty:
        sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
        sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

        df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

        st.subheader("Price + SMA with Signals")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10,5))
        plt.plot(df.index, df["close"], label="Close")
        plt.plot(df.index, df["SMA_fast"], label=f"SMA {sma_fast_period}")
        plt.plot(df.index, df["SMA_slow"], label=f"SMA {sma_slow_period}")

        # Plot buy signals
        buys = df[df["signal"] == 1]
        plt.scatter(buys.index, buys["close"], marker="^", color="green", label="Buy Signal", s=100)

        # Plot sell signals
        sells = df[df["signal"] == -1]
        plt.scatter(sells.index, sells["close"], marker="v", color="red", label="Sell Signal", s=100)

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("EUR/USD Close Price and SMA Signals")
        plt.grid(True)
        st.pyplot(plt)

        st.subheader("RSI")
        st.line_chart(df["RSI"])

        st.subheader("Latest Data")
        st.dataframe(df.tail(10))

    else:
        st.warning("No intraday data available.")
else:
    st.warning("⚠️ Please set your ALPHAVANTAGE_API_KEY in Streamlit secrets or environment variables.")
    
