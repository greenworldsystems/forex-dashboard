import os
import requests
import pandas as pd
import streamlit as st

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")
symbol, target = "EUR", "USD"
interval = "5min"  # options: 1min, 5min, 15min, 30min, 60min

@st.cache_data
def fetch_intraday_data():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY&from_symbol={symbol}&to_symbol={target}"
        f"&interval={interval}&outputsize=compact&apikey={API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get(f"Time Series FX ({interval})", {})
    if not data:
        st.warning("⚠️ No data returned. Check API key / usage limits.")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={"4. close": "close"}, inplace=True)
    return df

st.title("📈 Forex Dashboard: EUR/USD (Intraday)")

if API_KEY:
    df = fetch_intraday_data()
    if not df.empty:
        # Add indicators
        df["SMA_fast"] = df["close"].rolling(5).mean()
        df["SMA_slow"] = df["close"].rolling(20).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        RS = gain / loss
        df["RSI"] = 100 - (100 / (1 + RS))

        # Plot
        st.subheader("Price + SMA")
        st.line_chart(df[["close", "SMA_fast", "SMA_slow"]])

        st.subheader("RSI")
        st.line_chart(df["RSI"])

        st.subheader("Latest data")
        st.dataframe(df.tail(10))
    else:
        st.warning("No data to display.")
else:
    st.warning("⚠️ Please set ALPHAVANTAGE_API_KEY as env var or Streamlit secret.")

