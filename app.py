import os
import requests
import pandas as pd
import streamlit as st

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")
symbol, target = "EUR", "USD"

@st.cache_data
def fetch_daily_data():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={symbol}&to_symbol={target}"
        f"&outputsize=compact&apikey={API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get("Time Series FX (Daily)", {})
    if not data:
        st.warning("⚠️ No data returned. Check API key / usage limits.")
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
    return df

def backtest_sma_crossover(df):
    df = df.copy()
    df["position"] = 0
    df.loc[df["SMA_fast"] > df["SMA_slow"], "position"] = 1  # long
    df.loc[df["SMA_fast"] < df["SMA_slow"], "position"] = 0  # flat

    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]

    df["equity_curve"] = (1 + df["strategy_returns"].fillna(0)).cumprod()

    total_return = df["equity_curve"].iloc[-1] - 1
    max_drawdown = ((df["equity_curve"].cummax() - df["equity_curve"]) / df["equity_curve"].cummax()).max()

    return df, total_return, max_drawdown

st.title("📈 Forex Dashboard: EUR/USD with SMA, RSI & Backtest")

if API_KEY:
    df = fetch_daily_data()
    if not df.empty:
        # Sidebar sliders
        sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
        sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

        df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

        st.subheader("Price + SMA")
        st.line_chart(df[["close", "SMA_fast", "SMA_slow"]])

        st.subheader("RSI")
        st.line_chart(df["RSI"])

        # Backtest
        df_bt, total_ret, max_dd = backtest_sma_crossover(df)

        st.subheader("Backtest Equity Curve")
        st.line_chart(df_bt["equity_curve"])

        st.markdown(f"**Total Return:** {total_ret:.2%}")
        st.markdown(f"**Max Drawdown:** {max_dd:.2%}")

        st.subheader("Latest Data")
        st.dataframe(df.tail(10))
    else:
        st.warning("No data to display.")
else:
    st.warning("⚠️ Please set ALPHAVANTAGE_API_KEY as env var or Streamlit secret.")
    
