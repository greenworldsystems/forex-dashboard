import os
import json
import time
import threading

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Plotly for charts
import plotly.graph_objects as go

# Websocket client
import websocket

# API keys from Streamlit secrets or env
TWELVE_API_KEY = st.secrets.get("TWELVE_API_KEY") or os.getenv("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")

# Globals for WebSocket live feed
LIVE_DATA = []
WS_RUNNING = False

# Streamlit session state init
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "live_data" not in st.session_state:
    st.session_state.live_data = []

# Parameters and currency pair selector
CURRENCY_PAIRS = ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY", "AUD/USD"]
pair = st.sidebar.selectbox("Select Currency Pair", CURRENCY_PAIRS)
pair_twelve = pair.replace("/", "")  # e.g. "EURUSD"

# --- Functions for fetching historical data ---
@st.cache_data(ttl=600)
def fetch_intraday_twelve(pair_sym):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={pair_sym}&interval=15min&outputsize=500&apikey={TWELVE_API_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.warning("⚠️ No intraday data returned. Check API key / limits.")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df

@st.cache_data(ttl=600)
def fetch_daily_alpha():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={pair.split('/')[0]}&to_symbol={pair.split('/')[1]}"
        f"&outputsize=compact&apikey={ALPHAVANTAGE_API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get("Time Series FX (Daily)", {})
    if not data:
        st.warning("⚠️ No daily data returned from Alpha Vantage. Check API key / limits.")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.rename(columns={"4. close": "close"}, inplace=True)
    return df

# Indicator calculations
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

# Feature engineering for ML
def prepare_features(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["direction"] = np.where(df["returns"] > 0, 1, 0)
    df.dropna(inplace=True)
    return df

# ML models training
def train_models(df, feature_cols):
    X = df[feature_cols]
    y = df["direction"]
    split = int(len(df)*0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)

    # Predictions for whole dataset (for plotting)
    df["LR_pred"] = lr.predict(X)
    df["RF_pred"] = rf.predict(X)

    return lr_acc, rf_acc, df, rf

# Backtest
def backtest_strategy(df):
    df = df.copy()
    df["position"] = df["RF_pred"].shift(1)
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity_curve"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    total_return = df["equity_curve"].iloc[-1] - 1
    max_drawdown = ((df["equity_curve"].cummax() - df["equity_curve"]) / df["equity_curve"].cummax()).max()
    return df, total_return, max_drawdown

# --- WebSocket functions for live data ---

def on_message(ws, message):
    global LIVE_DATA
    data = json.loads(message)
    # Example data structure check for Twelve Data
    if "data" in data and data["data"]:
        tick = data["data"][0] if isinstance(data["data"], list) else data["data"]
        # tick dict: {'symbol': 'EURUSD', 'price': '1.1555', 'timestamp': '...'}
        # Convert to suitable dict
        tick_parsed = {
            "datetime": pd.to_datetime(tick.get("datetime", tick.get("timestamp", None))),
            "close": float(tick["close"]) if "close" in tick else float(tick["price"]),
            "open": float(tick.get("open", tick.get("price", 0))),
            "high": float(tick.get("high", tick.get("price", 0))),
            "low": float(tick.get("low", tick.get("price", 0))),
            "volume": float(tick.get("volume", 0)),
        }
        LIVE_DATA.append(tick_parsed)
        if len(LIVE_DATA) > 500:
            LIVE_DATA.pop(0)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    global WS_RUNNING
    WS_RUNNING = False
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket opened")
    subscribe_msg = {
        "action": "subscribe",
        "params": {
            "symbols": [pair_twelve]  # e.g. "EURUSD"
        }
    }
    ws.send(json.dumps(subscribe_msg))

def start_ws():
    global WS_RUNNING
    WS_RUNNING = True
    ws_url = "wss://ws.twelvedata.com/v1/quotes/forex"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        header={"Authorization": f"Bearer {TWELVE_API_KEY}"}
    )
    ws.run_forever()

# --- Streamlit UI ---

st.title("📈 Forex Dashboard: SMA, RSI, ML + Live Feed")

if not TWELVE_API_KEY or not ALPHAVANTAGE_API_KEY:
    st.warning("⚠️ Please set TWELVE_API_KEY and ALPHAVANTAGE_API_KEY in secrets or env variables.")
    st.stop()

# Sidebar sliders for indicators
sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# Fetch historical data: try 15-min intraday, fallback daily
df = fetch_intraday_twelve(pair_twelve)
if df.empty:
    df = fetch_daily_alpha()

if df.empty:
    st.error("No data available from APIs.")
    st.stop()

# Compute indicators
df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)
df = prepare_features(df)

# Train ML models
feature_cols = ["SMA_fast", "SMA_slow", "RSI"]
lr_acc, rf_acc, df, rf_model = train_models(df, feature_cols)

# Backtest
df_bt, total_ret, max_dd = backtest_strategy(df)

# Show ML performance & backtest
st.markdown(f"**Logistic Regression Accuracy:** {lr_acc:.2%}")
st.markdown(f"**Random Forest Accuracy:** {rf_acc:.2%}")
st.markdown(f"**Backtest Total Return:** {total_ret:.2%}")
st.markdown(f"**Backtest Max Drawdown:** {max_dd:.2%}")

# Plot price, SMA and ML predictions using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], mode="lines", name=f"SMA {sma_fast_period}"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], mode="lines", name=f"SMA {sma_slow_period}"))
fig.add_trace(go.Scatter(x=df.index, y=df["RF_pred"]*df["close"].max(), mode="markers", name="RF Buy/Sell Signal", marker=dict(color='green', size=5)))
st.plotly_chart(fig, use_container_width=True)

# RSI plot
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
st.plotly_chart(fig_rsi, use_container_width=True)

# --- Live WebSocket controls ---
col1, col2 = st.columns(2)
with col1:
    start_live = st.button("Start Live Feed")
with col2:
    stop_live = st.button("Stop Live Feed")

if start_live and st.session_state.ws_thread is None:
    st.session_state.ws_thread = threading.Thread(target=start_ws, daemon=True)
    st.session_state.ws_thread.start()
    st.success("Live feed started!")

if stop_live and st.session_state.ws_thread:
    WS_RUNNING = False
    st.session_state.ws_thread = None
    st.success("Live feed stopped!")

# Display live data sample
st.subheader("Live Data Sample (last 10 ticks)")
if LIVE_DATA:
    df_live = pd.DataFrame(LIVE_DATA).tail(10)
    st.dataframe(df_live)
else:
    st.write("No live data yet...")

# Note: You should implement live bar aggregation and indicator update in the future here.
