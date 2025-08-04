import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load API keys from Streamlit secrets or environment variables
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

# Supported currency pairs for selector
CURRENCY_PAIRS = ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY"]

# Streamlit app title
st.title("📈 Forex Dashboard: SMA, RSI, ML + Backtest + Live Feed")

# Currency pair selection
pair = st.sidebar.selectbox("Select Currency Pair", CURRENCY_PAIRS)
base, quote = pair.split("/")

# Indicator sliders
sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# ML model parameters sliders
rf_trees = st.sidebar.slider("Random Forest Trees", 10, 200, 50)
rf_depth = st.sidebar.slider("Random Forest Max Depth", 3, 20, 5)

@st.cache_data(show_spinner=False)
def fetch_intraday_data(symbol=f"{base}/{quote}", interval="15min"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVE_API_KEY}&format=JSON&outputsize=500"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.warning("⚠️ No intraday data returned. Check API key / usage limits.")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df

def compute_indicators(df):
    df = df.copy()
    df["SMA_fast"] = df["close"].rolling(sma_fast_period).mean()
    df["SMA_slow"] = df["close"].rolling(sma_slow_period).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    return df

def prepare_features(df):
    df = df.copy()
    df["SMA_diff"] = df["SMA_fast"] - df["SMA_slow"]
    df["RSI_diff"] = df["RSI"].diff()
    df["returns"] = df["close"].pct_change()
    df.dropna(inplace=True)

    feature_cols = ["SMA_fast", "SMA_slow", "RSI", "SMA_diff", "RSI_diff", "returns"]
    X = df[feature_cols]
    y = (df["returns"].shift(-1) > 0).astype(int)  # Predict if next period close is higher (1) or not (0)
    y = y.loc[X.index]
    return X, y, df

def backtest_strategy(df):
    df = df.copy()
    df["position"] = 0
    df.loc[df["SMA_fast"] > df["SMA_slow"], "position"] = 1
    df.loc[df["SMA_fast"] <= df["SMA_slow"], "position"] = 0
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    df["equity_curve"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    total_return = df["equity_curve"].iloc[-1] - 1
    max_drawdown = ((df["equity_curve"].cummax() - df["equity_curve"]) / df["equity_curve"].cummax()).max()
    return df, total_return, max_drawdown

def trading_simulator(df, signals, sl_pct=0.005, tp_pct=0.01):
    balance = 10000.0
    position = None
    entry_price = 0
    trade_log = []
    for i in range(len(signals)):
        signal = signals.iloc[i]
        price = df.loc[signal.name, "close"]
        if signal == 1 and position is None:
            position = "long"
            entry_price = price
            trade_log.append({"entry_time": signal.name, "entry_price": price, "type": "buy"})
        elif signal == 0 and position == "long":
            exit_price = price
            profit = exit_price - entry_price
            balance += profit
            trade_log[-1].update({"exit_time": signal.name, "exit_price": exit_price, "profit": profit})
            position = None
        elif position == "long":
            # Check stop loss or take profit
            if price <= entry_price * (1 - sl_pct) or price >= entry_price * (1 + tp_pct):
                exit_price = price
                profit = exit_price - entry_price
                balance += profit
                trade_log[-1].update({"exit_time": signal.name, "exit_price": exit_price, "profit": profit})
                position = None
    trades_df = pd.DataFrame(trade_log)
    total_profit = trades_df["profit"].sum() if not trades_df.empty else 0
    win_rate = (trades_df["profit"] > 0).mean() if not trades_df.empty else 0
    return balance, total_profit, len(trades_df), win_rate, trades_df

# Main app execution starts here
if not TWELVE_API_KEY:
    st.error("⚠️ Please set TWELVE_API_KEY as env var or Streamlit secret.")
    st.stop()

df = fetch_intraday_data()
if df.empty:
    st.warning("No data to display.")
    st.stop()

df = compute_indicators(df)
X, y, df = prepare_features(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Logistic Regression model
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Random Forest model
model_rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
feat_imp = model_rf.feature_importances_

# Backtest SMA strategy
df_bt, total_ret, max_dd = backtest_strategy(df)

# Trading simulator using Random Forest signals
def trading_simulator(df, signals, sl_pct=0.005, tp_pct=0.01):
    balance = 10000.0
    position = None
    entry_price = 0
    trade_log = []
    for timestamp, signal in signals.items():
        price = df.loc[timestamp, "close"]
        if signal == 1 and position is None:
            position = "long"
            entry_price = price
            trade_log.append({"entry_time": timestamp, "entry_price": price, "type": "buy"})
        elif signal == 0 and position == "long":
            exit_price = price
            profit = exit_price - entry_price
            balance += profit
            trade_log[-1].update({"exit_time": timestamp, "exit_price": exit_price, "profit": profit})
            position = None
        elif position == "long":
            # Check stop loss or take profit
            if price <= entry_price * (1 - sl_pct) or price >= entry_price * (1 + tp_pct):
                exit_price = price
                profit = exit_price - entry_price
                balance += profit
                trade_log[-1].update({"exit_time": timestamp, "exit_price": exit_price, "profit": profit})
                position = None
    trades_df = pd.DataFrame(trade_log)
    total_profit = trades_df["profit"].sum() if not trades_df.empty else 0
    win_rate = (trades_df["profit"] > 0).mean() if not trades_df.empty else 0
    return balance, total_profit, len(trades_df), win_rate, trades_df

# Display metrics
st.markdown(f"Data Source: Twelve Data (15min Intraday) for {pair}")
st.markdown(f"Logistic Regression Accuracy: {acc_lr:.2%}")
st.text("Confusion Matrix (Logistic Regression):")
st.write(cm_lr)
st.markdown(f"Random Forest Accuracy: {acc_rf:.2%}")
st.text("Confusion Matrix (Random Forest):")
st.write(cm_rf)

st.subheader("🌿 Random Forest Feature Importance")
feat_imp_df = pd.DataFrame({"feature": X.columns, "importance": feat_imp}).sort_values(by="importance", ascending=False)
st.bar_chart(feat_imp_df.set_index("feature"))

st.subheader("📊 Backtest Equity Curve (SMA Crossover)")
st.line_chart(df_bt["equity_curve"])
st.markdown(f"Total Return: {total_ret:.2%}")
st.markdown(f"Max Drawdown: {max_dd:.2%}")

st.subheader("🧪 Trading Simulator with Stop Loss & Take Profit")
st.markdown(f"Final Balance: ${balance:,.2f}")
st.markdown(f"Total Profit: ${total_profit:,.2f}")
st.markdown(f"Number of Trades: {num_trades}")
st.markdown(f"Win Rate: {win_rate:.2%}")

st.subheader("📈 Price + SMA + Buy/Sell Signals (Random Forest)")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], mode="lines", name=f"SMA {sma_fast_period}"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], mode="lines", name=f"SMA {sma_slow_period}"))

# Corrected buy/sell signal plotting using test subset indices
buy_signals = X_test.loc[y_pred_rf == 1]
sell_signals = X_test.loc[y_pred_rf == 0]

fig_price.add_trace(go.Scatter(
    x=buy_signals.index,
    y=df.loc[buy_signals.index, "close"],
    mode="markers",
    name="Buy Signals",
    marker=dict(symbol="triangle-up", color="green", size=10)
))
fig_price.add_trace(go.Scatter(
    x=sell_signals.index,
    y=df.loc[sell_signals.index, "close"],
    mode="markers",
    name="Sell Signals",
    marker=dict(symbol="triangle-down", color="red", size=10)
))

st.plotly_chart(fig_price, use_container_width=True)

st.subheader("📉 RSI")
st.line_chart(df["RSI"])

st.subheader("🔴 Live Feed (placeholder) - coming soon!")

st.subheader("📝 Trades Log")
if not trades_df.empty:
    st.dataframe(trades_df)
else:
    st.write("No trades executed in simulation.")

