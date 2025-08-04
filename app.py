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

# API Keys
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

st.title("📈 Forex Dashboard: SMA, RSI, ML + Backtest + Live Feed")

# Sidebar - Currency pair and parameters
currency_pairs = ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY"]
selected_pair = st.sidebar.selectbox("Select Currency Pair", currency_pairs)
base, quote = selected_pair.split("/")

sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# ML hyperparams sliders
max_depth = st.sidebar.slider("Random Forest Max Depth", 2, 20, 5)
n_estimators = st.sidebar.slider("Random Forest Number of Trees", 10, 200, 50)

@st.cache_data
def fetch_intraday_data(pair, interval="15min"):
    symbol = pair.replace("/", "")
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={pair}&interval={interval}&apikey={TWELVE_API_KEY}&format=JSON&outputsize=500"
    )
    r = requests.get(url)
    data = r.json()
    if data.get("status") != "ok":
        st.warning(f"⚠️ No intraday data returned. Check API key / usage limits.\nMessage: {data.get('message')}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "date", "close": "close", "open": "open", "high": "high", "low": "low"})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
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

def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["SMA_diff"] = df["SMA_fast"] - df["SMA_slow"]
    df["RSI_diff"] = df["RSI"].diff()
    df.dropna(inplace=True)
    X = df[["return", "SMA_diff", "RSI", "RSI_diff"]]
    return X

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

# Main
df = fetch_intraday_data(selected_pair)
if df.empty:
    st.stop()

df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

X = prepare_features(df)

# Target: 1 if next close price is higher than current, else 0
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df.dropna(inplace=True)
X = X.loc[df.index]

X_train, X_test, y_train, y_test = train_test_split(X, df["target"], test_size=0.3, shuffle=False)

# Train models
model_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
model_rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42).fit(X_train, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Backtest SMA crossover
df_bt, total_return, max_drawdown = backtest_sma_crossover(df)

st.markdown(f"Data Source: Twelve Data (15min Intraday) for {selected_pair}")
st.markdown(f"Logistic Regression Accuracy: {acc_lr:.2%}")
st.text(f"Confusion Matrix (Logistic Regression):\n{cm_lr}")
st.markdown(f"Random Forest Accuracy: {acc_rf:.2%}")
st.text(f"Confusion Matrix (Random Forest):\n{cm_rf}")

st.markdown("🌿 Random Forest Feature Importance")
feat_imp = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feat_imp)

st.markdown("📊 Backtest Equity Curve (SMA Crossover)")
st.line_chart(df_bt["equity_curve"])
st.markdown(f"Total Return: {total_return:.2%}")
st.markdown(f"Max Drawdown: {max_drawdown:.2%}")

# --- Safe trading simulation ---
if len(X_test) > 0:
    balance, total_profit, num_trades, win_rate, trades_df = trading_simulator(df.loc[X_test.index], pd.Series(y_pred_rf, index=X_test.index))

    st.markdown("🧪 Trading Simulator with Stop Loss & Take Profit")
    st.markdown(f"Final Balance: ${balance:,.2f}")
    st.markdown(f"Total Profit: ${total_profit:,.2f}")
    st.markdown(f"Number of Trades: {num_trades}")
    st.markdown(f"Win Rate: {win_rate:.2%}")

    if not trades_df.empty:
        st.markdown("📝 Trades Log")
        st.dataframe(trades_df)
else:
    st.warning("Not enough data to run trading simulation.")

# Price + SMA + Buy/Sell Signals Plot (Random Forest buy = 1, sell = 0)
buy_signals = df.loc[X_test.index][y_pred_rf == 1]
sell_signals = df.loc[X_test.index][y_pred_rf == 0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close Price"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], mode="lines", name=f"SMA {sma_fast_period}"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], mode="lines", name=f"SMA {sma_slow_period}"))

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["close"], mode="markers", name="Buy Signals",
                         marker=dict(color="green", size=8, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["close"], mode="markers", name="Sell Signals",
                         marker=dict(color="red", size=8, symbol="triangle-down")))

fig.update_layout(title=f"Price + SMA + Buy/Sell Signals ({selected_pair})",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# RSI plot
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
fig_rsi.update_layout(title="RSI", xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0, 100]))
st.plotly_chart(fig_rsi, use_container_width=True)

# Placeholder for live feed (to be implemented)
st.markdown("🔴 Live Feed (placeholder) - coming soon!")
    
