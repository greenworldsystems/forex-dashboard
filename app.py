import os
import requests
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

# Get API keys from env vars or Streamlit secrets
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

st.set_page_config(layout="wide")
st.title("📈 Forex Dashboard: SMA, RSI, ML + Backtest + Live Feed")

# Sidebar controls
pair = st.sidebar.selectbox(
    "Select currency pair",
    ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY", "AUD/USD"]
)

sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

rf_n_estimators = st.sidebar.slider("Random Forest Trees (n_estimators)", 10, 200, 50, step=10)
rf_max_depth = st.sidebar.slider("Random Forest Max Depth", 2, 20, 5)

# Parse currency symbols for API
base, quote = pair.split("/")

@st.cache_data
def fetch_intraday_data(symbol_base, symbol_quote, interval="15min", outputsize="30"):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol_base}/{symbol_quote}&interval={interval}"
        f"&outputsize={outputsize}&format=JSON&apikey={TWELVE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        st.warning(f"⚠️ No intraday data returned. Check API key / limits.\nMessage: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    })
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
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
    df["SMA_fast"] = df["close"].rolling(sma_fast_period).mean()
    df["SMA_slow"] = df["close"].rolling(sma_slow_period).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))

    # Features and target for ML model
    df["return"] = df["close"].pct_change().shift(-1)  # next period return
    df["target"] = (df["return"] > 0).astype(int)  # binary target: 1 if price goes up next period

    features = df[["SMA_fast", "SMA_slow", "RSI"]].dropna()
    target = df.loc[features.index, "target"]

    return features, target

def train_models(features, target):
    split = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=42
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    lr_acc = accuracy_score(y_test, y_pred_lr)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    lr_cm = confusion_matrix(y_test, y_pred_lr)
    rf_cm = confusion_matrix(y_test, y_pred_rf)

    return (lr, rf), (lr_acc, rf_acc), (lr_cm, rf_cm), X_test, y_test, y_pred_lr, y_pred_rf

def backtest_sma_crossover(df):
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

def trading_simulator(df, predictions, initial_balance=10000, stop_loss=0.01, take_profit=0.02):
    balance = initial_balance
    position = 0  # 1 for long, 0 for flat
    entry_price = 0
    trades = []
    wins = 0

    for i in range(1, len(df)):
        pred = predictions[i-1]
        price = df["close"].iloc[i]
        prev_price = df["close"].iloc[i-1]

        # Enter trade
        if position == 0 and pred == 1:
            position = 1
            entry_price = price
            trades.append({"entry_index": i, "entry_price": price, "exit_index": None, "exit_price": None, "profit": None})

        elif position == 1:
            change = (price - entry_price) / entry_price

            # Check stop loss / take profit
            if change <= -stop_loss or change >= take_profit or pred == 0:
                profit = (price - entry_price)
                balance += profit * 1000  # assume trading 1000 units
                trades[-1]["exit_index"] = i
                trades[-1]["exit_price"] = price
                trades[-1]["profit"] = profit * 1000
                if profit > 0:
                    wins += 1
                position = 0

    win_rate = wins / len(trades) if trades else 0
    total_profit = balance - initial_balance
    return balance, total_profit, len(trades), win_rate, trades

# Fetch data
df = fetch_intraday_data(base, quote)
if df.empty:
    st.warning("No intraday data to display.")
    st.stop()

# Compute indicators
df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

# Prepare features and target
features, target = prepare_features(df)

# Train models
(models_lr_rf, (lr_acc, rf_acc), (lr_cm, rf_cm), X_test, y_test, y_pred_lr, y_pred_rf) = train_models(features, target)

# Backtest SMA crossover
df_bt, total_return, max_drawdown = backtest_sma_crossover(df)

# Trading simulator using RF predictions
balance, total_profit, trades_count, win_rate, trades_log = trading_simulator(df.loc[X_test.index], y_pred_rf)

# Display metrics
st.markdown(f"**Data Source:** Twelve Data (15min Intraday) for {pair}")
st.markdown(f"Logistic Regression Accuracy: {lr_acc:.2%}")
st.markdown(f"Random Forest Accuracy: {rf_acc:.2%}")

st.subheader("Confusion Matrix (Logistic Regression):")
st.write(lr_cm)

st.subheader("Confusion Matrix (Random Forest):")
st.write(rf_cm)

st.subheader("🌿 Random Forest Feature Importance")
rf_model = models_lr_rf[1]
feat_imp = rf_model.feature_importances_
feat_names = features.columns
imp_df = pd.DataFrame({"feature": feat_names, "importance": feat_imp}).sort_values(by="importance", ascending=False)
st.dataframe(imp_df)

st.subheader("📊 Backtest Equity Curve (SMA Crossover)")
fig_backtest = go.Figure()
fig_backtest.add_trace(go.Scatter(x=df_bt.index, y=df_bt["equity_curve"], mode="lines", name="Equity Curve"))
st.plotly_chart(fig_backtest, use_container_width=True)
st.markdown(f"Total Return: {total_return:.2%}")
st.markdown(f"Max Drawdown: {max_drawdown:.2%}")

st.subheader("🧪 Trading Simulator with Stop Loss & Take Profit")
st.markdown(f"Final Balance: ${balance:,.2f}")
st.markdown(f"Total Profit: ${total_profit:,.2f}")
st.markdown(f"Number of Trades: {trades_count}")
st.markdown(f"Win Rate: {win_rate:.2%}")

st.subheader("📈 Price + SMA + Buy/Sell Signals (Random Forest)")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], mode="lines", name=f"SMA {sma_fast_period}"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], mode="lines", name=f"SMA {sma_slow_period}"))

buy_signals = df.loc[(y_pred_rf == 1)]
sell_signals = df.loc[(y_pred_rf == 0)]

# Align buy/sell signals with index of df (using X_test.index)
buy_signals = buy_signals.loc[X_test.index]
sell_signals = sell_signals.loc[X_test.index]

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
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
st.plotly_chart(fig_rsi, use_container_width=True)

st.subheader("🔴 Live Feed (placeholder) - coming soon!")
st.write("Live feed integration will be added soon.")

st.subheader("📝 Trades Log")
if trades_log:
    trades_df = pd.DataFrame(trades_log)
    st.dataframe(trades_df)
else:
    st.write("No trades executed yet.")
