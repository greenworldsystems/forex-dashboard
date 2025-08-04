import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide", page_title="Forex Dashboard with ML & Live Feed")

TWELVE_API_KEY = st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY")

# --- Data Fetching Functions ---

@st.cache_data(show_spinner=False)
def fetch_intraday_twelve(pair):
    symbol_encoded = pair.replace("/", "%2F")
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol_encoded}&interval=15min&outputsize=500&apikey={TWELVE_API_KEY}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "values" not in data:
            st.warning(f"⚠️ No intraday data from Twelve Data. Message: {data.get('message', '')}")
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching Twelve Data intraday: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_daily_alpha(pair):
    from_symbol, to_symbol = pair.split("/")
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={from_symbol}&to_symbol={to_symbol}"
        f"&outputsize=compact&apikey={ALPHAVANTAGE_API_KEY}"
    )
    try:
        r = requests.get(url)
        data = r.json().get("Time Series FX (Daily)", {})
        if not data:
            st.warning("⚠️ No daily data returned from Alpha Vantage.")
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(data, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close": "close"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching Alpha Vantage daily: {e}")
        return pd.DataFrame()

# --- Indicator Computation ---

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

# --- Feature Engineering for ML ---

def build_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)  # next period up/down
    df["SMA_diff"] = df["SMA_fast"] - df["SMA_slow"]
    df["RSI"] = df["RSI"].fillna(50)
    df.dropna(inplace=True)
    features = df[["SMA_fast", "SMA_slow", "SMA_diff", "RSI"]]
    target = df["target"]
    return features, target, df

# --- ML Models Training ---

def train_models(features, target):
    split = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    return (lr, rf, acc_lr, acc_rf, cm_lr, cm_rf, X_test.index, y_pred_lr, y_pred_rf, y_test)

# --- Backtesting SMA Crossover Strategy ---

def backtest_sma_crossover(df):
    df = df.copy()
    df["position"] = 0
    df.loc[df["SMA_fast"] > df["SMA_slow"], "position"] = 1
    df.loc[df["SMA_fast"] < df["SMA_slow"], "position"] = 0

    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    df["equity_curve"] = (1 + df["strategy_returns"].fillna(0)).cumprod()

    total_return = df["equity_curve"].iloc[-1] - 1
    max_drawdown = ((df["equity_curve"].cummax() - df["equity_curve"]) / df["equity_curve"].cummax()).max()
    return df, total_return, max_drawdown

# --- Trading Simulator with SL & TP ---

def run_trading_simulator(df, signals, stop_loss=0.005, take_profit=0.01, initial_balance=10000):
    balance = initial_balance
    position = 0  # 1=long, 0=flat
    entry_price = 0
    trades = []
    wins = 0

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        signal = signals.iloc[i]

        if position == 0 and signal == 1:  # enter long
            position = 1
            entry_price = price
            trades.append({"entry_index": i, "entry_price": price, "exit_price": None, "profit": None})
        elif position == 1:
            change = (price - entry_price) / entry_price
            # Check stop loss or take profit
            if change <= -stop_loss or change >= take_profit or signal == 0:
                position = 0
                exit_price = price
                profit = exit_price - entry_price
                trades[-1]["exit_price"] = exit_price
                trades[-1]["profit"] = profit
                balance += profit * (balance / entry_price)  # simulate position size proportional to balance
                if profit > 0:
                    wins += 1

    total_profit = balance - initial_balance
    win_rate = (wins / len(trades) * 100) if trades else 0
    return balance, total_profit, len(trades), win_rate, trades

# --- Plotting Functions with Plotly ---

def plot_price_sma_rsi(df, signals, pair, sma_fast_period, sma_slow_period):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], mode="lines", name=f"SMA {sma_fast_period}"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], mode="lines", name=f"SMA {sma_slow_period}"))

    buy_signals = df[signals == 1]
    sell_signals = df[signals == 0]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["close"], mode="markers",
                             marker=dict(symbol="triangle-up", color="green", size=10),
                             name="Buy Signals"))

    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["close"], mode="markers",
                             marker=dict(symbol="triangle-down", color="red", size=10),
                             name="Sell Signals"))

    fig.update_layout(title=f"{pair} Price with SMA & Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    fig.update_layout(title="RSI", xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(rf, features):
    importances = rf.feature_importances_
    fig = go.Figure(go.Bar(x=features, y=importances))
    fig.update_layout(title="Random Forest Feature Importance", xaxis_title="Feature", yaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(cm, title):
    import plotly.figure_factory as ff
    z = cm
    x = ["Pred 0", "Pred 1"]
    y = ["True 0", "True 1"]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Viridis")
    fig.update_layout(title_text=title, width=400, height=400)
    st.plotly_chart(fig)

# --- Main App ---

st.title("📈 Forex Dashboard: SMA, RSI, ML + Backtest + Live Feed")

pair = st.sidebar.selectbox("Select currency pair", ["EUR/USD", "USD/CAD", "GBP/USD", "USD/JPY", "AUD/USD"])

sma_fast_period = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
sma_slow_period = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# Fetch data
df = fetch_intraday_twelve(pair)
using_intraday = True
if df.empty:
    st.warning("⚠️ Twelve Data intraday unavailable, falling back to Alpha Vantage daily data.")
    df = fetch_daily_alpha(pair)
    using_intraday = False

if df.empty:
    st.error("No data available. Please check API keys and limits.")
    st.stop()

# Compute indicators
df = compute_indicators(df, sma_fast_period, sma_slow_period, rsi_period)

# ML features and target
features, target, df = build_features(df)

# Train ML models
lr, rf, acc_lr, acc_rf, cm_lr, cm_rf, test_index, y_pred_lr, y_pred_rf, y_test = train_models(features, target)

# Backtest SMA crossover
df_bt, total_ret, max_dd = backtest_sma_crossover(df)

# Simulator with RF predictions as signals
df_test = df.loc[test_index]
signals = pd.Series(y_pred_rf, index=test_index)
balance, total_profit, trades_count, win_rate, trades = run_trading_simulator(df_test, signals)

# Display metrics
st.markdown(f"### Data Source: {'Twelve Data (15min Intraday)' if using_intraday else 'Alpha Vantage (Daily Fallback)'}")
st.markdown(f"#### Logistic Regression Accuracy: {acc_lr*100:.2f}%")
st.markdown(f"#### Random Forest Accuracy: {acc_rf*100:.2f}%")
st.markdown("#### Confusion Matrix (Logistic Regression):")
plot_confusion_matrix(cm_lr, "Logistic Regression Confusion Matrix")
st.markdown("#### Confusion Matrix (Random Forest):")
plot_confusion_matrix(cm_rf, "Random Forest Confusion Matrix")

st.markdown("### 🌿 Random Forest Feature Importance")
plot_feature_importance(rf, features.columns)

st.markdown("### 📊 Backtest Equity Curve (SMA Crossover)")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["equity_curve"], mode="lines", name="Equity Curve"))
st.plotly_chart(fig_eq, use_container_width=True)
st.markdown(f"**Total Return:** {total_ret:.2%}")
st.markdown(f"**Max Drawdown:** {max_dd:.2%}")

st.markdown("### 🧪 Trading Simulator with Stop Loss & Take Profit")
st.markdown(f"Final Balance: ${balance:,.2f}")
st.markdown(f"Total Profit: ${total_profit:,.2f}")
st.markdown(f"Number of Trades: {trades_count}")
st.markdown(f"Win Rate: {win_rate:.2f}%")

st.markdown("### 📈 Price + SMA + Buy/Sell Signals (Random Forest)")
plot_price_sma_rsi(df_test, signals, pair, sma_fast_period, sma_slow_period)

st.markdown("### 📉 RSI")
plot_rsi(df)

# Placeholder for Live Feed (expandable)
if using_intraday:
    st.markdown("### 🔴 Live Feed (placeholder) - coming soon!")
else:
    st.info("Live feed disabled with daily data fallback.")

# Show recent trades log
if trades_count > 0:
    trades_df = pd.DataFrame(trades)
    st.markdown("### 📝 Trades Log")
    st.dataframe(trades_df)

