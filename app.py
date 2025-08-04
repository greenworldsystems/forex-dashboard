import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Get API keys
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY") or st.secrets.get("TWELVE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")

symbol = "EUR/USD"
twelve_interval = "15min"
alpha_symbol_from = "EUR"
alpha_symbol_to = "USD"

@st.cache_data(ttl=600)
def fetch_twelve_data():
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={twelve_interval}&apikey={TWELVE_API_KEY}&format=JSON&outputsize=500"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df["close"] = pd.to_numeric(df["close"])
    return df

@st.cache_data(ttl=600)
def fetch_alpha_daily_data():
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={alpha_symbol_from}&to_symbol={alpha_symbol_to}&outputsize=compact&apikey={ALPHAVANTAGE_API_KEY}"
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
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    return df

def prepare_ml_features(df):
    df = df.copy()
    df["return_1"] = df["close"].pct_change()
    df["sma_diff"] = df["SMA_fast"] - df["SMA_slow"]
    df["volatility"] = df["close"].rolling(5).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

def train_models(df, rf_n_estimators, rf_max_depth):
    features = ["return_1", "sma_diff", "volatility", "RSI"]
    X = df[features]
    y = df["target"]
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    preds_lr = logreg.predict(X_test)
    acc_lr = accuracy_score(y_test, preds_lr)
    cm_lr = confusion_matrix(y_test, preds_lr)

    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)
    cm_rf = confusion_matrix(y_test, preds_rf)

    # Add predictions to df
    df.loc[df.index[split_idx:], "pred_lr"] = preds_lr
    df.loc[df.index[split_idx:], "pred_rf"] = preds_rf

    return logreg, rf, acc_lr, cm_lr, acc_rf, cm_rf, df

def backtest_strategy(df, pred_column):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["strategy"] = df[pred_column].shift(1) * df["returns"]
    df["equity_curve"] = (1 + df["strategy"].fillna(0)).cumprod()
    total_return = df["equity_curve"].iloc[-1] - 1
    max_drawdown = ((df["equity_curve"].cummax() - df["equity_curve"]) / df["equity_curve"].cummax()).max()
    return df, total_return, max_drawdown

st.title("📈 Forex Dashboard: SMA, RSI & ML Prediction + Backtest")

if not (TWELVE_API_KEY and ALPHAVANTAGE_API_KEY):
    st.warning("⚠️ Please set TWELVE_API_KEY and ALPHAVANTAGE_API_KEY in secrets or env variables.")
else:
    df = fetch_twelve_data()
    source_used = "Twelve Data (15min Intraday)"
    if df.empty:
        df = fetch_alpha_daily_data()
        source_used = "Alpha Vantage Daily"

    if df.empty:
        st.error("No data available from either source.")
    else:
        sma_fast = st.sidebar.slider("SMA Fast Period", 2, 20, 5)
        sma_slow = st.sidebar.slider("SMA Slow Period", 10, 50, 20)
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        rf_n_estimators = st.sidebar.slider("RF Trees", 10, 200, 50)
        rf_max_depth = st.sidebar.slider("RF Max Depth", 2, 20, 5)

        df = compute_indicators(df, sma_fast, sma_slow, rsi_period)
        df_ml = prepare_ml_features(df)
        logreg, rf, acc_lr, cm_lr, acc_rf, cm_rf, df_ml = train_models(df_ml, rf_n_estimators, rf_max_depth)

        st.subheader(f"Data Source: {source_used}")

        st.subheader("✅ ML Model Accuracy")
        st.write(f"Logistic Regression Accuracy: {acc_lr:.2%}")
        st.write("Confusion Matrix (LR):")
        st.write(cm_lr)
        st.write(f"Random Forest Accuracy: {acc_rf:.2%}")
        st.write("Confusion Matrix (RF):")
        st.write(cm_rf)

        # Feature importance
        st.subheader("🌿 Random Forest Feature Importance")
        importance = pd.Series(rf.feature_importances_, index=["return_1", "sma_diff", "volatility", "RSI"])
        st.bar_chart(importance)

        # Backtest
        df_bt, total_ret, max_dd = backtest_strategy(df_ml, "pred_rf")
        st.subheader("📊 Backtest Equity Curve (Random Forest)")
        st.line_chart(df_bt["equity_curve"])
        st.write(f"Total Return: {total_ret:.2%}")
        st.write(f"Max Drawdown: {max_dd:.2%}")

        # Plot price & predictions
        st.subheader("📈 Price + SMA + ML Buy/Sell Predictions (RF)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml["close"], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml["SMA_fast"], mode='lines', name='SMA Fast'))
        fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml["SMA_slow"], mode='lines', name='SMA Slow'))

        buys = df_ml[df_ml["pred_rf"] == 1]
        sells = df_ml[df_ml["pred_rf"] == 0]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], mode='markers', name='RF Buy', marker=dict(color='green', size=8, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["close"], mode='markers', name='RF Sell', marker=dict(color='red', size=8, symbol='triangle-down')))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 RSI")
        st.line_chart(df["RSI"].dropna())
    
