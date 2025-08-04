import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# API keys from env or Streamlit secrets
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
    df.sort_index(inplace=True)
