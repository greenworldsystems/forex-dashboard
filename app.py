import os
import requests
import pandas as pd
import streamlit as st

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY")
symbol, target = "EUR", "USD"

@st.cache_data
def fetch_data():
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={symbol}&to_symbol={target}"
        f"&outputsize=compact&apikey={API_KEY}"
    )
    r = requests.get(url)
    data = r.json().get("Time Series FX (Daily)", {})
    df = pd.DataFrame.from_dict(data, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

st.title("📈 Forex Dashboard: EUR/USD")
st.markdown(
    "Simple demo: fetching daily close prices from Alpha Vantage "
    "and plotting as a chart."
)

if API_KEY:
    df = fetch_data()
    st.line_chart(df["4. close"])
    st.write("Latest data:")
    st.dataframe(df.tail(5))
else:
    st.warning("⚠️ Please set your ALPHAVANTAGE_API_KEY as env var or Streamlit secret.")
