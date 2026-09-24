"""Time Series Trend Miner - Streamlit dashboard.

Run:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fc

st.set_page_config(page_title="Time Series Trend Miner", page_icon="📈", layout="wide")
st.title("📈 Time Series Trend Miner")
st.caption("Temporal data mining: spot trends, seasonality and forecast what's next.")

# ---------- Sidebar: data source ----------
st.sidebar.header("Data source")
source = st.sidebar.radio("Get data from", ["Sample data", "Upload CSV"])

if source == "Sample data":
    kind = st.sidebar.selectbox("Dataset", ["sales", "weather", "stock"])
    raw = fc.generate_sample_data(kind)
    df = raw
else:
    file = st.sidebar.file_uploader("CSV file", type="csv")
    if file is None:
        st.info("👈 Upload a CSV with a date column and a numeric value column.")
        st.stop()
    raw = pd.read_csv(file)
    st.sidebar.write("Preview:", raw.head(3))
    date_col = st.sidebar.selectbox("Date column", raw.columns)
    value_col = st.sidebar.selectbox("Value column", [c for c in raw.columns if c != date_col])
    try:
        df = fc.load_series(raw, date_col, value_col)
    except ValueError as e:
        st.error(str(e))
        st.stop()

st.sidebar.header("Forecast settings")
method = st.sidebar.radio("Method", ["ARIMA"] + (["Prophet"] if fc.HAS_PROPHET else []))
if not fc.HAS_PROPHET:
    st.sidebar.caption("Install `prophet` to also enable Prophet forecasting.")
horizon = st.sidebar.slider("Days to forecast", 7, 90, 30)
period = st.sidebar.slider("Seasonality period (days)", 2, 30, 7)

c1, c2, c3 = st.columns(3)
c1.metric("Data points", len(df))
c2.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")
c3.metric("Latest value", f"{df['value'].iloc[-1]:,.2f}")

tab1, tab2, tab3 = st.tabs(["Trend & Seasonality", "Forecast", "Data"])

with tab1:
    st.subheader("Decomposition (observed = trend + seasonal + residual)")
    dec = fc.decompose(df, period=period)
    fig = go.Figure()
    for col, name in [("observed", "Observed"), ("trend", "Trend"), ("seasonal", "Seasonal")]:
        fig.add_trace(go.Scatter(x=dec["date"], y=dec[col], name=name, mode="lines"))
    fig.update_layout(height=450, legend=dict(orientation="h"))
    st.plotly_chart(fig)

with tab2:
    st.subheader(f"{horizon}-day forecast using {method}")
    with st.spinner("Fitting model..."):
        if method == "ARIMA":
            fc_df, order = fc.forecast_arima(df, horizon)
            st.caption(f"Best ARIMA order found: {order}")
        else:
            fc_df = fc.forecast_prophet(df, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="Actual", line=dict(color="#028090")))
    fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"], name="Forecast", line=dict(color="#EE6C4D")))
    fig.add_trace(go.Scatter(
        x=list(fc_df["date"]) + list(fc_df["date"])[::-1],
        y=list(fc_df["upper"]) + list(fc_df["lower"])[::-1],
        fill="toself", fillcolor="rgba(238,108,77,0.15)", line=dict(width=0),
        name="Confidence band", hoverinfo="skip",
    ))
    fig.update_layout(height=480, legend=dict(orientation="h"))
    st.plotly_chart(fig)

    st.download_button("Download forecast CSV", fc_df.to_csv(index=False), "forecast.csv", "text/csv")

with tab3:
    st.dataframe(df, hide_index=True)
    st.download_button("Download data CSV", df.to_csv(index=False), "timeseries.csv", "text/csv")
