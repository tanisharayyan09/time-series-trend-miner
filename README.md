# Time Series Trend Miner

A Streamlit dashboard for **temporal data mining**: spot trends and seasonality in time-based data and forecast what comes next.

**Live app:** https://YOUR-APP-NAME.streamlit.app

**Author:** Nishanthini M

## Features

- **Data input:** built-in sample data (sales, weather, stock) or your own CSV upload
- **Trend and seasonality:** splits a series into trend, seasonal and residual components
- **Forecasting:** ARIMA forecasts for 7 to 90 days, with automatic model selection and a confidence band
- **Adjustable settings:** forecast length and seasonality period (for example, 7 days for weekly patterns)
- **Downloads:** forecast CSV and cleaned data CSV
- **Optional Prophet support:** enabled automatically if `prophet` is installed

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (user interface) |
| `forecasting.py` | Data loading, decomposition and forecasting logic |
| `requirements.txt` | Python dependencies |
| `date_value_data.csv` | Sample file with 50 rows of date/value data for testing |

## CSV format

Your file needs at least 10 valid rows with:

- a **date column** (for example, `2025-01-01`)
- a **numeric value column**

Example:

```csv
date,value
2025-01-01,196.48
2025-01-02,215.06
2025-01-03,232.39
```

Rows with missing or invalid dates or values are dropped, and duplicate dates are removed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repo, set the branch to `main` and the main file path to `app.py`.
4. Click **Deploy**.

## Tech stack

Python, Streamlit, pandas, NumPy, statsmodels, Plotly
"Live app demo":
https://time-series-trend-miner-tanisharayyan09.streamlit.app/
