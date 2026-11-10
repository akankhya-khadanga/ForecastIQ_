# ForecastIQ — Streamlit Sales Forecasting Dashboard

This project includes a Streamlit frontend, the bundled ecommerce sales dataset, and the existing ML forecasting service. The dashboard works locally from the CSV and **does not require MongoDB**.

## 1. Set up the environment

Python 3.11–3.13 is recommended.

```bash
cd "project 2"
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Start the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL, normally **http://localhost:8501**. Open that URL in your browser. The first start trains the forecasting models and may take a little time.

## Dashboard graphs

- KPI cards: revenue, profit, order count, units sold, and profit margin, with previous-period comparisons
- Executive snapshot: top category, leading region, strongest month, and average order value
- Weekly sales pattern and interactive Plotly charts
- Monthly revenue trend
- Revenue share by category (doughnut chart)
- Monthly profit and revenue by region
- Top categories and regional comparisons
- Transaction quantity vs sales scatter plot
- Daily actual-vs-forecast chart with forecast interval
- Holdout model error comparison (MAE, RMSE, MAPE)
- XGBoost evaluation metrics (MAE, RMSE, R², MAPE) and a roughly six-month monthly revenue outlook
- XGBoost actual-vs-predicted and feature-importance charts
- Potential anomaly table and searchable transaction data

Use the sidebar to filter dates, categories, and regions. Upload a CSV to analyze a different dataset. Required uploaded columns are `Order Date` (or `Date`), `Sales`, and `Quantity`; `Unit Price` is recommended. Category, Region, and Profit are optional and receive defaults if missing.

## Dataset

The default dataset is `ecommerce_sales_2024_2025.csv`. Charts use the real rows in this file. Forecasts are model estimates and should not be interpreted as guaranteed future sales.

## Optional legacy Flask app

The original Flask app and authentication modules remain in the project. To use them, configure MongoDB variables in a local `.env` file based on `.env.example`, then run `python app.py`. This is separate from the Streamlit dashboard.

**Security:** do not commit `.env` or share database credentials. If a real MongoDB password was exposed, rotate it in MongoDB Atlas and update your local `.env`.
