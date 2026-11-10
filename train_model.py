import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# 1. LOAD DATA
df = pd.read_csv("ecommerce_sales_2024_2025.csv")

print("\nOriginal data:")
print(df.head())

print("\nShape:", df.shape)

# 2. DATA CLEANING
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

# Remove rows where date conversion failed
df.dropna(subset=["Order Date"], inplace=True)

# Remove rows where important numerical values are missing
df.dropna(
    subset=["Sales", "Quantity", "Unit Price"],
    inplace=True
)

# Sort chronologically
df.sort_values("Order Date", inplace=True)

df.reset_index(drop=True, inplace=True)

# 3. DAILY AGGREGATION
daily_df = (
    df.groupby("Order Date")
      .agg({
          "Sales": "sum",
          "Quantity": "sum"
      })
      .reset_index()
)


# Make sure every calendar day exists
daily_df = daily_df.set_index("Order Date")

daily_df = daily_df.asfreq("D", fill_value=0)

daily_df.reset_index(inplace=True)


print("\nDaily sales data:")
print(daily_df.head())

print("\nNumber of daily observations:", len(daily_df))

# 4. CALENDAR FEATURES
daily_df["Year"] = daily_df["Order Date"].dt.year

daily_df["Month"] = daily_df["Order Date"].dt.month

daily_df["Day"] = daily_df["Order Date"].dt.day

daily_df["DayOfWeek"] = (
    daily_df["Order Date"].dt.dayofweek
)

daily_df["Quarter"] = (
    daily_df["Order Date"].dt.quarter
)

daily_df["IsWeekend"] = (
    daily_df["DayOfWeek"] >= 5
).astype(int)

# 5. LAG FEATURES
daily_df["Sales_Lag_1"] = (
    daily_df["Sales"].shift(1)
)

daily_df["Sales_Lag_7"] = (
    daily_df["Sales"].shift(7)
)

daily_df["Sales_Lag_14"] = (
    daily_df["Sales"].shift(14)
)

daily_df["Sales_Lag_30"] = (
    daily_df["Sales"].shift(30)
)

daily_df["Rolling_Mean_7"] = (
    daily_df["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily_df["Rolling_Mean_14"] = (
    daily_df["Sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

daily_df["Rolling_Mean_30"] = (
    daily_df["Sales"]
    .shift(1)
    .rolling(30)
    .mean()
)

daily_df["Rolling_Std_7"] = (
    daily_df["Sales"]
    .shift(1)
    .rolling(7)
    .std()
)

# 7. REMOVE MISSING VALUES
daily_df.dropna(inplace=True)

daily_df.reset_index(drop=True, inplace=True)


print("\nData after feature engineering:")
print(daily_df.head())

print("\nFinal dataset shape:", daily_df.shape)

# 8. FEATURES AND TARGET
features = [

    "Quantity",

    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
    "IsWeekend",

    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Lag_14",
    "Sales_Lag_30",

    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_30",
    "Rolling_Std_7"
]


X = daily_df[features]

y = daily_df["Sales"]

# 9. TIME-BASED TRAIN / TEST SPLIT
# Do NOT use train_test_split()
# because this is a time-series problem.

split_index = int(len(daily_df) * 0.8)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\nTraining data:")
print(
    daily_df["Order Date"].iloc[0],
    "to",
    daily_df["Order Date"].iloc[split_index - 1]
)


print("\nTesting data:")
print(
    daily_df["Order Date"].iloc[split_index],
    "to",
    daily_df["Order Date"].iloc[-1]
)

# 10. XGBOOST MODEL
xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=4,

    min_child_weight=2,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=42,

    n_jobs=-1
)

# 11. TRAIN XGBOOST
print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost training completed!")

# 12. XGBOOST PREDICTION
xgb_pred = xgb_model.predict(X_test)

# 13. XGBOOST EVALUATION
xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)

xgb_mape = mean_absolute_percentage_error(
    y_test,
    xgb_pred
)


print("\n================================")
print("XGBOOST RESULTS")
print("================================")

print("MAE  :", round(xgb_mae, 2))

print("RMSE :", round(xgb_rmse, 2))

print("R2   :", round(xgb_r2, 4))


import matplotlib.pyplot as plt

# Create a comparison dataframe
results = pd.DataFrame({
    "Date": daily_df["Order Date"].iloc[split_index:].values,
    "Actual": y_test.values,
    "Predicted": xgb_pred
})

# Plot Actual vs Predicted
plt.figure(figsize=(14, 6))

plt.plot(
    results["Date"],
    results["Actual"],
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted"],
    label="Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales - XGBoost")

plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



import matplotlib.pyplot as plt

# Calculate monthly sales
monthly_sales = (
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"]
    .sum()
    .reset_index()
)

# Calculate month-to-month growth rate
monthly_sales["Growth_Rate"] = (
    monthly_sales["Sales"]
    .pct_change() * 100
)

# Remove first month because it has no previous month
monthly_sales = monthly_sales.dropna()

# Convert month to string
monthly_sales["Order Date"] = (
    monthly_sales["Order Date"].astype(str)
)

# Plot growth rate
plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Growth_Rate"],
    marker="o",
    linewidth=2
)

# Zero line = no growth
plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Monthly Sales Growth Rate")
plt.xlabel("Month")
plt.ylabel("Growth Rate (%)")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


#monthly sales trend
monthly_sales=(
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"]
.sum()
)
plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly sales trend")
plt.xlabel("Month")
plt.ylabel("sales")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


region_sales = (
    df.groupby("Region")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

plt.bar(
    region_sales.index,
    region_sales.values
)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================
# 14. FORECAST NEXT 6 MONTHS
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Make a copy of the historical daily data
forecast_df = daily_df.copy()

# Last date in our dataset
last_date = forecast_df["Order Date"].max()

# Forecast 6 months ahead
forecast_end = last_date + pd.DateOffset(months=6)

# Create all future dates
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    end=forecast_end,
    freq="D"
)

future_predictions = []

# We will use the recent average quantity
# as the future quantity assumption
future_quantity = forecast_df["Quantity"].tail(30).mean()


# ------------------------------------------------------------
# Generate predictions day by day
# ------------------------------------------------------------

for current_date in future_dates:

    # Calendar features
    year = current_date.year
    month = current_date.month
    day = current_date.day
    day_of_week = current_date.dayofweek
    quarter = current_date.quarter
    is_weekend = int(day_of_week >= 5)

    # Get all sales values available so far
    sales_history = forecast_df["Sales"].values

    # Lag features
    lag_1 = sales_history[-1]
    lag_7 = sales_history[-7]
    lag_14 = sales_history[-14]
    lag_30 = sales_history[-30]

    # Rolling features
    rolling_mean_7 = np.mean(sales_history[-7:])
    rolling_mean_14 = np.mean(sales_history[-14:])
    rolling_mean_30 = np.mean(sales_history[-30:])

    rolling_std_7 = np.std(
        sales_history[-7:],
        ddof=1
    )

    # Create one row for prediction
    future_row = pd.DataFrame({

        "Quantity": [future_quantity],

        "Year": [year],
        "Month": [month],
        "Day": [day],
        "DayOfWeek": [day_of_week],
        "Quarter": [quarter],
        "IsWeekend": [is_weekend],

        "Sales_Lag_1": [lag_1],
        "Sales_Lag_7": [lag_7],
        "Sales_Lag_14": [lag_14],
        "Sales_Lag_30": [lag_30],

        "Rolling_Mean_7": [rolling_mean_7],
        "Rolling_Mean_14": [rolling_mean_14],
        "Rolling_Mean_30": [rolling_mean_30],

        "Rolling_Std_7": [rolling_std_7]
    })

    # Predict sales
    predicted_sales = xgb_model.predict(
        future_row[features]
    )[0]

    # Prevent negative sales
    predicted_sales = max(0, predicted_sales)

    # Store prediction
    future_predictions.append(predicted_sales)

    # Add prediction to historical dataframe
    # so that the next day's lag features can use it
    new_row = pd.DataFrame({

        "Order Date": [current_date],

        "Sales": [predicted_sales],

        "Quantity": [future_quantity]
    })

    forecast_df = pd.concat(
        [forecast_df, new_row],
        ignore_index=True
    )


# ============================================================
# 15. CREATE FUTURE FORECAST DATAFRAME
# ============================================================

future_forecast = pd.DataFrame({

    "Order Date": future_dates,

    "Forecast_Sales": future_predictions

})

print("\nNext 6 months daily forecast:")
print(future_forecast.head())

print("\nForecast end date:")
print(future_forecast["Order Date"].max())


# ============================================================
# 16. MONTHLY HISTORICAL SALES
# ============================================================

historical_monthly = (
    daily_df
    .set_index("Order Date")["Sales"]
    .resample("MS")
    .sum()
    .reset_index()
)

historical_monthly.rename(
    columns={"Sales": "Sales"},
    inplace=True
)


# ============================================================
# 17. MONTHLY FUTURE FORECAST
# ============================================================

forecast_monthly = (
    future_forecast
    .set_index("Order Date")["Forecast_Sales"]
    .resample("MS")
    .sum()
    .reset_index()
)

forecast_monthly.rename(
    columns={"Forecast_Sales": "Forecast_Sales"},
    inplace=True
)


# ============================================================
# 18. DISPLAY FORECAST
# ============================================================

print("\n========================================")
print("NEXT 6 MONTHS SALES FORECAST")
print("========================================")

print(forecast_monthly)


# ============================================================
# 19. COMBINE HISTORICAL + FUTURE DATA
# ============================================================

historical_plot = historical_monthly.tail(12)

# ============================================================
# 20. PLOT HISTORICAL + FUTURE FORECAST
# ============================================================

plt.figure(figsize=(15, 7))

# Historical sales
plt.plot(
    historical_plot["Order Date"],
    historical_plot["Sales"],
    marker="o",
    linewidth=2,
    label="Historical Sales"
)

# Future forecast
plt.plot(
    forecast_monthly["Order Date"],
    forecast_monthly["Forecast_Sales"],
    marker="o",
    linewidth=2,
    linestyle="--",
    label="6-Month Forecast"
)

# Mark forecast starting point
plt.axvline(
    x=forecast_monthly["Order Date"].iloc[0],
    linestyle="--",
    label="Forecast Starts"
)

plt.title(
    "Historical Sales and Next 6 Months Forecast"
)

plt.xlabel("Month")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ============================================================
# 21. AUTOMATIC FORECAST INTERPRETATION
# ============================================================

first_forecast = forecast_monthly["Forecast_Sales"].iloc[0]

last_forecast = forecast_monthly["Forecast_Sales"].iloc[-1]

# Overall change
overall_change = (
    (last_forecast - first_forecast)
    / first_forecast
) * 100


# Determine overall direction
if overall_change > 5:

    direction = "increase"

elif overall_change < -5:

    direction = "decrease"

else:

    direction = "remain relatively stable"


print("\n")
print("========================================")
print("FORECAST INTERPRETATION")
print("========================================")

print(
    f"Based on the XGBoost forecast, sales are expected to "
    f"{direction} over the next 6 months."
)

print(
    f"The forecast starts at approximately "
    f"{first_forecast:,.2f} in the first forecast month "
    f"and reaches approximately "
    f"{last_forecast:,.2f} by the final forecast month."
)

print(
    f"This represents an overall change of "
    f"{overall_change:.2f}% from the beginning "
    f"to the end of the forecast period."
)


# Find highest and lowest forecast months
highest_index = forecast_monthly["Forecast_Sales"].idxmax()

lowest_index = forecast_monthly["Forecast_Sales"].idxmin()

highest_month = forecast_monthly.loc[
    highest_index,
    "Order Date"
]

lowest_month = forecast_monthly.loc[
    lowest_index,
    "Order Date"
]

highest_sales = forecast_monthly.loc[
    highest_index,
    "Forecast_Sales"
]

lowest_sales = forecast_monthly.loc[
    lowest_index,
    "Forecast_Sales"
]


print(
    f"The highest forecasted sales are expected in "
    f"{highest_month.strftime('%B %Y')}, "
    f"with approximately {highest_sales:,.2f} in sales."
)

print(
    f"The lowest forecasted sales are expected in "
    f"{lowest_month.strftime('%B %Y')}, "
    f"with approximately {lowest_sales:,.2f} in sales."
)


# Business interpretation
if overall_change > 5:

    print(
        "\nBusiness Interpretation:"
    )

    print(
        "The forecast indicates an overall upward sales trend. "
        "This may suggest increasing customer demand and "
        "potentially higher future revenue. The business "
        "may consider maintaining sufficient inventory and "
        "resources to handle the expected growth."
    )

elif overall_change < -5:

    print(
        "\nBusiness Interpretation:"
    )

    print(
        "The forecast indicates an overall downward sales trend. "
        "This may suggest weakening demand. The business should "
        "consider investigating product performance, customer "
        "demand, pricing, and regional sales patterns."
    )

else:

    print(
        "\nBusiness Interpretation:"
    )

    print(
        "The forecast indicates relatively stable sales over "
        "the next 6 months. The business may focus on maintaining "
        "current performance while identifying opportunities "
        "for additional growth."
    )
    
