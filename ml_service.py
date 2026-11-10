
import os
import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class MLService:
    def __init__(self, data_path="ecommerce_sales_2024_2025.csv"):
        self.data_path = data_path
        self.raw_df = None
        self.daily_df = None

        self.feature_cols = [
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
            "Rolling_Std_7",
        ]

        self.models = {}
        self.metrics = {}
        self.future_forecasts = {}
        self.historical_predictions = {}
        self.anomalies = []

        self.best_model_name = "XGBoost"
        self.is_trained = False

        if os.path.exists(self.data_path):
            self.load_and_train()
        else:
            print(f"Dataset file not found at: {self.data_path}")

    def load_and_train(self):
        """Load data, train models, and calculate historical anomalies."""
        print(f"Initializing ML pipeline with dataset: {self.data_path}")

        self._load_data()
        self._train_all_models()
        self._detect_anomalies()

        self.is_trained = True
        print(f"ML pipeline ready. Best model: {self.best_model_name}")

    def _load_data(self):
        """Clean and aggregate daily sales from the CSV file."""
        df = pd.read_csv(self.data_path)

        required_columns = [
            "Order Date",
            "Sales",
            "Quantity",
            "Unit Price",
            "Category",
            "Region",
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {missing_columns}"
            )

        df.drop_duplicates(inplace=True)

        df["Order Date"] = pd.to_datetime(
            df["Order Date"],
            dayfirst=True,
            errors="coerce",
        )

        for col in ["Sales", "Quantity", "Unit Price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(
            subset=["Order Date", "Sales", "Quantity", "Unit Price"],
            inplace=True,
        )

        df.sort_values("Order Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            raise ValueError("No valid sales records were found in the CSV.")

        self.raw_df = df

        # Aggregate transaction data into daily sales.
        daily = (
            df.groupby("Order Date")
            .agg({
                "Sales": "sum",
                "Quantity": "sum",
            })
            .reset_index()
        )

        # Include dates without transactions.
        daily = (
            daily.set_index("Order Date")
            .asfreq("D", fill_value=0)
            .reset_index()
        )

        # Calendar features.
        daily["Year"] = daily["Order Date"].dt.year
        daily["Month"] = daily["Order Date"].dt.month
        daily["Day"] = daily["Order Date"].dt.day
        daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
        daily["Quarter"] = daily["Order Date"].dt.quarter
        daily["IsWeekend"] = (
            daily["DayOfWeek"] >= 5
        ).astype(int)

        # Historical sales lag features.
        daily["Sales_Lag_1"] = daily["Sales"].shift(1)
        daily["Sales_Lag_7"] = daily["Sales"].shift(7)
        daily["Sales_Lag_14"] = daily["Sales"].shift(14)
        daily["Sales_Lag_30"] = daily["Sales"].shift(30)

        # Rolling statistics based only on previous sales.
        previous_sales = daily["Sales"].shift(1)

        daily["Rolling_Mean_7"] = previous_sales.rolling(7).mean()
        daily["Rolling_Mean_14"] = previous_sales.rolling(14).mean()
        daily["Rolling_Mean_30"] = previous_sales.rolling(30).mean()

        daily["Rolling_Std_7"] = (
            previous_sales.rolling(7).std().fillna(0)
        )

        # Remove rows without enough historical data.
        daily.dropna(subset=self.feature_cols, inplace=True)
        daily.reset_index(drop=True, inplace=True)

        if len(daily) < 2:
            raise ValueError(
                "Not enough historical data to train forecasting models. "
                "At least 32 consecutive calendar days are recommended."
            )

        self.daily_df = daily

    def _train_all_models(self):
        """Train and compare Seasonal Naive, Random Forest, and XGBoost."""
        daily = self.daily_df

        X = daily[self.feature_cols]
        y = daily["Sales"]

        # Chronological 80/20 train-test split.
        split_idx = int(len(daily) * 0.8)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                "Not enough data to train and evaluate models."
            )

        # 1. Seasonal Naive: predict using sales from seven days earlier.
        naive_pred = X_test["Sales_Lag_7"].values

        mae_naive = float(mean_absolute_error(y_test, naive_pred))
        rmse_naive = float(
            np.sqrt(mean_squared_error(y_test, naive_pred))
        )
        mape_naive = float(
            mean_absolute_percentage_error(y_test, naive_pred) * 100
        )

        # 2. Random Forest Regressor.
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        mae_rf = float(mean_absolute_error(y_test, rf_pred))
        rmse_rf = float(
            np.sqrt(mean_squared_error(y_test, rf_pred))
        )
        mape_rf = float(
            mean_absolute_percentage_error(y_test, rf_pred) * 100
        )

        self.models["random-forest"] = rf_model

        # 3. XGBoost Regressor.
        xgb_model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )

        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        mae_xgb = float(mean_absolute_error(y_test, xgb_pred))
        rmse_xgb = float(
            np.sqrt(mean_squared_error(y_test, xgb_pred))
        )
        mape_xgb = float(
            mean_absolute_percentage_error(y_test, xgb_pred) * 100
        )

        self.models["xgboost"] = xgb_model

        # Store model evaluation metrics.
        self.metrics = {
            "seasonal-naive": {
                "name": "Seasonal Naive",
                "mae": round(mae_naive, 2),
                "rmse": round(rmse_naive, 2),
                "mape": round(mape_naive, 1),
            },
            "random-forest": {
                "name": "Random Forest",
                "mae": round(mae_rf, 2),
                "rmse": round(rmse_rf, 2),
                "mape": round(mape_rf, 1),
            },
            "xgboost": {
                "name": "XGBoost",
                "mae": round(mae_xgb, 2),
                "rmse": round(rmse_xgb, 2),
                "mape": round(mape_xgb, 1),
            },
        }

        # Select the model with the lowest MAE.
        best_key = min(
            self.metrics,
            key=lambda key: self.metrics[key]["mae"],
        )

        self.best_model_name = self.metrics[best_key]["name"]

        # Generate historical predictions.
        self.historical_predictions["xgboost"] = np.maximum(
            0, xgb_model.predict(X)
        )

        self.historical_predictions["random-forest"] = np.maximum(
            0, rf_model.predict(X)
        )

        self.historical_predictions["seasonal-naive"] = np.maximum(
            0, daily["Sales_Lag_7"].values
        )

    def _detect_anomalies(self):
        """Identify dates where actual sales differ significantly from predictions."""
        daily = self.daily_df

        predictions = self.historical_predictions["xgboost"]
        actual = daily["Sales"].values

        residuals = actual - predictions
        std_resid = (
            np.std(residuals) if len(residuals) > 0 else 1.0
        )

        category_by_date = (
            self.raw_df.groupby(
                ["Order Date", "Category"]
            )["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
            .drop_duplicates(subset=["Order Date"])
            .set_index("Order Date")
        )

        anomalies = []

        for i in range(len(daily)):
            actual_value = float(actual[i])
            predicted_value = float(predictions[i])

            if predicted_value <= 0:
                predicted_value = 1.0

            deviation = (
                (actual_value - predicted_value) / predicted_value
            ) * 100

            z_score = abs(residuals[i]) / (
                std_resid if std_resid > 0 else 1.0
            )

            if abs(deviation) >= 28.0 or z_score >= 1.8:
                if abs(deviation) >= 60.0 or z_score >= 2.5:
                    severity = "high"
                elif abs(deviation) >= 35.0 or z_score >= 2.0:
                    severity = "medium"
                else:
                    severity = "low"

                date_value = daily["Order Date"].iloc[i]
                date_string = date_value.strftime("%Y-%m-%d")

                category = "General"

                if date_value in category_by_date.index:
                    category = str(
                        category_by_date.loc[date_value, "Category"]
                    )

                anomalies.append({
                    "id": f"AN-{1000 + len(anomalies) + 1}",
                    "date": date_string,
                    "product": f"{category} Segment",
                    "category": category,
                    "actual": round(actual_value, 2),
                    "predicted": round(predicted_value, 2),
                    "deviation": round(deviation, 2),
                    "severity": severity,
                })

        anomalies.sort(
            key=lambda item: item["date"],
            reverse=True,
        )

        self.anomalies = anomalies[:20]

    def get_sales_data(self):
        """Return historical daily sales for the dashboard."""
        if not self.is_trained:
            self.load_and_train()

        daily = self.daily_df

        top_categories = (
            self.raw_df.groupby(
                ["Order Date", "Category"]
            )["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
            .drop_duplicates(subset=["Order Date"])
            .set_index("Order Date")
        )

        rows = []

        for _, row in daily.iterrows():
            date_value = row["Order Date"]

            category = (
                top_categories.loc[date_value, "Category"]
                if date_value in top_categories.index
                else "Electronics"
            )

            rows.append({
                "date": date_value.strftime("%Y-%m-%d"),
                "product": f"{category} Item",
                "category": str(category),
                "actual_sales": round(float(row["Sales"]), 2),
            })

        return rows

    def get_forecast_data(self, model_key="xgboost", future_days=30):
        """Return historical predictions and future sales estimates."""
        if not self.is_trained:
            self.load_and_train()

        key = model_key.lower().strip()

        if key not in [
            "xgboost",
            "random-forest",
            "seasonal-naive",
        ]:
            key = "xgboost"

        if future_days < 1:
            raise ValueError("future_days must be at least 1.")

        daily = self.daily_df.copy()

        historical_predictions = self.historical_predictions.get(
            key,
            self.historical_predictions["xgboost"],
        )

        residuals = (
            daily["Sales"].values - historical_predictions
        )

        residual_std = (
            float(np.std(residuals))
            if len(residuals) > 0
            else 5000.0
        )

        rows = []

        # Historical actual and predicted sales.
        for i in range(len(daily)):
            date_string = daily["Order Date"].iloc[i].strftime(
                "%Y-%m-%d"
            )

            actual_value = float(daily["Sales"].iloc[i])
            forecast_value = float(historical_predictions[i])

            rows.append({
                "date": date_string,
                "actual": round(actual_value, 2),
                "forecast": round(forecast_value, 2),
                "lower": round(
                    max(0, forecast_value - 1.96 * residual_std), 2
                ),
                "upper": round(
                    forecast_value + 1.96 * residual_std, 2
                ),
            })

        # Generate future dates.
        last_date = daily["Order Date"].max()

        future_dates = [
            last_date + timedelta(days=step)
            for step in range(1, future_days + 1)
        ]

        forecast_history = list(daily["Sales"].values)

        future_quantity = float(
            daily["Quantity"].tail(30).mean()
        )

        future_predictions = []

        # Iterative multi-step forecasting.
        model = self.models.get(
            key,
            self.models["xgboost"],
        )

        for current_date in future_dates:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            day_of_week = current_date.dayofweek
            quarter = current_date.quarter
            is_weekend = int(day_of_week >= 5)

            lag_1 = forecast_history[-1]
            lag_7 = (
                forecast_history[-7]
                if len(forecast_history) >= 7
                else lag_1
            )
            lag_14 = (
                forecast_history[-14]
                if len(forecast_history) >= 14
                else lag_1
            )
            lag_30 = (
                forecast_history[-30]
                if len(forecast_history) >= 30
                else lag_1
            )

            recent_7 = forecast_history[-7:]
            recent_14 = forecast_history[-14:]
            recent_30 = forecast_history[-30:]

            rolling_mean_7 = float(np.mean(recent_7))
            rolling_mean_14 = float(np.mean(recent_14))
            rolling_mean_30 = float(np.mean(recent_30))

            rolling_std_7 = (
                float(np.std(recent_7, ddof=1))
                if len(recent_7) >= 7
                else 0.0
            )

            if key == "seasonal-naive":
                prediction = max(0, float(lag_7))
            else:
                feature_row = pd.DataFrame([{
                    "Quantity": future_quantity,
                    "Year": year,
                    "Month": month,
                    "Day": day,
                    "DayOfWeek": day_of_week,
                    "Quarter": quarter,
                    "IsWeekend": is_weekend,
                    "Sales_Lag_1": lag_1,
                    "Sales_Lag_7": lag_7,
                    "Sales_Lag_14": lag_14,
                    "Sales_Lag_30": lag_30,
                    "Rolling_Mean_7": rolling_mean_7,
                    "Rolling_Mean_14": rolling_mean_14,
                    "Rolling_Mean_30": rolling_mean_30,
                    "Rolling_Std_7": rolling_std_7,
                }])

                prediction = max(
                    0,
                    float(
                        model.predict(
                            feature_row[self.feature_cols]
                        )[0]
                    ),
                )

            future_predictions.append(prediction)
            forecast_history.append(prediction)

        # Append future estimates and approximate uncertainty bounds.
        for step, current_date in enumerate(future_dates):
            forecast_value = future_predictions[step]

            horizon_expansion = (
                1.0 + (step / future_days) * 0.4
            )

            band = residual_std * 1.96 * horizon_expansion

            rows.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "actual": None,
                "forecast": round(forecast_value, 2),
                "lower": round(
                    max(0, forecast_value - band), 2
                ),
                "upper": round(forecast_value + band, 2),
            })

        self.future_forecasts[key] = future_predictions

        return rows

    def get_anomalies_data(self):
        """Return detected historical sales anomalies."""
        if not self.is_trained:
            self.load_and_train()

        return self.anomalies

    def get_models_comparison_data(self):
        """Return evaluation metrics for the three forecasting models."""
        if not self.is_trained:
            self.load_and_train()

        order = [
            "Seasonal Naive",
            "Random Forest",
            "XGBoost",
        ]

        results = []

        for name in order:
            for metric in self.metrics.values():
                if metric["name"] == name:
                    results.append(metric)
                    break

        return results

    def get_category_breakdown_data(self):
        """Return category sales percentages for a doughnut chart."""
        if not self.is_trained:
            self.load_and_train()

        category_sales = self.raw_df.groupby("Category")["Sales"].sum()
        total_sales = category_sales.sum()

        results = []

        for category, value in category_sales.sort_values(
            ascending=False
        ).items():
            percentage = (
                (value / total_sales) * 100
                if total_sales > 0
                else 0
            )

            results.append({
                "category": str(category),
                "sales": round(float(percentage), 1),
                "total_sales": round(float(value), 2),
            })

        return results

    def get_top_products_data(self):
        """Return top category-region segments and their sales growth."""
        if not self.is_trained:
            self.load_and_train()

        df = self.raw_df.copy()

        segments = (
            df.groupby(["Category", "Region"])["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
            .head(5)
        )

        midpoint = (
            df["Order Date"].min()
            + (
                df["Order Date"].max()
                - df["Order Date"].min()
            ) / 2
        )

        first_half = df[df["Order Date"] <= midpoint]
        second_half = df[df["Order Date"] > midpoint]

        results = []

        for _, row in segments.iterrows():
            category = row["Category"]
            region = row["Region"]

            sales_first = first_half[
                (first_half["Category"] == category)
                & (first_half["Region"] == region)
            ]["Sales"].sum()

            sales_second = second_half[
                (second_half["Category"] == category)
                & (second_half["Region"] == region)
            ]["Sales"].sum()

            growth = (
                ((sales_second - sales_first) / sales_first) * 100
                if sales_first > 0
                else 0.0
            )

            results.append({
                "product": f"{category} ({region})",
                "sales": round(float(row["Sales"]), 2),
                "growth": round(float(growth), 1),
            })

        return results

    def get_dashboard_results(self):
        """Return test predictions, forecasts, metrics, and feature importance."""
        if not self.is_trained:
            self.load_and_train()

        if self.daily_df is None or self.daily_df.empty:
            raise RuntimeError("No training data is available.")

        daily = self.daily_df

        split_idx = int(len(daily) * 0.8)

        if split_idx >= len(daily) or split_idx < 1:
            raise RuntimeError(
                "Not enough daily rows to evaluate the model."
            )

        test_df = daily.iloc[split_idx:].copy()

        X_test = test_df[self.feature_cols]
        y_test = test_df["Sales"]

        xgb_model = self.models["xgboost"]

        predictions = np.maximum(
            0,
            xgb_model.predict(X_test),
        )

        # Historical monthly totals use all raw transactions.
        historical_monthly = (
            self.raw_df.assign(
                month=self.raw_df["Order Date"].dt.to_period("M")
            )
            .groupby("month")["Sales"]
            .sum()
        )

        # Forecast the next 183 days.
        future_rows = self.get_forecast_data(
            model_key="xgboost",
            future_days=183,
        )

        future_df = pd.DataFrame(future_rows)
        future_df["date"] = pd.to_datetime(future_df["date"])

        future_df = future_df[
            future_df["actual"].isna()
        ].copy()

        if future_df.empty:
            forecast_monthly = pd.Series(dtype=float)
        else:
            forecast_monthly = (
                future_df.set_index("date")["forecast"]
                .resample("MS")
                .sum()
            )

        metric = self.metrics["xgboost"]

        r2 = float(r2_score(y_test, predictions))

        return {
            "test": {
                "dates": test_df["Order Date"]
                .dt.strftime("%Y-%m-%d")
                .tolist(),
                "actual": [
                    float(value) for value in y_test.tolist()
                ],
                "predicted": [
                    float(value) for value in predictions
                ],
            },
            "forecast": {
                "historical_dates": [
                    str(period)
                    for period in historical_monthly.index[-12:]
                ],
                "historical_sales": [
                    float(value)
                    for value in historical_monthly.iloc[-12:].tolist()
                ],
                "forecast_dates": [
                    date.strftime("%Y-%m")
                    for date in forecast_monthly.index
                ],
                "forecast_sales": [
                    float(value)
                    for value in forecast_monthly.tolist()
                ],
            },
            "metrics": {
                "mae": float(metric["mae"]),
                "rmse": float(metric["rmse"]),
                "mape": float(metric["mape"]),
                "r2": r2,
            },
            "feature_importance": {
                "features": list(self.feature_cols),
                "values": [
                    float(value)
                    for value in xgb_model.feature_importances_
                ],
            },
        }

    def reload_dataset(self, new_path):
        """Reload and retrain using an updated or uploaded dataset."""
        self.data_path = new_path

        # Clear old results before retraining.
        self.models = {}
        self.metrics = {}
        self.future_forecasts = {}
        self.historical_predictions = {}
        self.anomalies = []
        self.is_trained = False

        self.load_and_train()

        return {
            "rows": len(self.raw_df),
            "days": len(self.daily_df),
            "best_model": self.best_model_name,
        }
