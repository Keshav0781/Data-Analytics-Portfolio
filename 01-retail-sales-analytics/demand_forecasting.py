"""
Demand Forecasting for Retail Sales (revised)
Author: Keshav Jha (updated)
Description: Time-series forecasting models for inventory planning and demand prediction.
Behavior:
 - Prefers real transactional dataset at data/processed/online_retail.csv (UCI Online Retail).
 - Falls back to synthetic generated data only if the real dataset is missing.
 - Aggregates to daily demand (Quantity) and revenue, creates lag/rolling features,
   trains models, evaluates, and saves metrics / predictions / models / plots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")


class DemandForecaster:
    def __init__(self, processed_csv: str = "data/processed/online_retail.csv", test_size_days: int = 60):
        self.processed_csv = Path(processed_csv)
        self.test_size_days = test_size_days
        self.raw_df = None
        self.daily = None
        self.features_df = None
        self.models = {}
        self.predictions = {}
        self.reports = Path("reports")
        self.models_dir = Path("models")
        self.reports.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # placeholders for evaluation
        self.X_test = None
        self.y_test = None
        self.test_dates = None

    def load_or_generate_data(self):
        """Load processed UCI Online Retail CSV if available, otherwise generate synthetic data."""
        if self.processed_csv.exists():
            print(f"Loading processed dataset from: {self.processed_csv}")
            df = pd.read_csv(self.processed_csv, low_memory=False)
            # Ensure expected columns
            expected_cols = {"InvoiceDate", "Quantity", "UnitPrice", "CustomerID", "InvoiceNo"}
            if not expected_cols.intersection(set(df.columns)):
                print("Warning: processed CSV doesn't have expected transactional columns. Falling back to synthetic.")
                self._generate_synthetic_transactions()
                return
            # normalize column names
            df.columns = [c.strip() for c in df.columns]
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            df = df.dropna(subset=["InvoiceDate"])
            self.raw_df = df
            print(f"Loaded {len(df):,} transactional rows.")
        else:
            print(f"Processed dataset not found at {self.processed_csv} — generating synthetic data instead.")
            self._generate_synthetic_transactions()

    def _generate_synthetic_transactions(self, months: int = 24):
        """Create a synthetic transactional dataset (only used as fallback)."""
        print("Generating synthetic transactional dataset (fallback).")
        start_date = pd.Timestamp("2022-01-01")
        end_date = start_date + pd.DateOffset(months=months)
        dates = pd.date_range(start_date, end_date, freq="D")
        categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
        rows = []
        rng = np.random.default_rng(42)
        order_id = 1
        for date in dates:
            for cat in categories:
                qty = max(0, int(rng.normal(loc=30 + (date.month % 12) * 2, scale=8)))
                unit_price = {
                    "Electronics": 200, "Clothing": 50, "Home & Garden": 75,
                    "Books": 15, "Sports": 80, "Beauty": 30
                }[cat] * float(rng.uniform(0.8, 1.2))
                # Many transactions in real world are multiple small orders; we convert to aggregated-like rows
                rows.append({
                    "InvoiceNo": f"SYN_{order_id:06d}",
                    "InvoiceDate": date,
                    "StockCode": f"SYN_{cat[:3].upper()}",
                    "Description": cat,
                    "Quantity": qty,
                    "UnitPrice": round(unit_price, 2),
                    "CustomerID": f"SYN_{rng.integers(1, 2000):05d}"
                })
                order_id += 1
        df = pd.DataFrame(rows)
        self.raw_df = df
        print(f"Generated synthetic dataset with {len(df):,} rows.")

    def aggregate_daily(self):
        """Aggregate transactional data to daily demand (sum of Quantity) and daily revenue."""
        if self.raw_df is None:
            raise ValueError("No raw transactional data. Run load_or_generate_data() first.")
        df = self.raw_df.copy()
        # Normalize column names used below
        cols = {c.lower(): c for c in df.columns}
        # Best-effort normalization
        date_col = None
        for candidate in ["InvoiceDate", "invoice_date", "date", "Invoice Date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            # try lowercase match
            for c in df.columns:
                if c.lower() == "invoicedate" or c.lower() == "date":
                    date_col = c
                    break
        if date_col is None:
            raise ValueError("Could not find a date column in transactional data.")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if "Quantity" not in df.columns and "quantity" in [c.lower() for c in df.columns]:
            # map lowercase quantity
            qty_col = [c for c in df.columns if c.lower() == "quantity"][0]
        else:
            qty_col = "Quantity"

        if qty_col not in df.columns:
            raise ValueError("Could not find Quantity column in transactional data.")

        # Some processed CSVs may have UnitPrice
        if "UnitPrice" in df.columns:
            df["Revenue"] = pd.to_numeric(df[qty_col], errors="coerce") * pd.to_numeric(df["UnitPrice"], errors="coerce")
        else:
            # fallback: approximate revenue = quantity * 1 (not ideal)
            df["Revenue"] = pd.to_numeric(df[qty_col], errors="coerce")

        # Keep positive quantities and valid dates
        df = df[pd.to_numeric(df[qty_col], errors="coerce") > 0]
        df = df.dropna(subset=[date_col])

        # Aggregate to daily
        df = df.set_index(date_col)
        daily = df.resample("D").agg({qty_col: "sum", "Revenue": "sum"})
        daily = daily.rename(columns={qty_col: "demand"}).fillna(0)
        daily = daily.reset_index().rename(columns={date_col: "date"})
        # add calendar features
        daily["year"] = daily["date"].dt.year
        daily["month"] = daily["date"].dt.month
        daily["quarter"] = daily["date"].dt.quarter
        daily["day_of_week"] = daily["date"].dt.weekday
        daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

        self.daily = daily
        daily.to_csv(self.reports / "daily_aggregated.csv", index=False)
        print(f"Aggregated to daily series with {len(daily):,} days. Saved to reports/daily_aggregated.csv")
        return daily

    def create_features(self, max_lag=30):
        """Create lag and rolling features required for modeling."""
        if self.daily is None:
            raise ValueError("Daily aggregated series missing. Run aggregate_daily() first.")
        df = self.daily.copy().sort_values("date").reset_index(drop=True)
        # lags
        for lag in [1, 7, 14, 30]:
            df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
        # rolling windows
        df["demand_ma_7"] = df["demand"].shift(1).rolling(window=7, min_periods=1).mean()
        df["demand_ma_30"] = df["demand"].shift(1).rolling(window=30, min_periods=1).mean()
        # season flags
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        # Drop initial NaNs introduced by lags
        df = df.dropna().reset_index(drop=True)
        self.features_df = df
        self.features_df.to_csv(self.reports / "features_prepared.csv", index=False)
        print(f"Prepared features: {len(df):,} records. Saved to reports/features_prepared.csv")
        return df

    def train_models(self):
        """Train models on historical data and keep test partition at the end (last N days)."""
        if self.features_df is None:
            raise ValueError("Features not prepared. Run create_features() first.")
        df = self.features_df.copy().sort_values("date").reset_index(drop=True)

        # Use last test_size_days as test
        if self.test_size_days >= len(df):
            raise ValueError("test_size_days too large for available data.")
        split_idx = len(df) - self.test_size_days
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)

        feature_cols = [
            "month", "quarter", "day_of_week", "is_weekend",
            "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_30",
            "demand_ma_7", "demand_ma_30", "is_holiday_season", "is_summer"
        ]
        # ensure feature cols exist (demand_lag_14 may not be present if we didn't create it)
        feature_cols = [c for c in feature_cols if c in df.columns]

        X_train = train[feature_cols].values
        y_train = train["demand"].values
        X_test = test[feature_cols].values
        y_test = test["demand"].values

        # keep for evaluation & plotting
        self.X_test = X_test
        self.y_test = y_test
        self.test_dates = pd.to_datetime(test["date"])

        print(f"Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        self.models["LinearRegression"] = lr
        self.predictions["LinearRegression"] = lr_pred

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        self.models["RandomForest"] = rf
        self.predictions["RandomForest"] = rf_pred

        # Moving average baseline: use demand_ma_30 from test set if present, otherwise use last available mean
        if "demand_ma_30" in test.columns:
            ma_pred = test["demand_ma_30"].values
        else:
            ma_pred = np.full_like(y_test, fill_value=np.mean(y_train))
        self.predictions["MovingAverage"] = ma_pred

        # Save models (best one later will be saved again)
        joblib.dump(lr, self.models_dir / "lr_initial.joblib")
        joblib.dump(rf, self.models_dir / "rf_initial.joblib")
        print("Initial models saved to models/ (lr_initial.joblib, rf_initial.joblib)")

        return self.models

    def evaluate(self):
        """Evaluate trained models and persist metrics + predictions."""
        if not self.predictions or self.X_test is None:
            raise ValueError("No predictions found. Run train_models() first.")
        results = {}
        metrics_rows = []
        for name, preds in self.predictions.items():
            # align lengths
            preds = np.array(preds)
            y_true = np.array(self.y_test)
            valid_mask = ~np.isnan(preds)
            preds = preds[valid_mask]
            y_eval = y_true[valid_mask]
            if len(y_eval) == 0:
                continue
            mae = mean_absolute_error(y_eval, preds)
            rmse = mean_squared_error(y_eval, preds, squared=False)
            # MAPE: avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                mape = np.mean(np.abs((y_eval - preds) / np.where(y_eval == 0, np.nan, y_eval))) * 100
                mape = np.nan_to_num(mape, nan=np.inf)
            r2 = r2_score(y_eval, preds)
            results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
            metrics_rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2})

        metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
        metrics_df.to_csv(self.reports / "forecast_metrics.csv", index=False)
        print("\nModel evaluation metrics saved to reports/forecast_metrics.csv")
        print(metrics_df.to_string(index=False))

        # Save predictions (aligned with test dates)
        preds_df = pd.DataFrame({"date": self.test_dates})
        for name, preds in self.predictions.items():
            preds_df[name] = np.round(preds).astype(int)
        preds_df["actual"] = np.round(self.y_test).astype(int)
        preds_df.to_csv(self.reports / "forecast_predictions.csv", index=False)
        print("Predictions saved to reports/forecast_predictions.csv")

        # Save the best model (by MAE) if any
        if not metrics_df.empty:
            best_name = metrics_df.iloc[0]["Model"]
            best_model = self.models.get(best_name) or self.models.get(best_name.replace(" ", ""))
            if best_model is not None:
                joblib.dump(best_model, self.models_dir / f"best_model_{best_name}.joblib")
                print(f"Best model ({best_name}) saved to models/best_model_{best_name}.joblib")

        return results

    def plot_evaluation(self):
        """Create and save evaluation plots (actual vs predicted, residuals, feature importance)."""
        if self.test_dates is None or self.y_test is None:
            raise ValueError("No test data available. Run train_models() and evaluate() first.")
        dates = pd.to_datetime(self.test_dates)
        preds_df = pd.read_csv(self.reports / "forecast_predictions.csv", parse_dates=["date"])
        # Actual vs Predicted (RandomForest & LinearRegression & MovingAverage if present)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, preds_df["actual"], label="Actual", linewidth=2)
        for col in ["RandomForest", "LinearRegression", "MovingAverage"]:
            if col in preds_df.columns:
                plt.plot(dates, preds_df[col], label=col)
        plt.title("Actual vs Predicted Demand (Test Window)")
        plt.xlabel("Date")
        plt.ylabel("Demand (units)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.reports / "forecast_actual_vs_pred.png")
        plt.close()
        print("Saved: reports/forecast_actual_vs_pred.png")

        # Feature importance for RandomForest
        if "RandomForest" in self.models:
            rf = self.models["RandomForest"]
            # get feature names used during training
            feature_cols = [c for c in self.features_df.columns if c.startswith(("demand_lag", "demand_ma", "month", "quarter", "day_of_week", "is_weekend", "is_holiday_season", "is_summer"))]
            # If model has feature_importances_
            if hasattr(rf, "feature_importances_"):
                importances = rf.feature_importances_
                # align length (safe fallback)
                if len(importances) == len(feature_cols):
                    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=True)
                    plt.figure(figsize=(8, 6))
                    plt.barh(fi_df["feature"], fi_df["importance"])
                    plt.title("RandomForest Feature Importance")
                    plt.tight_layout()
                    plt.savefig(self.reports / "feature_importance.png")
                    plt.close()
                    print("Saved: reports/feature_importance.png")

        # Residuals for best model if RandomForest exists
        if "RandomForest" in self.predictions:
            best_pred = np.array(self.predictions["RandomForest"])
            residuals = np.array(self.y_test) - best_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(best_pred, residuals, alpha=0.5)
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title("Residuals (RandomForest)")
            plt.tight_layout()
            plt.savefig(self.reports / "forecast_residuals.png")
            plt.close()
            print("Saved: reports/forecast_residuals.png")

    def generate_future(self, days_ahead: int = 30):
        """Generate a simple future forecast using the best available model."""
        # Try to find best model from saved metrics
        metrics_path = self.reports / "forecast_metrics.csv"
        if not metrics_path.exists():
            raise ValueError("Metrics file not found. Run evaluate() first.")
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            raise ValueError("No metrics found to pick best model.")
        best_model_name = metrics.sort_values("MAE").iloc[0]["Model"]
        # Load model
        candidate_path = self.models_dir / f"best_model_{best_model_name}.joblib"
        if not candidate_path.exists():
            # fallback: pick any existing model
            if "RandomForest" in self.models:
                best = self.models["RandomForest"]
                best_model_name = "RandomForest"
            elif "LinearRegression" in self.models:
                best = self.models["LinearRegression"]
                best_model_name = "LinearRegression"
            else:
                raise ValueError("No trained model available for forecasting.")
        else:
            best = joblib.load(candidate_path)

        # Build future feature rows
        last_date = pd.to_datetime(self.features_df["date"]).max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        history = self.features_df["demand"].copy().reset_index(drop=True)

        future_rows = []
        for i, d in enumerate(future_dates):
            # compute lag values from historical + previously predicted values (recursive)
            # For simplicity we use last available history (non-recursive approach)
            lag_1 = history.iloc[-1]
            lag_7 = history.tail(7).mean() if len(history) >= 7 else history.mean()
            lag_30 = history.tail(30).mean() if len(history) >= 30 else history.mean()
            ma_7 = lag_7
            ma_30 = lag_30
            row = {
                "month": d.month,
                "quarter": d.quarter,
                "day_of_week": d.weekday(),
                "is_weekend": 1 if d.weekday() >= 5 else 0,
                "demand_lag_1": lag_1,
                "demand_lag_7": lag_7,
                "demand_lag_30": lag_30,
                "demand_ma_7": ma_7,
                "demand_ma_30": ma_30,
                "is_holiday_season": 1 if d.month in [11, 12] else 0,
                "is_summer": 1 if d.month in [6, 7, 8] else 0
            }
            future_rows.append(row)

        future_X = pd.DataFrame(future_rows)
        # Keep only columns the model knows about
        model_features = [c for c in future_X.columns if c in self.features_df.columns]
        future_X = future_X[model_features]
        preds = best.predict(future_X)
        forecast_df = pd.DataFrame({"date": future_dates, "predicted_demand": np.round(preds).astype(int)})
        forecast_df.to_csv(self.reports / "future_forecast.csv", index=False)
        print(f"Generated {len(forecast_df)}-day forecast. Saved to reports/future_forecast.csv")
        return forecast_df


def main():
    forecaster = DemandForecaster(processed_csv="data/processed/online_retail.csv", test_size_days=60)
    forecaster.load_or_generate_data()
    forecaster.aggregate_daily()
    forecaster.create_features()
    forecaster.train_models()
    forecaster.evaluate()
    forecaster.plot_evaluation()
    future = forecaster.generate_future(days_ahead=30)
    print("\nFuture forecast (first 5 rows):")
    print(future.head())

if __name__ == "__main__":
    main()
