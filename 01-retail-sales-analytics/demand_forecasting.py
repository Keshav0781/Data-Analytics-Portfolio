"""
Demand Forecasting for Retail Sales
Author: Keshav Jha
Description: Time series forecasting models for inventory planning and demand prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self):
        """Initialize the demand forecasting system"""
        self.data = None
        self.forecast_data = None
        self.models = {}
        self.predictions = {}
        
    def generate_historical_data(self, months=24):
        """Generate realistic historical sales data for forecasting"""
        np.random.seed(42)
        
        # Generate daily data for specified months
        start_date = datetime(2022, 1, 1)
        end_date = start_date + timedelta(days=months * 30)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        
        data = []
        for date in dates:
            for category in categories:
                # Base demand
                base_demand = {'Electronics': 50, 'Clothing': 40, 'Home & Garden': 30,
                              'Books': 25, 'Sports': 35, 'Beauty': 45}[category]
                
                # Seasonal patterns
                month = date.month
                seasonal_factor = 1.0
                
                # Holiday season boost
                if month in [11, 12]:
                    seasonal_factor = 1.8
                elif month in [6, 7]:  # Summer
                    seasonal_factor = 1.3
                elif month in [2, 3]:  # Spring
                    seasonal_factor = 1.1
                
                # Day of week patterns
                weekday_factor = 1.0
                if date.weekday() in [5, 6]:  # Weekend
                    weekday_factor = 1.2
                elif date.weekday() == 4:  # Friday
                    weekday_factor = 1.15
                
                # Add trend (slight growth over time)
                days_since_start = (date - start_date).days
                trend_factor = 1 + (days_since_start * 0.0001)  # 0.01% daily growth
                
                # Calculate final demand with some randomness
                demand = base_demand * seasonal_factor * weekday_factor * trend_factor
                demand = max(0, int(demand * np.random.normal(1, 0.2)))
                
                # Calculate revenue (demand * average price)
                avg_prices = {'Electronics': 200, 'Clothing': 50, 'Home & Garden': 75,
                             'Books': 15, 'Sports': 80, 'Beauty': 30}
                revenue = demand * avg_prices[category] * np.random.uniform(0.8, 1.2)
                
                data.append({
                    'date': date,
                    'category': category,
                    'demand': demand,
                    'revenue': round(revenue, 2),
                    'year': date.year,
                    'month': date.month,
                    'quarter': date.quarter,
                    'day_of_week': date.weekday(),
                    'is_weekend': 1 if date.weekday() >= 5 else 0
                })
        
        self.data = pd.DataFrame(data)
        print(f"Generated {len(self.data):,} historical records for forecasting")
        return self.data
    
    def prepare_forecast_features(self, category=None):
        """Prepare features for demand forecasting"""
        if self.data is None:
            raise ValueError("Generate historical data first")
            
        # Filter by category if specified
        if category:
            forecast_data = self.data[self.data['category'] == category].copy()
        else:
            # Aggregate across all categories
            forecast_data = self.data.groupby('date').agg({
                'demand': 'sum',
                'revenue': 'sum',
                'year': 'first',
                'month': 'first',
                'quarter': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first'
            }).reset_index()
        
        # Sort by date
        forecast_data = forecast_data.sort_values('date').reset_index(drop=True)
        
        # Create lagged features
        forecast_data['demand_lag_1'] = forecast_data['demand'].shift(1)
        forecast_data['demand_lag_7'] = forecast_data['demand'].shift(7)  # Weekly lag
        forecast_data['demand_lag_30'] = forecast_data['demand'].shift(30)  # Monthly lag
        
        # Rolling averages
        forecast_data['demand_ma_7'] = forecast_data['demand'].rolling(window=7).mean()
        forecast_data['demand_ma_30'] = forecast_data['demand'].rolling(window=30).mean()
        
        # Seasonal indicators
        forecast_data['is_holiday_season'] = forecast_data['month'].isin([11, 12]).astype(int)
        forecast_data['is_summer'] = forecast_data['month'].isin([6, 7, 8]).astype(int)
        
        # Remove rows with NaN values from lagged features
        forecast_data = forecast_data.dropna().reset_index(drop=True)
        
        self.forecast_data = forecast_data
        print(f"Prepared {len(forecast_data)} records for forecasting")
        return forecast_data
    
    def build_forecasting_models(self, test_size=0.2):
        """Build multiple forecasting models"""
        if self.forecast_data is None:
            raise ValueError("Prepare forecast features first")
            
        # Define features for modeling
        feature_columns = [
            'month', 'quarter', 'day_of_week', 'is_weekend',
            'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
            'demand_ma_7', 'demand_ma_30', 'is_holiday_season', 'is_summer'
        ]
        
        X = self.forecast_data[feature_columns]
        y = self.forecast_data['demand']
        
        # Split data (time series split - last 20% for testing)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.test_dates = self.forecast_data['date'][split_idx:].reset_index(drop=True)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Model 1: Linear Regression
        print("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        self.models['Linear Regression'] = lr_model
        self.predictions['Linear Regression'] = lr_pred
        
        # Model 2: Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = rf_pred
        
        # Model 3: Simple Moving Average (baseline)
        print("Calculating Moving Average baseline...")
        ma_pred = self.forecast_data['demand_ma_30'][split_idx:].values
        self.predictions['Moving Average'] = ma_pred
        
        print("All models trained successfully!")
        return self.models
    
    def evaluate_models(self):
        """Evaluate and compare forecasting models"""
        if not self.predictions:
            raise ValueError("Build models first")
            
        results = {}
        
        print("\n=== MODEL EVALUATION RESULTS ===\n")
        print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 50)
        
        for model_name, predictions in self.predictions.items():
            # Remove any NaN values for evaluation
            valid_mask = ~np.isnan(predictions)
            y_true = self.y_test[valid_mask]
            y_pred = predictions[valid_mask]
            
            if len(y_pred) > 0:
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2
                }
                
                print(f"{model_name:<20} {mae:<10.2f} {rmse:<10.2f} {r2:<10.3f}")
        
        # Determine best model
        best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
        print(f"\nBest Model (lowest MAE): {best_model}")
        
        return results
    
    def create_forecast_visualizations(self):
        """Create comprehensive forecast visualizations"""
        if not self.predictions:
            raise ValueError("Build models first")
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Demand Forecasting Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted comparison
        ax1 = axes[0, 0]
        ax1.plot(self.test_dates, self.y_test.values, label='Actual', linewidth=2, color='blue')
        
        colors = ['red', 'green', 'orange']
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            valid_mask = ~np.isnan(predictions)
            dates_valid = self.test_dates[valid_mask]
            pred_valid = predictions[valid_mask]
            ax1.plot(dates_valid, pred_valid, label=f'{model_name}', 
                    linewidth=1.5, alpha=0.8, color=colors[i % len(colors)])
        
        ax1.set_title('Actual vs Predicted Demand')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Feature importance (Random Forest)
        if 'Random Forest' in self.models:
            feature_columns = [
                'month', 'quarter', 'day_of_week', 'is_weekend',
                'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
                'demand_ma_7', 'demand_ma_30', 'is_holiday_season', 'is_summer'
            ]
            importances = self.models['Random Forest'].feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            ax2 = axes[0, 1]
            ax2.barh(feature_imp['feature'], feature_imp['importance'])
            ax2.set_title('Feature Importance (Random Forest)')
            ax2.set_xlabel('Importance')
        
        # 3. Residuals plot
        ax3 = axes[1, 0]
        best_model_name = 'Random Forest'  # Usually performs best
        if best_model_name in self.predictions:
            residuals = self.y_test.values - self.predictions[best_model_name]
            ax3.scatter(self.predictions[best_model_name], residuals, alpha=0.6)
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_title(f'Residuals Plot ({best_model_name})')
            ax3.set_xlabel('Predicted Values')
            ax3.set_ylabel('Residuals')
        
        # 4. Monthly forecast accuracy
        ax4 = axes[1, 1]
        if len(self.test_dates) > 30:  # If we have enough data
            monthly_actual = []
            monthly_pred = []
            months = []
            
            for month in range(1, 13):
                month_mask = self.test_dates.dt.month == month
                if month_mask.sum() > 0:
                    monthly_actual.append(self.y_test[month_mask].mean())
                    monthly_pred.append(self.predictions['Random Forest'][month_mask].mean())
                    months.append(month)
            
            ax4.plot(months, monthly_actual, 'o-', label='Actual', linewidth=2)
            ax4.plot(months, monthly_pred, 's-', label='Predicted', linewidth=2)
            ax4.set_title('Monthly Average Demand')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Average Demand')
            ax4.legend()
            ax4.set_xticks(months)
        
        plt.tight_layout()
        plt.show()
    
    def generate_future_forecast(self, days_ahead=30):
        """Generate future demand predictions"""
        if not self.models:
            raise ValueError("Build models first")
            
        print(f"\n=== GENERATING {days_ahead}-DAY FORECAST ===\n")
        
        # Get the best model
        best_model = self.models.get('Random Forest', self.models.get('Linear Regression'))
        
        # Create future dates
        last_date = self.forecast_data['date'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        # This is a simplified forecast - in practice, you'd need to handle
        # the recursive nature of time series forecasting more carefully
        future_features = []
        for date in future_dates:
            features = {
                'month': date.month,
                'quarter': date.quarter,
                'day_of_week': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'is_holiday_season': 1 if date.month in [11, 12] else 0,
                'is_summer': 1 if date.month in [6, 7, 8] else 0,
                # Use recent averages for lagged features
                'demand_lag_1': self.forecast_data['demand'].tail(1).iloc[0],
                'demand_lag_7': self.forecast_data['demand'].tail(7).mean(),
                'demand_lag_30': self.forecast_data['demand'].tail(30).mean(),
                'demand_ma_7': self.forecast_data['demand'].tail(7).mean(),
                'demand_ma_30': self.forecast_data['demand'].tail(30).mean()
            }
            future_features.append(features)
        
        future_df = pd.DataFrame(future_features)
        future_predictions = best_model.predict(future_df)
        
        # Create forecast summary
        forecast_summary = pd.DataFrame({
            'date': future_dates,
            'predicted_demand': np.round(future_predictions).astype(int)
        })
        
        print("Future Demand Forecast:")
        print(forecast_summary.head(10))
        
        total_forecast = forecast_summary['predicted_demand'].sum()
        avg_daily = total_forecast / days_ahead
        print(f"\nForecast Summary ({days_ahead} days):")
        print(f"Total Expected Demand: {total_forecast:,} units")
        print(f"Average Daily Demand: {avg_daily:.0f} units")
        
        return forecast_summary

def main():
    """Main execution function"""
    print("Starting Demand Forecasting Analysis...")
    
    # Initialize forecaster
    forecaster = DemandForecaster()
    
    # Generate historical data
    print("Generating historical sales data...")
    forecaster.generate_historical_data(months=24)
    
    # Prepare features
    print("Preparing forecasting features...")
    forecaster.prepare_forecast_features()
    
    # Build models
    print("Building forecasting models...")
    forecaster.build_forecasting_models()
    
    # Evaluate models
    print("Evaluating model performance...")
    forecaster.evaluate_models()
    
    # Create visualizations
    print("Creating forecast visualizations...")
    forecaster.create_forecast_visualizations()
    
    # Generate future forecast
    print("Generating future forecasts...")
    future_forecast = forecaster.generate_future_forecast(days_ahead=30)
    
    print("\n=== FORECASTING ANALYSIS COMPLETE ===")
    print("Review the model performance metrics and visualizations above.")

if __name__ == "__main__":
    main()
