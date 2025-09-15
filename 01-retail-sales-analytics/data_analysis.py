"""
Retail Sales Data Analysis
Author: Keshav Jha
Description: Comprehensive analysis of e-commerce sales data for pattern identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RetailSalesAnalyzer:
    def __init__(self):
        """Initialize the retail sales analyzer"""
        self.data = None
        self.cleaned_data = None
        
    def generate_sample_data(self, n_records=10000):
        """Generate realistic sample e-commerce data"""
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        
        # Generate sample data
        data = []
        for i in range(n_records):
            order_date = np.random.choice(dates)
            category = np.random.choice(categories)
            
            # Seasonal effects
            month = order_date.month
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.5
            elif month in [6, 7]:  # Summer
                seasonal_multiplier = 1.2
                
            # Generate realistic prices based on category
            base_prices = {
                'Electronics': 200, 'Clothing': 50, 'Home & Garden': 75,
                'Books': 15, 'Sports': 80, 'Beauty': 30
            }
            
            price = base_prices[category] * np.random.uniform(0.5, 3.0) * seasonal_multiplier
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            data.append({
                'order_id': f'ORD_{i+1:06d}',
                'order_date': order_date,
                'category': category,
                'price': round(price, 2),
                'quantity': quantity,
                'total_amount': round(price * quantity, 2),
                'customer_id': f'CUST_{np.random.randint(1, 2000):05d}'
            })
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the sales data"""
        if self.data is None:
            raise ValueError("No data to clean. Generate or load data first.")
            
        # Create a copy for cleaning
        self.cleaned_data = self.data.copy()
        
        # Convert date column
        self.cleaned_data['order_date'] = pd.to_datetime(self.cleaned_data['order_date'])
        
        # Extract date features
        self.cleaned_data['year'] = self.cleaned_data['order_date'].dt.year
        self.cleaned_data['month'] = self.cleaned_data['order_date'].dt.month
        self.cleaned_data['quarter'] = self.cleaned_data['order_date'].dt.quarter
        self.cleaned_data['day_of_week'] = self.cleaned_data['order_date'].dt.day_name()
        
        # Remove any potential outliers (orders > 99th percentile)
        price_threshold = self.cleaned_data['total_amount'].quantile(0.99)
        self.cleaned_data = self.cleaned_data[self.cleaned_data['total_amount'] <= price_threshold]
        
        print(f"Data cleaned successfully. Shape: {self.cleaned_data.shape}")
        return self.cleaned_data
    
    def exploratory_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        if self.cleaned_data is None:
            raise ValueError("Clean the data first using clean_data() method.")
            
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        # Basic statistics
        print("1. BASIC STATISTICS:")
        print(f"Total Orders: {len(self.cleaned_data):,}")
        print(f"Total Revenue: ${self.cleaned_data['total_amount'].sum():,.2f}")
        print(f"Average Order Value: ${self.cleaned_data['total_amount'].mean():.2f}")
        print(f"Date Range: {self.cleaned_data['order_date'].min()} to {self.cleaned_data['order_date'].max()}")
        
        # Category analysis
        print("\n2. CATEGORY PERFORMANCE:")
        category_stats = self.cleaned_data.groupby('category').agg({
            'total_amount': ['count', 'sum', 'mean'],
            'quantity': 'sum'
        }).round(2)
        category_stats.columns = ['Orders', 'Revenue', 'Avg_Order_Value', 'Units_Sold']
        print(category_stats.sort_values('Revenue', ascending=False))
        
        # Monthly trends
        print("\n3. MONTHLY TRENDS:")
        monthly_stats = self.cleaned_data.groupby(['year', 'month']).agg({
            'total_amount': ['count', 'sum']
        }).round(2)
        monthly_stats.columns = ['Orders', 'Revenue']
        print(monthly_stats.tail(10))
        
        return category_stats, monthly_stats
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.cleaned_data is None:
            raise ValueError("Clean the data first.")
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Retail Sales Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue by Category
        category_revenue = self.cleaned_data.groupby('category')['total_amount'].sum().sort_values(ascending=True)
        axes[0,0].barh(category_revenue.index, category_revenue.values, color='skyblue')
        axes[0,0].set_title('Revenue by Product Category')
        axes[0,0].set_xlabel('Revenue ($)')
        
        # 2. Monthly Revenue Trend
        monthly_revenue = self.cleaned_data.groupby(self.cleaned_data['order_date'].dt.to_period('M'))['total_amount'].sum()
        axes[0,1].plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker='o', linewidth=2)
        axes[0,1].set_title('Monthly Revenue Trend')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Revenue ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Order Distribution by Day of Week
        day_orders = self.cleaned_data['day_of_week'].value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_orders = day_orders.reindex(day_order)
        axes[1,0].bar(day_orders.index, day_orders.values, color='lightcoral')
        axes[1,0].set_title('Orders by Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Number of Orders')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Price Distribution
        axes[1,1].hist(self.cleaned_data['total_amount'], bins=50, color='lightgreen', alpha=0.7)
        axes[1,1].set_title('Order Value Distribution')
        axes[1,1].set_xlabel('Order Value ($)')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
    def identify_patterns(self):
        """Identify key business patterns and insights"""
        print("\n=== KEY BUSINESS INSIGHTS ===\n")
        
        # Top performing categories
        top_category = self.cleaned_data.groupby('category')['total_amount'].sum().idxmax()
        print(f"1. TOP PERFORMING CATEGORY: {top_category}")
        
        # Peak sales day
        peak_day = self.cleaned_data['day_of_week'].value_counts().idxmax()
        print(f"2. PEAK SALES DAY: {peak_day}")
        
        # Seasonal patterns
        seasonal_revenue = self.cleaned_data.groupby('quarter')['total_amount'].sum()
        peak_quarter = seasonal_revenue.idxmax()
        print(f"3. PEAK SEASON: Q{peak_quarter} (Revenue: ${seasonal_revenue[peak_quarter]:,.2f})")
        
        # Customer insights
        customer_orders = self.cleaned_data.groupby('customer_id')['order_id'].count()
        repeat_customers = (customer_orders > 1).sum()
        repeat_rate = (repeat_customers / len(customer_orders)) * 100
        print(f"4. REPEAT CUSTOMER RATE: {repeat_rate:.1f}%")
        
        return {
            'top_category': top_category,
            'peak_day': peak_day,
            'peak_quarter': peak_quarter,
            'repeat_rate': repeat_rate
        }

def main():
    """Main execution function"""
    print("Starting Retail Sales Analysis...")
    
    # Initialize analyzer
    analyzer = RetailSalesAnalyzer()
    
    # Generate sample data
    print("Generating sample data...")
    analyzer.generate_sample_data(n_records=10000)
    
    # Clean data
    print("Cleaning data...")
    analyzer.clean_data()
    
    # Perform analysis
    print("Performing exploratory analysis...")
    analyzer.exploratory_analysis()
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations()
    
    # Identify patterns
    insights = analyzer.identify_patterns()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the generated visualizations and insights above.")

if __name__ == "__main__":
    main()
