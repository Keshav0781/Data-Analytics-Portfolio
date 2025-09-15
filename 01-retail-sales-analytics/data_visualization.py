"""
Advanced Data Visualization for Retail Analytics
Author: Keshav Jha
Description: Comprehensive visualization suite for retail sales data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class RetailDataVisualizer:
    def __init__(self):
        """Initialize the retail data visualizer"""
        self.data = None
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def load_sample_data(self):
        """Generate comprehensive sample data for visualization"""
        np.random.seed(42)
        
        # Generate 2 years of daily data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        regions = ['North', 'South', 'East', 'West']
        
        data = []
        for date in dates:
            for category in categories:
                for region in regions:
                    # Base sales with seasonal patterns
                    base_sales = {
                        'Electronics': 100, 'Clothing': 80, 'Home & Garden': 60,
                        'Books': 40, 'Sports': 70, 'Beauty': 90
                    }[category]
                    
                    # Regional variations
                    regional_multiplier = {
                        'North': 1.2, 'South': 1.0, 'East': 1.1, 'West': 0.9
                    }[region]
                    
                    # Seasonal effects
                    month = date.month
                    if category == 'Clothing':
                        seasonal = 1.5 if month in [3, 4, 9, 10] else 0.8
                    elif category == 'Electronics':
                        seasonal = 1.8 if month in [11, 12] else 1.0
                    elif category == 'Sports':
                        seasonal = 1.4 if month in [5, 6, 7, 8] else 0.9
                    else:
                        seasonal = 1.3 if month in [11, 12] else 1.0
                    
                    # Calculate final sales
                    daily_sales = base_sales * regional_multiplier * seasonal
                    daily_sales *= np.random.uniform(0.7, 1.3)
                    daily_sales = max(0, int(daily_sales))
                    
                    # Calculate revenue
                    avg_price = {
                        'Electronics': 250, 'Clothing': 60, 'Home & Garden': 80,
                        'Books': 20, 'Sports': 90, 'Beauty': 35
                    }[category]
                    
                    revenue = daily_sales * avg_price * np.random.uniform(0.8, 1.2)
                    
                    data.append({
                        'date': date,
                        'category': category,
                        'region': region,
                        'sales_units': daily_sales,
                        'revenue': round(revenue, 2),
                        'avg_price': round(revenue / daily_sales if daily_sales > 0 else avg_price, 2),
                        'year': date.year,
                        'month': date.month,
                        'quarter': date.quarter,
                        'day_of_week': date.day_name()
                    })
        
        self.data = pd.DataFrame(data)
        print(f"Generated {len(self.data):,} records for visualization")
        return self.data
    
    def create_executive_dashboard(self):
        """Create executive-level dashboard with key metrics"""
        if self.data is None:
            self.load_sample_data()
            
        # Set up the matplotlib style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Revenue Trend (Top spanning)
        ax1 = fig.add_subplot(gs[0, :2])
        monthly_revenue = self.data.groupby(['year', 'month'])['revenue'].sum().reset_index()
        monthly_revenue['period'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
        ax1.plot(monthly_revenue['period'], monthly_revenue['revenue'], 
                linewidth=3, marker='o', markersize=6, color='#1f77b4')
        ax1.fill_between(monthly_revenue['period'], monthly_revenue['revenue'], alpha=0.3, color='#1f77b4')
        ax1.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Revenue ($)')
        ax1.ticklabel_format(style='plain', axis='y')
        
        # 2. Category Performance (Top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        category_revenue = self.data.groupby('category')['revenue'].sum().sort_values(ascending=True)
        bars = ax2.barh(category_revenue.index, category_revenue.values, color=self.color_palette)
        ax2.set_title('Revenue by Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Revenue ($)')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'${width/1000000:.1f}M', ha='left', va='center')
        
        # 3. Regional Distribution (Middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        regional_data = self.data.groupby('region')['revenue'].sum()
        wedges, texts, autotexts = ax3.pie(regional_data.values, labels=regional_data.index, 
                                          autopct='%1.1f%%', colors=self.color_palette[:4])
        ax3.set_title('Revenue Distribution by Region', fontsize=14, fontweight='bold')
        
        # 4. Seasonal Heatmap (Middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        seasonal_data = self.data.groupby(['month', 'category'])['revenue'].sum().unstack()
        sns.heatmap(seasonal_data.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, cbar_kws={'shrink': .8})
        ax4.set_title('Seasonal Revenue Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Category')
        
        # 5. Daily Sales Pattern (Bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        daily_pattern = self.data.groupby('day_of_week')['sales_units'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_pattern = daily_pattern.reindex(day_order)
        bars = ax5.bar(daily_pattern.index, daily_pattern.values, color='lightcoral')
        ax5.set_title('Average Daily Sales Pattern', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Average Units Sold')
        plt.setp(ax5.get_xticklabels(), rotation=45)
        
        # 6. Price Analysis (Bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        category_prices = self.data.groupby('category')['avg_price'].mean().sort_values()
        ax6.barh(category_prices.index, category_prices.values, color='lightgreen')
        ax6.set_title('Average Price by Category', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Average Price ($)')
        
        # 7. Performance Metrics (Bottom)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Calculate key metrics
        total_revenue = self.data['revenue'].sum()
        total_units = self.data['sales_units'].sum()
        avg_order_value = total_revenue / len(self.data[self.data['sales_units'] > 0])
        best_category = self.data.groupby('category')['revenue'].sum().idxmax()
        
        # Create metrics text
        metrics_text = f"""
        KEY PERFORMANCE INDICATORS
        
        Total Revenue: ${total_revenue:,.0f}        Total Units Sold: {total_units:,}        Average Order Value: ${avg_order_value:.2f}
        
        Best Performing Category: {best_category}        Analysis Period: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}
        """
        
        ax7.text(0.5, 0.5, metrics_text, transform=ax7.transAxes, fontsize=12,
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax7.axis('off')
        
        plt.suptitle('Retail Sales Analytics - Executive Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
    def create_trend_analysis(self):
        """Create detailed trend analysis visualizations"""
        if self.data is None:
            self.load_sample_data()
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Category Trends Over Time
        monthly_category = self.data.groupby(['year', 'month', 'category'])['revenue'].sum().reset_index()
        monthly_category['period'] = pd.to_datetime(monthly_category[['year', 'month']].assign(day=1))
        
        ax1 = axes[0, 0]
        for i, category in enumerate(self.data['category'].unique()):
            cat_data = monthly_category[monthly_category['category'] == category]
            ax1.plot(cat_data['period'], cat_data['revenue'], 
                    marker='o', linewidth=2, label=category, color=self.color_palette[i])
        
        ax1.set_title('Revenue Trends by Category', fontweight='bold')
        ax1.set_ylabel('Revenue ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Growth Rate Analysis
        ax2 = axes[0, 1]
        quarterly_revenue = self.data.groupby(['year', 'quarter'])['revenue'].sum().reset_index()
        quarterly_revenue['prev_revenue'] = quarterly_revenue['revenue'].shift(1)
        quarterly_revenue['growth_rate'] = ((quarterly_revenue['revenue'] - quarterly_revenue['prev_revenue']) / quarterly_revenue['prev_revenue']) * 100
        quarterly_revenue['period'] = quarterly_revenue['year'].astype(str) + '-Q' + quarterly_revenue['quarter'].astype(str)
        
        bars = ax2.bar(range(len(quarterly_revenue)), quarterly_revenue['growth_rate'], 
                      color=['green' if x >= 0 else 'red' for x in quarterly_revenue['growth_rate']])
        ax2.set_title('Quarterly Growth Rate', fontweight='bold')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.set_xticks(range(len(quarterly_revenue)))
        ax2.set_xticklabels(quarterly_revenue['period'], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Regional Performance Comparison
        ax3 = axes[1, 0]
        regional_monthly = self.data.groupby(['year', 'month', 'region'])['revenue'].sum().reset_index()
        regional_monthly['period'] = pd.to_datetime(regional_monthly[['year', 'month']].assign(day=1))
        
        for i, region in enumerate(self.data['region'].unique()):
            region_data = regional_monthly[regional_monthly['region'] == region]
            ax3.plot(region_data['period'], region_data['revenue'], 
                    marker='s', linewidth=2, label=region, color=self.color_palette[i])
        
        ax3.set_title('Revenue Trends by Region', fontweight='bold')
        ax3.set_ylabel('Revenue ($)')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Market Share Evolution
        ax4 = axes[1, 1]
        category_share = self.data.groupby(['year', 'category'])['revenue'].sum().reset_index()
        category_share_pivot = category_share.pivot(index='year', columns='category', values='revenue')
        category_share_pct = category_share_pivot.div(category_share_pivot.sum(axis=1), axis=0) * 100
        
        bottom = np.zeros(len(category_share_pct))
        for i, category in enumerate(category_share_pct.columns):
            ax4.bar(category_share_pct.index, category_share_pct[category], 
                   bottom=bottom, label=category, color=self.color_palette[i % len(self.color_palette)])
            bottom += category_share_pct[category]
        
        ax4.set_title('Market Share by Category (%)', fontweight='bold')
        ax4.set_ylabel('Market Share (%)')
        ax4.set_xlabel('Year')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def create_customer_analysis(self):
        """Create customer behavior analysis visualizations"""
        if self.data is None:
            self.load_sample_data()
            
        # Generate customer-level data
        np.random.seed(42)
        customers = []
        for i in range(5000):
            # Customer segments
            segment = np.random.choice(['Premium', 'Regular', 'Budget'], p=[0.2, 0.5, 0.3])
            
            # Segment characteristics
            if segment == 'Premium':
                avg_order_value = np.random.normal(200, 50)
                purchase_frequency = np.random.poisson(8)
            elif segment == 'Regular':
                avg_order_value = np.random.normal(100, 30)
                purchase_frequency = np.random.poisson(4)
            else:
                avg_order_value = np.random.normal(50, 20)
                purchase_frequency = np.random.poisson(2)
            
            customers.append({
                'customer_id': f'CUST_{i+1:05d}',
                'segment': segment,
                'avg_order_value': max(10, avg_order_value),
                'purchase_frequency': max(1, purchase_frequency),
                'total_spent': max(10, avg_order_value) * max(1, purchase_frequency)
            })
        
        customer_df = pd.DataFrame(customers)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Customer Segmentation
        ax1 = axes[0, 0]
        segment_counts = customer_df['segment'].value_counts()
        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index, 
                                          autopct='%1.1f%%', colors=['gold', 'lightblue', 'lightcoral'])
        ax1.set_title('Customer Segmentation', fontweight='bold')
        
        # 2. Purchase Frequency Distribution
        ax2 = axes[0, 1]
        for segment in customer_df['segment'].unique():
            segment_data = customer_df[customer_df['segment'] == segment]
            ax2.hist(segment_data['purchase_frequency'], alpha=0.7, label=segment, bins=15)
        ax2.set_title('Purchase Frequency by Segment', fontweight='bold')
        ax2.set_xlabel('Annual Purchase Frequency')
        ax2.set_ylabel('Number of Customers')
        ax2.legend()
        
        # 3. Average Order Value Analysis
        ax3 = axes[1, 0]
        customer_df.boxplot(column='avg_order_value', by='segment', ax=ax3)
        ax3.set_title('Average Order Value by Segment', fontweight='bold')
        ax3.set_xlabel('Customer Segment')
        ax3.set_ylabel('Average Order Value ($)')
        plt.suptitle('')  # Remove automatic title
        
        # 4. Customer Value Distribution
        ax4 = axes[1, 1]
        ax4.scatter(customer_df['purchase_frequency'], customer_df['avg_order_value'], 
                   c=customer_df['total_spent'], cmap='viridis', alpha=0.6)
        ax4.set_title('Customer Value Analysis', fontweight='bold')
        ax4.set_xlabel('Purchase Frequency')
        ax4.set_ylabel('Average Order Value ($)')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Total Spent ($)')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function for visualization demo"""
    print("Starting Retail Data Visualization Suite...")
    
    # Initialize visualizer
    visualizer = RetailDataVisualizer()
    
    # Load sample data
    print("Loading sample retail data...")
    visualizer.load_sample_data()
    
    # Create executive dashboard
    print("Creating executive dashboard...")
    visualizer.create_executive_dashboard()
    
    # Create trend analysis
    print("Creating trend analysis...")
    visualizer.create_trend_analysis()
    
    # Create customer analysis
    print("Creating customer analysis...")
    visualizer.create_customer_analysis()
    
    print("\n=== VISUALIZATION SUITE COMPLETE ===")
    print("All visualizations have been generated successfully!")

if __name__ == "__main__":
    main()
