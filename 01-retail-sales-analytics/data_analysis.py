"""
Retail Sales Data Analysis
Author: Keshav Jha
Description: Comprehensive analysis of e-commerce sales data for pattern identification
Dataset: Online Retail (UCI Machine Learning Repository)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class RetailSalesAnalyzer:
    def __init__(self, input_file: str = "data/processed/online_retail.csv"):
        """Initialize with dataset path"""
        self.input_file = Path(input_file)
        self.data = None
        self.cleaned_data = None
        self.reports_path = Path("reports")
        self.reports_path.mkdir(exist_ok=True)

    def load_data(self):
        """Load dataset from processed CSV"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.input_file}")
        self.data = pd.read_csv(self.input_file, low_memory=False)
        return self.data

    def clean_data(self):
        """Clean and preprocess sales data"""
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")

        df = self.data.copy()
        df.columns = [c.strip() for c in df.columns]

        # Convert dates
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        # Remove cancelled transactions
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

        # Keep positive quantities
        df = df[pd.to_numeric(df['Quantity'], errors='coerce') > 0]

        # Drop missing customers
        df = df.dropna(subset=['CustomerID'])

        # Compute revenue
        df['Revenue'] = pd.to_numeric(df['UnitPrice'], errors='coerce') * pd.to_numeric(df['Quantity'], errors='coerce')

        # Extract features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

        self.cleaned_data = df
        print(f"Data cleaned successfully. Shape: {df.shape}")
        return df

    def exploratory_analysis(self):
        """Perform exploratory analysis and save results"""
        if self.cleaned_data is None:
            raise ValueError("Run clean_data() first.")

        df = self.cleaned_data

        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Total Orders: {len(df):,}")
        print(f"Total Revenue: £{df['Revenue'].sum():,.2f}")
        print(f"Average Order Value: £{df['Revenue'].mean():.2f}")
        print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

        # Category performance
        top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(20)
        top_products.to_csv(self.reports_path / "top_products.csv")

        # Monthly revenue
        monthly_revenue = df.set_index('InvoiceDate')['Revenue'].resample('M').sum()
        monthly_revenue.to_csv(self.reports_path / "monthly_revenue.csv")

        return top_products, monthly_revenue

    def create_visualizations(self):
        """Create and save key visualizations"""
        if self.cleaned_data is None:
            raise ValueError("Run clean_data() first.")

        df = self.cleaned_data

        # Revenue by top 20 products
        top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(20)
        plt.figure(figsize=(10,6))
        top_products.plot(kind='barh')
        plt.title("Top 20 Products by Revenue")
        plt.tight_layout()
        plt.savefig(self.reports_path / "top_products.png")
        plt.close()

        # Monthly revenue trend
        monthly_revenue = df.set_index('InvoiceDate')['Revenue'].resample('M').sum()
        plt.figure(figsize=(10,6))
        monthly_revenue.plot(marker='o')
        plt.title("Monthly Revenue Trend")
        plt.ylabel("Revenue (£)")
        plt.tight_layout()
        plt.savefig(self.reports_path / "monthly_revenue.png")
        plt.close()

        # Orders by day of week
        orders_by_day = df['DayOfWeek'].value_counts().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        )
        plt.figure(figsize=(8,5))
        orders_by_day.plot(kind='bar', color='skyblue')
        plt.title("Orders by Day of Week")
        plt.tight_layout()
        plt.savefig(self.reports_path / "orders_by_day.png")
        plt.close()

        print(f"Visualizations saved to {self.reports_path}")

    def identify_patterns(self):
        """Print and save key business insights"""
        if self.cleaned_data is None:
            raise ValueError("Run clean_data() first.")

        df = self.cleaned_data
        insights = {}

        # Peak products
        insights['Top_Product'] = df.groupby('Description')['Revenue'].sum().idxmax()

        # Peak day
        insights['Peak_Day'] = df['DayOfWeek'].value_counts().idxmax()

        # Peak quarter
        peak_quarter = df.groupby('Quarter')['Revenue'].sum().idxmax()
        insights['Peak_Quarter'] = f"Q{peak_quarter}"

        # Repeat customers
        customer_orders = df.groupby('CustomerID')['InvoiceNo'].nunique()
        repeat_rate = (customer_orders > 1).mean() * 100
        insights['Repeat_Rate'] = f"{repeat_rate:.1f}%"

        # Save insights
        pd.Series(insights).to_csv(self.reports_path / "business_insights.csv")

        print("=== BUSINESS INSIGHTS ===")
        for k, v in insights.items():
            print(f"{k}: {v}")

        return insights


def main():
    analyzer = RetailSalesAnalyzer()
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.exploratory_analysis()
    analyzer.create_visualizations()
    analyzer.identify_patterns()
    print("Analysis complete. Reports saved to /reports/")


if __name__ == "__main__":
    main()
