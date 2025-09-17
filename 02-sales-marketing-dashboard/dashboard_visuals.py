"""
Dashboard Visuals for Sales & Marketing Project
Author: Keshav Jha
Description: Generates visual reports from SQLite database queries
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
DB_PATH = "data/processed/sales_marketing.db"
OUT_DIR = "reports/visuals"

os.makedirs(OUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")

def get_data(query):
    """Run SQL query and return DataFrame"""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)

def plot_revenue_by_region():
    df = get_data("""
        SELECT c.region, SUM(s.revenue) AS revenue
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        GROUP BY c.region
        ORDER BY revenue DESC
    """)
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="region", y="revenue", palette="Blues_d")
    plt.title("Total Revenue by Region")
    plt.ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "revenue_by_region.png"))
    plt.close()
    print("✅ Saved revenue_by_region.png")

def plot_top_products():
    df = get_data("""
        SELECT p.category, SUM(s.revenue) AS revenue
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
        LIMIT 10
    """)
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, y="category", x="revenue", palette="viridis")
    plt.title("Top 10 Categories by Revenue")
    plt.xlabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_categories.png"))
    plt.close()
    print("✅ Saved top_categories.png")

def plot_monthly_sales_trend():
    df = get_data("""
        SELECT strftime('%Y-%m', s.date) AS month,
               SUM(s.revenue) AS revenue
        FROM sales s
        GROUP BY month
        ORDER BY month
    """)
    plt.figure(figsize=(10,4))
    sns.lineplot(data=df, x="month", y="revenue", marker="o")
    plt.xticks(rotation=45)
    plt.title("Monthly Revenue Trend")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "monthly_trend.png"))
    plt.close()
    print("✅ Saved monthly_trend.png")

def main():
    print("📊 Generating dashboard visuals...")
    plot_revenue_by_region()
    plot_top_products()
    plot_monthly_sales_trend()
    print("🎉 All visuals saved in reports/visuals/")

if __name__ == "__main__":
    main()
