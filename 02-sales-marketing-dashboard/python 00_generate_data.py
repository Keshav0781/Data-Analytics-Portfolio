"""
Generate Synthetic Sales & Marketing Data
Author: Keshav Jha
Description: Creates synthetic transaction-level sales data for analysis and dashboarding.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure output folder exists
os.makedirs("data/processed", exist_ok=True)

# Parameters
n_customers = 500
n_products = 50
n_transactions = 10000
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start_date, end_date, freq="D")

# Customer segments
segments = ["Premium", "Regular", "Budget"]
regions = ["North", "South", "East", "West"]

# Generate customers
customers = pd.DataFrame({
    "customer_id": [f"CUST_{i+1:04d}" for i in range(n_customers)],
    "segment": np.random.choice(segments, n_customers, p=[0.2, 0.5, 0.3]),
    "region": np.random.choice(regions, n_customers)
})

# Generate products
categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
products = pd.DataFrame({
    "product_id": [f"PROD_{i+1:03d}" for i in range(n_products)],
    "category": np.random.choice(categories, n_products),
    "base_price": np.random.randint(10, 500, n_products)
})

# Generate transactions
np.random.seed(42)
transactions = []
for i in range(n_transactions):
    cust = customers.sample(1).iloc[0]
    prod = products.sample(1).iloc[0]
    date = np.random.choice(date_range)
    quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
    price = prod.base_price * np.random.uniform(0.8, 1.2)
    revenue = round(price * quantity, 2)

    transactions.append({
        "transaction_id": f"TXN_{i+1:06d}",
        "date": date,
        "customer_id": cust.customer_id,
        "region": cust.region,
        "segment": cust.segment,
        "product_id": prod.product_id,
        "category": prod.category,
        "quantity": quantity,
        "unit_price": round(price, 2),
        "revenue": revenue
    })

sales = pd.DataFrame(transactions)

# Save outputs
customers.to_csv("data/processed/customers.csv", index=False)
products.to_csv("data/processed/products.csv", index=False)
sales.to_csv("data/processed/sales.csv", index=False)

print("✅ Synthetic data generated and saved in data/processed/")

