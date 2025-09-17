"""
Create SQLite Database from Synthetic Sales Data
Author: Keshav Jha
"""

import sqlite3
import pandas as pd
import os

# Paths
DATA_DIR = "data/processed"
DB_PATH = "data/processed/sales_marketing.db"

def create_database():
    # Remove existing DB if re-running
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load CSV files
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))

    # Write tables
    customers.to_sql("customers", conn, index=False, if_exists="replace")
    products.to_sql("products", conn, index=False, if_exists="replace")
    sales.to_sql("sales", conn, index=False, if_exists="replace")

    conn.commit()
    conn.close()
    print(f"✅ SQLite database created at {DB_PATH}")

if __name__ == "__main__":
    create_database()
