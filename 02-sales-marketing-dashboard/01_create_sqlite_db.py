"""
Create SQLite Database from Synthetic Sales Data
Author: Keshav Jha
"""

import sqlite3
import pandas as pd
import os

# Paths
DATA_DIR = "data/processed"
DB_PATH = os.path.join(DATA_DIR, "sales_marketing.db")

def create_database():
    print("📂 Looking for data in:", DATA_DIR)

    # Check files exist
    files = ["customers.csv", "products.csv", "sales.csv"]
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            print(f"❌ Missing file: {path}")
            return
        else:
            print(f"✅ Found {path}")

    # Remove existing DB if re-running
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("⚠️ Removed existing database.")

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)

    # Load CSVs
    print("📥 Loading CSVs...")
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))

    # Write tables
    print("📝 Writing tables to database...")
    customers.to_sql("customers", conn, index=False, if_exists="replace")
    products.to_sql("products", conn, index=False, if_exists="replace")
    sales.to_sql("sales", conn, index=False, if_exists="replace")

    conn.commit()
    conn.close()
    print(f"✅ SQLite database created at {DB_PATH}")

if __name__ == "__main__":
    create_database()
