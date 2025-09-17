"""
Create SQLite Database from Processed CSVs
Author: Keshav Jha
"""

import os
import sqlite3
import pandas as pd

def main():
    data_dir = "data/processed"
    db_path = os.path.join(data_dir, "sales_marketing.db")

    # Ensure folder exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Directory not found: {data_dir}")

    # Input files
    customers_csv = os.path.join(data_dir, "customers.csv")
    products_csv = os.path.join(data_dir, "products.csv")
    sales_csv = os.path.join(data_dir, "sales.csv")

    for f in [customers_csv, products_csv, sales_csv]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"❌ Missing file: {f}")

    print("📂 Loading CSV files...")
    customers = pd.read_csv(customers_csv)
    products = pd.read_csv(products_csv)
    sales = pd.read_csv(sales_csv)

    print("📝 Creating SQLite database...")
    conn = sqlite3.connect(db_path)

    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    sales.to_sql("sales", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    print(f"✅ Database created successfully: {db_path}")

if __name__ == "__main__":
    main()
