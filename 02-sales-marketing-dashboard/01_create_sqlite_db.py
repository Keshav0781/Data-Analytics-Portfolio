"""
Create SQLite Database from Processed CSVs
Author: Keshav Jha
"""

import os
import sqlite3
import pandas as pd

DATA_DIR = "data/processed"
DB_PATH = os.path.join(DATA_DIR, "sales_marketing.db")

def main():
    print(f"📂 Checking directory: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("❌ Data directory does not exist.")
        return

    # Input CSVs
    files = {
        "customers": os.path.join(DATA_DIR, "customers.csv"),
        "products": os.path.join(DATA_DIR, "products.csv"),
        "sales": os.path.join(DATA_DIR, "sales.csv"),
    }

    # Check files
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"❌ Missing {name} file: {path}")
            return
        else:
            print(f"✅ Found {name} file: {path}")

    print("📝 Creating SQLite database...")
    conn = sqlite3.connect(DB_PATH)

    for table_name, path in files.items():
        df = pd.read_csv(path)
        print(f"   → Writing {table_name} ({len(df)} rows)")
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    print(f"✅ Database created successfully: {DB_PATH}")

if __name__ == "__main__":
    main()
