"""
Run SQL queries against data/processed/sales_marketing.db and save results to reports/queries/
Author: Keshav Jha
"""

import sqlite3
import os
import pandas as pd

DB = "data/processed/sales_marketing.db"
QUERIES_FILE = "queries.sql"
OUT_DIR = "reports/queries"

os.makedirs(OUT_DIR, exist_ok=True)

def split_queries(sql_text):
    # naive split on semicolon for our file (works for these queries)
    parts = [q.strip() for q in sql_text.split(";") if q.strip()]
    return parts

def main():
    if not os.path.exists(DB):
        print("❌ Database not found:", DB)
        return
    if not os.path.exists(QUERIES_FILE):
        print("❌ queries.sql not found")
        return

    with open(QUERIES_FILE, "r") as f:
        sql_text = f.read()

    queries = split_queries(sql_text)
    conn = sqlite3.connect(DB)

    for i, q in enumerate(queries, start=1):
        name = f"query_{i:02d}"
        try:
            df = pd.read_sql_query(q, conn)
            out_csv = os.path.join(OUT_DIR, f"{name}.csv")
            df.to_csv(out_csv, index=False)
            print(f"✅ Saved {out_csv} ({len(df)} rows)")
        except Exception as e:
            print(f"❌ Failed to run query {i}: {e}")

    conn.close()
    print("All queries executed. Check reports/queries/ for CSV outputs.")

if __name__ == "__main__":
    main()
