# 02 - Sales & Marketing Insights (SQL + Dashboard)

## Project Overview
This project demonstrates SQL-based analytics and executive dashboarding for retail sales and marketing. It uses programmatically generated synthetic data (orders, products, customers, campaigns) to simulate a real business environment.

## What you get
- Synthetic dataset (CSV) and a SQLite DB for quick SQL queries.
- Ready-to-run SQL queries (monthly revenue, top products, campaign ROI, RFM).
- Python script that generates dashboard PNGs (monthly revenue, top products, campaign ROI, RFM scatter).
- Power BI instructions (DAX measures and visuals) so you can build an interactive dashboard.

## Files
- `00_generate_data.py` — generate synthetic CSVs (`data/processed/`)
- `01_create_sqlite_db.py` — create `data/sales_dashboard.db` (SQLite) from CSVs
- `queries.sql` — useful analysis queries you can run in SQLite/DB Browser
- `dashboard_visuals.py` — create visual PNGs saved to `reports/`
- `requirements.txt` — Python packages for this project
- `power_bi_instructions.md` — instructions & DAX formulas for Power BI

## How to run (step-by-step)
1. Create a virtual environment and install packages:
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
pip install -r 02-sales-marketing-dashboard/requirements.txt

