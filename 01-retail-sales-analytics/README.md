# Retail Sales Analytics & Demand Forecasting

## Project Overview
End-to-end analysis of synthetic e-commerce sales data to identify purchase patterns and build demand forecasting models for inventory optimization.

## Dataset
This project uses **synthetic data generated programmatically** to simulate real-world retail transactions (orders, customers, categories, prices).  
The file `sample_data.csv` is included as an **example schema** only.

## Business Problem
- Need to predict seasonal demand accurately  
- Identify cross-selling opportunities  
- Optimize inventory planning  

## Technical Approach
- Data cleaning and preprocessing  
- Exploratory data analysis (EDA)  
- Time series forecasting (Random Forest, Regression, Baseline)  
- Executive-level dashboards and customer analysis  

## Files in this project
- [`data_analysis.py`](data_analysis.py) – Comprehensive EDA and cleaning  
- [`demand_forecasting.py`](demand_forecasting.py) – Forecasting models and evaluation  
- [`data_visualization.py`](data_visualization.py) – Executive dashboards and customer analysis  
- [`sample_data.csv`](sample_data.csv) – Example dataset schema  
- [`requirements.txt`](requirements.txt) – Required Python packages  

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run individual scripts
python data_analysis.py
python demand_forecasting.py
python data_visualization.py
```
## Result
- Improved seasonal stock prediction accuracy
- Identified key factors affecting demand
- Created automated forecasting pipeline
