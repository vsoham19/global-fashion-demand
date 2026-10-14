# Global Fashion Demand Forecasting and Retail Inventory Optimization

A Big Data Analytics project that processes 6.4 million fashion retail transactions with Apache PySpark, forecasts demand at the store-category level using Spark MLlib, and converts those forecasts into inventory replenishment recommendations.

**Team:** Dhairya Thacker (25MCA060) and Soham Vaghasia (25MCA064)

---

## What This Project Does

1. **Ingests** six CSV files (6.42M transactions, 17,940 products, 35 stores, 7 countries)
2. **Cleans and integrates** them into a single analytical table via distributed joins
3. **Explores** demand patterns across country, store, category, season, weekday, and hour
4. **Engineers features** such as lagged demand, rolling averages, and discount flags using Window functions
5. **Forecasts** demand with three MLlib models (Random Forest, GBT, Linear Regression) tuned via CrossValidator
6. **Recommends inventory actions**: safety stock, reorder points, stockout/overstock flags, and order quantities for all 105 store-category segments
7. **Serves** the results through a Flask dashboard with four pages (Overview, Inventory, Insights, Models)

Key result: the Random Forest model achieves **R² = 0.8208** on a chronological test set, beating a naive baseline by ~26% RMSE.

---

## Prerequisites

- **Python 3.13** (3.10+ should work)
- **Java 8 or 11** (required by Spark — check with `java -version`)
- ~5 GB free disk space (dataset + Spark temp files)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/vsoham19/global-fashion-demand.git
cd global-fashion-demand

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the dataset (~235 MB download, ~1 GB extracted)
python download_data.py
```

## Running the Project

Run these steps in order:

```bash
# Step 1: Full pipeline (~20-30 min on a laptop)
# Ingestion -> Cleaning -> Integration -> EDA -> Feature Engineering
# -> Model Training with CrossValidation -> Inventory Engine
python main_pipeline.py

# Step 2: Generate all charts from the pipeline outputs (~30 seconds)
python visualize.py

# Step 3: Launch the dashboard
python app.py
# Open http://localhost:5000 in your browser
```

**Shortcut:** If you only need to rerun the inventory engine (it reads the Parquet output from a previous run, takes ~30 seconds instead of re-training models):

```bash
python run_inventory.py
```

**Windows users:** you can double-click `start_dashboard.bat` as a shortcut for step 3.

---

## Project Structure

```
global-fashion-demand/
├── main_pipeline.py        Main entry point: runs all phases end-to-end
├── download_data.py        Downloads the Kaggle dataset into data/raw/
├── run_inventory.py        Standalone inventory engine (reads Parquet)
├── visualize.py            Generates all matplotlib charts
├── app.py                  Flask dashboard backend (API + pages)
├── start_dashboard.bat     Windows launcher for the dashboard
├── requirements.txt        Python dependencies
│
├── src/                    Reusable pipeline modules
│   ├── config.py           Paths and Spark configuration
│   ├── ingestion.py        SparkSession creation and CSV loading
│   ├── preprocessing.py    Cleaning functions for all six tables
│   ├── features.py         Temporal, lag, and rolling feature engineering
│   ├── forecasting.py      MLlib pipeline builders and evaluation
│   └── evaluation.py       Metric computation (RMSE, MAE, R², sMAPE)
│
├── templates/              Dashboard HTML (Tailwind, dark-on-light theme)
│   ├── _head.html          Shared head and Tailwind config
│   ├── _nav.html           Shared navigation bar
│   ├── overview.html       KPIs and headline charts
│   ├── inventory.html      Risk summary and recommendations table
│   ├── insights.html       Geographic / Seasonal / Discount / Customer tabs
│   └── models.html         Model cards and evaluation results
│
├── data/                   (gitignored) Created by download_data.py
│   ├── raw/                Six source CSVs
│   └── parquet/            Spark outputs after running the pipeline
│
└── outputs/                (gitignored) Created by the pipeline
    ├── tables/             CSV and JSON results (metrics, recommendations)
    └── charts/             14 PNG visualizations
```

---

## Pipeline Phases at a Glance

| Phase | Script function | What happens |
|-------|----------------|--------------|
| 2 | `phase2_ingestion` | Load six CSVs into Spark DataFrames |
| 3 | `phase3_clean` | Standardize names, parse dates, drop 22K duplicates |
| 4 | `phase4_integration` | Four left joins (stores, customers, products, discounts) |
| 5 | `phase5_eda` | Nine distributed aggregations |
| 6/7 | `phase6_demand_aggregation` | Aggregate to date × store × category (686K rows), build lag/rolling features |
| 8 | `phase8_forecasting` | Chronological 70/15/15 split, train 3 models with CrossValidator |
| 10 | `phase10_retail_insights` | Payment methods, price bins, customer economics, yearly growth |
| 11 | `phase11_inventory_engine` | Safety stock, reorder points, risk flags, order recommendations |

---

## Model Results (Test Set)

| Model | RMSE | MAE | R² | sMAPE |
|-------|------|-----|-----|-------|
| Naive baseline | 12.87 | 6.24 | — | — |
| Linear Regression | 9.58 | 4.84 | 0.8194 | 47.7% |
| **Random Forest** | **9.54** | **4.43** | **0.8208** | 39.9% |
| Gradient Boosted Trees | 9.96 | 4.52 | 0.8045 | 40.6% |

Top predictive features: rolling_avg_7 (35%), rolling_avg_14 (21%), avg_discount (11%).

---

## Inventory Engine Summary

Across 105 store-category segments (7-day lead time, 95% service level):

- 14 segments flagged as **stockout risk** (13.3%), concentrated in Children's wear and the New York store
- 91 segments **healthy**, 0 overstock (current stock is simulated — the dataset has no inventory snapshots)
- Total recommended order value: **$3.2M**

Output: `outputs/tables/inventory_recommendations.csv` (24 columns per segment).

---

## Dashboard API Endpoints

`app.py` serves JSON for the frontend, reading precomputed files from `outputs/tables/`:

```
/api/summary                     Headline KPIs
/api/country_demand              Revenue by country
/api/store_demand                Store performance
/api/monthly_demand              Monthly trend
/api/category_demand             Category split
/api/model_metrics               RMSE/MAE/R²/sMAPE for all models
/api/feature_importances         RF and GBT importance rankings
/api/segment_metrics             Per-country and per-category evaluation
/api/inventory_summary           Risk counts and totals
/api/inventory_recommendations   Full 105-segment recommendation table
/api/demand_statistics           Demand mean/std per segment
```

Charts are served as static PNGs under `/charts/`.

---

## Notes and Known Limitations

- **The dataset is synthetic.** Performance numbers are internal comparisons, not real-world estimates.
- **Train/test split is chronological** (70/15/15 by date). Never shuffle time-series data — it leaks future information.
- **Inventory levels are simulated** since the dataset contains no stock records. The engine logic is real; only the current-stock input is synthetic.
- **Hour 0 shows negative revenue** in the raw data — likely returns timestamped at midnight. Left as-is.
- The Spark temp directory grows during training; if you run low on disk, delete `spark_tmp/` between runs.

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.13 |
| PySpark | 3.5.5 |
| Java | 8 |
| Flask | 3.x |
| Matplotlib | 3.10 |
| Tailwind CSS | CDN (dashboard styling) |
