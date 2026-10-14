import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
OUTPUT_CHARTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "charts")

SPARK_APP_NAME = "GlobalFashionDemand"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "6g"

DATA_FILES = {
    "transactions": "transactions.csv",
    "customers": "customers.csv",
    "products": "products.csv",
    "stores": "stores.csv",
    "employees": "employees.csv",
    "discounts": "discounts.csv",
}
