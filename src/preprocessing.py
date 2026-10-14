from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, to_timestamp, to_date, trim, lower, when, count, lit
)
from pyspark.sql.types import DoubleType, IntegerType


def standardize_column_names(df: DataFrame) -> DataFrame:
    for c in df.columns:
        new_name = c.strip().lower().replace(" ", "_").replace("-", "_")
        new_name = "".join(ch if ch.isalnum() or ch == "_" else "" for ch in new_name)
        new_name = "_".join(filter(None, new_name.split("_")))
        df = df.withColumnRenamed(c, new_name)
    return df


def clean_transactions(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)

    date_cols = [c for c in df.columns if "date" in c or c == "date"]
    for dc in date_cols:
        df = df.withColumn(dc, to_timestamp(col(dc), "yyyy-MM-dd HH:mm:ss"))

    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))

    if "discount" in df.columns:
        df = df.withColumn("discount", col("discount").cast(DoubleType()))
    if "quantity" in df.columns:
        df = df.withColumn("quantity", col("quantity").cast(IntegerType()))
    if "unit_price" in df.columns:
        df = df.withColumn("unit_price", col("unit_price").cast(DoubleType()))
    if "line_total" in df.columns:
        df = df.withColumn("line_total", col("line_total").cast(DoubleType()))
    if "invoice_total" in df.columns:
        df = df.withColumn("invoice_total", col("invoice_total").cast(DoubleType()))

    return df


def clean_customers(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)
    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))
    dob_col = [c for c in df.columns if "birth" in c]
    for dc in dob_col:
        df = df.withColumn(dc, to_date(col(dc), "yyyy-MM-dd"))
    return df


def clean_products(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)
    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))
    if "production_cost" in df.columns:
        df = df.withColumn("production_cost", col("production_cost").cast(DoubleType()))
    return df


def clean_stores(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)
    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))
    return df


def clean_employees(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)
    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))
    return df


def clean_discounts(df: DataFrame) -> DataFrame:
    df = standardize_column_names(df)
    for dc in ["start", "end"]:
        if dc in df.columns:
            df = df.withColumn(dc, to_date(col(dc), "yyyy-MM-dd"))
    str_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() == "string"]
    for sc in str_cols:
        df = df.withColumn(sc, trim(col(sc)))
    if "discont" in df.columns:
        df = df.withColumnRenamed("discont", "discount_pct")
    return df


def drop_duplicates(df: DataFrame, key_cols=None):
    before = df.count()
    if key_cols:
        df = df.dropDuplicates(key_cols)
    else:
        df = df.dropDuplicates()
    after = df.count()
    return df, before - after


def remove_null_key_rows(df: DataFrame, key_cols):
    for kc in key_cols:
        df = df.filter(col(kc).isNotNull())
    return df
