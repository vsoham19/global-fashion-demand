from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, year, month, quarter, dayofweek, hour, weekofyear,
    when, lag, round as spark_round, coalesce, lit, datediff, current_date
)
from pyspark.sql.types import DoubleType, IntegerType


def add_temporal_features(df: DataFrame, date_col: str = "date") -> DataFrame:
    df = df.withColumn("year", year(col(date_col)))
    df = df.withColumn("month", month(col(date_col)))
    df = df.withColumn("quarter", quarter(col(date_col)))
    df = df.withColumn("day_of_week", dayofweek(col(date_col)))
    df = df.withColumn("hour", hour(col(date_col)))
    df = df.withColumn("week_of_year", weekofyear(col(date_col)))
    return df


def add_revenue_features(df: DataFrame) -> DataFrame:
    if "quantity" in df.columns and "unit_price" in df.columns:
        df = df.withColumn(
            "gross_revenue",
            spark_round(col("quantity") * col("unit_price"), 2)
        )
    if "discount" in df.columns:
        df = df.withColumn("is_discounted", when(col("discount") > 0, 1).otherwise(0))
        if "gross_revenue" in df.columns:
            df = df.withColumn(
                "discounted_revenue",
                spark_round(col("gross_revenue") * (1 - col("discount")), 2)
            )
    return df


def add_price_tier(df: DataFrame, price_col: str = "unit_price") -> DataFrame:
    df = df.withColumn(
        "price_tier",
        when(col(price_col) < 30, "low")
        .when(col(price_col) < 80, "mid")
        .otherwise("high")
    )
    return df


def add_lag_features(df: DataFrame, partition_cols, order_col, value_col, lags=(1, 7, 14, 30)):
    w = Window.partitionBy(*partition_cols).orderBy(order_col)
    for lag_val in lags:
        df = df.withColumn(
            f"lag_{lag_val}",
            lag(col(value_col), lag_val).over(w)
        )
    return df


def add_rolling_features(df: DataFrame, partition_cols, order_col, value_col, windows=(7, 14, 30)):
    for w_size in windows:
        w = (
            Window.partitionBy(*partition_cols)
            .orderBy(order_col)
            .rowsBetween(-w_size + 1, 0)
        )
        df = df.withColumn(
            f"rolling_avg_{w_size}",
            spark_round(
                coalesce(
                    (col(value_col).cast(DoubleType())).over(w),
                    lit(0.0)
                ), 2
            )
        )
    return df


def build_demand_features(demand_df: DataFrame) -> DataFrame:
    demand_df = add_temporal_features(demand_df, "date")
    demand_df = add_price_tier(demand_df, "avg_unit_price")
    demand_df = add_lag_features(
        demand_df,
        partition_cols=["store_id", "product_id"],
        order_col="date",
        value_col="total_quantity",
        lags=(1, 7, 14, 30)
    )
    demand_df = add_rolling_features(
        demand_df,
        partition_cols=["store_id", "product_id"],
        order_col="date",
        value_col="total_quantity",
        windows=(7, 14, 30)
    )
    return demand_df
