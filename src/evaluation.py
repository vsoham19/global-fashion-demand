from pyspark.sql import DataFrame
from pyspark.sql.functions import col, abs as spark_abs, when, lit, avg, sum as spark_sum
from pyspark.sql.types import DoubleType
import os, json


def compute_metrics(predictions: DataFrame, label_col="total_quantity", prediction_col="prediction"):
    n = predictions.count()
    if n == 0:
        return {}

    sums = predictions.select(
        spark_sum((col(label_col) - col(prediction_col)) ** 2).alias("sse"),
        spark_sum(spark_abs(col(label_col) - col(prediction_col))).alias("abs_err"),
        spark_sum(col(label_col)).alias("actual_sum"),
        spark_sum(col(prediction_col)).alias("pred_sum"),
    ).collect()[0]

    rmse = (sums["sse"] / n) ** 0.5
    mae = sums["abs_err"] / n

    actual_mean = sums["actual_sum"] / n if sums["actual_sum"] else 0

    ss_total = predictions.select(
        spark_sum((col(label_col) - lit(actual_mean)) ** 2)
    ).collect()[0][0]
    r2 = 1 - (sums["sse"] / ss_total) if ss_total and ss_total > 0 else 0.0

    smape_val = 0.0
    try:
        smape_df = predictions.withColumn(
            "denom", (col(label_col) + col(prediction_col)) / 2
        ).withColumn(
            "smape_comp",
            when(col("denom") > 0, spark_abs(col(label_col) - col(prediction_col)) / col("denom")).otherwise(lit(0.0))
        )
        smape_val = smape_df.select(avg("smape_comp")).collect()[0][0] * 100
    except Exception:
        smape_val = (mae / actual_mean * 100) if actual_mean > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "smape": smape_val, "r2": r2, "n": n}


def baseline_last_period(test_df, train_df, partition_cols=["store_id", "product_id"]):
    from pyspark.sql.functions import max as spark_max
    w_order = None
    from pyspark.sql import Window
    w = Window.partitionBy(*partition_cols).orderBy("date")
    last_vals = train_df.withColumn("rn", spark_max("date").over(w)).where(col("rn") == col("date"))
    return last_vals


def compute_naive_metrics(test_df, baseline_col="total_quantity_lag1", prediction_col="prediction"):
    n = test_df.count()
    if n == 0:
        return {}
    sums = test_df.select(
        spark_sum((col("total_quantity") - col(prediction_col)) ** 2).alias("sse"),
        spark_sum(spark_abs(col("total_quantity") - col(prediction_col))).alias("abs_err"),
    ).collect()[0]
    rmse = (sums["sse"] / n) ** 0.5
    mae = sums["abs_err"] / n
    return {"naive_rmse": rmse, "naive_mae": mae, "n": n}


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def format_metrics_report(metrics: dict, model_name: str = "Model") -> str:
    lines = [f"=== {model_name} Evaluation ==="]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k.upper()}: {v:.4f}")
        else:
            lines.append(f"  {k.upper()}: {v}")
    return "\n".join(lines)
