import sys, io, os, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import PARQUET_DIR, OUTPUT_TABLES_DIR
from ingestion import get_spark_session

from pyspark.sql.functions import (
    col, sum as spark_sum, count, avg, round as spark_round,
    when, lit, greatest, least, datediff, countDistinct,
    max as spark_max, concat, stddev_pop, date_add, date_sub
)
from pyspark.sql.window import Window

OUTPUT_TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")


def main():
    print("=" * 60)
    print("PHASE 11: INVENTORY RECOMMENDATION ENGINE")
    print("=" * 60)

    LEAD_TIME_DAYS = 7
    SERVICE_LEVEL_Z = 1.65
    FORECAST_HORIZON_DAYS = 30

    spark = get_spark_session()

    demand_path = os.path.join(PARQUET_DIR, "demand")
    print(f"\n  Reading demand data from {demand_path}...")
    demand = spark.read.parquet(demand_path)
    total_rows = demand.count()
    print(f"  Loaded {total_rows:,} rows")

    max_date = demand.agg(spark_max("date")).collect()[0][0]
    print(f"  Historical data ends: {max_date}")
    print(f"  Lead time: {LEAD_TIME_DAYS} days | Service level: 95% (Z={SERVICE_LEVEL_Z})")
    print(f"  Forecast horizon: {FORECAST_HORIZON_DAYS} days")

    # Step 1: Demand statistics per store x category
    print("\n  [1/7] Computing demand statistics per store x category...")
    demand_stats = (
        demand.groupBy("store_id", "country", "city", "category")
        .agg(
            avg("total_quantity").alias("avg_daily_demand"),
            stddev_pop("total_quantity").alias("std_daily_demand"),
            spark_round(avg("total_revenue"), 2).alias("avg_daily_revenue"),
            spark_round(avg("avg_unit_price"), 2).alias("avg_unit_price"),
            count("*").alias("days_of_history"),
            spark_max("date").alias("last_demand_date")
        )
        .withColumn("std_daily_demand",
            when(col("std_daily_demand").isNull(), col("avg_daily_demand") * 0.3)
            .otherwise(col("std_daily_demand")))
        .withColumn("std_daily_demand", greatest(col("std_daily_demand"), lit(1.0)))
    )
    demand_stats.cache()
    n_segments = demand_stats.count()
    print(f"  Found {n_segments} store x category segments")

    # Step 2: Forecast upcoming demand using recent 7-day trend
    print("  [2/7] Forecasting upcoming demand (recent 7-day trend)...")
    max_date_row = demand.agg(spark_max("date").alias("max_d")).collect()[0]
    max_d = max_date_row["max_d"]
    cutoff_date = max_d - __import__("datetime").timedelta(days=6)
    print(f"    Last date: {max_d}, cutoff: {cutoff_date}")

    recent_7day = (
        demand.filter(col("date") >= lit(cutoff_date))
        .groupBy("store_id", "category")
        .agg(
            avg("total_quantity").alias("recent_7day_avg"),
            spark_sum("total_quantity").alias("recent_7day_total"),
            count("*").alias("days_in_window")
        )
    )
    recent_7day.show(10, truncate=False)

    forecasted = (
        demand_stats.join(recent_7day, ["store_id", "category"], "left")
        .withColumn("forecast_7day_demand",
            when(col("recent_7day_total").isNotNull() & (col("days_in_window") > 0),
                 col("recent_7day_total"))
            .otherwise(col("avg_daily_demand") * 7))
        .withColumn("forecast_30day_demand",
            when(col("recent_7day_total").isNotNull() & (col("days_in_window") > 0),
                 col("recent_7day_total") * (30.0 / greatest(col("days_in_window"), lit(1))))
            .otherwise(col("avg_daily_demand") * 30))
        .withColumn("forecast_daily_avg", col("forecast_30day_demand") / 30)
    )

    # Step 3: Safety stock calculation
    print("  [3/7] Calculating safety stock (95% service level)...")
    forecasted = forecasted.withColumn(
        "safety_stock",
        spark_round(
            lit(SERVICE_LEVEL_Z) * col("std_daily_demand") * lit(LEAD_TIME_DAYS ** 0.5), 0
        ).cast("integer")
    )
    forecasted = forecasted.withColumn("safety_stock", greatest(col("safety_stock"), lit(5)))

    # Step 4: Required inventory
    print("  [4/7] Computing required inventory levels...")
    forecasted = forecasted.withColumn(
        "reorder_point",
        spark_round(col("forecast_daily_avg") * LEAD_TIME_DAYS + col("safety_stock"), 0).cast("integer")
    )
    forecasted = forecasted.withColumn(
        "required_inventory",
        spark_round(col("forecast_daily_avg") * FORECAST_HORIZON_DAYS + col("safety_stock"), 0).cast("integer")
    )

    # Step 5: Simulate current stock
    print("  [5/7] Simulating current stock levels...")
    forecasted = forecasted.withColumn(
        "simulated_current_stock",
        spark_round(
            col("avg_daily_demand") * lit(10) *
            (lit(0.6) + (col("store_id") % 10).cast("double") / lit(10) * lit(0.8)),
            0
        ).cast("integer")
    )

    # Step 6: Risk flagging
    print("  [6/7] Flagging stockout and overstock risks...")
    forecasted = forecasted.withColumn("stock_gap", col("required_inventory") - col("simulated_current_stock"))
    forecasted = forecasted.withColumn(
        "coverage_days",
        spark_round(col("simulated_current_stock") / col("forecast_daily_avg"), 1)
    )
    forecasted = forecasted.withColumn(
        "risk_status",
        when(col("coverage_days") < lit(LEAD_TIME_DAYS) * 0.7, "STOCKOUT_RISK")
        .when(col("coverage_days") > lit(FORECAST_HORIZON_DAYS) * 1.3, "OVERSTOCK_RISK")
        .otherwise("OK")
    )
    forecasted = forecasted.withColumn(
        "risk_severity",
        when(col("risk_status") == "STOCKOUT_RISK",
             when(col("coverage_days") < 3, "CRITICAL")
             .when(col("coverage_days") < 5, "HIGH").otherwise("MEDIUM"))
        .when(col("risk_status") == "OVERSTOCK_RISK",
              when(col("coverage_days") > 45, "HIGH").otherwise("LOW"))
        .otherwise("NONE")
    )

    # Step 7: Replenishment recommendation
    print("  [7/7] Generating replenishment recommendations...")
    forecasted = forecasted.withColumn(
        "recommended_order_qty",
        when(col("stock_gap") > 0, col("stock_gap")).otherwise(lit(0))
    )
    forecasted = forecasted.withColumn(
        "recommended_action",
        when((col("risk_status") == "STOCKOUT_RISK") & (col("recommended_order_qty") > 0),
             concat(lit("ORDER "), col("recommended_order_qty").cast("string"), lit(" units immediately")))
        .when((col("risk_status") == "OVERSTOCK_RISK") & (col("stock_gap") < 0),
             concat(lit("REDUCE next order by "), (col("stock_gap") * -1).cast("string"), lit(" units")))
        .otherwise("No action needed")
    )
    forecasted = forecasted.withColumn(
        "priority_rank",
        when(col("risk_severity") == "CRITICAL", 1)
        .when(col("risk_severity") == "HIGH", 2)
        .when(col("risk_severity") == "MEDIUM", 3)
        .when(col("risk_severity") == "LOW", 4).otherwise(5)
    )
    forecasted = forecasted.withColumn(
        "estimated_excess_value",
        when(col("stock_gap") < 0, spark_round((col("stock_gap") * -1) * col("avg_unit_price"), 2))
        .otherwise(lit(0.0))
    )

    inventory_recs = (
        forecasted.select(
            "priority_rank", "risk_severity", "risk_status",
            "store_id", "country", "city", "category",
            "avg_daily_demand", "std_daily_demand", "forecast_daily_avg",
            "forecast_7day_demand", "forecast_30day_demand",
            "safety_stock", "reorder_point", "required_inventory",
            "simulated_current_stock", "stock_gap", "coverage_days",
            "recommended_order_qty", "recommended_action",
            "estimated_excess_value", "avg_unit_price", "avg_daily_revenue", "days_of_history"
        )
        .orderBy("priority_rank", col("stock_gap").desc())
    )
    inventory_recs.cache()

    # Summary
    total = inventory_recs.count()
    stockout_count = inventory_recs.filter(col("risk_status") == "STOCKOUT_RISK").count()
    overstock_count = inventory_recs.filter(col("risk_status") == "OVERSTOCK_RISK").count()
    ok_count = inventory_recs.filter(col("risk_status") == "OK").count()
    total_order_value = inventory_recs.agg(
        spark_round(spark_sum(col("recommended_order_qty") * col("avg_unit_price")), 2)
    ).collect()[0][0]
    total_excess_value = inventory_recs.agg(
        spark_round(spark_sum("estimated_excess_value"), 2)
    ).collect()[0][0]

    print(f"\n  {'=' * 55}")
    print(f"  INVENTORY RISK SUMMARY")
    print(f"  {'=' * 55}")
    print(f"  Total segments analyzed:  {total}")
    print(f"  Stockout risk segments:   {stockout_count} ({stockout_count/total*100:.1f}%)")
    print(f"  Overstock risk segments:  {overstock_count} ({overstock_count/total*100:.1f}%)")
    print(f"  Healthy segments:         {ok_count} ({ok_count/total*100:.1f}%)")
    print(f"  Total recommended orders: ${total_order_value:,.2f}")
    print(f"  Total excess inventory:   ${total_excess_value:,.2f}")
    print(f"  {'=' * 55}")

    print(f"\n  --- Top 15 CRITICAL/HIGH Priority Recommendations ---")
    inventory_recs.filter(col("priority_rank") <= 2).show(15, truncate=False)

    print(f"\n  --- Demand Statistics (all 105 segments) ---")
    demand_stats.select(
        "store_id", "country", "city", "category",
        spark_round("avg_daily_demand", 1).alias("avg_demand"),
        spark_round("std_daily_demand", 1).alias("std_demand"),
        "days_of_history"
    ).orderBy("store_id", "category").show(30, truncate=False)

    # Save
    os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
    inv_pdf = inventory_recs.toPandas()
    inv_path = os.path.join(OUTPUT_TABLES_DIR, "inventory_recommendations.csv")
    inv_pdf.to_csv(inv_path, index=False)
    print(f"\n  Saved inventory_recommendations.csv ({len(inv_pdf)} rows)")

    stats_pdf = demand_stats.toPandas()
    stats_path = os.path.join(OUTPUT_TABLES_DIR, "demand_statistics.csv")
    stats_pdf.to_csv(stats_path, index=False)
    print(f"  Saved demand_statistics.csv ({len(stats_pdf)} rows)")

    summary = {
        "total_segments": total,
        "stockout_risk": stockout_count,
        "overstock_risk": overstock_count,
        "healthy": ok_count,
        "total_order_value": float(total_order_value) if total_order_value else 0,
        "total_excess_value": float(total_excess_value) if total_excess_value else 0,
        "lead_time_days": LEAD_TIME_DAYS,
        "service_level": "95%",
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
    }
    with open(os.path.join(OUTPUT_TABLES_DIR, "inventory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved inventory_summary.json")

    print(f"\n{'=' * 60}")
    print(f"PHASE 11 COMPLETE")
    print(f"{'=' * 60}")

    spark.stop()


if __name__ == "__main__":
    main()
