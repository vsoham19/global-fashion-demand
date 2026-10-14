import sys, io, os, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PARQUET_DIR,
    MODELS_DIR, OUTPUT_TABLES_DIR, OUTPUT_CHARTS_DIR
)
from ingestion import get_spark_session, load_csv, save_parquet
from preprocessing import (
    clean_transactions, clean_customers, clean_products,
    clean_stores, clean_employees, clean_discounts,
    drop_duplicates, remove_null_key_rows
)
from features import add_temporal_features, add_revenue_features, add_price_tier
from evaluation import compute_metrics, save_metrics, format_metrics_report

from pyspark.sql.functions import (
    col, sum as spark_sum, count, avg, round as spark_round,
    year, month, quarter, dayofweek, hour, weekofyear,
    when, lag, datediff, countDistinct, min as spark_min, max as spark_max,
    concat, lit, lower, coalesce, abs as spark_abs, date_add
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def phase2_ingestion(spark):
    print("\n" + "=" * 60)
    print("PHASE 2: SPARK INGESTION")
    print("=" * 60)

    datasets = {}
    for name in ["transactions", "customers", "products", "stores", "employees", "discounts"]:
        t0 = time.time()
        df = load_csv(spark, name)
        cnt = df.count()
        elapsed = time.time() - t0
        print(f"  {name}: {cnt:,} rows | {len(df.columns)} cols | {elapsed:.1f}s")
        df.printSchema()
        datasets[name] = df

    return datasets


def phase3_clean(datasets):
    print("\n" + "=" * 60)
    print("PHASE 3: DATA CLEANING")
    print("=" * 60)

    transactions = clean_transactions(datasets["transactions"])
    transactions, dup_dropped = drop_duplicates(transactions, ["invoice_id", "line"])
    print(f"  transactions: removed {dup_dropped} duplicates")

    customers = clean_customers(datasets["customers"])
    customers, dup_dropped = drop_duplicates(customers, ["customer_id"])
    print(f"  customers: removed {dup_dropped} duplicates")

    products = clean_products(datasets["products"])
    products, dup_dropped = drop_duplicates(products, ["product_id"])
    print(f"  products: removed {dup_dropped} duplicates")

    stores = clean_stores(datasets["stores"])
    stores, dup_dropped = drop_duplicates(stores, ["store_id"])
    print(f"  stores: removed {dup_dropped} duplicates")

    employees = clean_employees(datasets["employees"])
    employees, dup_dropped = drop_duplicates(employees, ["employee_id"])
    print(f"  employees: removed {dup_dropped} duplicates")

    discounts = clean_discounts(datasets["discounts"])
    discounts, dup_dropped = drop_duplicates(discounts)
    print(f"  discounts: removed {dup_dropped} duplicates")

    transactions = remove_null_key_rows(transactions, ["invoice_id", "product_id", "store_id"])
    print(f"  transactions after null removal: {transactions.count():,}")

    return {
        "transactions": transactions, "customers": customers,
        "products": products, "stores": stores,
        "employees": employees, "discounts": discounts
    }


def phase4_integration(cleaned):
    print("\n" + "=" * 60)
    print("PHASE 4: DATA INTEGRATION")
    print("=" * 60)

    t = cleaned["transactions"]

    t = t.join(cleaned["stores"], "store_id", "left")
    print(f"  after stores join: {t.count():,}")

    customers_clean = cleaned["customers"].drop("city", "country")
    t = t.join(customers_clean, "customer_id", "left")
    print(f"  after customers join: {t.count():,}")

    t = t.join(
        cleaned["products"].select("product_id", "category", "sub_category", "production_cost"),
        "product_id", "left"
    )
    print(f"  after products join: {t.count():,}")

    t = t.join(
        cleaned["discounts"].select(
            col("start").alias("discount_start"),
            col("end").alias("discount_end"),
            col("discount_pct"),
            col("category").alias("discount_category"),
            col("sub_category").alias("discount_sub_category")
        ),
        (t.category == col("discount_category")) &
        (t.sub_category == col("discount_sub_category")) &
        (t.date >= col("discount_start")) &
        (t.date <= col("discount_end")),
        "left"
    )
    print(f"  after discounts join: {t.count():,}")

    t = t.withColumn(
        "discount",
        when(col("discount").isNull() & col("discount_pct").isNotNull(), col("discount_pct"))
        .otherwise(coalesce(col("discount"), lit(0.0)))
    )

    return t


def phase5_eda(integrated):
    print("\n" + "=" * 60)
    print("PHASE 5: EXPLORATORY BDA")
    print("=" * 60)

    results = {}

    country_demand = (
        integrated.groupBy("country")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_round(avg("line_total"), 2).alias("avg_line_total"),
            count("*").alias("transaction_count")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Demand by Country ---")
    country_demand.show(20, truncate=False)
    results["country_demand"] = country_demand

    store_demand = (
        integrated.groupBy("store_id", "country", "city")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units"),
            count("*").alias("transaction_count")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Demand by Store ---")
    store_demand.show(40, truncate=False)
    results["store_demand"] = store_demand

    monthly_demand = (
        integrated
        .withColumn("year", year("date"))
        .withColumn("month", month("date"))
        .groupBy("year", "month")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units"),
            count("*").alias("transaction_count")
        )
        .orderBy("year", "month")
    )
    print("\n--- Monthly Demand Trend ---")
    monthly_demand.show(30, truncate=False)
    results["monthly_demand"] = monthly_demand

    category_demand = (
        integrated.groupBy("category")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units"),
            count("*").alias("transaction_count")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Demand by Category ---")
    category_demand.show(20, truncate=False)
    results["category_demand"] = category_demand

    subcategory_demand = (
        integrated.groupBy("category", "sub_category")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Demand by Sub-Category ---")
    subcategory_demand.show(30, truncate=False)
    results["subcategory_demand"] = subcategory_demand

    weekday_demand = (
        integrated
        .withColumn("day_of_week", dayofweek("date"))
        .groupBy("day_of_week")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units")
        )
        .orderBy("day_of_week")
    )
    print("\n--- Demand by Day of Week ---")
    weekday_demand.show(20, truncate=False)
    results["weekday_demand"] = weekday_demand

    hourly_demand = (
        integrated
        .withColumn("hour_of_day", hour("date"))
        .groupBy("hour_of_day")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units")
        )
        .orderBy("hour_of_day")
    )
    print("\n--- Demand by Hour ---")
    hourly_demand.show(25, truncate=False)
    results["hourly_demand"] = hourly_demand

    discount_analysis = (
        integrated
        .withColumn("has_discount", when(col("discount") > 0, "Discounted").otherwise("Full Price"))
        .groupBy("has_discount")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units"),
            count("*").alias("transaction_count"),
            spark_round(avg("line_total"), 2).alias("avg_line_total")
        )
    )
    print("\n--- Discount vs Full Price ---")
    discount_analysis.show(20, truncate=False)
    results["discount_analysis"] = discount_analysis

    country_category = (
        integrated.groupBy("country", "category")
        .agg(
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_sum("quantity").alias("total_units")
        )
        .orderBy("country", col("total_revenue").desc())
    )
    print("\n--- Country x Category Demand ---")
    country_category.show(40, truncate=False)
    results["country_category"] = country_category

    return results


def phase6_demand_aggregation(integrated):
    print("\n" + "=" * 60)
    print("PHASE 6/7: DEMAND AGGREGATION & FEATURE ENGINEERING")
    print("=" * 60)

    print("\n  Aggregating at date x store x category grain...")
    demand = (
        integrated
        .withColumn("demand_date", col("date").cast("date"))
        .groupBy("demand_date", "store_id", "country", "city", "category", "sub_category")
        .agg(
            spark_sum("quantity").alias("total_quantity"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_round(avg("unit_price"), 2).alias("avg_unit_price"),
            spark_round(avg("discount"), 4).alias("avg_discount"),
            count("*").alias("line_count"),
            countDistinct("product_id").alias("unique_products")
        )
        .withColumnRenamed("demand_date", "date")
    )

    demand = demand.withColumn("year", year("date"))
    demand = demand.withColumn("month", month("date"))
    demand = demand.withColumn("quarter", quarter("date"))
    demand = demand.withColumn("day_of_week", dayofweek("date"))
    demand = demand.withColumn("week_of_year", weekofyear("date"))
    demand = demand.withColumn(
        "is_discounted",
        when(col("avg_discount") > 0, 1).otherwise(0)
    )
    demand = demand.withColumn(
        "price_tier",
        when(col("avg_unit_price") < 30, "low")
        .when(col("avg_unit_price") < 80, "mid")
        .otherwise("high")
    )

    w = Window.partitionBy("store_id", "category").orderBy("date")
    for lag_val in [1, 7, 14, 30]:
        demand = demand.withColumn(f"lag_{lag_val}", lag("total_quantity", lag_val).over(w))

    for w_size in [7, 14, 30]:
        w_rolling = (
            Window.partitionBy("store_id", "category")
            .orderBy("date")
            .rowsBetween(-w_size + 1, 0)
        )
        demand = demand.withColumn(
            f"rolling_avg_{w_size}",
            spark_round(avg("total_quantity").over(w_rolling), 2)
        )

    fill_cols = [f"lag_{l}" for l in [1, 7, 14, 30]] + [f"rolling_avg_{w}" for w in [7, 14, 30]]
    demand = demand.fillna(0, subset=fill_cols)

    print(f"  Demand dataset: {demand.count():,} rows | {len(demand.columns)} cols")
    demand.printSchema()

    return demand


def phase8_forecasting(demand):
    print("\n" + "=" * 60)
    print("PHASE 8: FORECASTING WITH HYPERPARAMETER TUNING")
    print("=" * 60)

    feature_cols_numeric = [
        "year", "month", "quarter", "day_of_week", "week_of_year",
        "avg_unit_price", "avg_discount", "is_discounted",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_avg_7", "rolling_avg_14", "rolling_avg_30"
    ]
    feature_cols_categorical = ["store_id", "category"]
    target_col = "total_quantity"
    all_features = feature_cols_numeric + [f"{c}_idx" for c in feature_cols_categorical]

    demand_ordered = demand.orderBy("date")
    total = demand_ordered.count()
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    train_df = demand_ordered.limit(train_end)
    remaining = demand_ordered.subtract(train_df).orderBy("date")
    val_df = remaining.limit(val_end - train_end)
    test_df = remaining.subtract(val_df).orderBy("date")

    print(f"  Train: {train_df.count():,} | Val: {val_df.count():,} | Test: {test_df.count():,}")

    train_df = train_df.fillna(0, subset=feature_cols_numeric)
    val_df = val_df.fillna(0, subset=feature_cols_numeric)
    test_df = test_df.fillna(0, subset=feature_cols_numeric)

    train_df.cache()
    train_df.count()
    test_df.cache()
    test_df.count()

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in feature_cols_categorical
    ]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")

    results = {}
    all_preds = {}

    # --- Model 1: Random Forest with CrossValidation ---
    print("\n--- Random Forest (with CrossValidation) ---")
    rf = RandomForestRegressor(
        featuresCol="features", labelCol=target_col, maxBins=100, seed=42
    )
    rf_paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [20, 30])
        .addGrid(rf.maxDepth, [6, 8, 10])
        .build()
    )
    rf_pipeline = Pipeline(stages=indexers + [assembler, rf])
    rf_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="rmse"
    )
    rf_cv = CrossValidator(
        estimator=rf_pipeline,
        estimatorParamMaps=rf_paramGrid,
        evaluator=rf_evaluator,
        numFolds=3,
        seed=42,
        parallelism=1
    )
    t0 = time.time()
    rf_cv_model = rf_cv.fit(train_df)
    rf_time = time.time() - t0
    print(f"  RF CV training time: {rf_time:.1f}s")

    best_rf = rf_cv_model.bestModel
    rf_preds = best_rf.transform(test_df)
    rf_preds.cache()
    rf_preds.count()
    rf_metrics = compute_metrics(rf_preds, target_col, "prediction")
    print(format_metrics_report(rf_metrics, "Random Forest (Best CV)"))

    rf_stages = best_rf.stages
    rf_fitted = None
    for s in rf_stages:
        if hasattr(s, "featureImportances"):
            rf_fitted = s
            break
    if rf_fitted:
        importances = rf_fitted.featureImportances.toArray()
        fi = list(zip(all_features, importances))
        fi.sort(key=lambda x: x[1], reverse=True)
        print("  Top 10 Feature Importances:")
        for fname, imp in fi[:10]:
            print(f"    {fname}: {imp:.4f}")

    all_preds["rf"] = rf_preds.select("date", "store_id", "category", target_col, "prediction")
    results["rf"] = rf_metrics

    # --- Model 2: GBT Regressor with CrossValidation ---
    print("\n--- Gradient Boosted Trees (with CrossValidation) ---")
    gbt = GBTRegressor(
        featuresCol="features", labelCol=target_col, maxBins=100, seed=42
    )
    gbt_paramGrid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [15, 20])
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxBins, [100])
        .build()
    )
    gbt_indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in feature_cols_categorical
    ]
    gbt_pipeline = Pipeline(stages=gbt_indexers + [assembler, gbt])
    gbt_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="rmse"
    )
    gbt_cv = CrossValidator(
        estimator=gbt_pipeline,
        estimatorParamMaps=gbt_paramGrid,
        evaluator=gbt_evaluator,
        numFolds=2,
        seed=42,
        parallelism=1
    )
    t0 = time.time()
    gbt_cv_model = gbt_cv.fit(train_df)
    gbt_time = time.time() - t0
    print(f"  GBT CV training time: {gbt_time:.1f}s")

    best_gbt = gbt_cv_model.bestModel
    gbt_preds = best_gbt.transform(test_df)
    gbt_preds.cache()
    gbt_preds.count()
    gbt_metrics = compute_metrics(gbt_preds, target_col, "prediction")
    print(format_metrics_report(gbt_metrics, "Gradient Boosted Trees (Best CV)"))

    gbt_stages = best_gbt.stages
    for s in gbt_stages:
        if hasattr(s, "featureImportances"):
            importances = s.featureImportances.toArray()
            fi_gbt = list(zip(all_features, importances))
            fi_gbt.sort(key=lambda x: x[1], reverse=True)
            print("  Top 10 Feature Importances:")
            for fname, imp in fi_gbt[:10]:
                print(f"    {fname}: {imp:.4f}")
            break

    all_preds["gbt"] = gbt_preds.select("date", "store_id", "category", target_col, "prediction")
    results["gbt"] = gbt_metrics

    # --- Model 3: Linear Regression ---
    print("\n--- Linear Regression ---")
    lr = LinearRegression(
        featuresCol="features", labelCol=target_col, maxIter=50, regParam=0.1
    )
    lr_indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in feature_cols_categorical
    ]
    lr_pipeline = Pipeline(stages=lr_indexers + [assembler, lr])

    t0 = time.time()
    lr_model = lr_pipeline.fit(train_df)
    lr_time = time.time() - t0
    print(f"  LR training time: {lr_time:.1f}s")

    lr_preds = lr_model.transform(test_df)
    lr_preds.cache()
    lr_preds.count()
    lr_metrics = compute_metrics(lr_preds, target_col, "prediction")
    print(format_metrics_report(lr_metrics, "Linear Regression"))

    lr_stages = lr_model.stages
    for s in lr_stages:
        if hasattr(s, "coefficients"):
            coeffs = s.coefficients.toArray()
            fi_lr = list(zip(all_features, coeffs))
            fi_lr.sort(key=lambda x: abs(x[1]), reverse=True)
            print("  Top 10 Coefficients (by magnitude):")
            for fname, c in fi_lr[:10]:
                print(f"    {fname}: {c:.4f}")
            break

    all_preds["lr"] = lr_preds.select("date", "store_id", "category", target_col, "prediction")
    results["lr"] = lr_metrics

    # --- Naive Baseline ---
    print("\n--- Naive Baseline (lag_1) ---")
    baseline_test = test_df.filter(col("lag_1").isNotNull())
    n = baseline_test.count()
    baseline_rmse = 0
    baseline_mae = 0
    if n > 0:
        baseline_sums = baseline_test.select(
            spark_sum((col(target_col) - col("lag_1")) ** 2).alias("sse"),
            spark_sum(spark_abs(col(target_col) - col("lag_1"))).alias("abs_err"),
        ).collect()[0]
        baseline_rmse = (baseline_sums["sse"] / n) ** 0.5
        baseline_mae = baseline_sums["abs_err"] / n
        print(f"  Naive RMSE: {baseline_rmse:.4f}")
        print(f"  Naive MAE:  {baseline_mae:.4f}")
        print(f"  Naive N:    {n:,}")

    results["naive"] = {"rmse": baseline_rmse, "mae": baseline_mae, "smape": 0, "r2": 0, "n": n}

    # --- Per-Country/Category Evaluation (using best model = RF) ---
    print("\n--- Per-Country Evaluation (Random Forest) ---")
    country_metrics = {}
    country_list = [r["country"] for r in demand.groupBy("country").count().orderBy(col("count").desc()).collect()]
    for ctry in country_list[:7]:
        ctry_preds = rf_preds.filter(col("country") == ctry)
        if ctry_preds.count() > 10:
            cm = compute_metrics(ctry_preds, target_col, "prediction")
            country_metrics[ctry] = cm
            print(f"  {ctry}: RMSE={cm['rmse']:.2f} | MAE={cm['mae']:.2f} | R2={cm['r2']:.4f} | N={cm['n']:,}")

    print("\n--- Per-Category Evaluation (Random Forest) ---")
    cat_metrics = {}
    for cat in ["Feminine", "Masculine", "Children"]:
        cat_preds = rf_preds.filter(col("category") == cat)
        if cat_preds.count() > 10:
            cm = compute_metrics(cat_preds, target_col, "prediction")
            cat_metrics[cat] = cm
            print(f"  {cat}: RMSE={cm['rmse']:.2f} | MAE={cm['mae']:.2f} | R2={cm['r2']:.4f} | N={cm['n']:,}")

    # Save predicted vs actual for plotting
    pred_vs_actual = all_preds["rf"].toPandas()
    pred_vs_actual.to_csv(os.path.join(OUTPUT_TABLES_DIR, "predicted_vs_actual.csv"), index=False)
    print("  Saved predicted_vs_actual.csv")

    # Save per-segment metrics
    seg_metrics = {"country": country_metrics, "category": cat_metrics}
    with open(os.path.join(OUTPUT_TABLES_DIR, "segment_metrics.json"), "w") as f:
        json.dump({k: {mk: {ik: iv for ik, iv in mv.items()} for mk, mv in v.items()} for k, v in seg_metrics.items()}, f, indent=2, default=str)
    print("  Saved segment_metrics.json")

    # Save feature importances
    fi_data = {"rf": [(f, float(v)) for f, v in fi], "gbt": [(f, float(v)) for f, v in fi_gbt]}
    with open(os.path.join(OUTPUT_TABLES_DIR, "feature_importances.json"), "w") as f:
        json.dump(fi_data, f, indent=2)
    print("  Saved feature_importances.json")

    return {
        "rf_model": best_rf, "rf_preds": rf_preds, "rf_metrics": rf_metrics,
        "gbt_model": best_gbt, "gbt_preds": gbt_preds, "gbt_metrics": gbt_metrics,
        "lr_model": lr_model, "lr_preds": lr_preds, "lr_metrics": lr_metrics,
        "train_df": train_df, "test_df": test_df,
        "country_metrics": country_metrics, "cat_metrics": cat_metrics
    }


def phase10_retail_insights(integrated):
    print("\n" + "=" * 60)
    print("PHASE 10: RETAIL OPTIMIZATION & BUSINESS INSIGHTS")
    print("=" * 60)

    top_products = (
        integrated.groupBy("product_id", "category", "sub_category")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue")
        )
        .orderBy(col("total_units").desc())
        .limit(20)
    )
    print("\n--- Top 20 Products by Units Sold ---")
    top_products.show(20, truncate=False)

    top_stores = (
        integrated.groupBy("store_id", "country", "city")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            countDistinct("customer_id").alias("unique_customers")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Store Performance ---")
    top_stores.show(40, truncate=False)

    seasonal = (
        integrated
        .withColumn("month", month("date"))
        .groupBy("month")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue")
        )
        .orderBy("month")
    )
    print("\n--- Seasonal Demand Pattern ---")
    seasonal.show(20, truncate=False)

    discount_impact = (
        integrated
        .withColumn("has_discount", when(col("discount") > 0, "Discounted").otherwise("Full Price"))
        .groupBy("category", "has_discount")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(avg("line_total"), 2).alias("avg_line_total")
        )
        .orderBy("category", "has_discount")
    )
    print("\n--- Discount Impact by Category ---")
    discount_impact.show(30, truncate=False)

    payment_methods = (
        integrated.groupBy("payment_method")
        .agg(
            count("*").alias("transaction_count"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_round(avg("line_total"), 2).alias("avg_transaction_value")
        )
        .orderBy(col("total_revenue").desc())
    )
    print("\n--- Payment Method Analysis ---")
    payment_methods.show(20, truncate=False)

    price_distribution = (
        integrated
        .withColumn("price_bin",
            when(col("unit_price") < 20, "0-20")
            .when(col("unit_price") < 50, "20-50")
            .when(col("unit_price") < 100, "50-100")
            .when(col("unit_price") < 200, "100-200")
            .otherwise("200+")
        )
        .groupBy("price_bin")
        .agg(
            count("*").alias("transaction_count"),
            spark_round(avg("quantity"), 2).alias("avg_quantity"),
            spark_round(avg("line_total"), 2).alias("avg_line_total")
        )
        .orderBy("price_bin")
    )
    print("\n--- Price Distribution Analysis ---")
    price_distribution.show(20, truncate=False)

    revenue_per_customer = (
        integrated.groupBy("country")
        .agg(
            countDistinct("customer_id").alias("unique_customers"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            spark_round(spark_sum("line_total") / countDistinct("customer_id"), 2).alias("revenue_per_customer"),
            spark_round(spark_sum("quantity") / countDistinct("customer_id"), 2).alias("units_per_customer")
        )
        .orderBy(col("revenue_per_customer").desc())
    )
    print("\n--- Revenue Per Customer by Country ---")
    revenue_per_customer.show(20, truncate=False)

    yearly_growth = (
        integrated
        .withColumn("year", year("date"))
        .groupBy("year")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue"),
            countDistinct("store_id").alias("active_stores")
        )
        .orderBy("year")
    )
    print("\n--- Yearly Growth ---")
    yearly_growth.show(20, truncate=False)

    transaction_type = (
        integrated.groupBy("transaction_type")
        .agg(
            count("*").alias("count"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue")
        )
        .orderBy(col("count").desc())
    )
    print("\n--- Transaction Type Breakdown ---")
    transaction_type.show(10, truncate=False)

    top_subcategory_country = (
        integrated.groupBy("country", "sub_category")
        .agg(
            spark_sum("quantity").alias("total_units"),
            spark_round(spark_sum("line_total"), 2).alias("total_revenue")
        )
        .orderBy("country", col("total_revenue").desc())
    )
    print("\n--- Top Sub-Category per Country ---")
    top_subcategory_country.show(30, truncate=False)

    return {
        "top_products": top_products, "top_stores": top_stores,
        "seasonal": seasonal, "discount_impact": discount_impact,
        "payment_methods": payment_methods,
        "price_distribution": price_distribution,
        "revenue_per_customer": revenue_per_customer,
        "yearly_growth": yearly_growth,
        "transaction_type": transaction_type,
        "top_subcategory_country": top_subcategory_country
    }


def phase11_inventory_engine(demand, forecast_results):
    from pyspark.sql.functions import stddev, stddev_pop, datediff, greatest, least
    from datetime import datetime, timedelta

    print("\n" + "=" * 60)
    print("PHASE 11: INVENTORY RECOMMENDATION ENGINE")
    print("=" * 60)

    LEAD_TIME_DAYS = 7
    SERVICE_LEVEL_Z = 1.65
    FORECAST_HORIZON_DAYS = 30
    SAFETY_STOCK_DAYS = 3

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

    # Step 2: Forecast upcoming demand using recent trend
    print("  [2/7] Forecasting upcoming demand...")
    recent_window = Window.partitionBy("store_id", "category").orderBy("date")
    recent_7day = (
        demand
        .withColumn("rn", spark_max("date").over(recent_window))
        .where(col("date") >= datediff(col("rn"), lit(6)))
        .groupBy("store_id", "category")
        .agg(
            avg("total_quantity").alias("recent_7day_avg"),
            spark_sum("total_quantity").alias("recent_7day_total")
        )
    )
    forecasted = (
        demand_stats.join(recent_7day, ["store_id", "category"], "left")
        .withColumn("forecast_7day_demand",
            when(col("recent_7day_total").isNotNull(), col("recent_7day_total"))
            .otherwise(col("avg_daily_demand") * 7))
        .withColumn("forecast_30day_demand",
            when(col("recent_7day_total").isNotNull(), col("recent_7day_total") * (30.0 / 7.0))
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
    forecasted = forecasted.withColumn(
        "safety_stock",
        greatest(col("safety_stock"), lit(5))
    )

    # Step 4: Required inventory
    print("  [4/7] Computing required inventory levels...")
    forecasted = forecasted.withColumn(
        "reorder_point",
        spark_round(
            col("forecast_daily_avg") * LEAD_TIME_DAYS + col("safety_stock"), 0
        ).cast("integer")
    )
    forecasted = forecasted.withColumn(
        "required_inventory",
        spark_round(
            col("forecast_daily_avg") * FORECAST_HORIZON_DAYS + col("safety_stock"), 0
        ).cast("integer")
    )

    # Step 5: Simulate current stock (based on ~10 days of historical average demand)
    print("  [5/7] Simulating current stock levels...")
    forecasted = forecasted.withColumn(
        "simulated_current_stock",
        spark_round(
            col("avg_daily_demand") * lit(10) * 
            (lit(0.6) + (col("store_id") % 10).cast("double") / lit(10) * lit(0.8)),
            0
        ).cast("integer")
    )

    # Step 6: Risk flagging and gap calculation
    print("  [6/7] Flagging stockout and overstock risks...")
    forecasted = forecasted.withColumn(
        "stock_gap",
        col("required_inventory") - col("simulated_current_stock")
    )
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
             .when(col("coverage_days") < 5, "HIGH")
             .otherwise("MEDIUM"))
        .when(col("risk_status") == "OVERSTOCK_RISK",
              when(col("coverage_days") > 45, "HIGH")
              .otherwise("LOW"))
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
        when(col("risk_status") == "STOCKOUT_RISK", 
             concat(lit("ORDER "), col("recommended_order_qty").cast("string"), 
                    lit(" units immediately")))
        .when((col("risk_status") == "OVERSTOCK_RISK") & (col("stock_gap") < 0),
             concat(lit("REDUCE next order by "), 
                    (col("stock_gap") * -1).cast("string"), lit(" units")))
        .otherwise("No action needed")
    )
    forecasted = forecasted.withColumn(
        "priority_rank",
        when(col("risk_severity") == "CRITICAL", 1)
        .when(col("risk_severity") == "HIGH", 2)
        .when(col("risk_severity") == "MEDIUM", 3)
        .when(col("risk_severity") == "LOW", 4)
        .otherwise(5)
    )
    forecasted = forecasted.withColumn(
        "estimated_stockout_date",
        when(col("risk_status") == "STOCKOUT_RISK",
             date_add(col("last_demand_date"), 
                      (col("coverage_days") + LEAD_TIME_DAYS).cast("integer")))
        .otherwise(lit(None))
    )
    forecasted = forecasted.withColumn(
        "estimated_excess_value",
        when(col("stock_gap") < 0,
             spark_round((col("stock_gap") * -1) * col("avg_unit_price"), 2))
        .otherwise(lit(0.0))
    )

    # Finalize output
    inventory_recs = (
        forecasted
        .select(
            "priority_rank", "risk_severity", "risk_status",
            "store_id", "country", "city", "category",
            "avg_daily_demand", "std_daily_demand", "forecast_daily_avg",
            "forecast_7day_demand", "forecast_30day_demand",
            "safety_stock", "reorder_point", "required_inventory",
            "simulated_current_stock", "stock_gap", "coverage_days",
            "recommended_order_qty", "recommended_action",
            "estimated_stockout_date", "estimated_excess_value",
            "avg_unit_price", "avg_daily_revenue", "days_of_history"
        )
        .orderBy("priority_rank", col("stock_gap").desc())
    )
    inventory_recs.cache()

    # Print summary
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

    print(f"\n  {'=' * 50}")
    print(f"  INVENTORY RISK SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Total segments analyzed:  {total}")
    print(f"  Stockout risk segments:   {stockout_count} ({stockout_count/total*100:.1f}%)")
    print(f"  Overstock risk segments:  {overstock_count} ({overstock_count/total*100:.1f}%)")
    print(f"  Healthy segments:         {ok_count} ({ok_count/total*100:.1f}%)")
    print(f"  Total recommended orders: ${total_order_value:,.2f}")
    print(f"  Total excess inventory:   ${total_excess_value:,.2f}")
    print(f"  {'=' * 50}")

    print(f"\n  --- Top 15 CRITICAL/HIGH Priority Recommendations ---")
    inventory_recs.filter(col("priority_rank") <= 2).show(15, truncate=False)

    print(f"\n  --- Demand Statistics (sample) ---")
    demand_stats.select(
        "store_id", "country", "city", "category",
        spark_round("avg_daily_demand", 1).alias("avg_demand"),
        spark_round("std_daily_demand", 1).alias("std_demand"),
        "days_of_history"
    ).orderBy("store_id", "category").show(20, truncate=False)

    # Save outputs
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
    import json as _json
    with open(os.path.join(OUTPUT_TABLES_DIR, "inventory_summary.json"), "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"  Saved inventory_summary.json")

    return {
        "inventory_recs": inventory_recs,
        "demand_stats": demand_stats,
        "summary": summary
    }


def save_outputs(integrated, demand, eda_results, forecast_results, insights, inventory_results=None):
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    integrated_path = os.path.join(PARQUET_DIR, "integrated")
    integrated.coalesce(20).write.mode("overwrite").parquet(integrated_path)
    print(f"  Saved integrated data to {integrated_path}")

    demand_path = os.path.join(PARQUET_DIR, "demand")
    demand.coalesce(10).write.mode("overwrite").parquet(demand_path)
    print(f"  Saved demand data to {demand_path}")

    for name, df in eda_results.items():
        if hasattr(df, "toPandas"):
            pdf = df.toPandas()
            csv_path = os.path.join(OUTPUT_TABLES_DIR, f"{name}.csv")
            pdf.to_csv(csv_path, index=False)
            print(f"  Saved {name}")

    all_metrics = {
        "rf": forecast_results["rf_metrics"],
        "gbt": forecast_results["gbt_metrics"],
        "lr": forecast_results["lr_metrics"]
    }
    save_metrics(all_metrics, os.path.join(OUTPUT_TABLES_DIR, "model_metrics.json"))
    print("  Saved model metrics")

    for name, df in insights.items():
        if hasattr(df, "toPandas"):
            pdf = df.toPandas()
            csv_path = os.path.join(OUTPUT_TABLES_DIR, f"{name}.csv")
            pdf.to_csv(csv_path, index=False)
            print(f"  Saved {name}")

    if inventory_results:
        print("  Inventory outputs already saved in phase11")

    print("  All outputs saved.")


def main():
    print("=" * 60)
    print("GLOBAL FASHION DEMAND FORECASTING WITH PYSPARK")
    print("=" * 60)

    t_start = time.time()

    spark = get_spark_session()
    print(f"  SparkSession created: {spark.sparkContext.appName}")

    datasets = phase2_ingestion(spark)

    cleaned = phase3_clean(datasets)

    integrated = phase4_integration(cleaned)

    eda_results = phase5_eda(integrated)

    demand = phase6_demand_aggregation(integrated)

    forecast_results = phase8_forecasting(demand)

    insights = phase10_retail_insights(integrated)

    inventory_results = phase11_inventory_engine(demand, forecast_results)

    save_outputs(integrated, demand, eda_results, forecast_results, insights, inventory_results)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE | Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    spark.stop()


if __name__ == "__main__":
    main()
