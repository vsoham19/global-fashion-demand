from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, when, lit
import os


FEATURE_COLS_NUMERIC = [
    "year", "month", "quarter", "day_of_week", "week_of_year",
    "avg_unit_price", "avg_discount", "is_discounted",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_avg_7", "rolling_avg_14", "rolling_avg_30"
]

FEATURE_COLS_CATEGORICAL = ["store_id", "product_id"]

TARGET_COL = "total_quantity"


def prepare_features(df: DataFrame) -> DataFrame:
    fill_vals = {}
    for c in FEATURE_COLS_NUMERIC:
        if c in df.columns:
            fill_vals[c] = 0.0
    if fill_vals:
        df = df.fillna(fill_vals)
    return df


def build_lr_pipeline():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in FEATURE_COLS_CATEGORICAL
        if c in FEATURE_COLS_NUMERIC or True
    ]
    all_features = FEATURE_COLS_NUMERIC + [f"{c}_idx" for c in FEATURE_COLS_CATEGORICAL]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
    lr = LinearRegression(featuresCol="features", labelCol=TARGET_COL, maxIter=100, regParam=0.1)
    return Pipeline(stages=indexers + [assembler, lr])


def build_rf_pipeline():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in FEATURE_COLS_CATEGORICAL
    ]
    all_features = FEATURE_COLS_NUMERIC + [f"{c}_idx" for c in FEATURE_COLS_CATEGORICAL]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
    rf = RandomForestRegressor(
        featuresCol="features", labelCol=TARGET_COL,
        numTrees=50, maxDepth=10, seed=42
    )
    return Pipeline(stages=indexers + [assembler, rf])


def split_chronological(df: DataFrame, train_ratio=0.7, val_ratio=0.15):
    total = df.count()
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    ordered = df.orderBy("date")
    train = ordered.limit(train_end)
    remaining = ordered.subtract(train).orderBy("date")
    val = remaining.limit(val_end - train_end)
    test = remaining.subtract(val).orderBy("date")

    return train, val, test


def evaluate_model(predictions: DataFrame, label_col=TARGET_COL, prediction_col="prediction"):
    evaluators = {
        "rmse": RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse"),
        "mae": RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae"),
        "r2": RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2"),
    }
    results = {}
    for name, evaluator in evaluators.items():
        results[name] = evaluator.evaluate(predictions)
    return results


def compute_smape(predictions: DataFrame, label_col=TARGET_COL, prediction_col="prediction"):
    df = predictions.withColumn(
        "smape_val",
        when(
            (col(label_col) + col(prediction_col)) > 0,
            spark_abs(col(label_col) - col(prediction_col)) / ((col(label_col) + col(prediction_col)) / 2)
        ).otherwise(lit(0.0))
    )
    total = df.count()
    if total == 0:
        return 0.0
    smape_sum = df.select("smape_val").rdd.map(lambda r: r[0]).sum()
    return smape_sum / total * 100


def train_and_evaluate(train_df, val_df, test_df, model_type="rf"):
    train_df = prepare_features(train_df)
    val_df = prepare_features(val_df)
    test_df = prepare_features(test_df)

    if model_type == "lr":
        pipeline = build_lr_pipeline()
    else:
        pipeline = build_rf_pipeline()

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    metrics = evaluate_model(predictions)
    smape = compute_smape(predictions)
    metrics["smape"] = smape

    return model, predictions, metrics


def save_model(model, path):
    model.write().overwrite().save(path)
