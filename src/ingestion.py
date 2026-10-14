import os
from pyspark.sql import SparkSession
from config import (
    RAW_DATA_DIR, PARQUET_DIR, SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEMORY, DATA_FILES
)


def get_spark_session():
    import os
    temp_dir = os.path.join(os.path.dirname(PROJECT_ROOT) if False else ".", "spark_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    return (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.parquet.compression", "snappy")
        .config("spark.local.dir", temp_dir)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def load_csv(spark, file_key, header=True, infer_schema=True):
    path = os.path.join(RAW_DATA_DIR, DATA_FILES[file_key])
    return (
        spark.read
        .option("header", header)
        .option("inferSchema", infer_schema)
        .option("multiLine", "true")
        .csv(path)
    )


def save_parquet(df, path, partition_cols=None):
    writer = df.write.mode("overwrite").parquet(path)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save()
