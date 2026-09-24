# set environment variable
# export ICEBERG_WAREHOUSE="/home/sbit/projects/spark-banking-training/data/warehouse/iceberg"
# export BI_OUTPUT="/home/sbit/projects/bi_projects/bi_output"

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------- config
# Path to the folder the DWH team's lab 11b wrote to: <project>/data/warehouse/iceberg
WAREHOUSE = os.environ.get("ICEBERG_WAREHOUSE")
if not WAREHOUSE or not os.path.isdir(WAREHOUSE):
    sys.exit(f"Set ICEBERG_WAREHOUSE to the DWH iceberg folder. Got: {WAREHOUSE!r}")

# Where BI writes its own report outputs. Keep it OUTSIDE the DWH warehouse.
BI_OUTPUT = os.environ.get("BI_OUTPUT", os.path.join(os.getcwd(), "bi_output"))

# MUST match the DWH side: Spark 3.5 + Iceberg 1.5.2 (see bank_session.py)
ICEBERG_VERSION = "1.5.2"


def get_bi_spark(app_name="bi-iceberg-reader"):
    """SparkSession that sees the DWH Iceberg warehouse as catalog `dwh`."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages",
                f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION}")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        # `dwh` = the shared Hadoop catalog (same settings as `local` in bank_session.py)
        .config("spark.sql.catalog.dwh", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.dwh.type", "hadoop")
        .config("spark.sql.catalog.dwh.warehouse", WAREHOUSE)
        # so plain `silver.txns` works without typing `dwh.` every time
        .config("spark.sql.defaultCatalog", "dwh")
        .config("spark.sql.shuffle.partitions", 8)
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def title(text):
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


spark = get_bi_spark()
spark.sparkContext.setLogLevel("ERROR")

# ====================================================================
title("1. What did the DWH team publish?")
# ====================================================================
spark.sql("SHOW NAMESPACES IN dwh").show()
spark.sql("SHOW TABLES IN dwh.silver").show(truncate=False)
spark.sql("DESCRIBE TABLE dwh.silver.txns").show(50, truncate=False)

# ====================================================================
title("2. Spark SQL: daily successful spend by channel")
# ====================================================================
daily_sql = spark.sql("""
    SELECT  to_date(txn_ts)          AS txn_date,
            channel,
            count(*)                 AS txn_count,
            round(sum(amount), 2)    AS total_amount
    FROM    silver.txns                      -- same as dwh.silver.txns
    WHERE   status = 'SUCCESS'
    GROUP BY to_date(txn_ts), channel
    ORDER BY txn_date, channel
""")
daily_sql.show(20, truncate=False)

# ====================================================================
title("3. PySpark API: top 10 cities by spend")
# ====================================================================
txns = spark.table("dwh.silver.txns")

top_cities = (
    txns.filter(F.col("status") == "SUCCESS")
        .groupBy("city")
        .agg(F.count("*").alias("txn_count"),
             F.round(F.sum("amount"), 2).alias("total_amount"),
             F.round(F.avg("amount"), 2).alias("avg_amount"))
        .orderBy(F.desc("total_amount"))
        .limit(10)
)
top_cities.show(truncate=False)

# ====================================================================
title("4. Snapshots + time travel (see the table as it was before MERGE)")
# ====================================================================
spark.sql("""
    SELECT snapshot_id, committed_at, operation
    FROM   dwh.silver.txns.snapshots
    ORDER BY committed_at
""").show(truncate=False)

first_id = spark.sql(
    "SELECT snapshot_id FROM dwh.silver.txns.snapshots ORDER BY committed_at LIMIT 1"
).collect()[0][0]

old = spark.read.option("snapshot-id", first_id).table("dwh.silver.txns")
print("rows in oldest kept snapshot:", old.count())
print("rows in latest snapshot     :", txns.count())
# SQL version of the same thing:
#   SELECT count(*) FROM dwh.silver.txns VERSION AS OF <snapshot_id>

# ====================================================================
title("5. Save report outputs in the BI team's own folder")
# ====================================================================
# Parquet for other Spark/Power BI jobs, CSV for Excel users.
daily_sql.write.mode("overwrite").parquet(f"{BI_OUTPUT}/daily_channel_spend")
top_cities.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv(f"{BI_OUTPUT}/top_cities_csv")

# Small result -> pandas, ready for charts / notebooks
# print(top_cities.toPandas().head())

print("Reports written to:", BI_OUTPUT)

spark.stop()
