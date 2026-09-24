import os
import sys
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DecimalType,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from bank_session import get_spark, step, RAW, WAREHOUSE

C360_WAREHOUSE = f"{WAREHOUSE}/c360"
spark = get_spark(
    "assignment-customer360",
    delta=True,
    iceberg=True,
    extra={"spark.sql.catalog.local.warehouse": C360_WAREHOUSE},
)
BRONZE = f"{C360_WAREHOUSE}/bronze"
QUARANTINE = f"{C360_WAREHOUSE}/quarantine"

TXN_SCHEMA = StructType([
    StructField("txn_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("txn_ts", StringType(), True),
    StructField("amount", DecimalType(18, 2), True),
    StructField("currency", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("txn_type", StringType(), True),
    StructField("counterparty_account", StringType(), True),
    StructField("status", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("city", StringType(), True),
])

ACCOUNTS_SCHEMA = StructType([
    StructField("account_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("account_type", StringType(), True),
    StructField("branch_code", StringType(), True),
    StructField("open_date", StringType(), True),
    StructField("status", StringType(), True),
    StructField("balance", DecimalType(18, 2), True),
    StructField("currency", StringType(), True),
])

CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", StringType(), True),
    StructField("full_name", StringType(), True),
    StructField("dob_year", IntegerType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("income_band", StringType(), True),
    StructField("kyc_status", StringType(), True),
    StructField("risk_rating", StringType(), True),
    StructField("onboarded_date", StringType(), True),
    StructField("segment", StringType(), True),
])

LOANS_SCHEMA = StructType([
    StructField("loan_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product", StringType(), True),
    StructField("principal", DecimalType(18, 2), True),
    StructField("interest_rate", DecimalType(18, 2), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("emi", DecimalType(18, 2), True),
    StructField("disbursed_date", StringType(), True),
    StructField("outstanding", DecimalType(18, 2), True),
    StructField("dpd", IntegerType(), True),
    StructField("status", StringType(), True),
])

CARDS_SCHEMA = StructType([
    StructField("card_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("card_type", StringType(), True),
    StructField("network", StringType(), True),
    StructField("issue_date", StringType(), True),
    StructField("expiry_date", StringType(), True),
    StructField("status", StringType(), True),
    StructField("credit_limit", DecimalType(18, 2), True),
])

FRAUD_SCHEMA = StructType([
    StructField("txn_id", StringType(), True),
    StructField("is_fraud", IntegerType(), True),
    StructField("reported_date", StringType(), True),
    StructField("fraud_type", StringType(), True),
])

BRANCH_SCHEMA = StructType([
    StructField("branch_code", StringType(), True),
    StructField("branch_name", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("region", StringType(), True),
    StructField("ifsc", StringType(), True),
])


def read_csv(path, schema):
    return spark.read.option("header", True).schema(schema).csv(path)


def write_delta(df, path, partition_by=None):
    writer = df.write.format("delta").mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.save(path)


step(1, "Bronze")
transactions = read_csv(f"{RAW}/transactions/*.csv", TXN_SCHEMA)
bronze_transactions = (
    transactions
    .withColumn("source_file", F.input_file_name())
    .withColumn("ingest_date", F.to_date(F.regexp_extract(F.input_file_name(), r"dt=(\d{4}-\d{2}-\d{2})", 1)))
    .withColumn("ingest_ts", F.current_timestamp())
)
bronze_transactions.write.mode("overwrite").partitionBy("ingest_date").parquet(f"{BRONZE}/transactions")
print(f"bronze/transactions rows: {bronze_transactions.count():,}")
print("bronze/transactions by ingest_date:")
bronze_transactions.groupBy("ingest_date").count().orderBy("ingest_date").show()

bronze_updates = read_csv(f"{RAW}/txn_updates.csv", TXN_SCHEMA).withColumn("source_file", F.input_file_name()).withColumn("ingest_ts", F.current_timestamp())
bronze_updates.write.mode("overwrite").parquet(f"{BRONZE}/txn_updates")
print(f"bronze/txn_updates rows: {bronze_updates.count():,}")

for name, schema in [
    ("accounts", ACCOUNTS_SCHEMA),
    ("customers", CUSTOMERS_SCHEMA),
    ("loans", LOANS_SCHEMA),
    ("cards", CARDS_SCHEMA),
    ("fraud_labels", FRAUD_SCHEMA),
    ("branches", BRANCH_SCHEMA),
]:
    df = read_csv(f"{RAW}/{name}.csv", schema).withColumn("ingest_ts", F.current_timestamp())
    df.write.mode("overwrite").parquet(f"{BRONZE}/{name}")
    print(f"bronze/{name} rows: {df.count():,}")

step(2, "Silver")
accounts = read_csv(f"{RAW}/accounts.csv", ACCOUNTS_SCHEMA)
customers = read_csv(f"{RAW}/customers.csv", CUSTOMERS_SCHEMA)
loans = read_csv(f"{RAW}/loans.csv", LOANS_SCHEMA)
cards = read_csv(f"{RAW}/cards.csv", CARDS_SCHEMA)
fraud_labels = read_csv(f"{RAW}/fraud_labels.csv", FRAUD_SCHEMA)
branches = read_csv(f"{RAW}/branches.csv", BRANCH_SCHEMA)

bronze_count = bronze_transactions.count()
dedup_window = Window.partitionBy("txn_id").orderBy(F.monotonically_increasing_id())
deduped = bronze_transactions.withColumn("rn", F.row_number().over(dedup_window)).filter(F.col("rn") == 1).drop("rn")
duplicates_removed = bronze_count - deduped.count()

txn_columns = [
    "txn_id", "account_id", "txn_ts", "amount", "currency", "channel",
    "merchant_category", "txn_type", "counterparty_account", "status",
    "device_id", "city",
]
updates = bronze_updates.select(*txn_columns)
original = (
    deduped
    .join(updates.select("txn_id"), "txn_id", "left_anti")
    .select(*txn_columns)
    .withColumn("record_source", F.lit("ORIGINAL"))
)
corrections = updates.withColumn("record_source", F.lit("CORRECTION"))
corrected = original.unionByName(corrections)
rows_after_corrections = corrected.count()

cleaned = (
    corrected
    .withColumn("channel", F.upper(F.trim(F.col("channel"))))
    .withColumn("merchant_category", F.when(F.trim(F.coalesce(F.col("merchant_category"), F.lit(""))) == "", F.lit("UNKNOWN")).otherwise(F.col("merchant_category")))
    .withColumn("txn_ts", F.to_timestamp("txn_ts"))
    .withColumn("posting_date", F.to_date("txn_ts"))
)

account_ref = accounts.select("account_id", "customer_id", "branch_code")
checked = cleaned.join(account_ref, "account_id", "left")
checked = checked.withColumn(
    "bad_reason",
    F.when(F.col("amount") < 0, F.lit("NEGATIVE_AMOUNT"))
     .when(F.col("customer_id").isNull(), F.lit("ORPHAN_ACCOUNT"))
)

quarantine = checked.filter(F.col("bad_reason").isNotNull())
quarantine.select(
    "txn_id", "account_id", "customer_id", "branch_code", "txn_ts", "posting_date", "amount", "bad_reason"
).write.mode("overwrite").parquet(f"{QUARANTINE}/transactions")

fraud_ref = F.broadcast(fraud_labels.select("txn_id", "is_fraud", "fraud_type"))
silver_transactions = (
    checked.filter(F.col("bad_reason").isNull())
    .drop("bad_reason")
    .join(fraud_ref, "txn_id", "left")
    .withColumn("is_fraud", F.coalesce(F.col("is_fraud"), F.lit(0)) == 1)
    .select(
        "txn_id", "account_id", "customer_id", "branch_code", "txn_ts", "posting_date",
        "amount", "channel", "merchant_category", "txn_type", "status", "is_fraud", "fraud_type", "record_source"
    )
)
write_delta(silver_transactions, f"{C360_WAREHOUSE}/silver/transactions", "posting_date")

negative_count = quarantine.filter(F.col("bad_reason") == "NEGATIVE_AMOUNT").count()
orphan_count = quarantine.filter(F.col("bad_reason") == "ORPHAN_ACCOUNT").count()
print(f"duplicate rows removed: {duplicates_removed:,}")
print(f"rows after corrections: {rows_after_corrections:,}")
print(f"quarantine NEGATIVE_AMOUNT: {negative_count:,}")
print(f"quarantine ORPHAN_ACCOUNT: {orphan_count:,}")
print(f"silver/transactions rows: {silver_transactions.count():,}")
print("silver/transactions status counts:")
silver_transactions.groupBy("status").count().orderBy("status").show()
print(f"silver/transactions is_fraud=true: {silver_transactions.filter(F.col('is_fraud')).count():,}")
print(f"silver/transactions txn_id unique: {silver_transactions.select('txn_id').distinct().count() == silver_transactions.count()}")

silver_customers = customers.withColumn("age", F.lit(2026) - F.col("dob_year")).withColumn(
    "age_band",
    F.when(F.col("age").between(18, 24), "18-24")
     .when(F.col("age").between(25, 34), "25-34")
     .when(F.col("age").between(35, 44), "35-44")
     .when(F.col("age").between(45, 59), "45-59")
     .otherwise("60+")
)
write_delta(silver_customers, f"{C360_WAREHOUSE}/silver/customers")
print(f"silver/customers rows: {silver_customers.count():,}")
print("silver/customers age_band counts:")
silver_customers.groupBy("age_band").count().orderBy("age_band").show()

silver_loans = loans.withColumn(
    "asset_class",
    F.when(F.col("status") == "CLOSED", "CLOSED")
     .when(F.col("dpd") == 0, "STANDARD")
     .when(F.col("dpd").between(1, 30), "SMA-0")
     .when(F.col("dpd").between(31, 60), "SMA-1")
     .when(F.col("dpd").between(61, 90), "SMA-2")
     .when(F.col("dpd") > 90, "NPA")
)
write_delta(silver_loans, f"{C360_WAREHOUSE}/silver/loans")
print(f"silver/loans rows: {silver_loans.count():,}")
print("silver/loans asset_class counts:")
silver_loans.groupBy("asset_class").count().orderBy("asset_class").show()

step(3, "Gold")
spark.sql("CREATE DATABASE IF NOT EXISTS local.gold")

account_counts = accounts.groupBy("customer_id").agg(F.count("account_id").alias("num_accounts"))
account_balances = accounts.filter(F.col("status") != "CLOSED").groupBy("customer_id").agg(F.sum("balance").cast("decimal(18,2)").alias("total_balance"))

success_txns = silver_transactions.filter(F.col("status") == "SUCCESS")
txn_summary = success_txns.groupBy("customer_id").agg(
    F.count("txn_id").alias("txn_count"),
    F.sum(F.when(F.col("txn_type") == "DEBIT", F.col("amount")).otherwise(F.lit(0).cast("decimal(18,2)"))).cast("decimal(18,2)").alias("debit_amount"),
    F.sum(F.when(F.col("txn_type") == "CREDIT", F.col("amount")).otherwise(F.lit(0).cast("decimal(18,2)"))).cast("decimal(18,2)").alias("credit_amount"),
)

channel_counts = success_txns.groupBy("customer_id", "channel").agg(F.count("txn_id").alias("channel_count"))
channel_window = Window.partitionBy("customer_id").orderBy(F.desc("channel_count"), F.asc("channel"))
preferred_channel = channel_counts.withColumn("rn", F.row_number().over(channel_window)).filter(F.col("rn") == 1).select("customer_id", "channel").withColumnRenamed("channel", "preferred_channel")

loan_summary = silver_loans.filter(F.col("status") == "ACTIVE").groupBy("customer_id").agg(
    F.count("loan_id").alias("active_loans"),
    F.sum("outstanding").cast("decimal(18,2)").alias("loan_outstanding"),
    F.sum("emi").cast("decimal(18,2)").alias("monthly_emi"),
    F.max("dpd").alias("max_dpd"),
)

card_summary = cards.filter(F.col("status") == "ACTIVE").groupBy("customer_id").agg(F.count("card_id").alias("active_cards"))
fraud_summary = silver_transactions.filter(F.col("is_fraud")).groupBy("customer_id").agg(F.count("txn_id").alias("fraud_txn_count"))

customer_360 = (
    silver_customers.select("customer_id", "segment", "risk_rating", "kyc_status", "age_band")
    .join(account_counts, "customer_id", "left")
    .join(account_balances, "customer_id", "left")
    .join(txn_summary, "customer_id", "left")
    .join(preferred_channel, "customer_id", "left")
    .join(loan_summary, "customer_id", "left")
    .join(card_summary, "customer_id", "left")
    .join(fraud_summary, "customer_id", "left")
    .fillna({
        "num_accounts": 0,
        "total_balance": 0,
        "txn_count": 0,
        "debit_amount": 0,
        "credit_amount": 0,
        "preferred_channel": "NONE",
        "active_loans": 0,
        "loan_outstanding": 0,
        "monthly_emi": 0,
        "active_cards": 0,
        "fraud_txn_count": 0,
    })
    .withColumn(
        "relationship_tier",
        F.when(F.col("num_accounts") == 0, "NO_ACCOUNT")
         .when((F.col("total_balance") >= F.lit(1000000)) | (F.col("segment") == "WEALTH"), "PLATINUM")
         .when(F.col("total_balance") >= F.lit(200000), "GOLD")
         .otherwise("SILVER")
    )
    .withColumn(
        "attention_flag",
        F.coalesce(F.col("max_dpd") > 90, F.lit(False))
        | (F.col("fraud_txn_count") > 0)
        | (F.col("kyc_status") != "VERIFIED")
    )
    .select(
        "customer_id", "segment", "risk_rating", "kyc_status", "age_band", "num_accounts", "total_balance",
        "txn_count", "debit_amount", "credit_amount", "preferred_channel", "active_loans", "loan_outstanding",
        "monthly_emi", "max_dpd", "active_cards", "fraud_txn_count", "relationship_tier", "attention_flag"
    )
)

customer_360.createOrReplaceTempView("customer_360")
spark.sql("DROP TABLE IF EXISTS local.gold.customer_360")
spark.sql("CREATE TABLE local.gold.customer_360 USING iceberg AS SELECT * FROM customer_360")
print(f"gold/customer_360 rows: {customer_360.count():,}")
print("relationship_tier counts:")
customer_360.groupBy("relationship_tier").count().orderBy("relationship_tier").show()
print("preferred_channel counts:")
customer_360.groupBy("preferred_channel").count().orderBy("preferred_channel").show()
print(f"attention_flag=true: {customer_360.filter(F.col('attention_flag')).count():,}")

active_loans = silver_loans.filter(F.col("status") == "ACTIVE")
products = active_loans.select("product").distinct()
asset_classes = active_loans.select("asset_class").distinct()
loan_grid = products.crossJoin(asset_classes)
loan_agg = active_loans.groupBy("product", "asset_class").agg(
    F.count("loan_id").alias("loan_count"),
    F.sum("outstanding").cast("decimal(18,2)").alias("outstanding"),
)
loan_totals = active_loans.groupBy("product").agg(F.sum("outstanding").cast("decimal(18,2)").alias("product_total"))
npa_totals = active_loans.filter(F.col("asset_class") == "NPA").groupBy("product").agg(F.sum("outstanding").cast("decimal(18,2)").alias("npa_total"))
loan_portfolio = (
    loan_grid.join(loan_agg, ["product", "asset_class"], "left")
    .join(loan_totals, "product", "left")
    .join(npa_totals, "product", "left")
    .fillna({"loan_count": 0, "outstanding": 0, "npa_total": 0})
    .withColumn("share_pct", F.round(F.when(F.col("product_total") != 0, F.col("outstanding") / F.col("product_total") * 100).otherwise(0), 2))
    .withColumn("gnpa_pct", F.round(F.when(F.col("product_total") != 0, F.col("npa_total") / F.col("product_total") * 100).otherwise(0), 2))
    .select("product", "asset_class", "loan_count", "outstanding", "share_pct", "gnpa_pct")
)
loan_portfolio.createOrReplaceTempView("loan_portfolio")
spark.sql("DROP TABLE IF EXISTS local.gold.loan_portfolio")
spark.sql("CREATE TABLE local.gold.loan_portfolio USING iceberg AS SELECT * FROM loan_portfolio")
print(f"gold/loan_portfolio rows: {loan_portfolio.count():,}")
print("gnpa_pct by product:")
loan_portfolio.select("product", "gnpa_pct").distinct().orderBy("product").show()

fraud_daily = (
    success_txns.groupBy("posting_date", "channel")
    .agg(
        F.count("txn_id").alias("txn_count"),
        F.sum("amount").cast("decimal(18,2)").alias("txn_amount"),
        F.sum(F.when(F.col("is_fraud"), 1).otherwise(0)).alias("fraud_count"),
        F.sum(F.when(F.col("is_fraud"), F.col("amount")).otherwise(F.lit(0).cast("decimal(18,2)"))).cast("decimal(18,2)").alias("fraud_amount"),
    )
    .withColumn("fraud_rate_bps", F.round(F.col("fraud_count") / F.col("txn_count") * 10000, 1))
    .select("posting_date", "channel", "txn_count", "txn_amount", "fraud_count", "fraud_amount", "fraud_rate_bps")
)
fraud_daily.createOrReplaceTempView("fraud_daily")
spark.sql("DROP TABLE IF EXISTS local.gold.channel_fraud_daily")
spark.sql("CREATE TABLE local.gold.channel_fraud_daily USING iceberg AS SELECT * FROM fraud_daily")
print(f"gold/channel_fraud_daily rows: {fraud_daily.count():,}")
print(f"gold/channel_fraud_daily total fraud_count: {fraud_daily.agg(F.sum('fraud_count')).first()[0]:,}")
print("UPI over 3 days:")
fraud_daily.filter(F.col("channel") == "UPI").agg(F.sum("txn_count").alias("txn_count"), F.sum("fraud_count").alias("fraud_count")).show()

spark.stop()
