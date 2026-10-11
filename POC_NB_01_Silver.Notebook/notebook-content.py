# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "db44b2e5-406d-4faa-a02a-02c6d6bc145d",
# META       "default_lakehouse_name": "POC_LH_BRP_SILVER",
# META       "default_lakehouse_workspace_id": "cf8e7010-b147-4129-9b1b-e9d49f595be2"
# META     },
# META     "environment": {
# META       "environmentId": "63afde11-ede9-a8ac-4a54-40df3049bfaa",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# PARAMETERS CELL ********************

run_id = "manual_20261009"
as_of_date = "2026-08-28"
alert_threshold = 0.10


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

alert_threshold = float(alert_threshold)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

"""Shared Spark helpers embedded in each generated POC notebook."""

import json
import re
import time

import notebookutils
from pyspark.sql import SparkSession, functions as F

WORKSPACE_ID = "cf8e7010-b147-4129-9b1b-e9d49f595be2"
BRONZE_ID = "ff748fb3-b044-4729-a322-e39f97cafe1a"
SILVER_ID = "db44b2e5-406d-4faa-a02a-02c6d6bc145d"
GOLD_ID = "95f39639-d268-486d-818e-ca16976ad318"


def lakehouse_root(item_id: str) -> str:
    return f"abfss://{WORKSPACE_ID}@onelake.dfs.fabric.microsoft.com/{item_id}"


def require_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("run_id must contain 1-80 letters, numbers, _ or -.")
    return value


def write_candidate(frame, name: str, run_id: str) -> int:
    if not name.startswith("POC_"):
        raise ValueError("Only POC candidate names may be written.")
    path = f"{lakehouse_root(SILVER_ID)}/Files/runs/{require_run_id(run_id)}/{name}"
    frame.write.format("delta").mode("overwrite").save(path)
    return frame.count()


def read_candidate(spark, name: str, run_id: str):
    path = f"{lakehouse_root(SILVER_ID)}/Files/runs/{require_run_id(run_id)}/{name}"
    return spark.read.format("delta").load(path)


def write_evidence(name: str, run_id: str, value: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid evidence name.")
    path = f"{lakehouse_root(SILVER_ID)}/Files/runs/{require_run_id(run_id)}"
    notebookutils.fs.mkdirs(path)
    notebookutils.fs.put(f"{path}/{name}.json", json.dumps(value, default=str), True)


def source_table(spark, name: str):
    allowed = {
        "sanitized_raw_vehicle_data", "sanitized_raw_claims",
        "sanitized_raw_model_family", "sanitized_raw_fleet_observations",
    }
    if name not in allowed:
        raise ValueError("Unexpected Bronze source.")
    return spark.read.format("delta").load(
        f"{lakehouse_root(BRONZE_ID)}/Tables/POC/{name}"
    )


def active_spark():
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("An active Fabric Spark session is required.")
    spark.conf.set("spark.sql.shuffle.partitions", "16")
    return spark


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

"""Build run-isolated survival observations and bounded usage features.

Registration starts age follow-up; administrative censoring uses as_of_date.
Invalid claim chronology quarantines affected vehicles from survival entirely,
rather than silently treating them as failure-free. These are exploratory curves,
not a claim of production Dataiku parity.
"""

from pyspark.sql import Window


def build_silver(run_id: str, as_of_date: str, min_events: int = 10) -> dict:
    started = time.monotonic()
    spark = active_spark()
    require_run_id(run_id)
    if notebookutils.fs.exists(
        f"{lakehouse_root(GOLD_ID)}/Files/outbound/{run_id}/_COMPLETE.json"
    ):
        raise ValueError("A published run is immutable; use a new run_id.")
    vehicles = source_table(spark, "sanitized_raw_vehicle_data")
    models = source_table(spark, "sanitized_raw_model_family")
    claims = source_table(spark, "sanitized_raw_claims")
    fleet = source_table(spark, "sanitized_raw_fleet_observations")
    if vehicles.count() != vehicles.select("SERIAL_NO").distinct().count():
        raise ValueError("Vehicle key is not unique; refusing a fan-out join.")
    if models.count() != models.select("MODEL_NO").distinct().count():
        raise ValueError("Model key is not unique.")
    reference = models.select(
        F.col("MODEL_NO").alias("PRODUCT_CODE"),
        F.col("MODEL_YEAR").cast("int").alias("model_year"),
        F.col("PRODUCT_LINE_GROUP").alias("product_line"),
    )
    population = vehicles.select(
        "SERIAL_NO", "PRODUCT_CODE", "PLANT", "FLEET_REGION",
        "REGISTRATION_DATE",
    ).join(reference, "PRODUCT_CODE", "left").select(
        "SERIAL_NO", "product_line", "model_year", "PLANT", "FLEET_REGION",
        F.col("REGISTRATION_DATE").alias("origin_date"),
    )
    full_population = population
    invalid_population = population.filter(
        F.col("origin_date").isNull()
        | (F.col("origin_date") > F.to_date(F.lit(as_of_date)))
        | F.col("product_line").isNull() | F.col("model_year").isNull()
        | F.col("PLANT").isNull()
    )
    invalid_ids = invalid_population.select("SERIAL_NO")
    joined_claims = claims.join(
        population.select("SERIAL_NO", "origin_date"), "SERIAL_NO", "inner"
    )
    invalid_claims = joined_claims.filter(
        F.col("FAILURE_DATE").isNull()
        | (F.col("FAILURE_DATE") < F.col("origin_date"))
        | (F.col("FAILURE_DATE") > F.to_date(F.lit(as_of_date)))
        | (F.col("FAILURE_DATE") > F.col("REPAIR_DATE"))
    )
    quarantined = invalid_ids.union(
        invalid_claims.select("SERIAL_NO")
    ).distinct()
    population = population.join(quarantined, "SERIAL_NO", "left_anti").cache()
    cohort_sizes = full_population.groupBy("product_line", "model_year").agg(
        F.count("*").alias("source_vehicles")
    ).join(
        population.groupBy("product_line", "model_year").agg(
            F.count("*").alias("retained_vehicles")
        ), ["product_line", "model_year"], "left",
    ).fillna({"retained_vehicles": 0}).withColumn(
        "retention_ratio", F.col("retained_vehicles") / F.col("source_vehicles")
    )
    valid_claims = claims.join(
        population.select("SERIAL_NO"), "SERIAL_NO", "inner"
    )
    first_overall = valid_claims.groupBy("SERIAL_NO").agg(
        F.min("FAILURE_DATE").alias("first_event_date")
    ).withColumn("event_type", F.lit("ALL_FAILURES"))
    first_family = valid_claims.filter(F.col("PART_FAMILY").isNotNull()).groupBy(
        "SERIAL_NO", "PART_FAMILY"
    ).agg(F.min("FAILURE_DATE").alias("first_event_date")).select(
        "SERIAL_NO", F.col("PART_FAMILY").alias("event_type"), "first_event_date"
    )
    events = first_overall.unionByName(first_family)
    cohort_events = events.join(
        population.select("SERIAL_NO", "product_line", "model_year"),
        "SERIAL_NO", "inner",
    )
    groups = cohort_events.groupBy(
        "product_line", "model_year", "event_type"
    ).agg(F.count("*").alias("distinct_events")).filter(
        F.col("distinct_events") >= min_events
    )
    observations = population.join(
        groups.select("product_line", "model_year", "event_type"),
        ["product_line", "model_year"], "inner",
    ).join(events, ["SERIAL_NO", "event_type"], "left").withColumn(
        "event", F.col("first_event_date").isNotNull().cast("int")
    ).withColumn(
        "duration",
        F.datediff(
            F.coalesce("first_event_date", F.to_date(F.lit(as_of_date))),
            "origin_date",
        ).cast("double"),
    ).withColumn("axis", F.lit("age_days"))
    eligible_fleet = fleet.filter(
        ~F.col("METAIS_DELETED")
        & F.col("VALID_OBSERVATION_FLAG")
        & (F.col("USAGE_UNIT") == "KM")
        & F.col("VEHICLE_USAGE").isNotNull()
        & (F.col("VEHICLE_USAGE") >= 0)
        & (F.col("MEASURE_DATE") >= F.col("REGISTRATION_DATE"))
        & (F.col("MEASURE_DATE") <= F.to_date(F.lit(as_of_date)))
    )
    latest_km = eligible_fleet.groupBy("SERIAL_NO").agg(
        F.max("VEHICLE_USAGE").alias("observed_km")
    )
    first_window = Window.partitionBy("SERIAL_NO", "PART_FAMILY").orderBy(
        "FAILURE_DATE", "CLAIM_ID", "PART_NO"
    )
    km_family = valid_claims.filter(
        F.col("PART_FAMILY").isNotNull()
    ).withColumn("_first", F.row_number().over(first_window)).filter(
        F.col("_first") == 1
    ).select(
        "SERIAL_NO", F.col("PART_FAMILY").alias("event_type"),
        F.col("ODOMETER_KM").alias("event_km"),
    )
    overall_window = Window.partitionBy("SERIAL_NO").orderBy(
        "FAILURE_DATE", "CLAIM_ID", "PART_NO"
    )
    km_overall = valid_claims.withColumn(
        "_first", F.row_number().over(overall_window)
    ).filter(F.col("_first") == 1).select(
        "SERIAL_NO", F.lit("ALL_FAILURES").alias("event_type"),
        F.col("ODOMETER_KM").alias("event_km"),
    )
    mileage = observations.join(
        km_family.unionByName(km_overall), ["SERIAL_NO", "event_type"], "left"
    ).join(latest_km, "SERIAL_NO", "left").withColumn(
        "duration",
        F.when(F.col("event") == 1, F.col("event_km")).otherwise(
            F.col("observed_km")
        ).cast("double"),
    ).filter(
        F.col("duration").isNotNull() & (F.col("duration") >= 0)
        & (F.col("duration") <= 300000)
    ).withColumn("axis", F.lit("odometer_km"))
    observations = observations.select(
        "SERIAL_NO", "product_line", "model_year", "event_type",
        "duration", "event", "axis",
    ).unionByName(mileage.select(
        "SERIAL_NO", "product_line", "model_year", "event_type",
        "duration", "event", "axis",
    ))
    observation_rows = write_candidate(
        observations, "POC_survival_analysis_observations", run_id,
    )
    features = eligible_fleet.select(
        "SERIAL_NO", "PRODUCT_CODE", "PLANT", "FLEET_REGION",
        "MEASURE_DATE", "VEHICLE_USAGE", "REGISTRATION_DATE",
    ).join(reference, "PRODUCT_CODE", "inner").select(
        "SERIAL_NO", "product_line", "model_year", "PLANT", "FLEET_REGION",
        "MEASURE_DATE", F.col("VEHICLE_USAGE").cast("double").alias("target_km"),
        F.datediff("MEASURE_DATE", "REGISTRATION_DATE").alias("age_days"),
    ).filter(F.col("target_km") <= 300000).withColumn(
        "vehicle_fold", F.pmod(F.xxhash64("SERIAL_NO"), F.lit(5))
    ).orderBy(F.xxhash64("SERIAL_NO", "MEASURE_DATE")).limit(100000)
    feature_rows = features.count()
    if feature_rows < 100:
        raise ValueError("Insufficient valid usage training rows.")
    features.coalesce(1).write.mode("overwrite").parquet(
        f"{lakehouse_root(SILVER_ID)}/Files/runs/{run_id}/usage_features"
    )
    quality = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "vehicle_rows": vehicles.count(),
        "invalid_population_rows": invalid_population.count(),
        "invalid_claim_rows": invalid_claims.count(),
        "quarantined_vehicles": quarantined.count(),
        "retained_survival_vehicles": population.count(),
        "candidate_event_groups": groups.count(),
        "minimum_cohort_retention": cohort_sizes.agg(
            F.min("retention_ratio")
        ).first()[0],
        "cohort_sizes": [row.asDict() for row in cohort_sizes.collect()],
        "survival_observation_rows": observation_rows,
        "usage_feature_rows": feature_rows,
        "mileage_outlier_limit_km": 300000,
        "excluded_fleet_rows": fleet.count() - eligible_fleet.count(),
        "statistical_status": "exploratory_not_production_validated",
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_evidence("silver", run_id, quality)
    population.unpersist()
    print(json.dumps(quality, indent=2))
    return quality


silver_result = build_silver(run_id, as_of_date)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
