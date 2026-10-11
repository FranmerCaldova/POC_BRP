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


# CELL ********************

alert_threshold = float(alert_threshold)


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


# CELL ********************

"""Publish only validated candidates and complete run-scoped Parquet exports."""

import datetime


def publish_gold(run_id: str) -> dict:
    started = time.monotonic()
    spark = active_spark()
    require_run_id(run_id)
    candidate_root = f"{lakehouse_root(SILVER_ID)}/Files/runs/{run_id}"
    dq = json.loads(notebookutils.fs.head(candidate_root + "/dq.json", 65536))
    if dq["status"] != "passed":
        raise ValueError("A passing run-scoped DQ result is required.")
    tables = [
        "POC_fail_rate_probabilities", "POC_claim_rate_actuals",
        "POC_fail_rate_alerts", "POC_partition_metrics",
    ]
    outbound = f"{lakehouse_root(GOLD_ID)}/Files/outbound/{run_id}"
    if notebookutils.fs.exists(outbound + "/_COMPLETE.json"):
        raise ValueError("Export is already complete; use a new run_id to rerun.")
    counts = {}
    schemas = {}
    for name in tables:
        frame = read_candidate(spark, name, run_id).withColumn("run_id", F.lit(run_id))
        counts[name] = frame.count()
        schemas[name] = frame.schema.jsonValue()
        path = f"{lakehouse_root(GOLD_ID)}/Tables/{name}"
        if notebookutils.fs.exists(path + "/_delta_log"):
            frame.write.format("delta").mode("overwrite").option(
                "replaceWhere", f"run_id = '{run_id}'"
            ).save(path)
        else:
            frame.write.format("delta").mode("overwrite").save(path)
        frame.write.mode("overwrite").parquet(f"{outbound}/{name}")
    usage = spark.read.parquet(candidate_root + "/usage_predictions.parquet")
    name = "POC_vehicle_usage_ml_predictions"
    counts[name] = usage.count()
    schemas[name] = usage.schema.jsonValue()
    path = f"{lakehouse_root(GOLD_ID)}/Tables/{name}"
    if notebookutils.fs.exists(path + "/_delta_log"):
        usage.write.format("delta").mode("overwrite").option(
            "replaceWhere", f"run_id = '{run_id}'"
        ).save(path)
    else:
        usage.write.format("delta").mode("overwrite").save(path)
    usage.write.mode("overwrite").parquet(f"{outbound}/{name}")
    report = read_candidate(spark, "POC_fail_rate_probabilities", run_id).filter(
        (F.col("axis") == "age_days") & (F.col("horizon") == 365)
    ).withColumn("run_id", F.lit(run_id))
    name = "POC_ppq_fail_rate_report"
    counts[name] = report.count()
    schemas[name] = report.schema.jsonValue()
    path = f"{lakehouse_root(GOLD_ID)}/Tables/{name}"
    if notebookutils.fs.exists(path + "/_delta_log"):
        report.write.format("delta").mode("overwrite").option(
            "replaceWhere", f"run_id = '{run_id}'"
        ).save(path)
    else:
        report.write.format("delta").mode("overwrite").save(path)
    report.write.mode("overwrite").parquet(f"{outbound}/{name}")
    observations = read_candidate(
        spark, "POC_survival_analysis_observations", run_id,
    ).withColumn("run_id", F.lit(run_id))
    silver_table = f"{lakehouse_root(SILVER_ID)}/Tables/POC_survival_analysis_observations"
    if notebookutils.fs.exists(silver_table + "/_delta_log"):
        observations.write.format("delta").mode("overwrite").option(
            "replaceWhere", f"run_id = '{run_id}'"
        ).save(silver_table)
    else:
        observations.write.format("delta").mode("overwrite").save(silver_table)
    dq_table = spark.read.format("delta").load(
        f"{lakehouse_root(GOLD_ID)}/Tables/POC_dq_results"
    ).filter(F.col("run_id") == run_id)
    counts["POC_dq_results"] = dq_table.count()
    schemas["POC_dq_results"] = dq_table.schema.jsonValue()
    dq_table.write.mode("overwrite").parquet(f"{outbound}/POC_dq_results")
    stages = []
    stage_evidence = {}
    for stage in ("silver", "survival", "usage", "dq"):
        evidence = json.loads(
            notebookutils.fs.head(candidate_root + f"/{stage}.json", 65536)
        )
        stage_evidence[stage] = evidence
        stages.append((run_id, stage, "completed", evidence["elapsed_seconds"]))
    stage_frame = spark.createDataFrame(
        stages, "run_id string, stage string, status string, elapsed_seconds double",
    )
    path = f"{lakehouse_root(GOLD_ID)}/Tables/POC_pipeline_stage_metrics"
    if notebookutils.fs.exists(path + "/_delta_log"):
        stage_frame.write.format("delta").mode("overwrite").option(
            "replaceWhere", f"run_id = '{run_id}'"
        ).save(path)
    else:
        stage_frame.write.format("delta").mode("overwrite").save(path)
    stage_frame.write.mode("overwrite").parquet(f"{outbound}/POC_pipeline_stage_metrics")
    counts["POC_pipeline_stage_metrics"] = stage_frame.count()
    schemas["POC_pipeline_stage_metrics"] = stage_frame.schema.jsonValue()
    all_curves = spark.read.format("delta").load(
        f"{lakehouse_root(GOLD_ID)}/Tables/POC_fail_rate_probabilities"
    )
    start_date = datetime.date(2014, 1, 1)
    end_date = datetime.date(datetime.datetime.now(datetime.timezone.utc).year + 2, 12, 31)
    calendar = spark.range((end_date - start_date).days + 1).select(
        F.date_add(F.lit(start_date), F.col("id").cast("int")).alias("date"),
    ).withColumn("calendar_year", F.year("date")).withColumn(
        "calendar_month", F.month("date"),
    ).withColumn("calendar_day", F.dayofmonth("date"))
    dimensions = {
        "POC_products": all_curves.select("product_line").distinct(),
        "POC_model_years": all_curves.select("model_year").distinct(),
        "POC_event_types": all_curves.select("event_type").distinct(),
        "POC_calendar": calendar,
    }
    for name, frame in dimensions.items():
        counts[name] = frame.count()
        schemas[name] = frame.schema.jsonValue()
        frame.write.format("delta").mode("overwrite").save(
            f"{lakehouse_root(GOLD_ID)}/Tables/{name}"
        )
        frame.write.mode("overwrite").parquet(f"{outbound}/{name}")
    usage_evidence = stage_evidence["usage"]
    pointer = spark.createDataFrame([(
        run_id, stage_evidence["silver"]["as_of_date"], None,
        usage_evidence["mae_km"], usage_evidence["baseline_mae_km"],
        usage_evidence["rmse_km"], usage_evidence["model_version"],
        stage_evidence["survival"]["alert_threshold"],
    )], (
        "run_id string, as_of_date string, publication_seconds double, "
        "usage_mae_km double, usage_baseline_mae_km double, usage_rmse_km double, "
        "usage_model_version string, alert_threshold double"
    ))
    schemas["POC_run_manifest"] = pointer.schema.jsonValue()
    manifest = {
        "run_id": run_id, "status": "published", "table_rows": counts,
        "outbound_path": outbound,
        "table_schemas": schemas,
        "statistical_status": "exploratory_not_production_validated",
    }
    for name, count in counts.items():
        observed = spark.read.parquet(f"{outbound}/{name}").count()
        if observed != count:
            raise ValueError(f"Export count mismatch for {name}: {observed} != {count}")
        published = spark.read.format("delta").load(
            f"{lakehouse_root(GOLD_ID)}/Tables/{name}"
        )
        if name not in dimensions:
            published = published.filter(F.col("run_id") == run_id)
        published = published.count()
        if published != count:
            raise ValueError(f"Gold count mismatch for {name}: {published} != {count}")
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
    pointer = pointer.withColumn(
        "publication_seconds", F.lit(manifest["elapsed_seconds"]),
    )
    notebookutils.fs.put(outbound + "/_manifest.json", json.dumps(manifest), True)
    notebookutils.fs.put(outbound + "/_COMPLETE.json", json.dumps(manifest), True)
    pointer.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        f"{lakehouse_root(GOLD_ID)}/Tables/POC_run_manifest"
    )
    notebookutils.fs.put(
        f"{lakehouse_root(GOLD_ID)}/Files/published.json", json.dumps(manifest), True,
    )
    write_evidence("publish", run_id, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


# CELL ********************

publish_result = publish_gold(run_id)

