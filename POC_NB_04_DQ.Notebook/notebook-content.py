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

"""Persist checks even on failure, then block publication on failed checks."""

from pyspark.sql import Window


def read_run_evidence(name: str, run_id: str) -> dict:
    path = f"{lakehouse_root(SILVER_ID)}/Files/runs/{run_id}/{name}.json"
    return json.loads(notebookutils.fs.head(path, 65536))


def run_dq(run_id: str) -> dict:
    started = time.monotonic()
    spark = active_spark()
    silver = read_run_evidence("silver", run_id)
    survival = read_run_evidence("survival", run_id)
    usage = read_run_evidence("usage", run_id)
    checks = []

    def check(name, passed, actual, threshold, severity="blocking"):
        checks.append((
            run_id, name, "passed" if passed else "failed",
            severity, float(actual), float(threshold),
        ))

    check("missing_population_attributes", silver["invalid_population_rows"] == 0,
          silver["invalid_population_rows"], 0)
    retained = silver["retained_survival_vehicles"] / silver["vehicle_rows"]
    check("cohort_population_retained", retained >= 0.5, retained, 0.5)
    minimum_retention = silver["minimum_cohort_retention"]
    check("individual_cohort_retention", minimum_retention >= 0.5,
          minimum_retention, 0.5)
    check("source_vehicle_volume", 100000 <= silver["vehicle_rows"] <= 200000,
          silver["vehicle_rows"], 100000)
    check("survival_models_available", survival["fitted_partitions"] > 0,
          survival["fitted_partitions"], 1)
    ratio = usage["mae_km"] / max(usage["baseline_mae_km"], 1.0)
    check("usage_holdout_vs_baseline", ratio <= 1.2, ratio, 1.2)
    curves = read_candidate(spark, "POC_fail_rate_probabilities", run_id)
    invalid_probability = curves.filter(
        (F.col("model_status") == "supported")
        & (F.col("failure_probability").isNull()
           | (F.col("failure_probability") < 0)
           | (F.col("failure_probability") > 1))
    ).count()
    check("probability_range", invalid_probability == 0, invalid_probability, 0)
    ordered = Window.partitionBy(
        "product_line", "model_year", "event_type", "axis"
    ).orderBy("horizon")
    invalid_direction = curves.filter(F.col("model_status") == "supported").withColumn(
        "previous_risk", F.lag("failure_probability").over(ordered)
    ).filter(F.col("failure_probability") < F.col("previous_risk") - 1e-10).count()
    check("failure_curve_monotonicity", invalid_direction == 0,
          invalid_direction, 0)
    check("invalid_claim_chronology", silver["invalid_claim_rows"] == 0,
          silver["invalid_claim_rows"], 0, severity="warning")
    check("invalid_fleet_followup", silver["excluded_fleet_rows"] == 0,
          silver["excluded_fleet_rows"], 0, severity="warning")
    baseline_path = f"{lakehouse_root(GOLD_ID)}/Files/published.json"
    if notebookutils.fs.exists(baseline_path):
        baseline = json.loads(notebookutils.fs.head(baseline_path, 65536))
        old_usage = read_run_evidence("usage", baseline["run_id"])
        scale = max(old_usage["prediction_std"], 1.0)
        shift = abs(usage["prediction_mean"] - old_usage["prediction_mean"]) / scale
        check("prediction_mean_shift_standardized", shift <= 0.5, shift, 0.5)
        volume_ratio = usage["prediction_rows"] / max(old_usage["prediction_rows"], 1)
        check("prediction_volume_ratio", 0.5 <= volume_ratio <= 1.5,
              volume_ratio, 0.5)
    else:
        checks.append((
            run_id, "prediction_drift", "baseline_unavailable",
            "information", None, None,
        ))
    frame = spark.createDataFrame(
        checks,
        "run_id string, check_name string, status string, severity string, "
        "actual double, threshold double",
    ).withColumn("checked_at", F.current_timestamp()).withColumn(
        "checked_date", F.to_date("checked_at"),
    )
    path = f"{lakehouse_root(GOLD_ID)}/Tables/POC_dq_results"
    if notebookutils.fs.exists(path + "/_delta_log"):
        frame.write.format("delta").mode("overwrite").option("mergeSchema", "true").option(
            "replaceWhere", f"run_id = '{require_run_id(run_id)}'"
        ).save(path)
    else:
        frame.write.format("delta").mode("overwrite").save(path)
    blocked = [row[1] for row in checks if row[2] == "failed" and row[3] == "blocking"]
    result = {
        "run_id": run_id, "status": "failed" if blocked else "passed",
        "blocking_checks": blocked, "checks": len(checks),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_evidence("dq", run_id, result)
    print(json.dumps(result, indent=2))
    if blocked:
        raise ValueError("Gold publication blocked by DQ: " + ", ".join(blocked))
    return result


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dq_result = run_dq(run_id)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
