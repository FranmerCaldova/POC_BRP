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

"""Fit lifelines Kaplan-Meier curves in parallel; score in code, not PREDICT."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from lifelines import KaplanMeierFitter

OUTPUT_SCHEMA = (
    "product_line string, model_year int, event_type string, axis string, "
    "horizon double, population long, events long, at_risk long, "
    "failure_probability double, ci_lower double, ci_upper double, "
    "observed_claim_fraction double, model_status string"
)


def fit_survival_partition(frame: pd.DataFrame) -> pd.DataFrame:
    axis = str(frame["axis"].iloc[0])
    horizons = (
        [0, 90, 180, 365, 730, 1095, 1460, 1825, 3650]
        if axis == "age_days"
        else [0, 1000, 5000, 10000, 20000, 40000, 80000, 120000]
    )
    population = len(frame)
    events = int(frame["event"].sum())
    if population > 250000:
        raise ValueError("Partition exceeds the explicit pandas memory bound.")
    eligible = population >= 100 and events >= 10
    model = None
    if eligible:
        model = KaplanMeierFitter()
        model.fit(frame["duration"], frame["event"], timeline=horizons)
    rows = []
    for horizon in horizons:
        at_risk = int((frame["duration"] >= horizon).sum())
        supported = (
            eligible and at_risk >= 30
            and horizon <= float(frame["duration"].max())
        )
        probability = lower = upper = None
        if supported and model is not None:
            probability = float(1.0 - model.predict(horizon))
            ci = model.confidence_interval_.loc[float(horizon)]
            lower = float(1.0 - ci.iloc[1])
            upper = float(1.0 - ci.iloc[0])
        rows.append({
            "product_line": str(frame["product_line"].iloc[0]),
            "model_year": int(frame["model_year"].iloc[0]),
            "event_type": str(frame["event_type"].iloc[0]),
            "axis": axis,
            "horizon": float(horizon),
            "population": int(population),
            "events": events,
            "at_risk": at_risk,
            "failure_probability": probability,
            "ci_lower": lower,
            "ci_upper": upper,
            "observed_claim_fraction": float(
                ((frame["event"] == 1) & (frame["duration"] <= horizon)).sum()
                / population
            ),
            "model_status": (
                "supported" if supported
                else "unsupported_horizon" if eligible
                else "insufficient_support"
            ),
        })
    return pd.DataFrame(rows)


def run_survival(run_id: str, alert_threshold: float) -> dict:
    started = time.monotonic()
    if not 0 < alert_threshold < 1:
        raise ValueError("alert_threshold must be between zero and one.")
    spark = active_spark()
    observations = read_candidate(
        spark, "POC_survival_analysis_observations", run_id,
    )
    curves = observations.groupBy(
        "product_line", "model_year", "event_type", "axis"
    ).applyInPandas(fit_survival_partition, OUTPUT_SCHEMA).cache()
    curve_rows = write_candidate(curves, "POC_fail_rate_probabilities", run_id)
    actuals = curves.select(
        "product_line", "model_year", "event_type", "axis", "horizon",
        "population", "events", "observed_claim_fraction",
    )
    actual_rows = write_candidate(actuals, "POC_claim_rate_actuals", run_id)
    alerts = curves.filter(
        (F.col("axis") == "age_days") & (F.col("horizon") == 365)
        & (F.col("model_status") == "supported")
        & (F.col("failure_probability") > alert_threshold)
    ).withColumn("alert_threshold", F.lit(alert_threshold)).withColumn(
        "alert_rule", F.lit("Exploratory 365-day risk; not a recall decision")
    )
    alert_rows = write_candidate(alerts, "POC_fail_rate_alerts", run_id)
    partition_metrics = curves.filter(F.col("horizon") == 0).select(
        "product_line", "model_year", "event_type", "axis",
        "population", "events", "model_status",
    )
    metric_rows = write_candidate(
        partition_metrics, "POC_partition_metrics", run_id,
    )
    fitted = partition_metrics.filter(
        F.col("model_status") != "insufficient_support"
    ).count()
    if fitted == 0:
        raise ValueError("No survival partition has sufficient support.")
    mlflow.set_experiment("POC_EXP_PPQ_SURVIVAL_DEMO")
    with mlflow.start_run(run_name=f"POC_survival_{run_id}") as parent:
        mlflow.log_params({
            "pipeline_run_id": run_id, "method": "KaplanMeierFitter",
            "event_type_mapping": "PART_FAMILY plus ALL_FAILURES",
            "censoring": "administrative snapshot cutoff",
            "minimum_population": 100, "minimum_events": 10,
            "minimum_at_risk": 30, "alert_threshold": alert_threshold,
            "scoring": "notebook Python; lifelines is not supported by PREDICT",
        })
        mlflow.log_metrics({
            "candidate_partitions": metric_rows,
            "fitted_partitions": fitted, "curve_rows": curve_rows,
            "alerts": alert_rows,
        })
        metrics_frame = partition_metrics.toPandas()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition_metrics.csv"
            metrics_frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        parent_id = parent.info.run_id
        experiment_id = parent.info.experiment_id
    evidence = {
        "run_id": run_id, "curve_rows": curve_rows, "actual_rows": actual_rows,
        "alert_threshold": alert_threshold,
        "alert_rows": alert_rows, "candidate_partitions": metric_rows,
        "fitted_partitions": fitted, "skipped_partitions": metric_rows - fitted,
        "mlflow_run_id": parent_id, "experiment_id": experiment_id,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_evidence("survival", run_id, evidence)
    curves.unpersist()
    print(json.dumps(evidence, indent=2))
    return evidence


survival_result = run_survival(run_id, alert_threshold)

