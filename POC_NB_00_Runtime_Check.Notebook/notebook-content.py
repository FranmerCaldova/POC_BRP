# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "environment": {
# META       "environmentId": "63afde11-ede9-a8ac-4a54-40df3049bfaa",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

"""Run in a new Fabric Spark notebook after creation has been authorized.

This script does not install packages or access Bronze. Aggregate validation
evidence is written only to the newly created POC Silver lakehouse.
Install the approved pinned environment first. Missing dependencies and failed
assertions deliberately propagate to the notebook and pipeline.
"""

import importlib.metadata
import json
import sys

import notebookutils
import pandas as pd
from lifelines import KaplanMeierFitter
from pyspark.sql import SparkSession


def fit_partition(frame: pd.DataFrame) -> pd.DataFrame:
    model = KaplanMeierFitter()
    model.fit(frame["duration"], event_observed=frame["event"])
    survival = float(model.predict(2.0))
    return pd.DataFrame(
        {
            "partition": [str(frame["partition"].iloc[0])],
            "rows": [int(len(frame))],
            "events": [int(frame["event"].sum())],
            "survival_at_2": [survival],
            "python_version": [sys.version.split()[0]],
            "lifelines_version": [importlib.metadata.version("lifelines")],
        }
    )


def main() -> None:
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Run this script in an active Fabric Spark notebook.")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            f"Expected Python 3.13 for the Runtime 2.0 test; got {sys.version}."
        )
    if not spark.version.startswith("4.1."):
        raise RuntimeError(
            f"Expected Spark 4.1 for the Runtime 2.0 test; got {spark.version}."
        )

    versions = {
        name: importlib.metadata.version(name)
        for name in ("lifelines", "pandas", "numpy", "scipy", "pyarrow")
    }
    versions["python"] = sys.version.split()[0]
    versions["spark"] = spark.version

    # At time 2, the two events leave survival 3/4 * 2/3 = 1/2.
    frame = pd.DataFrame(
        {
            "partition": ["driver"] * 4,
            "duration": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 1, 0, 0],
        }
    )
    driver = fit_partition(frame)
    if abs(float(driver["survival_at_2"].iloc[0]) - 0.5) > 1e-10:
        raise AssertionError("Driver Kaplan-Meier result differs from 0.5.")

    records = [
        (f"cohort_{group:03d}", duration, event)
        for group in range(128)
        for duration, event in zip(
            (1.0, 2.0, 3.0, 4.0), (1, 1, 0, 0), strict=True
        )
    ]
    source = spark.createDataFrame(
        records, "partition string, duration double, event int"
    ).repartition(8, "partition")
    result = source.groupBy("partition").applyInPandas(
        fit_partition,
        schema=(
            "partition string, rows long, events long, survival_at_2 double, "
            "python_version string, lifelines_version string"
        ),
    ).collect()
    if len(result) != 128 or len({row.partition for row in result}) != 128:
        raise AssertionError("Expected exactly 128 distinct synthetic groups.")
    for row in result:
        if row.rows != 4 or row.events != 2:
            raise AssertionError(f"Incorrect support for {row.partition}.")
        if abs(row.survival_at_2 - 0.5) > 1e-10:
            raise AssertionError(f"Incorrect survival for {row.partition}.")
        if row.python_version != versions["python"]:
            raise AssertionError("Driver and worker Python versions differ.")
        if row.lifelines_version != versions["lifelines"]:
            raise AssertionError("Driver and worker lifelines versions differ.")
    evidence = json.dumps(
            {
                "status": "passed",
                "versions": versions,
                "synthetic_groups_verified": len(result),
                "expected_survival_at_2": 0.5,
                "bronze_accessed": False,
                "bronze_written": False,
            },
            indent=2,
    )
    destination = (
        "abfss://cf8e7010-b147-4129-9b1b-e9d49f595be2"
        "@onelake.dfs.fabric.microsoft.com/"
        "db44b2e5-406d-4faa-a02a-02c6d6bc145d/"
        "Files/validation/runtime_check.json"
    )
    notebookutils.fs.mkdirs(destination.rsplit("/", 1)[0])
    notebookutils.fs.put(destination, evidence, True)
    print(evidence)


if __name__ == "__main__":
    main()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

"""Read the complete Bronze vehicle table through Spark; print aggregates only."""

import json

import notebookutils
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("An active Fabric Spark session is required.")
    root = (
        "abfss://cf8e7010-b147-4129-9b1b-e9d49f595be2"
        "@onelake.dfs.fabric.microsoft.com/"
        "ff748fb3-b044-4729-a322-e39f97cafe1a/Tables/POC/"
    )
    vehicles = spark.read.format("delta").load(
        root + "sanitized_raw_vehicle_data"
    )
    models = spark.read.format("delta").load(root + "sanitized_raw_model_family")
    fleet = spark.read.format("delta").load(
        root + "sanitized_raw_fleet_observations"
    )
    summary = vehicles.agg(
        F.count("*").alias("rows"),
        F.countDistinct("SERIAL_NO").alias("distinct_vehicles"),
        *[
            F.sum(F.col(name).isNull().cast("long")).alias("null_" + name)
            for name in vehicles.columns
        ],
        F.min("REGISTRATION_DATE").alias("min_registration"),
        F.max("REGISTRATION_DATE").alias("max_registration"),
        F.min("DELIVERY_DATE").alias("min_delivery"),
        F.max("DELIVERY_DATE").alias("max_delivery"),
    ).first()
    if summary is None:
        raise RuntimeError("Vehicle aggregate returned no row.")
    unmatched = vehicles.join(
        models.select("MODEL_NO").distinct(),
        vehicles.PRODUCT_CODE == models.MODEL_NO,
        "left_anti",
    ).count()
    paired = fleet.select(
        "SERIAL_NO", "MEASURE_DATE"
    ).join(
        vehicles.select("SERIAL_NO", "DELIVERY_DATE", "REGISTRATION_DATE"),
        "SERIAL_NO",
        "inner",
    )
    temporal = paired.agg(
        F.count("*").alias("joined_observations"),
        F.sum(
            (F.col("MEASURE_DATE") < F.col("DELIVERY_DATE")).cast("long")
        ).alias("before_delivery"),
        F.sum(
            (F.col("MEASURE_DATE") < F.col("REGISTRATION_DATE")).cast("long")
        ).alias("before_registration"),
    ).first()
    if temporal is None:
        raise RuntimeError("Temporal aggregate returned no row.")
    evidence = json.dumps(
            {
                "vehicle_summary": summary.asDict(),
                "schema": vehicles.schema.jsonValue(),
                "unmatched_product_codes": unmatched,
                "temporal_summary": temporal.asDict(),
                "source_writes": False,
            },
            default=str,
            indent=2,
    )
    destination = (
        "abfss://cf8e7010-b147-4129-9b1b-e9d49f595be2"
        "@onelake.dfs.fabric.microsoft.com/"
        "db44b2e5-406d-4faa-a02a-02c6d6bc145d/"
        "Files/validation/vehicle_profile.json"
    )
    notebookutils.fs.mkdirs(destination.rsplit("/", 1)[0])
    notebookutils.fs.put(destination, evidence, True)
    print(evidence)


if __name__ == "__main__":
    main()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
