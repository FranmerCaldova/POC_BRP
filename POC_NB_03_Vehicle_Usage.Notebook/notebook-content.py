# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.12"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "db44b2e5-406d-4faa-a02a-02c6d6bc145d",
# META       "default_lakehouse_name": "POC_LH_BRP_SILVER",
# META       "default_lakehouse_workspace_id": "cf8e7010-b147-4129-9b1b-e9d49f595be2"
# META     }
# META   }
# META }

# CELL ********************

# MAGIC %%configure
{"numCores": 2}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# PARAMETERS CELL ********************

run_id = "manual_20261009"
as_of_date = "2026-08-28"
alert_threshold = 0.10


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

alert_threshold = float(alert_threshold)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

"""Pure Python usage training; no Spark import or session.

Read bounded run-scoped features from the attached POC Silver lakehouse.
Evaluate later observations from held-out vehicles, never random row splits.
"""

import importlib.metadata
import json
import re
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_usage(run_id: str, as_of_date: str) -> dict:
    started = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run_id.")
    root = Path("/lakehouse/default/Files/runs") / run_id
    if not root.is_dir():
        raise RuntimeError("The POC Silver lakehouse must be attached as default.")
    data = pd.read_parquet(root / "usage_features")
    if len(data) > 100000:
        raise ValueError("Usage features exceed the declared memory bound.")
    if data.empty or data["target_km"].isna().any():
        raise ValueError("Usage target is missing.")
    data["MEASURE_DATE"] = pd.to_datetime(data["MEASURE_DATE"])
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=365)
    train = data[
        (data["vehicle_fold"] != 0) & (data["MEASURE_DATE"] < cutoff)
    ]
    test = data[
        (data["vehicle_fold"] == 0) & (data["MEASURE_DATE"] >= cutoff)
    ]
    if len(train) < 100 or len(test) < 100:
        raise ValueError("Insufficient vehicle-separated temporal holdout.")
    if set(train["SERIAL_NO"]) & set(test["SERIAL_NO"]):
        raise AssertionError("Vehicle leakage detected.")
    numeric = ["age_days", "model_year"]
    categorical = ["product_line", "PLANT", "FLEET_REGION"]
    columns = numeric + categorical
    model = Pipeline([
        ("features", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("missing", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ])),
        ("regressor", RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=30,
            n_jobs=2, random_state=42,
        )),
    ])
    mlflow.set_experiment("POC_EXP_PPQ_USAGE")
    with mlflow.start_run(run_name=f"POC_usage_{run_id}") as run:
        model.fit(train[columns], train["target_km"])
        estimate = model.predict(test[columns])
        baseline = np.full(len(test), train["target_km"].median())
        metrics = {
            "mae_km": float(mean_absolute_error(test["target_km"], estimate)),
            "rmse_km": float(np.sqrt(
                mean_squared_error(test["target_km"], estimate)
            )),
            "baseline_mae_km": float(
                mean_absolute_error(test["target_km"], baseline)
            ),
            "train_rows": float(len(train)), "test_rows": float(len(test)),
        }
        mlflow.log_params({
            "pipeline_run_id": run_id, "as_of_date": as_of_date,
            "temporal_cutoff": str(cutoff.date()),
            "vehicle_holdout_fold": 0,
            "n_estimators": 80, "max_depth": 10, "min_samples_leaf": 30,
            "training_row_limit": 100000,
        })
        mlflow.log_metrics(metrics)
        sample = train[columns].head(5)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
        )
        version = mlflow.register_model(
            info.model_uri, "POC_ML_VEHICLE_USAGE",
        )
        latest = data.sort_values("MEASURE_DATE").drop_duplicates(
            "SERIAL_NO", keep="last"
        ).copy()
        latest["predicted_km"] = model.predict(latest[columns])
        if not np.isfinite(latest["predicted_km"]).all():
            raise ValueError("Model produced nonfinite predictions.")
        latest["run_id"] = run_id
        latest["model_version"] = str(version.version)
        latest["mlflow_run_id"] = run.info.run_id
        latest["model_status"] = "exploratory"
        # Spark cannot read Parquet TIMESTAMP(NANOS); fail rather than truncate.
        latest.to_parquet(
            root / "usage_predictions.parquet", index=False, engine="pyarrow",
            coerce_timestamps="us", allow_truncated_timestamps=False,
        )
        evidence = {
            "run_id": run_id, **metrics, "prediction_rows": len(latest),
            "experiment_id": run.info.experiment_id,
            "mlflow_run_id": run.info.run_id,
            "model_name": "POC_ML_VEHICLE_USAGE",
            "model_version": str(version.version),
            "prediction_mean": float(latest["predicted_km"].mean()),
            "prediction_std": float(latest["predicted_km"].std()),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("scikit-learn", "pandas", "numpy")
            },
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        evidence["versions"]["mlflow"] = mlflow.__version__
        (root / "usage.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8",
        )
    print(json.dumps(evidence, indent=2))
    return evidence


usage_result = train_usage(run_id, as_of_date)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
