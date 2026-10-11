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

"""Verify MLflow artifacts in Fabric and seed a lightweight demo experiment.

Copy tracking evidence only, not training or publication. Historical runs and
Gold stay unchanged. All artifact reads and temporary files remain in Fabric.
"""

import hashlib
import json
import re
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def validate_mlflow(pipeline_run_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", pipeline_run_id):
        raise ValueError("Invalid pipeline run identifier.")
    root = Path("/lakehouse/default/Files/runs")
    source_pipeline = pipeline_run_id
    evidence_directory = root / source_pipeline
    survival_evidence = json.loads(
        (evidence_directory / "survival.json").read_text("utf-8"),
    )
    usage_evidence = json.loads(
        (evidence_directory / "usage.json").read_text("utf-8"),
    )
    if any(
        evidence.get("run_id") != source_pipeline
        for evidence in (survival_evidence, usage_evidence)
    ):
        raise ValueError("Stage evidence is not bound to the requested pipeline.")
    source_id = survival_evidence["mlflow_run_id"]
    usage_id = usage_evidence["mlflow_run_id"]
    client = MlflowClient()
    source = client.get_run(source_id)
    expected_metrics = {
        "candidate_partitions", "fitted_partitions", "curve_rows", "alerts",
    }
    if source.info.status != "FINISHED":
        raise ValueError("The source tracking run is not finished.")
    if set(source.data.metrics) != expected_metrics:
        raise ValueError("Unexpected survival metrics.")
    if (
        "run_id" in source.data.params
        or source.data.params.get("pipeline_run_id") != source_pipeline
    ):
        raise ValueError("Source tracking identifiers are not separated.")
    mlflow.set_experiment("POC_EXP_PPQ_SURVIVAL_DEMO")
    experiment = mlflow.get_experiment_by_name("POC_EXP_PPQ_SURVIVAL_DEMO")
    if experiment is None:
        raise RuntimeError("Demo experiment was not created.")
    copies = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.source_mlflow_run_id = '{source_id}'",
        max_results=2,
    )
    if len(copies) > 1:
        raise ValueError("Multiple demo copies exist for the same source run.")
    with tempfile.TemporaryDirectory() as directory:
        original = Path(client.download_artifacts(
            source_id, "partition_metrics.csv", directory,
        ))
        frame = pd.read_csv(original)
        if len(frame) != survival_evidence["candidate_partitions"]:
            raise ValueError("Partition CSV row count differs from the validated run.")
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if source.info.experiment_id == experiment.experiment_id:
            target_id = source_id
        elif copies:
            target_id = copies[0].info.run_id
        else:
            with mlflow.start_run(
                run_name=f"POC_survival_demo_copy_{source_pipeline}",
                tags={
                    "source_mlflow_run_id": source_id,
                    "evidence_kind": "tracking_copy_not_retraining",
                },
            ) as target:
                mlflow.log_params(source.data.params)
                mlflow.log_metrics(source.data.metrics)
                mlflow.log_artifact(str(original))
                target_id = target.info.run_id
        target = client.get_run(target_id)
        if target.info.status != "FINISHED":
            raise ValueError("Demo tracking run is not finished.")
        if target.data.metrics != source.data.metrics:
            raise ValueError("Demo tracking metrics differ from the source.")
        if target.data.params != source.data.params:
            raise ValueError("Demo tracking parameters differ from the source.")
        target_directory = Path(directory) / "demo"
        target_directory.mkdir()
        copied = Path(client.download_artifacts(
            target_id, "partition_metrics.csv", str(target_directory),
        ))
        if hashlib.sha256(copied.read_bytes()).hexdigest() != digest:
            raise ValueError("Demo artifact differs from the source.")
        usage = client.get_run(usage_id)
        if (
            usage.info.status != "FINISHED"
            or "run_id" in usage.data.params
            or usage.data.params.get("pipeline_run_id") != source_pipeline
        ):
            raise ValueError("Usage tracking identifiers or status are invalid.")
        model_artifacts = client.list_artifacts(usage_id, "model")
        if not any(item.path == "model/MLmodel" for item in model_artifacts):
            raise ValueError("Usage model metadata is missing.")
        metadata_directory = Path(directory) / "usage"
        metadata_directory.mkdir()
        metadata = Path(client.download_artifacts(
            usage_id, "model/MLmodel", str(metadata_directory),
        ))
        if metadata.stat().st_size == 0:
            raise ValueError("Usage model metadata is empty.")
    evidence = {
        "status": "passed", "source_pipeline_run_id": source_pipeline,
        "source_survival_mlflow_run_id": source_id,
        "demo_experiment_name": experiment.name,
        "demo_experiment_id": experiment.experiment_id,
        "demo_mlflow_run_id": target_id,
        "demo_run_kind": (
            "original_training_run" if target_id == source_id
            else "tracking_copy_not_retraining"
        ),
        "aggregate_metrics": target.data.metrics,
        "partition_csv_rows": len(frame), "partition_csv_sha256": digest,
        "source_and_demo_artifacts_readable_and_identical": True,
        "usage_mlflow_run_id": usage_id,
        "usage_model_metadata_readable": True,
        "gold_republished": False, "source_runs_modified": False,
    }
    target_path = root / source_pipeline / "mlflow_validation.json"
    if not target_path.parent.is_dir():
        raise RuntimeError("Attach the POC Silver lakehouse before validation.")
    target_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return evidence


mlflow_validation = validate_mlflow(run_id)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
