import os
import sys

# Allow this script to use relative imports regardless of the folder name.
if not __package__:
    _here = os.path.dirname(os.path.abspath(__file__))   # .../nuT_.../inf_scripts
    _pkg_dir = os.path.dirname(_here)                     # .../nuT_...
    _pkg_name = os.path.basename(_pkg_dir)                # nuT_...
    sys.path.insert(0, os.path.dirname(_pkg_dir))         # parent of nuT_...
    __package__ = f"{_pkg_name}.inf_scripts"
    import importlib as _il
    _il.import_module(_pkg_name)
    _il.import_module(__package__)

import yaml
import argparse
import logging
import random
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..training import (
    NuTStandardModel,
    BinaryClassificationTaskLogits,
    EnergyReconstruction,
    DirectionReconstructionWithKappa,
    BinaryCrossEntropyWithLogitsLoss,
    LogCoshLoss,
    VonMisesFisher3DLoss,
)
from .. import (
    KM3NeTNodesAsTimeSeries,
    KM3NeTHitsSequence,
)
from .. import detector as _detector_module
from ..dataloader import PrometheusEventDataset, collate_fn
from ..script_supporting_functions import build_backbone

logger = logging.getLogger(__name__)


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Per-task MLflow logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_energy(results):
    """Scatter plot, residual histogram, and summary metrics for energy reco."""
    if "initial_state_energy" not in results.columns or "energy_pred" not in results.columns:
        logger.warning("Energy columns missing — skipping energy MLflow logging.")
        return

    true_e  = results["initial_state_energy"].values
    pred_e  = results["energy_pred"].values

    # Work in log10 space — consistent with how the model was trained
    log_true = np.log10(true_e)
    log_pred = np.log10(pred_e)
    residuals = log_pred - log_true

    mlflow.log_metric("energy/n_events",      int(len(results)))
    mlflow.log_metric("energy/mae_log10",     float(np.mean(np.abs(residuals))))
    mlflow.log_metric("energy/median_bias",   float(np.median(residuals)))
    mlflow.log_metric("energy/resolution_iqr",
                      float((np.percentile(residuals, 75) - np.percentile(residuals, 25)) / 2))

    # Scatter: log10(E_true) vs log10(E_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(log_true, log_pred, s=1, alpha=0.2, rasterized=True)
    lims = [min(log_true.min(), log_pred.min()) - 0.1,
            max(log_true.max(), log_pred.max()) + 0.1]
    ax.plot(lims, lims, "r--", lw=1, label="ideal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("log$_{10}$(E$_\\mathrm{true}$ / GeV)")
    ax.set_ylabel("log$_{10}$(E$_\\mathrm{pred}$ / GeV)")
    ax.set_title("Energy reconstruction")
    ax.legend(fontsize=8)
    mlflow.log_figure(fig, "energy_scatter.png")
    plt.close(fig)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=100, color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", lw=1, ls="--")
    ax.set_xlabel("log$_{10}$(E$_\\mathrm{pred}$) $-$ log$_{10}$(E$_\\mathrm{true}$)")
    ax.set_ylabel("Events")
    ax.set_title("Energy residuals")
    mlflow.log_figure(fig, "energy_residuals.png")
    plt.close(fig)

    # Resolution vs true energy
    bins = np.linspace(log_true.min(), log_true.max(), 20)
    bin_idx = np.digitize(log_true, bins)
    centers, medians, resols = [], [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() < 10:
            continue
        res = residuals[mask]
        centers.append(0.5 * (bins[b - 1] + bins[b]))
        medians.append(np.median(res))
        resols.append((np.percentile(res, 75) - np.percentile(res, 25)) / 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(centers, medians, "o-", ms=4)
    ax1.axhline(0, color="red", lw=1, ls="--")
    ax1.set_ylabel("Median bias")
    ax2.plot(centers, resols, "o-", ms=4, color="darkorange")
    ax2.set_xlabel("log$_{10}$(E$_\\mathrm{true}$ / GeV)")
    ax2.set_ylabel("Resolution (IQR/2)")
    fig.suptitle("Energy resolution vs true energy")
    mlflow.log_figure(fig, "energy_resolution_vs_energy.png")
    plt.close(fig)


def _log_classification(results):
    """ROC curve and score distributions for track/shower classification."""
    if "target_pred" not in results.columns:
        logger.warning("Classification column missing — skipping classification MLflow logging.")
        return

    # Derive true track label from truth columns if available
    if "initial_state_type" not in results.columns or "interaction" not in results.columns:
        logger.warning("Truth columns for track label missing — skipping classification MLflow logging.")
        return

    from sklearn.metrics import roc_curve, auc as sk_auc

    true_track = ((results["initial_state_type"].abs() == 14) & (results["interaction"] == 1)).astype(int).values
    logits     = results["target_pred"].values
    scores     = 1 / (1 + np.exp(-logits))   # sigmoid

    mlflow.log_metric("classification/n_events", int(len(results)))
    mlflow.log_metric("classification/frac_track", float(true_track.mean()))

    fpr, tpr, _ = roc_curve(true_track, scores)
    roc_auc     = sk_auc(fpr, tpr)
    mlflow.log_metric("classification/auc", float(roc_auc))

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — track/shower")
    ax.legend()
    mlflow.log_figure(fig, "classification_roc.png")
    plt.close(fig)

    # Score distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[true_track == 0], bins=60, alpha=0.6, label="Cascade", density=True)
    ax.hist(scores[true_track == 1], bins=60, alpha=0.6, label="Track",   density=True)
    ax.set_xlabel("Track score (sigmoid)")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions")
    ax.legend()
    mlflow.log_figure(fig, "classification_scores.png")
    plt.close(fig)


def _log_direction(results):
    """Opening-angle distribution for direction reconstruction."""
    pred_cols = ["dir_x_pred", "dir_y_pred", "dir_z_pred"]
    true_cols = ["part_dir_x", "part_dir_y", "part_dir_z"]

    if not all(c in results.columns for c in pred_cols + true_cols):
        logger.warning("Direction columns missing — skipping direction MLflow logging.")
        return

    pred_dir = results[pred_cols].values
    true_dir = results[true_cols].values

    # Normalise (model outputs unit vectors, but normalise defensively)
    pred_dir = pred_dir / np.linalg.norm(pred_dir, axis=1, keepdims=True)
    true_dir = true_dir / np.linalg.norm(true_dir, axis=1, keepdims=True)

    cos_angle = np.clip((pred_dir * true_dir).sum(axis=1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    mlflow.log_metric("direction/n_events",          int(len(results)))
    mlflow.log_metric("direction/median_angle_deg",  float(np.median(angle_deg)))
    mlflow.log_metric("direction/p68_angle_deg",
                      float(np.percentile(angle_deg, 68)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(angle_deg, bins=100, color="steelblue", edgecolor="none")
    ax.axvline(np.median(angle_deg), color="red", lw=1, ls="--",
               label=f"Median = {np.median(angle_deg):.2f}°")
    ax.set_xlabel("Opening angle (°)")
    ax.set_ylabel("Events")
    ax.set_title("Direction reconstruction")
    ax.legend()
    mlflow.log_figure(fig, "direction_opening_angle.png")
    plt.close(fig)


_TASK_LOGGERS = {
    "Energy reconstruction":        _log_energy,
    "Track shower classification":  _log_classification,
    "Direction reconstruction":     _log_direction,
}


def _log_to_mlflow(config, results, task_name, ckpt_path, mlflow_run_name, output_path):
    os.environ["MLFLOW_TRACKING_URI"]      = "http://caemlflow.in2p3.fr:5000"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "kapoor"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "enn2it8hs8r02zu57zwp52j621ci3058"

    experiment_name = config["logs"]["experiment_name"]

    # Default: append "[inference]" to the training run name so it is grouped
    # in the same experiment but never writes into the training run.
    if mlflow_run_name is None:
        mlflow_run_name = config["logs"]["run_name"] + " [inference]"

    logger.info(f"Logging to MLflow — experiment: '{experiment_name}', run: '{mlflow_run_name}'")

    with mlflow.start_run(
        experiment_name=experiment_name,
        run_name=mlflow_run_name,
    ):
        # Tag with the checkpoint that was used
        mlflow.set_tag("ckpt_path",  ckpt_path)
        mlflow.set_tag("task",       task_name)
        mlflow.set_tag("n_events",   str(len(results)))

        # Dispatch to the task-specific logging helper
        log_fn = _TASK_LOGGERS.get(task_name)
        if log_fn is not None:
            log_fn(results)
        else:
            logger.warning(f"No MLflow logger defined for task '{task_name}'.")

        # Always upload the predictions parquet as an artifact
        mlflow.log_artifact(output_path, artifact_path="predictions")

    logger.info("MLflow logging complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_file, ckpt_path, output_path, selection_path=None, n_events=None,
         mlflow_run_name=None):

    config = load_config(config_file)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    features = config["data"]["features"]
    truth = config["data"]["truth"]
    idx_dict = {feat: idx for idx, feat in enumerate(features)}
    updated_features = features.copy()
    for id_label in ["is_signal", "string_id"]:
        if id_label in features:
            updated_features.remove(id_label)

    logger.info(f"Features: {features}")
    logger.info(f"Truth columns: {truth}")

    # Build data_definition — no perturbation during inference
    _detector_cls = getattr(_detector_module, config["detector"]["name"])
    data_definition = KM3NeTHitsSequence(
        detector=_detector_cls(),
        node_definition=KM3NeTNodesAsTimeSeries(
            input_feature_names=features,
            max_hits=config["node_definition"]["max_hits"],
            trig_name=config["node_definition"]["trig_name"],
            unique=config["node_definition"]["unique"],
        ),
        input_feature_names=features,
        perturbation_dict=None,
    )

    # Build event selection
    if selection_path is not None:
        selection = pd.read_parquet(selection_path)["event_no"].tolist()
        logger.info(f"Loaded {len(selection)} events from {selection_path}")
    else:
        conn = sqlite3.connect(config["dataloader"]["path"])
        selection = pd.read_sql_query(
            f"SELECT event_no FROM {config['dataloader']['truth_table_name']}", conn
        )["event_no"].tolist()
        conn.close()
        logger.info(f"Using all {len(selection)} events from the database")

    if n_events is not None and n_events < len(selection):
        random.seed(config["training"]["seed"])
        selection = random.sample(selection, n_events)
        logger.info(f"Randomly subsampled to {len(selection)} events")

    # Inference dataloader — no shuffle, no train/val split
    dataset = PrometheusEventDataset(
        db_path=config["dataloader"]["path"],
        pulse_table=config["dataloader"]["pulsemap"],
        truth_table=config["dataloader"]["truth_table_name"],
        features=features,
        truth_columns=truth,
        data_definition=data_definition,
        selection=selection,
        labels=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Inference on {len(dataset)} events ({len(dataloader)} batches)")

    # Rebuild backbone and task — must match the training config exactly
    backbone = build_backbone(config, features, updated_features, idx_dict)

    task_name = config["task"]
    tasks = {
        "Track shower classification": BinaryClassificationTaskLogits(
            hidden_size=backbone.nb_outputs,
            target_labels="track",
            loss_function=BinaryCrossEntropyWithLogitsLoss(),
        ),
        "Energy reconstruction": EnergyReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels="initial_state_energy",
            loss_function=LogCoshLoss(),
            transform_prediction_and_target=lambda x: torch.log10(x),
        ),
        "Direction reconstruction": DirectionReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            target_labels=["part_dir_x", "part_dir_y", "part_dir_z"],
            loss_function=VonMisesFisher3DLoss(),
        ),
    }

    model = NuTStandardModel(
        backbone=backbone,
        tasks=[tasks[task_name]],
        optimizer_class=AdamW,
        optimizer_kwargs=config["optimizer"]["parameters"],
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": config["optimizer"]["scheduler_patience"]},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    # Load checkpoint weights
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    # Switch every task head to inference mode (applies transform_inference
    # instead of transform_prediction_training during forward)
    for task in model._tasks:
        task.inference()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")
    model = model.to(device)
    model.eval()

    prediction_labels = model.prediction_labels  # e.g. ["energy_pred"]

    all_event_nos = []
    all_preds = []
    all_truth = {col: [] for col in truth}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # model.forward accepts a list of batch dicts (consistent with training)
            preds = model([batch_dev])  # List[Tensor], one per task

            # Concatenate task outputs along feature dim: shape [B, n_pred_labels]
            pred_tensor = torch.cat(preds, dim=1).cpu().numpy()
            all_preds.append(pred_tensor)

            all_event_nos.append(batch["event_no"].cpu().numpy())

            for col in truth:
                if col in batch:
                    all_truth[col].append(batch[col].cpu().numpy())

    # Assemble results DataFrame
    results = pd.DataFrame({"event_no": np.concatenate(all_event_nos)})

    preds_np = np.concatenate(all_preds, axis=0)
    for i, lbl in enumerate(prediction_labels):
        results[lbl] = preds_np[:, i]

    for col in truth:
        if all_truth[col]:
            results[col] = np.concatenate(all_truth[col])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(results)} predictions → {output_path}")

    # Log metrics, figures, and the parquet artifact to MLflow
    _log_to_mlflow(config, results, task_name, ckpt_path, mlflow_run_name, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run inference with a trained nuT model checkpoint."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training config yaml (same file used during training).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the .ckpt checkpoint file to load.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output parquet file (predictions + truth).",
    )
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional parquet file with an 'event_no' column specifying which "
             "events to run inference on. If omitted, all events in the DB are used.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=None,
        help="If set, randomly subsample this many events from the selection.",
    )
    parser.add_argument(
        "--mlflow_run_name",
        default=None,
        help="MLflow run name for inference results. Defaults to the training "
             "run_name from the config with ' [inference]' appended.",
    )

    args = parser.parse_args()

    main(
        config_file=args.config,
        ckpt_path=args.ckpt,
        output_path=args.output,
        selection_path=args.selection,
        n_events=args.events,
        mlflow_run_name=args.mlflow_run_name,
    )
