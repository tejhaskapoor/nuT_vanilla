import os
import sys

# Allow this script to use relative imports regardless of the folder name.
# When run directly (__package__ is None/empty), we resolve the package name
# from the directory and inject it so that "from . import ..." works below.
if not __package__:
    _here = os.path.dirname(os.path.abspath(__file__))   # .../nuT_.../scripts
    _pkg_dir = os.path.dirname(_here)                     # .../nuT_...
    _pkg_name = os.path.basename(_pkg_dir)                # nuT_...
    sys.path.insert(0, os.path.dirname(_pkg_dir))         # parent of nuT_...
    __package__ = f"{_pkg_name}.scripts"
    import importlib as _il
    _il.import_module(_pkg_name)
    _il.import_module(__package__)

import yaml
from glob import glob
from typing import Dict, Callable, Optional, Tuple, List, Union
import argparse
import logging

import numpy as np
import pandas as pd
import random
import mlflow
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

# --- nuT standalone imports (relative — work regardless of folder name) ---
from .. import (
    KM3NeTNodesAsTimeSeries,
    KM3NeTHitsSequence,
)
from ..training import (
    NuTStandardModel,
    BinaryClassificationTaskLogits,
    EnergyReconstruction,
    DirectionReconstructionWithKappa,
    BinaryCrossEntropyWithLogitsLoss,
    LogCoshLoss,
    VonMisesFisher3DLoss,
)
from .. import detector as _detector_module
from ..constants import PROMETHEUS_GEOMETRY_TABLE_DIR
from ..labels import Track
from ..dataloader import make_train_validation_dataloader
from ..script_supporting_functions import log_model_complexity, build_backbone, GPUUtilizationLogger, log_dataset_sizes

import math
import sqlite3

logger = logging.getLogger(__name__)



def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_file):

    config = load_config(config_file)

    # Set the seed
    pl.seed_everything(config['training']['seed'])

    # Ensure that all operations are deterministic on GPU for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Allow TF32 on Ampere+ GPUs for matmul operations: ~2× faster than full
    # float32 with negligible accuracy loss for neural network training.
    torch.set_float32_matmul_precision('high')

    # Initialise logger (mlflow or tensorboard)
    _logger_backend = config["logs"].get("logger", "mlflow")
    pl_logger = None

    if _logger_backend == "mlflow":
        os.makedirs(config["logs"]["dir"], exist_ok=True)

        os.environ["MLFLOW_TRACKING_URI"] = "http://caemlflow.in2p3.fr:5000"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "kapoor"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "enn2it8hs8r02zu57zwp52j621ci3058"

        pl_logger = MLFlowLogger(
            experiment_name=config["logs"]["experiment_name"],
            run_name=config["logs"]["run_name"],
            tracking_uri="http://caemlflow.in2p3.fr:5000",
            run_id=config["logs"]["run_id"] if config["logs"]["run_id"] is not None else None,
        )

    elif _logger_backend == "tensorboard":
        os.makedirs(config["logs"]["dir"], exist_ok=True)

        pl_logger = TensorBoardLogger(
            save_dir=config["logs"]["dir"],
            name=config["logs"]["experiment_name"],
            version=config["logs"]["run_name"],
        )

    else:
        raise ValueError(f"Unknown logger backend '{_logger_backend}'. Choose 'mlflow' or 'tensorboard'.")

    # Data definition
    features = config["data"]["features"]
    truth = config["data"]["truth"]
    idx_dict = {feat: idx for idx, feat in enumerate(features)}
    updated_features = features.copy()
    for id_label in ['is_signal', 'string_id']:
        if id_label in features:
            updated_features.remove(id_label)

    logger.info(f"Features: {features}")
    logger.info(f"Truth: {truth}")

    # Data pipeline: detector standardization → hit sequence construction
    _detector_cls = getattr(_detector_module, config["detector"]["name"])
    data_definition = KM3NeTHitsSequence(
        detector=_detector_cls(),
        node_definition=KM3NeTNodesAsTimeSeries(
            input_feature_names=features,
            max_hits=config["node_definition"]["max_hits"],
            trig_name=config["node_definition"]["trig_name"],
            unique=config["node_definition"]["unique"]
        ),
        input_feature_names=features,
        perturbation_dict={'t': 1, 'charge': 0.25},
    )

    # Do the selection
    selection = pd.read_parquet(config["dataloader"]["selection_train"])['event_no'].values
    truth_conn = sqlite3.connect(config["dataloader"]["path"])
    c_truth = truth_conn.cursor()
    c_truth.execute('SELECT event_no, initial_state_type, interaction FROM mc_truth')
    mc_truth_info = np.array([r for r in c_truth.fetchall()])

    mc_truth_pd = pd.DataFrame()
    mc_truth_pd['event_no'] = mc_truth_info[:, 0]
    mc_truth_pd['initial_state_type'] = mc_truth_info[:, 1]
    mc_truth_pd['interaction'] = mc_truth_info[:, 2]
    mc_truth_pd = mc_truth_pd[np.isin(mc_truth_pd['event_no'].values, selection)]

    tracks = mc_truth_pd[mc_truth_pd['interaction'] == 1]['event_no'].astype(int).values
    cascades = mc_truth_pd[mc_truth_pd['interaction'] == 2]['event_no'].astype(int).values

    logger.info(f'{len(tracks)} tracks in dataset')
    logger.info(f'{len(cascades)} cascades in dataset')

    if config["dataloader"]["events"]:
        logger.info(f'Randomly choosing {config["dataloader"]["events"]} of tracks/cascades')
        random.seed(config["training"]["seed"])
        tracks_selection = random.sample(list(tracks), config["dataloader"]["events"])
        cascades_selection = random.sample(list(cascades), config["dataloader"]["events"])
        logger.info(f'Example tracks {tracks_selection[:5]}...')

    # Create dataloaders (standalone — no graphnet)
    track_label = Track(
        pid_key="initial_state_type",
        interaction_key="interaction",
    )

    labels = {"track": track_label}
    # Direction task: KM3NeT data has part_dir_x/y/z in the truth table directly.
    # Prometheus data stores azimuth/zenith instead, so we derive x/y/z at batch time.
    if (config["task"] == "Direction reconstruction" and
            config["dataloader"].get("direction_source", "prometheus") == "prometheus"):
        labels["part_dir_x"] = lambda t: torch.cos(t["initial_state_azimuth"]) * torch.sin(t["initial_state_zenith"])
        labels["part_dir_y"] = lambda t: torch.sin(t["initial_state_azimuth"]) * torch.sin(t["initial_state_zenith"])
        labels["part_dir_z"] = lambda t: torch.cos(t["initial_state_zenith"])

    (training_dataloader, validation_dataloader,) = make_train_validation_dataloader(
        db_path=config["dataloader"]["path"],
        pulse_table=config["dataloader"]["pulsemap"],
        truth_table=config["dataloader"]["truth_table_name"],
        features=features,
        truth_columns=truth,
        data_definition=data_definition,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        selection=tracks_selection + cascades_selection,
        test_size=config["dataloader"]["validation_size"],
        labels=labels,
    )

    log_dataset_sizes(training_dataloader, validation_dataloader, pl_logger)

    # The model backbone — class is set by config["backbone"]["name"]
    backbone = build_backbone(config, features, updated_features, idx_dict)

    log_model_complexity(backbone, training_dataloader, pl_logger)

    # The task definition (standalone — no graphnet)
    tasks = {
        'Track shower classification': BinaryClassificationTaskLogits(
            hidden_size=backbone.nb_outputs,
            target_labels='track',
            loss_function=BinaryCrossEntropyWithLogitsLoss(),
        ),

        'Energy reconstruction': EnergyReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels='initial_state_energy',
            loss_function=LogCoshLoss(),
            transform_prediction_and_target=lambda x: torch.log10(x),
        ),

        'Direction reconstruction': DirectionReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            target_labels=['part_dir_x', 'part_dir_y', 'part_dir_z'],
            loss_function=VonMisesFisher3DLoss(),
        ),
    }

    # The model (standalone — replaces graphnet's StandardModel)
    model = NuTStandardModel(
        backbone=backbone,
        tasks=[tasks[config["task"]]],
        optimizer_class=AdamW,
        optimizer_kwargs=config["optimizer"]["parameters"],
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": config["optimizer"]["scheduler_patience"],
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )

    if config["models"]["pretrain_path"] is not None:
        logger.info(f'Loading weights from pretrained model: {config["models"]["pretrain_path"]}...')
        model.load_state_dict(torch.load(config["models"]["pretrain_path"])['state_dict'])
    else:
        logger.info(f'Creating model with random weight initialization')

    # torch.compile disabled: Inductor cannot handle the dynamic indexing in the
    # pairwise attention masks (DU/DOM/PMT/Euclidean/Causality). Model runs in
    # eager mode.
    logger.info('Running model in eager mode (torch.compile disabled)')

    # Log which SDPA kernel PyTorch selected (Flash Attention preferred for speed)
    logger.info(f"Flash kernel enabled       : {torch.backends.cuda.flash_sdp_enabled()}")
    logger.info(f"Mem-efficient kernel enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    logger.info(f"Math kernel enabled        : {torch.backends.cuda.math_sdp_enabled()}")

    # Callbacks
    callback_ckpt_best = ModelCheckpoint(
        dirpath=os.path.join(
            config["models"]["to_store_path"],
            config["logs"]["experiment_name"],
            config["logs"]["run_name"]
        ),
        save_top_k=1, monitor="val_loss", mode="min",
        filename=config["logs"]["run_name"] + "_{epoch}_{val_loss:.4f}"
    )

    callbacks = [
        TQDMProgressBar(),
        callback_ckpt_best,
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
        ),
        ModelSummary(max_depth=10),
        GPUUtilizationLogger(log_every_n_steps=config["logs"]["steps"]),
    ]

    # Train using pl.Trainer directly (replaces model.fit())
    trainer = pl.Trainer(
        max_epochs=config["training"]["fit"].get("max_epochs", 100),
        accelerator=config["training"]["fit"].get("accelerator", "gpu"),
        devices=config["training"]["fit"].get("devices", [0]),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=config["logs"]["steps"],
        fast_dev_run=config["training"]["test_no_batches"] if config["training"]["test_no_batches"] is not None else False,
        profiler='simple',
    )

    trainer.fit(
        model,
        training_dataloader,
        validation_dataloader,
        ckpt_path=config["models"]["ckpt_path"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train transformer model with a configuration yaml file."
    )

    parser.add_argument(
        "--config",
        help="Path to yaml file with arguments",
        default="",
    )

    args, unknown = parser.parse_known_args()

    main(args.config)
