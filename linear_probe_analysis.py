"""
linear_probe_analysis.py
========================
Evaluate per-block representations of a trained nuT_vanilla model using linear probes.

For each encoder block (plus a "block -1" = post-embedding, pre-transformer baseline),
the CLS token is extracted via forward hooks. A linear probe (Ridge / LogisticRegression)
is then fitted on each depth's representations and evaluated against:

  - log10(initial_state_energy)   [primary regression, R² and MAE]
  - is_track label                [binary classification, AUC]
  - log10(n_pulses)               [structural regression, R²]

Results are plotted as performance vs. block depth, with baselines for:
  - a randomly initialised (untrained) model
  - a simple input-feature aggregate model (mean/sum of raw hit features per event)

Usage
-----
python linear_probe_analysis.py \\
    --config  configs/pone-pro-energy-config.yaml \\
    --ckpt    /path/to/best.ckpt \\
    --output  linear_probe_results.png \\
    [--events 10000]            # number of events to use (default: all)
    [--batch_size 256]
    [--no_random_baseline]      # skip random-init baseline (saves time)
"""

import os
import sys
import argparse
import logging
import random
import sqlite3

# ---------------------------------------------------------------------------
# sys.path bootstrap — same pattern as inf_scripts/pone-pro-infer.py
# ---------------------------------------------------------------------------
if not __package__:
    _here = os.path.dirname(os.path.abspath(__file__))   # .../nuT_...
    _pkg_name = os.path.basename(_here)
    sys.path.insert(0, os.path.dirname(_here))            # parent of nuT_...
    __package__ = _pkg_name
    import importlib as _il
    _il.import_module(_pkg_name)

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .training import (
    NuTStandardModel,
    EnergyReconstruction,
    BinaryClassificationTaskLogits,
    LogCoshLoss,
    BinaryCrossEntropyWithLogitsLoss,
)
from . import KM3NeTNodesAsTimeSeries, KM3NeTHitsSequence
from . import detector as _detector_module
from .dataloader import PrometheusEventDataset, collate_fn
from .script_supporting_functions import build_backbone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_output_stem(config):
    """Build a unique filename stem from config: <model>_<task>_<n_train_events>ev.

    Pieces
    ------
    model  : config["backbone"]["name"]          (e.g. nuT_vanilla, nuT_PROMETHEUS)
    task   : config["task"]                      (e.g. energy, classification)
    n_ev   : row-count of config["dataloader"]["selection_train"] if the file
             exists, otherwise "allev".
    """
    model_name = config["backbone"]["name"]
    task = config.get("task", "unknown")
    task_clean = task.replace(" ", "_").replace("/", "-")

    sel_file = config["dataloader"].get("selection_train")
    n_train = None
    if sel_file and os.path.isfile(sel_file):
        n_train = len(pd.read_parquet(sel_file))

    events_str = f"{n_train}ev" if n_train is not None else "allev"
    return f"{model_name}_{task_clean}_{events_str}"


# ---------------------------------------------------------------------------
# Hook-based representation extraction
# ---------------------------------------------------------------------------

def extract_representations(backbone, dataloader, device, direction_cols=None):
    """Run dataloader through backbone and collect per-block representations.

    For nuT_vanilla: CLS token (position 0) is extracted at every block, since
    CLS is present throughout.

    For nuT / nuT_PROMETHEUS: hits_blocks operate without a CLS token, so a
    masked mean over real hit positions is used instead (padding excluded via
    backbone._last_hit_mask).  evt_blocks include the CLS token and use
    position 0 as usual.

    Parameters
    ----------
    direction_cols : list[str] or None
        Truth column names for direction reconstruction (e.g.
        ``["initial_state_azimuth", "initial_state_zenith"]``).
        Only columns actually present in the batch are collected.

    Returns
    -------
    reps : dict
        Keys: -1 (pre-first-block), 0 .. n_blocks-1 (post-block i).
        Values: np.ndarray of shape [N_events, model_dim].
    truth : dict
        Keys: 'energy', 'is_track', 'n_pulses', 'initial_state_type',
        'interaction', and any direction_cols that were found.
        Values: np.ndarray of shape [N_events].
    raw_feats : np.ndarray
        Per-event aggregate of raw hit features, shape [N_events, n_agg_features].
        Used for the input-features baseline.
    """
    direction_cols = direction_cols or []

    # Detect model architecture
    is_vanilla = hasattr(backbone, "blocks")
    if is_vanilla:
        n_blocks_actual = len(backbone.blocks)
    else:
        n_hits_blocks = len(backbone.hits_blocks)
        n_evt_blocks  = len(backbone.evt_blocks)
        n_blocks_actual = n_hits_blocks + n_evt_blocks

    reps = {k: [] for k in range(-1, n_blocks_actual)}
    truth_lists = {k: [] for k in ("energy", "is_track", "n_pulses",
                                   "initial_state_type", "interaction")}
    for col in direction_cols:
        truth_lists[col] = []
    raw_feat_lists = []

    handles = []

    # ---- helper: masked mean over hit positions ----------------------------
    def _masked_mean(x_3d):
        """[B, L, D] → [B, D], averaging only over real (non-padded) hits."""
        mask = backbone._last_hit_mask          # [B, L] bool, True = real hit
        x_masked = x_3d * mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return (x_masked.sum(dim=1) / counts).detach().cpu().float()

    if is_vanilla:
        # ---- nuT_vanilla: CLS always at position 0 -------------------------
        def pre_hook_block0(_module, args):
            # args[0]: [B, 1+L, model_dim]
            reps[-1].append(args[0][:, 0, :].detach().cpu().float())

        handles.append(backbone.blocks[0].register_forward_pre_hook(pre_hook_block0))

        def make_post_hook(block_idx):
            def hook(_module, _args, output):
                # output: [B, 1+L, model_dim]
                reps[block_idx].append(output[:, 0, :].detach().cpu().float())
            return hook

        for i, block in enumerate(backbone.blocks):
            handles.append(block.register_forward_hook(make_post_hook(i)))

    else:
        # ---- nuT / nuT_PROMETHEUS: two-phase blocks ------------------------
        # hits_blocks: no CLS, sequence is [B, L, model_dim] → masked mean
        # evt_blocks:  CLS at position 0, sequence is [B, 1+L, model_dim]

        def pre_hook_hits0(_module, args):
            # args[0]: [B, L, model_dim] — no CLS yet
            reps[-1].append(_masked_mean(args[0]))

        handles.append(backbone.hits_blocks[0].register_forward_pre_hook(pre_hook_hits0))

        def make_hits_post_hook(block_idx):
            def hook(_module, _args, output):
                # output: [B, L, model_dim]
                reps[block_idx].append(_masked_mean(output))
            return hook

        for i, block in enumerate(backbone.hits_blocks):
            handles.append(block.register_forward_hook(make_hits_post_hook(i)))

        def make_evt_post_hook(block_idx):
            def hook(_module, _args, output):
                # output: [B, 1+L, model_dim] — CLS at position 0
                reps[block_idx].append(output[:, 0, :].detach().cpu().float())
            return hook

        for i, block in enumerate(backbone.evt_blocks):
            handles.append(block.register_forward_hook(make_evt_post_hook(n_hits_blocks + i)))

    backbone.eval()
    backbone.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representations", leave=False):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Run backbone (hooks fire automatically)
            backbone(batch_dev)

            # ---- truth labels -----------------------------------------------
            energy = batch["initial_state_energy"].cpu().float().numpy()
            interaction = batch["interaction"].cpu().float().numpy()
            pid = batch["initial_state_type"].cpu().float().numpy()
            n_pulses = batch["n_pulses"].cpu().float().numpy()

            # Track: muon-neutrino CC interaction (pid ±14, interaction==1)
            is_track = ((np.abs(pid) == 14) & (interaction == 1)).astype(np.float32)

            truth_lists["energy"].append(energy)
            truth_lists["is_track"].append(is_track)
            truth_lists["n_pulses"].append(n_pulses)
            truth_lists["initial_state_type"].append(pid)
            truth_lists["interaction"].append(interaction)

            for col in direction_cols:
                if col in batch:
                    truth_lists[col].append(
                        batch[col].cpu().float().numpy()
                    )

            # ---- raw per-event feature aggregates ---------------------------
            # Aggregate hit features (mean and sum) per event using batch index.
            x_hits = batch["x"].cpu().float().numpy()       # [N_total, n_feats]
            batch_idx = batch["batch"].cpu().long().numpy()  # [N_total]
            B = int(batch_idx.max()) + 1
            n_feats = x_hits.shape[1]

            agg_mean = np.zeros((B, n_feats), dtype=np.float32)
            agg_sum = np.zeros((B, n_feats), dtype=np.float32)
            counts = np.zeros(B, dtype=np.float32)

            np.add.at(agg_sum, batch_idx, x_hits)
            np.add.at(counts, batch_idx, 1.0)
            counts_safe = np.maximum(counts, 1.0)[:, None]
            agg_mean = agg_sum / counts_safe

            # Concatenate mean and sum features
            raw_event = np.concatenate([agg_mean, agg_sum], axis=1)  # [B, 2*n_feats]
            raw_feat_lists.append(raw_event)

    # Remove hooks
    for h in handles:
        h.remove()

    # Concatenate across batches
    reps_np = {k: np.concatenate(v, axis=0) for k, v in reps.items()}
    truth_np = {
        k: np.concatenate(v, axis=0)
        for k, v in truth_lists.items()
        if len(v) > 0
    }
    raw_feats_np = np.concatenate(raw_feat_lists, axis=0)

    return reps_np, truth_np, raw_feats_np


# ---------------------------------------------------------------------------
# Linear probe fitting helpers
# ---------------------------------------------------------------------------

def probe_regression(X, y, test_frac=0.2, alpha=1.0, seed=42):
    """Fit Ridge regression, return (R², MAE) on held-out test set."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_frac))
    tr_idx, te_idx = idx[:split], idx[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = Ridge(alpha=alpha)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = float(np.mean(np.abs(y_pred - y_te)))
    return r2, mae


def probe_classification(X, y, test_frac=0.2, seed=42):
    """Fit logistic regression, return AUC on held-out test set."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_frac))
    tr_idx, te_idx = idx[:split], idx[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    y_tr, y_te = y[tr_idx].astype(int), y[te_idx].astype(int)

    # Guard: skip if only one class in train or test
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, scores))


def run_all_probes(reps, truth_np, raw_feats_np, direction_cols=None):
    """Run probes at all block depths and return results dict.

    Parameters
    ----------
    direction_cols : list[str] or None
        Direction truth column names (e.g. ``["initial_state_azimuth",
        "initial_state_zenith"]``).  Only columns present in ``truth_np``
        are probed; absent columns are silently skipped.
    """
    direction_cols = [c for c in (direction_cols or []) if c in truth_np]

    log_energy = np.log10(np.clip(truth_np["energy"], 1e-6, None))
    log_npulses = np.log10(np.clip(truth_np["n_pulses"].astype(float), 1.0, None))
    is_track = truth_np["is_track"]

    depths = sorted(reps.keys())  # -1, 0, 1, ..., n_blocks-1
    results = {
        "depths": depths,
        "energy_r2": [], "energy_mae": [],
        "npulses_r2": [],
        "track_auc": [],
        "direction_cols": direction_cols,
    }
    for col in direction_cols:
        results[f"{col}_r2"] = []

    for d in tqdm(depths, desc="Fitting probes"):
        X = reps[d]

        r2_e, mae_e = probe_regression(X, log_energy)
        r2_n, _ = probe_regression(X, log_npulses)
        auc_t = probe_classification(X, is_track)

        results["energy_r2"].append(r2_e)
        results["energy_mae"].append(mae_e)
        results["npulses_r2"].append(r2_n)
        results["track_auc"].append(auc_t)

        for col in direction_cols:
            r2_dir, _ = probe_regression(X, truth_np[col])
            results[f"{col}_r2"].append(r2_dir)

    # Raw-feature baseline (no transformer at all)
    r2_raw, mae_raw = probe_regression(raw_feats_np, log_energy)
    r2_raw_n, _ = probe_regression(raw_feats_np, log_npulses)
    auc_raw = probe_classification(raw_feats_np, is_track)
    results["raw_energy_r2"] = r2_raw
    results["raw_energy_mae"] = mae_raw
    results["raw_npulses_r2"] = r2_raw_n
    results["raw_track_auc"] = auc_raw

    for col in direction_cols:
        r2_dir_raw, _ = probe_regression(raw_feats_np, truth_np[col])
        results[f"raw_{col}_r2"] = r2_dir_raw

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, random_results, output_path, meta=None):
    """Generate a 3×3 summary figure.

    Row 0: Energy R²             | Energy MAE              | Track AUC
    Row 1: n_pulses R² (sanity)  | initial_state_azimuth R²| initial_state_zenith R²
    Row 2: muon_azimuth R²       | muon_zenith R²           | model metadata

    Direction panels are hidden when the corresponding column was not found in
    the truth table.  The fixed grid positions make cross-run comparisons easy
    even when some columns are absent.
    """
    meta = meta or {}
    depths = results["depths"]
    x_labels = ["emb"] + [str(d) for d in depths if d >= 0]
    x_pos = list(range(len(depths)))

    # Map each direction column to a fixed (row, col) cell so the layout is
    # stable regardless of which columns happen to be present.
    _dir_panel_map = {
        "initial_state_azimuth": (1, 1),
        "initial_state_zenith":  (1, 2),
        "muon_azimuth":          (2, 0),
        "muon_zenith":           (2, 1),
    }
    _dir_colors = {
        "initial_state_azimuth": "mediumpurple",
        "initial_state_zenith":  "indianred",
        "muon_azimuth":          "teal",
        "muon_zenith":           "goldenrod",
    }
    # Track which cells are occupied so unoccupied ones can be hidden
    _occupied = set()

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # ── suptitle with model identity ───────────────────────────────────────
    model_name = meta.get("model_name", "")
    trained_task = meta.get("trained_task", "")
    title = "Linear probe performance vs. block depth"
    if model_name:
        title += f"   |   model: {model_name}"
    if trained_task:
        title += f"   |   trained for: {trained_task}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # ── helper: shared x-axis formatting ───────────────────────────────────
    def _fmt_x(ax):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Block index  (emb = post-embedding)")
        ax.grid(True, alpha=0.3)

    # ── Panel (0,0): Energy R² ─────────────────────────────────────────────
    _occupied.add((0, 0))
    ax = axes[0, 0]
    ax.plot(x_pos, results["energy_r2"], "o-", color="steelblue",
            label="Trained model", lw=2, ms=6)
    if random_results is not None:
        ax.plot(x_pos, random_results["energy_r2"], "s--", color="grey",
                label="Random init", lw=1.5, ms=5, alpha=0.7)
    ax.axhline(results["raw_energy_r2"], color="tomato", ls=":", lw=1.5,
               label=f"Input features R²={results['raw_energy_r2']:.3f}")
    ax.set_ylabel("R²")
    ax.set_title("Energy (log₁₀ GeV) — R²")
    ax.legend(fontsize=8)
    _fmt_x(ax)

    # ── Panel (0,1): Energy MAE ────────────────────────────────────────────
    _occupied.add((0, 1))
    ax = axes[0, 1]
    ax.plot(x_pos, results["energy_mae"], "o-", color="steelblue",
            label="Trained model", lw=2, ms=6)
    if random_results is not None:
        ax.plot(x_pos, random_results["energy_mae"], "s--", color="grey",
                label="Random init", lw=1.5, ms=5, alpha=0.7)
    ax.axhline(results["raw_energy_mae"], color="tomato", ls=":", lw=1.5,
               label=f"Input features MAE={results['raw_energy_mae']:.3f}")
    ax.set_ylabel("MAE (log₁₀ units)")
    ax.set_title("Energy (log₁₀ GeV) — MAE")
    ax.legend(fontsize=8)
    ax.invert_yaxis()  # lower MAE is better
    _fmt_x(ax)

    # ── Panel (0,2): Track AUC ─────────────────────────────────────────────
    _occupied.add((0, 2))
    ax = axes[0, 2]
    ax.plot(x_pos, results["track_auc"], "o-", color="darkorange",
            label="Trained model", lw=2, ms=6)
    if random_results is not None:
        ax.plot(x_pos, random_results["track_auc"], "s--", color="grey",
                label="Random init", lw=1.5, ms=5, alpha=0.7)
    ax.axhline(results["raw_track_auc"], color="darkorange", ls=":", lw=1.5,
               label=f"Input features AUC={results['raw_track_auc']:.3f}")
    ax.set_ylabel("AUC")
    ax.set_title("Track/shower classification — AUC\n(emergent capability)")
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8)
    _fmt_x(ax)

    # ── Panel (1,0): n_pulses R²  [sanity check / debugging] ──────────────
    _occupied.add((1, 0))
    ax = axes[1, 0]
    ax.plot(x_pos, results["npulses_r2"], "^-", color="seagreen",
            label="Trained model", lw=2, ms=6)
    if random_results is not None:
        ax.plot(x_pos, random_results["npulses_r2"], "^--", color="grey",
                label="Random init", lw=1.5, ms=5, alpha=0.7)
    ax.axhline(results["raw_npulses_r2"], color="seagreen", ls=":", lw=1.5,
               label=f"Input features R²={results['raw_npulses_r2']:.3f}")
    ax.set_ylabel("R²")
    ax.set_title("log₁₀(n_pulses) — R²\n(sanity check / debugging)")
    ax.legend(fontsize=8)
    _fmt_x(ax)

    # ── Direction panels — fixed positions, hidden if column absent ─────────
    direction_cols = results.get("direction_cols", [])
    for col, (row, col_idx) in _dir_panel_map.items():
        ax = axes[row, col_idx]
        if col in direction_cols:
            _occupied.add((row, col_idx))
            key = f"{col}_r2"
            color = _dir_colors[col]
            ax.plot(x_pos, results[key], "o-", color=color,
                    label="Trained model", lw=2, ms=6)
            if random_results is not None and key in random_results:
                ax.plot(x_pos, random_results[key], "s--", color="grey",
                        label="Random init", lw=1.5, ms=5, alpha=0.7)
            raw_val = results.get(f"raw_{col}_r2", float("nan"))
            ax.axhline(raw_val, color=color, ls=":", lw=1.5,
                       label=f"Input features R²={raw_val:.3f}")
            ax.set_ylabel("R²")
            # Note on muon direction panels that they only cover track events
            extra = "\n(tracks only — emergent)" if col.startswith("muon") else "\n(emergent capability)"
            ax.set_title(f"Direction — {col} R²{extra}")
            ax.legend(fontsize=8)
            _fmt_x(ax)
        else:
            ax.set_visible(False)

    # ── Panel (2,2): model metadata text box ──────────────────────────────
    _occupied.add((2, 2))
    ax = axes[2, 2]
    ax.axis("off")

    n_params = meta.get("n_params")
    n_params_str = f"{n_params:,}" if n_params is not None else "N/A"
    n_events_str = (
        f"{meta['n_events']:,}" if meta.get("n_events") is not None else "N/A"
    )
    info_lines = [
        ("Model",         meta.get("model_name", "N/A")),
        ("Blocks",        str(meta.get("n_blocks", "N/A"))),
        ("Parameters",    n_params_str),
        ("Trained task",  meta.get("trained_task", "N/A")),
        ("Events probed", n_events_str),
        ("Checkpoint",    meta.get("ckpt", "N/A")),
    ]
    header = "Model information"
    body = "\n".join(f"  {k:<14} {v}" for k, v in info_lines)
    note = (
        "\nNote:\n"
        "  If a model was trained for task A,\n"
        "  probing it for tasks B and C is a\n"
        "  lightweight check of what the\n"
        "  representations incidentally encode.\n"
        "  Poor performance on B or C is\n"
        "  expected and fine — the model was\n"
        "  never asked to learn them.\n"
        "  Scores should always be interpreted\n"
        "  relative to the supervised task.\n"
        "  n_pulses R² is a structural sanity\n"
        "  check: a good backbone should encode\n"
        "  event size, but it is not a physics\n"
        "  target.\n"
        "  Muon direction is only meaningful\n"
        "  for CC nu_mu (track) events."
    )
    ax.text(
        0.05, 0.97, f"{header}\n{'─' * 32}\n{body}\n{note}",
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                  edgecolor="goldenrod", linewidth=1.2),
    )

    # Hide any cells that were never assigned a plot
    for r in range(3):
        for c in range(3):
            if (r, c) not in _occupied:
                axes[r, c].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved probe analysis figure → {output_path}")


# ---------------------------------------------------------------------------
# Model + dataloader construction (reused from infer script)
# ---------------------------------------------------------------------------

def build_model_and_loader(config, ckpt_path, n_events, batch_size,
                           selection_path=None, extra_truth_cols=None):
    """Construct backbone, wrap in NuTStandardModel, load checkpoint, build dataloader.

    Parameters
    ----------
    extra_truth_cols : list[str] or None
        Additional truth columns to load (e.g. direction columns).
        They are appended to the required set only if not already present.
    """
    features = config["data"]["features"]
    truth = config["data"]["truth"]
    idx_dict = {feat: idx for idx, feat in enumerate(features)}
    updated_features = [f for f in features if f not in ("is_signal", "string_id")]

    # Ensure required truth columns are loaded
    required_truth = list(truth)
    for col in ("initial_state_energy", "initial_state_type", "interaction"):
        if col not in required_truth:
            required_truth.append(col)
    for col in (extra_truth_cols or []):
        if col not in required_truth:
            required_truth.append(col)

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

    # Use the training selection file so we only touch events that training
    # already validated as clean (no NULL columns, correct schema, etc.).
    # Fall back to reading all events from the DB only if neither the CLI
    # argument nor the config key is set.
    sel_file = selection_path or config["dataloader"].get("selection_train")
    if sel_file is not None and os.path.isfile(sel_file):
        selection = pd.read_parquet(sel_file)["event_no"].tolist()
        logger.info(f"Loaded {len(selection)} events from selection file: {sel_file}")
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

    dataset = PrometheusEventDataset(
        db_path=config["dataloader"]["path"],
        pulse_table=config["dataloader"]["pulsemap"],
        truth_table=config["dataloader"]["truth_table_name"],
        features=features,
        truth_columns=required_truth,
        data_definition=data_definition,
        selection=selection,
        labels=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["dataloader"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False,
    )

    backbone = build_backbone(config, features, updated_features, idx_dict)

    model = NuTStandardModel(
        backbone=backbone,
        tasks=[
            EnergyReconstruction(
                hidden_size=backbone.nb_outputs,
                target_labels="initial_state_energy",
                loss_function=LogCoshLoss(),
                transform_prediction_and_target=lambda x: torch.log10(x),
            )
        ],
        optimizer_class=AdamW,
        optimizer_kwargs=config["optimizer"]["parameters"],
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": config["optimizer"]["scheduler_patience"]},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    )

    if ckpt_path is not None:
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    return model, loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_file, ckpt_path, output_dir, n_events=None, batch_size=256,
         include_random_baseline=True, selection_path=None):

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(config_file)

    # If a full file path was passed instead of a directory, use its parent dir
    if os.path.splitext(output_dir)[1]:
        output_dir = os.path.dirname(output_dir) or "."

    # Auto-generate unique filenames from config metadata
    stem = build_output_stem(config)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stem}.png")
    logger.info(f"Output stem: {stem}  →  {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    bb_cfg = config["backbone"]
    if "no_blocks" in bb_cfg:
        n_blocks = bb_cfg["no_blocks"]
    else:
        n_blocks = bb_cfg["no_hits_blocks"] + bb_cfg["no_evt_blocks"]
    logger.info(f"Model has {n_blocks} encoder blocks")

    # ── Detect available direction columns in the truth table ──────────────
    _candidate_dir_cols = (
        "initial_state_azimuth", "initial_state_zenith",
        "muon_azimuth", "muon_zenith",
    )
    _db_path = config["dataloader"]["path"]
    if not os.path.isfile(_db_path):
        raise FileNotFoundError(
            f"Database file not found: {_db_path!r}\n"
            "Check the 'dataloader.path' key in your config."
        )
    _conn = sqlite3.connect(_db_path)
    _cursor = _conn.execute(
        f"PRAGMA table_info({config['dataloader']['truth_table_name']})"
    )
    _db_cols = {row[1] for row in _cursor.fetchall()}
    _conn.close()
    direction_cols = [c for c in _candidate_dir_cols if c in _db_cols]
    if direction_cols:
        logger.info(f"Direction columns found in truth table: {direction_cols}")
    else:
        logger.info("No direction columns found in truth table — direction probes skipped")

    # ── Trained model ──────────────────────────────────────────────────────
    logger.info("Building trained model …")
    trained_model, loader = build_model_and_loader(
        config, ckpt_path, n_events, batch_size,
        selection_path=selection_path,
        extra_truth_cols=direction_cols,
    )
    trained_backbone = trained_model.backbone

    # ── Collect metadata for figure and CSV ───────────────────────────────
    n_params = sum(p.numel() for p in trained_model.parameters())
    meta = {
        "model_name":   config["backbone"]["name"],
        "n_blocks":     n_blocks,
        "n_params":     n_params,
        "trained_task": config.get("task", "Unknown"),
        "n_events":     len(loader.dataset),
        "ckpt":         os.path.basename(ckpt_path) if ckpt_path else "N/A",
    }
    logger.info(
        f"Model: {meta['model_name']}  |  blocks: {n_blocks}  "
        f"|  params: {n_params:,}  |  task: {meta['trained_task']}"
    )

    logger.info(f"Extracting representations from {len(loader.dataset)} events …")
    reps, truth_np, raw_feats_np = extract_representations(
        trained_backbone, loader, device,
        direction_cols=direction_cols,
    )

    logger.info("Running linear probes on trained representations …")
    trained_results = run_all_probes(
        reps, truth_np, raw_feats_np, direction_cols=direction_cols
    )

    # Print summary table to stdout
    print("\n── Linear probe results (trained model) ──────────────────────")
    print(f"{'Depth':>6}  {'Energy R²':>10}  {'Energy MAE':>11}  {'Track AUC':>10}  {'Npulse R²':>10}")
    for i, d in enumerate(trained_results["depths"]):
        label = "emb" if d == -1 else str(d)
        print(f"{label:>6}  {trained_results['energy_r2'][i]:>10.4f}  "
              f"{trained_results['energy_mae'][i]:>11.4f}  "
              f"{trained_results['track_auc'][i]:>10.4f}  "
              f"{trained_results['npulses_r2'][i]:>10.4f}")
    print(f"{'raw':>6}  {trained_results['raw_energy_r2']:>10.4f}  "
          f"{trained_results['raw_energy_mae']:>11.4f}  "
          f"{trained_results['raw_track_auc']:>10.4f}  "
          f"{trained_results['raw_npulses_r2']:>10.4f}")
    print()

    # ── Random-init baseline ──────────────────────────────────────────────
    random_results = None
    if include_random_baseline:
        logger.info("Building random-init model (same architecture, random weights) …")
        random_model, _ = build_model_and_loader(
            config, ckpt_path=None,
            n_events=n_events,
            batch_size=batch_size,
            selection_path=selection_path,
            extra_truth_cols=direction_cols,
        )
        random_backbone = random_model.backbone

        logger.info("Extracting representations from random model …")
        reps_rand, _, _ = extract_representations(
            random_backbone, loader, device,
            direction_cols=direction_cols,
        )
        logger.info("Running linear probes on random representations …")
        random_results = run_all_probes(
            reps_rand, truth_np, raw_feats_np, direction_cols=direction_cols
        )

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_results(trained_results, random_results, output_path, meta=meta)

    # ── Save numeric results to CSV ────────────────────────────────────────
    csv_path = output_path.replace(".png", ".csv")
    rows = []
    for i, d in enumerate(trained_results["depths"]):
        row = {
            # metadata columns (repeated on every row for self-contained CSV)
            "model_name":   meta["model_name"],
            "trained_task": meta["trained_task"],
            "n_blocks":     meta["n_blocks"],
            "n_params":     meta["n_params"],
            "n_events":     meta["n_events"],
            "ckpt":         meta["ckpt"],
            # probe results
            "depth": d,
            "label": "emb" if d == -1 else str(d),
            "energy_r2_trained":   trained_results["energy_r2"][i],
            "energy_mae_trained":  trained_results["energy_mae"][i],
            "track_auc_trained":   trained_results["track_auc"][i],
            "npulses_r2_trained":  trained_results["npulses_r2"][i],
        }
        for col in direction_cols:
            row[f"{col}_r2_trained"] = trained_results[f"{col}_r2"][i]

        if random_results is not None:
            row.update({
                "energy_r2_random":   random_results["energy_r2"][i],
                "energy_mae_random":  random_results["energy_mae"][i],
                "track_auc_random":   random_results["track_auc"][i],
                "npulses_r2_random":  random_results["npulses_r2"][i],
            })
            for col in direction_cols:
                row[f"{col}_r2_random"] = random_results[f"{col}_r2"][i]

        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Saved numeric results → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe analysis of nuT_vanilla block representations.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True,
                        help="Path to training config YAML.")
    parser.add_argument("--ckpt", required=True,
                        help="Path to trained .ckpt checkpoint.")
    parser.add_argument("--output_dir", default=".",
                        help="Directory where output files are written. "
                             "Filenames are auto-generated as "
                             "<model>_<task>_<n_train_events>ev.{png,csv}.")
    parser.add_argument("--events", type=int, default=None,
                        help="Number of events to use (default: all).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_random_baseline", action="store_true",
                        help="Skip the random-init baseline (faster).")
    parser.add_argument("--selection", default=None,
                        help="Path to a parquet file with an 'event_no' column "
                             "specifying which events to use. If omitted, the "
                             "selection_train path from the config is used.")
    args = parser.parse_args()

    main(
        config_file=args.config,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        n_events=args.events,
        batch_size=args.batch_size,
        include_random_baseline=not args.no_random_baseline,
        selection_path=args.selection,
    )
