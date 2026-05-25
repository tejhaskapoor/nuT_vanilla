"""
Standalone DataLoader for Prometheus SQLite databases.

Replaces graphnet's make_train_validation_dataloader without any
graphnet or torch_geometric dependency.

Batch format returned by collate_fn:
    {
        "x":        Tensor[N_total, n_features]   — all pulses concatenated
        "batch":    LongTensor[N_total]            — event index per pulse
        "n_pulses": LongTensor[B]                  — pulse count per event
        <truth_col>: Tensor[B]                     — one per truth column
        <label_key>: Tensor[B]                     — one per computed label
    }
"""

import logging
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PrometheusEventDataset(Dataset):
    """
    Reads Prometheus events from a SQLite database one event at a time.

    Each __getitem__ call:
      1. Queries the pulse table for all hits of one event.
      2. Runs data_definition (KM3NeTHitsSequence) for standardisation /
         subsampling / perturbation.
      3. Attaches truth columns and any computed label tensors.

    The truth table is loaded into memory once at construction time for
    fast per-event lookups; pulse rows are read lazily from SQLite so that
    large datasets never fully reside in RAM.

    Args:
        db_path:         Path to the SQLite database file.
        pulse_table:     Table name containing per-hit rows  (e.g. "total").
        truth_table:     Table name containing per-event truth (e.g. "mc_truth").
        features:        Ordered list of column names to read from pulse_table.
                         Must match the feature list used to build data_definition.
        truth_columns:   Column names to read from truth_table and forward to
                         each batch dict (e.g. ["initial_state_energy", "pid"]).
        data_definition: A KM3NeTHitsSequence instance (callable).  Called as
                             data_definition(
                                 input_features=Tensor[n_pulses, n_feat],
                                 input_feature_names=List[str],
                                 truth_dicts=[Dict[str, Tensor]],
                             )
                         and must return a dict with at least "x" and "n_pulses".
        selection:       List of event_no integers to include.  None = all events.
        labels:          Dict mapping a label name to a callable that takes a
                         truth dict (keys = truth_columns, values = scalar Tensors)
                         and returns a scalar Tensor.
                         Example:
                             {"track": lambda t: ((t["pid"].abs()==14) &
                                                  (t["interaction_type"]==1)).float()}
    """

    def __init__(
        self,
        db_path: str,
        pulse_table: str,
        truth_table: str,
        features: List[str],
        truth_columns: List[str],
        data_definition,
        selection: Optional[List[int]] = None,
        labels: Optional[Dict[str, Callable]] = None,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.pulse_table = pulse_table
        self.truth_table = truth_table
        self.features = features
        self.truth_columns = truth_columns
        self.data_definition = data_definition
        self.labels = labels or {}

        # Load truth table into a DataFrame for fast index-based access.
        # Connections must be opened and closed here; worker processes will
        # receive this DataFrame via pickle (no open connection is held).
        # When a selection is provided, push the filter into SQL to avoid
        # loading the entire table (which can be millions of rows) into RAM.
        all_cols = ["event_no"] + [c for c in truth_columns if c != "event_no"]
        cols_sql = ", ".join(all_cols)
        conn = sqlite3.connect(db_path)
        if selection is not None and len(selection) <= 999:
            # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999.
            # Convert to plain Python ints — numpy int64 values are not matched
            # correctly by SQLite's parameter binding.
            sel_ints = [int(e) for e in selection]
            placeholders = ",".join("?" * len(sel_ints))
            truth_df = pd.read_sql_query(
                f"SELECT {cols_sql} FROM {truth_table} "
                f"WHERE event_no IN ({placeholders})",
                conn, params=sel_ints,
            )
        else:
            truth_df = pd.read_sql_query(
                f"SELECT {cols_sql} FROM {truth_table}", conn
            )
            if selection is not None:
                truth_df = truth_df[truth_df["event_no"].isin(selection)]
        conn.close()

        if len(truth_df) == 0:
            raise RuntimeError(
                f"No events found in '{truth_table}' for the given selection. "
                f"Check that the database path, table name, and event_no values are correct."
            )

        self.truth_df = truth_df.reset_index(drop=True)
        self.event_nos: List[int] = self.truth_df["event_no"].tolist()

        logger.info(
            f"PrometheusEventDataset: {len(self.event_nos)} events "
            f"from '{db_path}' (pulse table: '{pulse_table}')"
        )

    def __len__(self) -> int:
        return len(self.event_nos)

    def __getitem__(self, idx: int) -> Dict:
        event_no = self.event_nos[idx]

        # ── 1. Read pulse rows for this event ────────────────────────────────
        # A new connection is opened per call so that DataLoader's worker
        # processes (num_workers > 0) each have their own connection.
        cols_sql = ", ".join(self.features)
        conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
        pulse_df = pd.read_sql_query(
            f"SELECT {cols_sql} FROM {self.pulse_table} "
            f"WHERE event_no = {event_no}",
            conn,
        )
        conn.close()

        pulse_tensor = torch.from_numpy(pulse_df.values.astype("float32"))
        # shape: [n_pulses_raw, n_features]

        # ── 2. Build per-event truth dict (scalar tensors) ───────────────────
        truth_row = self.truth_df.iloc[idx]
        truth_dict: Dict[str, torch.Tensor] = {
            col: torch.tensor(float(truth_row[col]), dtype=torch.float32)
            for col in self.truth_columns
        }

        # ── 3. Apply data_definition (standardisation / subsampling) ─────────
        # truth_dicts is a one-element list so KM3NeTHitsSequence can forward
        # the truth values into the result dict automatically.
        result: Dict = self.data_definition(
            input_features=pulse_tensor,
            input_feature_names=self.features,
            truth_dicts=[truth_dict],
        )
        # result now contains at least: {"x": Tensor, "n_pulses": Tensor}
        # plus any truth keys forwarded by KM3NeTHitsSequence.

        # ── 4. Computed labels ────────────────────────────────────────────────
        for label_key, label_fn in self.labels.items():
            result[label_key] = label_fn(truth_dict)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Merges a list of per-event dicts into a single batched dict.

    Input
    -----
    batch : list of B dicts, each containing:
        "x"        : Tensor[n_pulses_i, n_features]
        "n_pulses" : Tensor scalar (int)
        <other>    : Tensor scalar (truth / label values)

    Output
    ------
    dict with:
        "x"        : Tensor[N_total, n_features]   (all pulses concatenated)
        "batch"    : LongTensor[N_total]            (event index per pulse)
        "n_pulses" : LongTensor[B]                  (pulses per event)
        <other>    : Tensor[B]                      (stacked scalars)
    """
    xs = [item["x"] for item in batch]

    # Derive n_pulses from actual tensor shapes (ignores the stored scalar so
    # that any subsampling already applied by KM3NeTNodesAsTimeSeries is respected).
    n_pulses = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    
    not_ok_pulses = n_pulses == 0
    if not_ok_pulses.any():
        new_xs, new_n_pulses = [], []
        for ii in range(len(xs)):
            if n_pulses[ii] > 0:
                new_xs.append(xs[ii])
                new_n_pulses.append(n_pulses[ii])

        xs = new_xs
        n_pulses = torch.tensor(new_n_pulses, dtype=torch.long)

    x_cat = torch.cat(xs, dim=0)                                    # [N_total, D]
    batch_idx = torch.repeat_interleave(n_pulses)                                                                # [N_total]

    result: Dict = {
        "x":        x_cat,
        "batch":    batch_idx,
        "n_pulses": n_pulses,
    }

    # Stack every other key as a [B] tensor.
    other_keys = [k for k in batch[0] if k not in ("x", "n_pulses")]
    for key in other_keys:
        values = [item[key] for ii, item in enumerate(batch) if not not_ok_pulses[ii].item()]
        result[key] = torch.stack(values)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# High-level factory (like graphnet's make_train_validation_dataloader)
# ─────────────────────────────────────────────────────────────────────────────

def make_train_validation_dataloader(
    db_path: str,
    pulse_table: str,
    truth_table: str,
    features: List[str],
    truth_columns: List[str],
    data_definition,
    batch_size: int,
    selection: Optional[List[int]] = None,
    labels: Optional[Dict[str, Callable]] = None,
    test_size: float = 0.33,
    seed: int = 42,
    num_workers: int = 4,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders from a Prometheus SQLite database.

    Replaces graphnet.training.utils.make_train_validation_dataloader.

    Args:
        db_path:          Path to SQLite database.
        pulse_table:      Name of the pulse / hit table (e.g. "total").
        truth_table:      Name of the truth table       (e.g. "mc_truth").
        features:         Ordered feature column names (pulse table).
        truth_columns:    Truth column names to include in each batch dict.
        data_definition:  KM3NeTHitsSequence instance used for processing.
        batch_size:       Number of events per batch.
        selection:        List of event_no integers to use.  None = all events.
        labels:           Dict of computed labels {key: callable(truth_dict)}.
        test_size:        Fraction of selection reserved for validation.
        seed:             Random seed for the train/val split.
        num_workers:      DataLoader worker processes.
        persistent_workers: Keep workers alive between epochs (requires
                          num_workers > 0).

    Returns:
        (train_dataloader, val_dataloader)
    """
    # If no selection is provided, use all events in the truth table.
    if selection is None:
        conn = sqlite3.connect(db_path)
        all_events = pd.read_sql_query(
            f"SELECT event_no FROM {truth_table}", conn
        )["event_no"].tolist()
        conn.close()
        selection = all_events

    train_sel, val_sel = train_test_split(
        selection, test_size=test_size, random_state=seed
    )
    logger.info(
        f"Split: {len(train_sel)} train events, {len(val_sel)} val events"
    )

    shared_kwargs = dict(
        db_path=db_path,
        pulse_table=pulse_table,
        truth_table=truth_table,
        features=features,
        truth_columns=truth_columns,
        data_definition=data_definition,
        labels=labels,
    )

    train_dataset = PrometheusEventDataset(selection=train_sel, **shared_kwargs)
    val_dataset   = PrometheusEventDataset(selection=val_sel,   **shared_kwargs)

    _persistent = persistent_workers and num_workers > 0

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=_persistent,
        pin_memory=torch.cuda.is_available(),  # page-locked memory for faster GPU transfers
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
