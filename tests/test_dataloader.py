"""
Integration tests for nuT_no_graphnet/dataloader.py against the real
Prometheus SQLite example database and nuT_PROMETHEUS model.

Complements test_dataloader.py (which uses a mock in-memory database)
with tests that verify:

  1. PrometheusEventDataset.__getitem__ returns a plain dict,
     NOT a torch_geometric.data.Data object.
  2. collate_fn assembles a dict batch with the correct keys/shapes.
  3. make_train_validation_dataloader mirrors graphnet's
     make_train_validation_dataloader (same split logic, same batch format).
  4. nuT_PROMETHEUS.forward() accepts the dict batch produced by the
     dataloader (i.e. data["x"] / data["batch"] path is exercised).
  5. Computed labels arrive in the batch dict.

Run from the graphnet root:
    python nuT_no_graphnet/tests/test_dataloader_model_compat.py
or via pytest:
    python -m pytest nuT_no_graphnet/tests/test_dataloader_model_compat.py -v
"""

import math
import os
import sqlite3
import sys
import importlib as _il

import pandas as pd
import pytest
import torch

# Resolve package dir (parent of this tests/ directory) dynamically.
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = os.path.basename(_pkg_dir)
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))

nuT_PROMETHEUS = _il.import_module(_P).nuT_PROMETHEUS

_m_dl = _il.import_module(f"{_P}.dataloader")
PrometheusEventDataset          = _m_dl.PrometheusEventDataset
collate_fn                      = _m_dl.collate_fn
make_train_validation_dataloader = _m_dl.make_train_validation_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Database / feature config
#
# The 'total' table columns (excluding event_no):
#   sensor_id, sensor_pos_x, sensor_pos_y, sensor_pos_z, sensor_string_id, t
#
# We read five numeric columns as pulse features.  'sensor_string_id' is
# aliased as 'string_id' in IDX_DICT so the model can build a STRING mask.
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "data", "examples", "sqlite", "prometheus", "prometheus-events.db",
    )
)
PULSE_TABLE  = "total"
TRUTH_TABLE  = "mc_truth"

# Columns read from the 'total' table (must exist in the database).
FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "sensor_string_id"]

# idx_dict for nuT_PROMETHEUS:
#   - Keys align with what the model looks up inside array_to_sequence output.
#   - 'string_id' → index 4 (sensor_string_id data); used for STRING mask.
#   - 'is_signal' is absent → only 'string_id' is filtered out.
#   - Remaining processed features: sensor_pos_x, sensor_pos_y, sensor_pos_z, t
#     → n_features = 4
IDX_DICT = {
    "sensor_pos_x": 0,
    "sensor_pos_y":  1,
    "sensor_pos_z":  2,
    "t":             3,
    "string_id":     4,   # sensor_string_id column; excluded from x, used in mask
}

# Available truth columns in the 'mc_truth' table.
TRUTH_COLUMNS = ["injection_energy", "dummy_pid"]

BATCH_SIZE = 4
MAX_HITS   = 50   # small for speed

# ─────────────────────────────────────────────────────────────────────────────
# Minimal data_definition
#
# Replaces KM3NeTHitsSequence for these integration tests.
# The Prometheus SQLite example DB uses different feature names than the
# KM3NeT defaults hard-coded in KM3NeTNodesAsTimeSeries, so we provide a
# lightweight stand-in that:
#   • applies simple per-feature normalisation,
#   • sorts pulses by time,
#   • truncates to max_hits,
#   • returns the expected dict {x, n_pulses, <truth keys>}.
# ─────────────────────────────────────────────────────────────────────────────

class _PrometheusDataDef:
    """Lightweight data_definition for Prometheus example data."""

    _SCALES = {
        "sensor_pos_x":    100.0,
        "sensor_pos_y":    100.0,
        "sensor_pos_z":    100.0,
        "t":               1.05e4,
        "sensor_string_id": 1.0,
    }

    def __init__(self, max_hits: int = 300) -> None:
        self.max_hits = max_hits

    def __call__(self, input_features, input_feature_names, truth_dicts=None):
        x = input_features.clone().float()
        for i, name in enumerate(input_feature_names):
            x[:, i] = x[:, i] / self._SCALES.get(name, 1.0)
        x = x[torch.argsort(x[:, input_feature_names.index("t")])]
        x = x[: self.max_hits]
        result = {
            "x":        x,
            "n_pulses": torch.tensor(x.shape[0], dtype=torch.int32),
        }
        if truth_dicts is not None:
            for d in truth_dicts:
                for key, val in d.items():
                    result[key] = val
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared model
#
# n_features = features kept after removing 'string_id' (and 'is_signal', which
# is absent) from IDX_DICT = 4 (pos_x, pos_y, pos_z, t).
# ─────────────────────────────────────────────────────────────────────────────

_TO_REMOVE = {"is_signal", "string_id"}
_N_FEATURES = len([k for k in IDX_DICT if k not in _TO_REMOVE])  # 4

_MODEL = nuT_PROMETHEUS(
    idx_dict              = IDX_DICT,
    emb_dims              = 64,
    seq_length            = MAX_HITS,
    emb_type              = "nuT",
    n_features            = _N_FEATURES,
    abs_position_encoding = True,
    refractive_index      = 1.33,
    masks                 = ["Causality", "Euclidean", "STRING"],
    mode                  = "concat",
    pairwise_dims         = 16,
    num_heads             = 4,
    dropout_attn          = 0.0,
    hidden_dim            = 64,
    dropout_FFNN          = 0.0,
    no_hits_blocks        = 2,
    no_evt_blocks         = 1,
)
_MODEL.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data_def():
    return _PrometheusDataDef(max_hits=MAX_HITS)


@pytest.fixture(scope="module")
def n_events_in_db():
    conn = sqlite3.connect(DB_PATH)
    n = pd.read_sql_query(
        f"SELECT COUNT(*) AS n FROM {TRUTH_TABLE}", conn
    ).iloc[0]["n"]
    conn.close()
    return int(n)


@pytest.fixture(scope="module")
def dataset(data_def):
    return PrometheusEventDataset(
        db_path         = DB_PATH,
        pulse_table     = PULSE_TABLE,
        truth_table     = TRUTH_TABLE,
        features        = FEATURES,
        truth_columns   = TRUTH_COLUMNS,
        data_definition = data_def,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseSanity:
    def test_db_exists(self):
        assert os.path.exists(DB_PATH), f"Test database not found: {DB_PATH}"

    def test_events_present(self, n_events_in_db):
        assert n_events_in_db > 0, "Database has no events"
        print(f"\n  DB events: {n_events_in_db}")

    def test_required_columns_in_pulse_table(self):
        conn = sqlite3.connect(DB_PATH)
        cols = [
            row[1]
            for row in conn.execute(f"PRAGMA table_info({PULSE_TABLE})").fetchall()
        ]
        conn.close()
        for f in FEATURES:
            assert f in cols, f"Column '{f}' missing from '{PULSE_TABLE}'"

    def test_required_columns_in_truth_table(self):
        conn = sqlite3.connect(DB_PATH)
        cols = [
            row[1]
            for row in conn.execute(f"PRAGMA table_info({TRUTH_TABLE})").fetchall()
        ]
        conn.close()
        for col in TRUTH_COLUMNS:
            assert col in cols, f"Column '{col}' missing from '{TRUTH_TABLE}'"


class TestDatasetReturnsDicts:
    """PrometheusEventDataset must return plain dicts, not PyG Data objects."""

    def test_dataset_length(self, dataset, n_events_in_db):
        assert len(dataset) == n_events_in_db, (
            f"Dataset length {len(dataset)} != DB events {n_events_in_db}"
        )

    def test_item_is_dict(self, dataset):
        item = dataset[0]
        # Must be a plain Python dict — NOT a torch_geometric.data.Data object.
        assert isinstance(item, dict), (
            f"Expected dict, got {type(item).__name__}. "
            "Dataloader must produce plain dicts, not PyG Data."
        )

    def test_item_has_required_keys(self, dataset):
        item = dataset[0]
        for key in ("x", "n_pulses") + tuple(TRUTH_COLUMNS):
            assert key in item, f"Missing key '{key}' in dataset item"

    def test_x_is_2d_float32(self, dataset):
        item = dataset[0]
        assert item["x"].ndim == 2,           "x should be 2-D [n_pulses, n_features]"
        assert item["x"].shape[1] == len(FEATURES), (
            f"x should have {len(FEATURES)} feature columns, got {item['x'].shape[1]}"
        )
        assert item["x"].dtype == torch.float32

    def test_n_pulses_is_scalar(self, dataset):
        item = dataset[0]
        assert item["n_pulses"].ndim == 0, "n_pulses should be a scalar tensor"

    def test_truth_values_are_scalars(self, dataset):
        item = dataset[0]
        for col in TRUTH_COLUMNS:
            assert item[col].ndim == 0, f"Truth '{col}' should be a scalar tensor"

    def test_n_pulses_bounded_by_max_hits(self, dataset):
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            assert item["n_pulses"].item() <= MAX_HITS, (
                f"Event {i}: n_pulses={item['n_pulses'].item()} > MAX_HITS={MAX_HITS}"
            )


class TestCollateFn:
    """collate_fn must produce a dict batch with correct keys and shapes."""

    def test_batch_is_dict(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        assert isinstance(batch, dict), (
            f"Batch must be dict, got {type(batch).__name__}. "
            "Must NOT be a PyG Batch or Data object."
        )

    def test_required_batch_keys(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        for key in ("x", "batch", "n_pulses") + tuple(TRUTH_COLUMNS):
            assert key in batch, f"Missing batch key '{key}'"

    def test_x_shape(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        total_pulses = batch["n_pulses"].sum().item()
        assert batch["x"].shape == (total_pulses, len(FEATURES)), (
            f"x shape: expected ({total_pulses}, {len(FEATURES)}), "
            f"got {tuple(batch['x'].shape)}"
        )

    def test_batch_index_shape_and_range(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        total_pulses = batch["n_pulses"].sum().item()
        assert batch["batch"].shape == (total_pulses,)
        assert batch["batch"].min().item() == 0
        assert batch["batch"].max().item() == BATCH_SIZE - 1

    def test_n_pulses_shape(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        assert batch["n_pulses"].shape == (BATCH_SIZE,)

    def test_n_pulses_sums_to_total(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        assert batch["n_pulses"].sum().item() == batch["x"].shape[0]

    def test_truth_tensors_shape(self, dataset):
        items = [dataset[i] for i in range(BATCH_SIZE)]
        batch = collate_fn(items)
        for col in TRUTH_COLUMNS:
            assert batch[col].shape == (BATCH_SIZE,), (
                f"{col} shape: expected ({BATCH_SIZE},), got {tuple(batch[col].shape)}"
            )


class TestModelAcceptsDictBatch:
    """nuT_PROMETHEUS.forward() must accept the dict produced by the dataloader."""

    def test_forward_on_single_batch(self, dataset):
        batch = collate_fn([dataset[i] for i in range(BATCH_SIZE)])

        # Explicit check: input is a dict, not PyG Data
        assert isinstance(batch, dict), "batch must be dict before passing to model"

        with torch.no_grad():
            output = _MODEL(batch)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (BATCH_SIZE, _MODEL.nb_outputs), (
            f"Expected ({BATCH_SIZE}, {_MODEL.nb_outputs}), got {tuple(output.shape)}"
        )
        assert not torch.any(torch.isnan(output)), "Model output contains NaN"

    def test_model_uses_dict_keys(self, dataset):
        """Verify the model actually follows the dict['x']/dict['batch'] code path."""
        batch = collate_fn([dataset[i] for i in range(BATCH_SIZE)])
        # Ensure keys are present — if the model tried data.x it would fail here
        assert "x"     in batch
        assert "batch" in batch
        with torch.no_grad():
            _ = _MODEL(batch)   # should not raise AttributeError


class TestMakeTrainValidationDataloader:
    """make_train_validation_dataloader mirrors graphnet's factory function."""

    def _make_loaders(self, data_def, **kwargs):
        defaults = dict(
            db_path         = DB_PATH,
            pulse_table     = PULSE_TABLE,
            truth_table     = TRUTH_TABLE,
            features        = FEATURES,
            truth_columns   = TRUTH_COLUMNS,
            data_definition = data_def,
            batch_size      = BATCH_SIZE,
            test_size       = 0.33,
            seed            = 42,
            num_workers     = 0,
            persistent_workers = False,
        )
        defaults.update(kwargs)
        return make_train_validation_dataloader(**defaults)

    def test_returns_two_dataloaders(self, data_def):
        from torch.utils.data import DataLoader as TorchDataLoader
        train_dl, val_dl = self._make_loaders(data_def)
        assert isinstance(train_dl, TorchDataLoader)
        assert isinstance(val_dl,   TorchDataLoader)

    def test_split_covers_all_events(self, data_def, n_events_in_db):
        train_dl, val_dl = self._make_loaders(data_def)
        assert len(train_dl.dataset) + len(val_dl.dataset) == n_events_in_db

    def test_split_respects_test_size(self, data_def, n_events_in_db):
        train_dl, val_dl = self._make_loaders(data_def, test_size=0.33)
        n_val = len(val_dl.dataset)
        expected = math.ceil(n_events_in_db * 0.33)
        assert abs(n_val - expected) <= 1, (
            f"Val size {n_val} not close to expected {expected}"
        )

    def test_train_batch_is_dict(self, data_def):
        train_dl, _ = self._make_loaders(data_def)
        batch = next(iter(train_dl))
        assert isinstance(batch, dict), (
            f"Train batch must be dict, got {type(batch).__name__}"
        )

    def test_val_batch_is_dict(self, data_def):
        _, val_dl = self._make_loaders(data_def)
        batch = next(iter(val_dl))
        assert isinstance(batch, dict), (
            f"Val batch must be dict, got {type(batch).__name__}"
        )

    def test_model_accepts_train_and_val_batches(self, data_def):
        train_dl, val_dl = self._make_loaders(data_def)
        train_batch = next(iter(train_dl))
        val_batch   = next(iter(val_dl))
        with torch.no_grad():
            t_out = _MODEL(train_batch)
            v_out = _MODEL(val_batch)
        assert t_out.shape[0] == BATCH_SIZE
        assert v_out.shape[0] == BATCH_SIZE
        assert not torch.any(torch.isnan(t_out))
        assert not torch.any(torch.isnan(v_out))

    def test_computed_labels_in_batch(self, data_def):
        labels = {"is_nu_mu": lambda t: (t["dummy_pid"].abs() == 14).float()}
        train_dl, _ = self._make_loaders(data_def, labels=labels)
        batch = next(iter(train_dl))
        assert "is_nu_mu" in batch, "Computed label 'is_nu_mu' missing from batch"
        assert batch["is_nu_mu"].shape == (BATCH_SIZE,)
        assert torch.all((batch["is_nu_mu"] == 0) | (batch["is_nu_mu"] == 1)), (
            "is_nu_mu should be binary (0 or 1)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run directly (without pytest)
# ─────────────────────────────────────────────────────────────────────────────

def _run_all():
    dd = _PrometheusDataDef(max_hits=MAX_HITS)

    conn = sqlite3.connect(DB_PATH)
    n_ev = int(
        pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {TRUTH_TABLE}", conn).iloc[0]["n"]
    )
    conn.close()

    ds = PrometheusEventDataset(
        db_path=DB_PATH, pulse_table=PULSE_TABLE, truth_table=TRUTH_TABLE,
        features=FEATURES, truth_columns=TRUTH_COLUMNS, data_definition=dd,
    )

    sanity = TestDatabaseSanity()
    sanity.test_db_exists()
    sanity.test_events_present(n_ev)
    sanity.test_required_columns_in_pulse_table()
    sanity.test_required_columns_in_truth_table()
    print("TEST 1 (Database sanity): PASSED")

    dict_tests = TestDatasetReturnsDicts()
    dict_tests.test_dataset_length(ds, n_ev)
    dict_tests.test_item_is_dict(ds)
    dict_tests.test_item_has_required_keys(ds)
    dict_tests.test_x_is_2d_float32(ds)
    dict_tests.test_n_pulses_is_scalar(ds)
    dict_tests.test_truth_values_are_scalars(ds)
    dict_tests.test_n_pulses_bounded_by_max_hits(ds)
    print("TEST 2 (Dataset returns dicts): PASSED")

    col_tests = TestCollateFn()
    col_tests.test_batch_is_dict(ds)
    col_tests.test_required_batch_keys(ds)
    col_tests.test_x_shape(ds)
    col_tests.test_batch_index_shape_and_range(ds)
    col_tests.test_n_pulses_shape(ds)
    col_tests.test_n_pulses_sums_to_total(ds)
    col_tests.test_truth_tensors_shape(ds)
    print("TEST 3 (collate_fn dict batch): PASSED")

    model_tests = TestModelAcceptsDictBatch()
    model_tests.test_forward_on_single_batch(ds)
    model_tests.test_model_uses_dict_keys(ds)
    print("TEST 4 (model accepts dict batch): PASSED")

    loader_tests = TestMakeTrainValidationDataloader()
    loader_tests.test_returns_two_dataloaders(dd)
    loader_tests.test_split_covers_all_events(dd, n_ev)
    loader_tests.test_split_respects_test_size(dd, n_ev)
    loader_tests.test_train_batch_is_dict(dd)
    loader_tests.test_val_batch_is_dict(dd)
    loader_tests.test_model_accepts_train_and_val_batches(dd)
    loader_tests.test_computed_labels_in_batch(dd)
    print("TEST 5 (make_train_validation_dataloader): PASSED")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    _run_all()
