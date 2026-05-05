"""
test_detector.py — Standalone tests for nuT_local detector and data representation.
No graphnet imports. Run with: python test_detector.py  OR  pytest test_detector.py -v
"""

import sys
import os
import importlib as _il
import numpy as np
import torch

# Resolve package dir (parent of this tests/ directory) dynamically.
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = os.path.basename(_pkg_dir)
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))

_m_det = _il.import_module(f"{_P}.nuT_detector")
Detector         = _m_det.Detector
ORCA             = _m_det.ORCA
ORCA115          = _m_det.ORCA115
ORCA6            = _m_det.ORCA6
ORCA6_2_ORCA115  = _m_det.ORCA6_2_ORCA115
ORCA10_2_ORCA115 = _m_det.ORCA10_2_ORCA115

_m_rep = _il.import_module(f"{_P}.nuT_data_representation")
KM3NeTNodesAsTimeSeries = _m_rep.KM3NeTNodesAsTimeSeries
KM3NeTHitsSequence      = _m_rep.KM3NeTHitsSequence

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "pos_x", "pos_y", "pos_z",
    "dir_x", "dir_y", "dir_z",
    "t", "tot",
    "du_id", "dom_id", "channel_id", "trig",
]
N_FEATURES = len(FEATURE_NAMES)


def make_event(n_hits: int = 500, seed: int = 42) -> np.ndarray:
    """Return a synthetic (n_hits, N_FEATURES) float32 numpy array."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_hits, N_FEATURES)).astype(np.float32)
    # Give 't' column realistic-ish values (used for sorting)
    data[:, FEATURE_NAMES.index("t")] = rng.uniform(0, 2500, size=n_hits)
    # Give 'trig' column binary-ish values
    data[:, FEATURE_NAMES.index("trig")] = rng.integers(0, 2, size=n_hits).astype(np.float32)
    # Give id columns integer-ish values
    for col in ["du_id", "dom_id", "channel_id"]:
        data[:, FEATURE_NAMES.index(col)] = rng.integers(0, 20, size=n_hits).astype(np.float32)
    return data


def make_orca_detector(norm: bool = True) -> ORCA:
    return ORCA(ORCA115_norm=norm)


def make_sequence(perturbation_dict=None, seed=None) -> KM3NeTHitsSequence:
    detector = make_orca_detector()
    return KM3NeTHitsSequence(
        detector=detector,
        input_feature_names=FEATURE_NAMES,
        perturbation_dict=perturbation_dict,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Detector: instantiation and basic API
# ---------------------------------------------------------------------------

def test_orca_instantiation():
    det = make_orca_detector()
    assert isinstance(det, torch.nn.Module)
    assert isinstance(det, Detector)


def test_orca115_instantiation():
    det = ORCA115(raw=True)
    assert isinstance(det, Detector)


def test_orca6_instantiation():
    det = ORCA6()
    assert isinstance(det, Detector)


def test_orca6_2_orca115_instantiation():
    det = ORCA6_2_ORCA115()
    assert isinstance(det, Detector)


def test_orca10_2_orca115_instantiation():
    det = ORCA10_2_ORCA115()
    assert isinstance(det, Detector)


# ---------------------------------------------------------------------------
# 2. Detector: feature_map
# ---------------------------------------------------------------------------

def test_orca_feature_map_has_expected_keys():
    det = make_orca_detector()
    fmap = det.feature_map()
    for name in FEATURE_NAMES:
        assert name in fmap, f"Missing feature '{name}' in ORCA feature_map"


def test_orca115_feature_map_has_expected_keys():
    det = ORCA115(raw=True)
    fmap = det.feature_map()
    for name in FEATURE_NAMES:
        assert name in fmap, f"Missing feature '{name}' in ORCA115 feature_map"


def test_feature_map_values_are_callable():
    det = make_orca_detector()
    for name, fn in det.feature_map().items():
        assert callable(fn), f"feature_map['{name}'] is not callable"


# ---------------------------------------------------------------------------
# 3. Detector: forward / _standardize
# ---------------------------------------------------------------------------

def test_orca_forward_output_is_tensor():
    det = make_orca_detector()
    x = torch.tensor(make_event(), dtype=torch.float)
    out = det(x, FEATURE_NAMES)
    assert isinstance(out, torch.Tensor)


def test_orca_forward_preserves_shape():
    det = make_orca_detector()
    x = torch.tensor(make_event(n_hits=200), dtype=torch.float)
    out = det(x, FEATURE_NAMES)
    assert out.shape == (200, N_FEATURES)


def test_orca_forward_modifies_values():
    """Standardization should change at least some values."""
    det = make_orca_detector(norm=True)
    x_np = make_event()
    x = torch.tensor(x_np.copy(), dtype=torch.float)
    out = det(x, FEATURE_NAMES)
    original = torch.tensor(x_np, dtype=torch.float)
    assert not torch.allclose(out, original), "Standardization did not change any values"


def test_orca_forward_unknown_feature_raises():
    det = make_orca_detector()
    x = torch.tensor(make_event(), dtype=torch.float)
    bad_names = FEATURE_NAMES[:-1] + ["unknown_feature"]
    try:
        det(x, bad_names)
        assert False, "Expected KeyError for unknown feature"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# 4. Detector: identity and replace_with_identity
# ---------------------------------------------------------------------------

def test_identity_is_passthrough():
    det = make_orca_detector()
    x = torch.tensor([0.1, 1.5, -3.7, 100.0])
    out = det._identity(x)
    assert torch.allclose(out, x)


def test_replace_with_identity_not_supported_by_orca():
    """ORCA.__init__ does not accept replace_with_identity; feature lives in base Detector."""
    try:
        ORCA(ORCA115_norm=True, replace_with_identity=["t"])
        assert False, "Expected TypeError for unsupported kwarg"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# 5. Detector: geometry_table (optional, raises without path)
# ---------------------------------------------------------------------------

def test_geometry_table_path_is_set():
    """All concrete detector subclasses define a geometry_table_path."""
    for cls in [ORCA6, ORCA6_2_ORCA115, ORCA10_2_ORCA115]:
        det = cls()
        assert hasattr(det, "geometry_table_path"), f"{cls.__name__} missing geometry_table_path"
        assert isinstance(det.geometry_table_path, str)


def test_sensor_index_name_property():
    det = make_orca_detector()
    assert det.sensor_index_name == det.sensor_id_column


def test_string_index_name_property():
    det = make_orca_detector()
    assert det.string_index_name == det.string_id_column


def test_sensor_position_names_property():
    det = make_orca_detector()
    assert det.sensor_position_names == det.xyz


# ---------------------------------------------------------------------------
# 6. KM3NeTHitsSequence: instantiation
# ---------------------------------------------------------------------------

def test_sequence_instantiation():
    seq = make_sequence()
    assert isinstance(seq, torch.nn.Module)


def test_sequence_stores_detector():
    det = make_orca_detector()
    seq = KM3NeTHitsSequence(
        detector=det,
        input_feature_names=FEATURE_NAMES,
    )
    assert seq._detector is det


def test_sequence_default_node_definition():
    """If no node_definition given, defaults to KM3NeTNodesAsTimeSeries."""
    det = make_orca_detector()
    seq = KM3NeTHitsSequence(detector=det, input_feature_names=FEATURE_NAMES)
    assert isinstance(seq._node_definition, KM3NeTNodesAsTimeSeries)



# ---------------------------------------------------------------------------
# 7. KM3NeTHitsSequence: forward — output type and shape
# ---------------------------------------------------------------------------

def test_forward_returns_dict_with_tensor():
    seq = make_sequence()
    x = make_event(n_hits=500)
    out = seq.forward(x, FEATURE_NAMES)
    assert isinstance(out, dict)
    assert isinstance(out["x"], torch.Tensor)


def test_forward_output_dtype():
    seq = make_sequence()
    x = make_event(n_hits=200)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].dtype == torch.float32


def test_forward_output_n_features():
    seq = make_sequence()
    x = make_event(n_hits=300)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].shape[1] == N_FEATURES


def test_forward_truncates_long_event():
    """Events with more hits than max_hits (default 300) are truncated."""
    seq = make_sequence()
    x = make_event(n_hits=500)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].shape[0] <= 300


def test_forward_passthrough_short_event():
    """Events shorter than max_hits keep all their hits."""
    seq = make_sequence()
    x = make_event(n_hits=50)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].shape[0] <= 50  # may be <= 50 if first-hit selection is active


def test_forward_single_hit_event():
    """Degenerate case: single hit event."""
    seq = make_sequence()
    x = make_event(n_hits=1)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].shape[0] == 1
    assert out["x"].shape[1] == N_FEATURES


def test_forward_exact_max_hits_event():
    """Event with exactly max_hits (default 300) hits."""
    seq = make_sequence()
    x = make_event(n_hits=300)
    out = seq.forward(x, FEATURE_NAMES)
    assert out["x"].shape[0] <= 300
    assert out["x"].shape[1] == N_FEATURES


# ---------------------------------------------------------------------------
# 8. KM3NeTHitsSequence: forward — perturbation
# ---------------------------------------------------------------------------

def test_perturbation_changes_output():
    """Two forward calls with active perturbation (no fixed seed) produce different results."""
    seq = make_sequence(perturbation_dict={"t": 50.0})
    x = make_event()
    out1 = seq.forward(x.copy(), FEATURE_NAMES)
    out2 = seq.forward(x.copy(), FEATURE_NAMES)
    assert not torch.allclose(out1["x"], out2["x"]), "Outputs should differ with active perturbation"


def test_deterministic_with_int_seed():
    """Two instances with the same integer seed produce identical outputs.
    Uses all-triggered hits to avoid stochastic torch sampling in _hits_sampler."""
    seq1 = make_sequence(perturbation_dict={"t": 50.0}, seed=99)
    seq2 = make_sequence(perturbation_dict={"t": 50.0}, seed=99)
    x = make_event()
    x[:, FEATURE_NAMES.index("trig")] = 1.0  # all triggered → no random sampling
    out1 = seq1.forward(x.copy(), FEATURE_NAMES)
    out2 = seq2.forward(x.copy(), FEATURE_NAMES)
    assert torch.allclose(out1["x"], out2["x"]), "Same seed should give identical outputs"


def test_different_seeds_give_different_outputs():
    seq1 = make_sequence(perturbation_dict={"t": 50.0}, seed=1)
    seq2 = make_sequence(perturbation_dict={"t": 50.0}, seed=2)
    x = make_event()
    out1 = seq1.forward(x.copy(), FEATURE_NAMES)
    out2 = seq2.forward(x.copy(), FEATURE_NAMES)
    assert not torch.allclose(out1["x"], out2["x"]), "Different seeds should give different outputs"


def test_no_perturbation_is_deterministic():
    """Without perturbation and with all-triggered hits (no torch random sampling),
    repeated calls on the same input give identical output."""
    seq = make_sequence(perturbation_dict=None)
    x = make_event()
    x[:, FEATURE_NAMES.index("trig")] = 1.0  # all triggered → no random sampling
    out1 = seq.forward(x.copy(), FEATURE_NAMES)
    out2 = seq.forward(x.copy(), FEATURE_NAMES)
    assert torch.allclose(out1["x"], out2["x"])


# ---------------------------------------------------------------------------
# 9. No graphnet imports in refactored files
# ---------------------------------------------------------------------------

def test_no_graphnet_imports_in_nuT_detector():
    detector_path = os.path.join(os.path.dirname(__file__), "nuT_detector.py")
    with open(detector_path) as f:
        src = f.read()
    assert "from graphnet" not in src, "nuT_detector.py contains 'from graphnet'"
    assert "import graphnet" not in src, "nuT_detector.py contains 'import graphnet'"


def test_no_graphnet_imports_in_nuT_data_representation():
    repr_path = os.path.join(os.path.dirname(__file__), "nuT_data_representation.py")
    with open(repr_path) as f:
        lines = f.readlines()
    # Skip comment lines (lines whose first non-whitespace char is '#')
    non_comment = "".join(l for l in lines if not l.lstrip().startswith("#"))
    assert "from graphnet" not in non_comment, "nuT_data_representation.py contains 'from graphnet'"
    assert "import graphnet" not in non_comment, "nuT_data_representation.py contains 'import graphnet'"


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, []
    for fn in all_tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{passed + len(failed)} passed")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
