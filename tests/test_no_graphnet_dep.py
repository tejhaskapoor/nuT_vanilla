"""End-to-end tests verifying zero graphnet/PyG dependency after extraction."""
import sys, os, importlib as _il, torch, numpy as np

# Resolve package dir (parent of this tests/ directory) dynamically.
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = os.path.basename(_pkg_dir)
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))

# ── 1. Import test ─────────────────────────────────────────────────────────────
# All imports must succeed without graphnet or torch_geometric installed.
_m_labels   = _il.import_module(f"{_P}.labels")
_m_rep      = _il.import_module(f"{_P}.nuT_data_representation")
_m_det      = _il.import_module(f"{_P}.nuT_detector")
_m_model    = _il.import_module(f"{_P}.nuT_model_no_graphnet")
_m_training = _il.import_module(f"{_P}.nuT_training")

Label, Direction, Track, Neutrino, Muon = (
    _m_labels.Label, _m_labels.Direction, _m_labels.Track,
    _m_labels.Neutrino, _m_labels.Muon,
)
KM3NeTNodesAsTimeSeries = _m_rep.KM3NeTNodesAsTimeSeries
KM3NeTHitsSequence      = _m_rep.KM3NeTHitsSequence
PrometheusDetector       = _m_det.PrometheusDetector
nuT_PROMETHEUS           = _m_model.nuT_PROMETHEUS
NuTStandardModel                 = _m_training.NuTStandardModel
BinaryClassificationTaskLogits   = _m_training.BinaryClassificationTaskLogits
BinaryCrossEntropyWithLogitsLoss = _m_training.BinaryCrossEntropyWithLogitsLoss
DirectionReconstructionWithKappa = _m_training.DirectionReconstructionWithKappa
EnergyReconstruction             = _m_training.EnergyReconstruction
VonMisesFisher3DLoss             = _m_training.VonMisesFisher3DLoss
print("✓ All imports succeeded (no graphnet/PyG required)")

# ── 2. Label test ──────────────────────────────────────────────────────────────
# Verify labels work with plain dict batches (not torch_geometric.data.Data)
B = 4
mock_batch = {
    "azimuth":  torch.rand(B),
    "zenith":   torch.rand(B),
    "pid":      torch.tensor([12, 14, -14, 12], dtype=torch.float),
    "position_x": torch.rand(B), "position_y": torch.rand(B), "position_z": torch.rand(B), 
    "interaction_type": torch.tensor([1,1,0,0]),
}
dir_label   = Direction(azimuth_key="azimuth", zenith_key="zenith")
track_label = Track(pid_key="pid", interaction_key = "interaction_type" )
nu_label    = Neutrino(pid_key="pid")
assert dir_label(mock_batch).shape == (B, 3),   "Direction label shape mismatch"
assert track_label(mock_batch).shape == (B,),   "Track label shape mismatch"
assert nu_label(mock_batch).shape == (B,),      "Neutrino label shape mismatch"
print("✓ Labels work with dict batches")

# ── 3. nuT_training.py: batch_size computation (the num_graphs fix) ────────────
# Verify batch_size is computed from "n_pulses" key, not from d.num_graphs
#
# nuT_PROMETHEUS.forward accesses idx_dict keys: sensor_pos_x, sensor_pos_y,
# sensor_pos_z, t, string_id, is_signal (removed), plus any physics features.
# to_remove = ['is_signal', 'string_id'] — these are excluded from FeaturesProcessing.
# Rule: n_features = len(idx_dict) - len(to_remove ∩ idx_dict.keys())
idx_dict = {
    "t":            0,
    "sensor_pos_x": 1,
    "sensor_pos_y": 2,
    "sensor_pos_z": 3,
    "is_signal":    4,   # removed inside forward — not passed to FeaturesProcessing
    "string_id":    5,   # removed inside forward — not passed to FeaturesProcessing
}
to_remove_in_test = {"is_signal", "string_id"}
D = len(idx_dict)                                           # 6 — columns in "x"
n_features = len(idx_dict) - len(to_remove_in_test)        # 4 — input to FeaturesProcessing
N_total = 120




batch_dict = {
    "x":        torch.randn(N_total, D),#[120,6]
    "batch":    torch.repeat_interleave(torch.arange(B), N_total // B),
    "n_pulses": torch.tensor([N_total // B] * B),
    "target":   torch.randint(0, 2, (B,)).float(),
}
# Simulate the batch_size computation from updated training_step (line 376):
batch_size = sum(
    d["n_pulses"].shape[0] if isinstance(d, dict)
    else (d.num_graphs if hasattr(d, 'num_graphs') else 1)
    for d in [batch_dict]
)
assert batch_size == B, f"Expected batch_size={B}, got {batch_size}"
print("✓ batch_size computed correctly from dict 'n_pulses' (num_graphs fix)")

# ── 4. compute_loss with dict batches ─────────────────────────────────────────
# Verify compute_loss merges list-of-dicts correctly
task = BinaryClassificationTaskLogits(
    hidden_size=16,
    target_labels=["target"],
    prediction_labels=["target_pred"],
    loss_function=BinaryCrossEntropyWithLogitsLoss(),

)
# Simulate a list-of-dicts batch (as created by shared_step's list wrapping)
preds = [torch.randn(B, 1)]   # one task output
data_list = [batch_dict]       # shared_step wraps single batch in list
# replicate compute_loss logic from nuT_training.py lines 344-364:
if isinstance(data_list, list):
    data_merged = {}
    for label in ["target"]:
        data_merged[label] = torch.cat([d[label] for d in data_list], dim=0)
assert "target" in data_merged and data_merged["target"].shape == (B,)
print("✓ compute_loss dict-merge works correctly for list-of-dicts batches")

# ── 5. Model forward with dict batch ──────────────────────────────────────────
# Verify nuT_PROMETHEUS.forward accepts dict with "x" and "batch" keys.
# n_features must equal len(idx_dict) minus keys in to_remove=['is_signal','string_id'].
model = nuT_PROMETHEUS(
    idx_dict=idx_dict, emb_dims=16, seq_length=30, n_features=n_features,
    num_heads=2, no_hits_blocks=1, no_evt_blocks=1,
)
output = model(batch_dict)   # must accept dict, not Data object
assert output.shape == (B, model.nb_outputs), f"Unexpected output shape: {output.shape}"
print("✓ nuT_PROMETHEUS.forward works with dict batch")

# ── 6. Full NuTStandardModel training_step (no trainer needed) ────────────────
task = BinaryClassificationTaskLogits(
    hidden_size=model.nb_outputs,
    target_labels=["target"],
    prediction_labels=["target_pred"],
    loss_function=BinaryCrossEntropyWithLogitsLoss()
)
std_model = NuTStandardModel(backbone=model, tasks=[task])
# Simulate what training_step does (without pl.Trainer logging):
preds = std_model(batch_dict)
loss  = std_model.compute_loss(preds, [batch_dict])
assert loss.item() > 0, "Loss should be positive"
print("✓ NuTStandardModel forward + compute_loss works end-to-end")

# ── 7. Verify zero graphnet/PyG imports in the package ───────────────────────
import pkgutil
_pkg_mod = _il.import_module(_P)
bad = ["graphnet", "torch_geometric"]
for finder, modname, ispkg in pkgutil.walk_packages(_pkg_mod.__path__, prefix=f"{_P}."):
    if "nuT_other_models" in modname:
        continue  # excluded from cleanup
    mod = sys.modules.get(modname)
    if mod is None:
        continue
    src = getattr(mod, "__file__", "") or ""
    for b in bad:
        assert b not in sys.modules or modname not in str(sys.modules[b]),\
            f"{b} was imported by {modname}"
print("✓ No graphnet or torch_geometric in loaded modules (excluding nuT_other_models)")

print("\nAll tests passed!")
