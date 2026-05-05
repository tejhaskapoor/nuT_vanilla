"""Full test for Step 6: nuT_no_graphnet/nuT_training.py

Tests all standalone components:
- Loss functions (LogCoshLoss, BinaryCrossEntropyWithLogitsLoss, VonMisesFisher3DLoss)
- Task heads (EnergyReconstruction, DirectionReconstructionWithKappa, BinaryClassificationTaskLogits)
- NuTStandardModel (backbone + tasks combined)

Run from the graphnet root directory:
    python test_step6.py
"""

import torch
import torch.nn as nn
import sys
import os
import importlib as _il
from unittest.mock import Mock

# Resolve package dir (parent of this tests/ directory) dynamically.
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = os.path.basename(_pkg_dir)
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))

_m_pkg      = _il.import_module(_P)
_m_training = _il.import_module(f"{_P}.nuT_training")

nuT_PROMETHEUS                   = _m_pkg.nuT_PROMETHEUS
NuTStandardModel                 = _m_training.NuTStandardModel
LossFunction                     = _m_training.LossFunction
LogCoshLoss                      = _m_training.LogCoshLoss
BinaryCrossEntropyWithLogitsLoss = _m_training.BinaryCrossEntropyWithLogitsLoss
VonMisesFisher3DLoss             = _m_training.VonMisesFisher3DLoss
Task                             = _m_training.Task
EnergyReconstruction             = _m_training.EnergyReconstruction
DirectionReconstructionWithKappa = _m_training.DirectionReconstructionWithKappa
BinaryClassificationTask         = _m_training.BinaryClassificationTask
BinaryClassificationTaskLogits   = _m_training.BinaryClassificationTaskLogits
eps_like                         = _m_training.eps_like

# ============================================================
# 1. Test imports
# ============================================================
print("=" * 60)
print("TEST 1: Imports")
print("=" * 60)

print("All imports OK")

# ============================================================
# 2. Test loss functions
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Loss Functions")
print("=" * 60)

# LogCoshLoss
loss_fn = LogCoshLoss()
pred = torch.randn(16, 1)
target = torch.randn(16, 1)
loss = loss_fn(pred, target)
assert loss.shape == (), f"LogCoshLoss should return scalar, got {loss.shape}"
assert loss.item() >= 0, f"LogCoshLoss should be non-negative, got {loss.item()}"
print(f"  LogCoshLoss: {loss.item():.4f} OK")

# LogCoshLoss with weights
weights = torch.ones(16, 1)
loss_w = loss_fn(pred, target, weights=weights)
assert torch.allclose(loss, loss_w, atol=1e-6), "Weights=1 should give same loss"
print(f"  LogCoshLoss with weights: OK")

# BinaryCrossEntropyWithLogitsLoss
loss_fn = BinaryCrossEntropyWithLogitsLoss()
pred = torch.randn(16, 1)
target = torch.randint(0, 2, (16, 1)).float()
loss = loss_fn(pred, target)
assert loss.shape == (), f"BCEWithLogits should return scalar, got {loss.shape}"
assert loss.item() >= 0, f"BCEWithLogits should be non-negative, got {loss.item()}"
print(f"  BinaryCrossEntropyWithLogitsLoss: {loss.item():.4f} OK")

# VonMisesFisher3DLoss
loss_fn = VonMisesFisher3DLoss()
# pred: [N, 4] = [dir_x, dir_y, dir_z, kappa]
direction = torch.randn(16, 3)
direction = direction / direction.norm(dim=-1, keepdim=True)  # normalize
kappa = torch.abs(torch.randn(16, 1)) + 1.0  # positive kappa
pred = torch.cat([direction, kappa], dim=-1)
target = torch.randn(16, 3)
target = target / target.norm(dim=-1, keepdim=True)  # normalize
loss = loss_fn(pred, target)
assert loss.shape == (), f"vMF loss should return scalar, got {loss.shape}"
print(f"  VonMisesFisher3DLoss: {loss.item():.4f} OK")

# Test vMF loss backward pass (gradients flow)
pred_grad = pred.clone().requires_grad_(True)
loss = loss_fn(pred_grad, target)
loss.backward()
assert pred_grad.grad is not None, "vMF loss should produce gradients"
assert not torch.any(torch.isnan(pred_grad.grad)), "vMF gradients should not be NaN"
print(f"  VonMisesFisher3DLoss backward: OK")

# ============================================================
# 3. Test task heads
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Task Heads")
print("=" * 60)

hidden_size = 256
batch_size = 8
x = torch.randn(batch_size, hidden_size)

# EnergyReconstruction
task = EnergyReconstruction(
    hidden_size=hidden_size,
    target_labels='initial_state_energy',
    loss_function=LogCoshLoss(),
    transform_prediction_and_target=lambda x: torch.log10(x),
)
out = task(x)
assert out.shape == (batch_size, 1), f"Energy output shape: expected ({batch_size}, 1), got {out.shape}"
assert torch.all(out > 0), "Energy predictions should be positive (softplus)"
print(f"  EnergyReconstruction output shape: {out.shape} OK")

# DirectionReconstructionWithKappa
task = DirectionReconstructionWithKappa(
    hidden_size=hidden_size,
    target_labels=['part_dir_x', 'part_dir_y', 'part_dir_z'],
    loss_function=VonMisesFisher3DLoss(),
)
out = task(x)
assert out.shape == (batch_size, 4), f"Direction output shape: expected ({batch_size}, 4), got {out.shape}"
# Check direction is approximately unit norm
dir_norm = torch.norm(out[:, :3], dim=1)
assert torch.allclose(dir_norm, torch.ones_like(dir_norm), atol=1e-5), \
    f"Direction should be unit norm, got norms: {dir_norm}"
assert torch.all(out[:, 3] > 0), "Kappa should be positive"
print(f"  DirectionReconstructionWithKappa output shape: {out.shape} OK")

# BinaryClassificationTaskLogits
task = BinaryClassificationTaskLogits(
    hidden_size=hidden_size,
    target_labels='track',
    loss_function=BinaryCrossEntropyWithLogitsLoss(),
)
out = task(x)
assert out.shape == (batch_size, 1), f"Classification output shape: expected ({batch_size}, 1), got {out.shape}"
print(f"  BinaryClassificationTaskLogits output shape: {out.shape} OK")

# BinaryClassificationTask (sigmoid)
task = BinaryClassificationTask(
    hidden_size=hidden_size,
    target_labels='track',
    loss_function=BinaryCrossEntropyWithLogitsLoss(),
)
out = task(x)
assert out.shape == (batch_size, 1), f"Classification output shape: expected ({batch_size}, 1), got {out.shape}"
assert torch.all(out >= 0) and torch.all(out <= 1), "Sigmoid output should be in [0, 1]"
print(f"  BinaryClassificationTask output shape: {out.shape} OK")

# ============================================================
# 4. Test task compute_loss
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Task compute_loss")
print("=" * 60)

# Energy task loss
task = EnergyReconstruction(
    hidden_size=hidden_size,
    target_labels='initial_state_energy',
    loss_function=LogCoshLoss(),
    transform_prediction_and_target=lambda x: torch.log10(x),
)
pred = task(x)
data_dict = {'initial_state_energy': torch.abs(torch.randn(batch_size)) + 1.0}
loss = task.compute_loss(pred, data_dict)
assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
print(f"  EnergyReconstruction compute_loss: {loss.item():.4f} OK")

# Direction task loss
task = DirectionReconstructionWithKappa(
    hidden_size=hidden_size,
    target_labels=['part_dir_x', 'part_dir_y', 'part_dir_z'],
    loss_function=VonMisesFisher3DLoss(),
)
pred = task(x)
target_dir = torch.randn(batch_size, 3)
target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
data_dict = {
    'part_dir_x': target_dir[:, 0],
    'part_dir_y': target_dir[:, 1],
    'part_dir_z': target_dir[:, 2],
}
loss = task.compute_loss(pred, data_dict)
assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
print(f"  DirectionReconstructionWithKappa compute_loss: {loss.item():.4f} OK")

# Classification task loss
task = BinaryClassificationTaskLogits(
    hidden_size=hidden_size,
    target_labels='track',
    loss_function=BinaryCrossEntropyWithLogitsLoss(),
)
pred = task(x)
data_dict = {'track': torch.randint(0, 2, (batch_size,)).float()}
loss = task.compute_loss(pred, data_dict)
assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
print(f"  BinaryClassificationTaskLogits compute_loss: {loss.item():.4f} OK")

# ============================================================
# 5. Test NuTStandardModel with nuT_PROMETHEUS backbone
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: NuTStandardModel with nuT_PROMETHEUS")
print("=" * 60)

FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "charge", "string_id", "is_signal"]
IDX_DICT = {feat: idx for idx, feat in enumerate(FEATURES)}

backbone_config = {
    "idx_dict": IDX_DICT, "emb_dims": 256, "seq_length": 300,
    "emb_type": "nuT", "n_features": 5, "abs_position_encoding": True,
    "refractive_index": 1.33, "masks": ["Causality", "Euclidean", "STRING"],
    "mode": "concat", "pairwise_dims": 64, "num_heads": 8,
    "dropout_attn": 0.0, "hidden_dim": 256, "dropout_FFNN": 0.0,
    "no_hits_blocks": 4, "no_evt_blocks": 2,
}

backbone = nuT_PROMETHEUS(**backbone_config)

task = EnergyReconstruction(
    hidden_size=backbone.nb_outputs,
    target_labels='initial_state_energy',
    loss_function=LogCoshLoss(),
    transform_prediction_and_target=lambda x: torch.log10(x),
)

model = NuTStandardModel(
    backbone=backbone,
    tasks=[task],
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs={'lr': 1e-3},
    scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_kwargs={"patience": 5},
    scheduler_config={"frequency": 1, "monitor": "val_loss"},
)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model created: {n_params:,} parameters")

# Create mock data in flat format (as graphnet's dataloader produces)
batch_size = 4
seq_length = 300
n_features = len(FEATURES)

data = Mock()
data.x = torch.randn(batch_size * seq_length, n_features)
data.batch = torch.repeat_interleave(torch.arange(batch_size), seq_length)
data.num_graphs = batch_size
data.__getitem__ = lambda self, key: getattr(self, key)

# Add target label
data.initial_state_energy = torch.abs(torch.randn(batch_size)) + 1.0

model.eval()
with torch.no_grad():
    preds = model(data)

assert len(preds) == 1, f"Expected 1 prediction, got {len(preds)}"
assert preds[0].shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {preds[0].shape}"
print(f"  Forward pass output shape: {preds[0].shape} OK")

# Test configure_optimizers
opt_config = model.configure_optimizers()
assert "optimizer" in opt_config, "configure_optimizers should return dict with 'optimizer'"
assert "lr_scheduler" in opt_config, "configure_optimizers should return dict with 'lr_scheduler'"
print(f"  configure_optimizers: OK")

# ============================================================
# 6. Test with multiple tasks
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: NuTStandardModel with multiple tasks")
print("=" * 60)

backbone2 = nuT_PROMETHEUS(**backbone_config)

tasks = [
    BinaryClassificationTaskLogits(
        hidden_size=backbone2.nb_outputs,
        target_labels='track',
        loss_function=BinaryCrossEntropyWithLogitsLoss(),
    ),
    EnergyReconstruction(
        hidden_size=backbone2.nb_outputs,
        target_labels='initial_state_energy',
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
    ),
]

model2 = NuTStandardModel(
    backbone=backbone2,
    tasks=tasks,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs={'lr': 1e-3},
)

data2 = Mock()
data2.x = torch.randn(batch_size * seq_length, n_features)
data2.batch = torch.repeat_interleave(torch.arange(batch_size), seq_length)
data2.num_graphs = batch_size
data2.__getitem__ = lambda self, key: getattr(self, key)
data2.track = torch.randint(0, 2, (batch_size,)).float()
data2.initial_state_energy = torch.abs(torch.randn(batch_size)) + 1.0

model2.eval()
with torch.no_grad():
    preds2 = model2(data2)

assert len(preds2) == 2, f"Expected 2 predictions, got {len(preds2)}"
assert preds2[0].shape == (batch_size, 1), f"Classification shape: {preds2[0].shape}"
assert preds2[1].shape == (batch_size, 1), f"Energy shape: {preds2[1].shape}"
print(f"  Multi-task forward pass: {[p.shape for p in preds2]} OK")

# Test target/prediction labels
assert model2.target_labels == ['track', 'initial_state_energy'], \
    f"target_labels: {model2.target_labels}"
print(f"  target_labels: {model2.target_labels} OK")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL STEP 6 TESTS PASSED!")
print("=" * 60)
