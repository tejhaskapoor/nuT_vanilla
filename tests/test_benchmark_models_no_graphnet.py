import torch
import sys
import os
import importlib as _il

# Resolve package dir (parent of this tests/ directory) dynamically.
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = os.path.basename(_pkg_dir)
if os.path.dirname(_pkg_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(_pkg_dir))

nuT_PROMETHEUS = _il.import_module(_P).nuT_PROMETHEUS

FEATURES = ["sensor_pos_x","sensor_pos_y","sensor_pos_z","t","charge","string_id","is_signal"]
IDX_DICT = {feat: idx for idx, feat in enumerate(FEATURES)}

config = {
    "idx_dict": IDX_DICT, "emb_dims": 256, "seq_length": 300,
    "emb_type": "nuT", "n_features": 5, "abs_position_encoding": True,
    "refractive_index": 1.33, "masks": ["Causality", "Euclidean", "STRING"],
    "mode": "concat", "pairwise_dims": 64, "num_heads": 8,
    "dropout_attn": 0.0, "hidden_dim": 256, "dropout_FFNN": 0.0,
    "no_hits_blocks": 4, "no_evt_blocks": 2,
}

model = nuT_PROMETHEUS(**config)
model.eval()
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")

# Create mock data (the test uses Mock objects with .x and .batch)
from unittest.mock import Mock
data = Mock()
batch_size = 4
data.x = torch.randn(batch_size * 300, 7)  # flat [n, d] format for array_to_sequence
data.batch = torch.repeat_interleave(torch.arange(batch_size), 300)

with torch.no_grad():
    output = model(data)

print(f"Output shape: {output.shape}")
assert output.shape == (batch_size, 256), f"Expected (4, 256), got {output.shape}"
print("Step 4 PASSED!")
