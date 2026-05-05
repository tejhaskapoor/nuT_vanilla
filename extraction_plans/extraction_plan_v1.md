# Plan: Extract nuT from GraphNet

## Context
The nuT transformer model (`nuT_local/`) is built on top of the graphnet framework but doesn't use most of graphnet's functionality. The goal is to make nuT fully standalone so it runs without graphnet installed, improving portability and reducing dependency overhead.

## Files in scope
- `nuT_local/nuT_components_layers.py` — Encoder_block (already `nn.Module`)
- `nuT_local/nuT_components_embedding.py` — Feature/positional/pairwise processing + mask classes
- `nuT_local/nuT_model_no_graphnet.py` — Main model classes (nuT, nuT_PROMETHEUS)
- `nuT_local/nuT_data_representation.py` — Data pipeline (KM3NeTNodesAsTimeSeries, KM3NeTHitsSequence)
- `nuT_local/__init__.py` — Package exports
- `nuT_local/test_benchmark_models.py` — Tests

Files to **ignore**: `nuT_model.py`, `nuT_model_optimized.py`, `nuT_model_advanced.py`

---

## Step 1: Clean unused imports in `nuT_components_layers.py`

**Changes:**
- Remove `from torch_geometric.data import Data` (line 7, unused)
- Remove `from torch_geometric.utils import to_dense_batch` (line 11, unused)
- Remove `from pytorch_lightning import LightningModule` (line 12, unused)

`Encoder_block` already inherits from `nn.Module` — no functional change.

**Test:** `python -c "from nuT_local.nuT_components_layers import Encoder_block; print('OK')"`

---

## Step 2: Replace `LightningModule` with `nn.Module` in `nuT_components_embedding.py`

**Changes:**
- Remove `from pytorch_lightning import LightningModule` (line 9)
- Change base class of `CausalityMask` (line 98), `EuclideanMask` (line 137), `DotProductMask` (line 164), `IdsMask` (line 185) from `LightningModule` to `nn.Module`

None of these classes use any Lightning-specific features.

**Test:**
```python
from nuT_local.nuT_components_embedding import CausalityMask, EuclideanMask, IdsMask
import torch
m = CausalityMask(1.33); assert m(torch.randn(2, 10, 4)).shape == (2, 10, 10)
m = EuclideanMask(50); assert m(torch.randn(2, 10, 3)).shape == (2, 10, 10)
m = IdsMask(); assert m(torch.randn(2, 10)).shape == (2, 10, 10)
```

---

## Step 3: Inline `array_to_sequence` into `nuT_model_no_graphnet.py`

**Changes:**
- Copy the `array_to_sequence` function from `src/graphnet/models/utils.py` (lines 69-110) into `nuT_model_no_graphnet.py` as a module-level function. It's ~15 lines of pure PyTorch.
- Remove `from graphnet.models.utils import array_to_sequence` (line 25)
- Remove unused `from torch_geometric.utils import to_dense_batch` (line 27)

**Test:**
```python
import torch
from nuT_local.nuT_model_no_graphnet import array_to_sequence
x = torch.randn(20, 5)
batch = torch.tensor([0]*7 + [1]*6 + [2]*7)
padded, mask, lengths = array_to_sequence(x, batch)
assert padded.shape == (3, 7, 5)
assert mask.shape == (3, 7)
```

---

## Step 4: Replace `Model` base class with `nn.Module` in `nuT_model_no_graphnet.py`

**Changes:**
- Remove `from graphnet.models import Model` (line 8)
- Change `class nuT(Model)` → `class nuT(nn.Module)` (line 32)
- Change `class nuT_PROMETHEUS(Model)` → `class nuT_PROMETHEUS(nn.Module)` (line 223)
- Change `forward(self, data: Data)` type hint to `forward(self, data)` on both classes (lines 138, 326), since we no longer need `torch_geometric.data.Data`
- Remove `from torch_geometric.data import Data` (line 28)

This is safe because:
- `super().__init__()` is called with no args (lines 72, 263) — same for `nn.Module`
- `self.nb_inputs` and `self.nb_outputs` are set manually (not inherited)
- No `Model`-specific methods (save/load/from_config) are called

After this step, `nuT_model_no_graphnet.py` has **zero** graphnet/torch_geometric imports.

**Test:** Run existing benchmark tests — output shape should be `(batch_size, model.nb_outputs)`.

---

## Step 5: Decouple `nuT_data_representation.py` from graphnet

This is the largest step. Remove all 3 graphnet imports (lines 10-12). Two complete options are provided — choose one at implementation time.

---

### Option A: Remove `KM3NeTHitsSequence`, keep only `KM3NeTNodesAsTimeSeries`

**Rationale:** `KM3NeTHitsSequence` is only needed inside the graphnet data pipeline (Detector standardization, perturbation, truth labels). Tests and standalone inference pass raw tensors directly to the model.

**Changes to `nuT_data_representation.py`:**
1. Remove all 3 graphnet imports (lines 10-12)
2. Remove `KM3NeTHitsSequence` class entirely (lines 270-309)
3. Add a standalone `NodeDefinition` base class (~25 lines) at the top of the file:
   ```python
   import logging
   from abc import abstractmethod

   class NodeDefinition(nn.Module):
       """Standalone base class for node definitions."""
       def __init__(self, input_feature_names=None):
           super().__init__()
           self._logger = logging.getLogger(self.__class__.__name__)
           if input_feature_names:
               self._output_feature_names = self._define_output_feature_names(input_feature_names)

       @property
       def nb_outputs(self):
           return len(self._output_feature_names)

       def forward(self, x):
           return self._construct_nodes(x)

       def warning(self, msg): self._logger.warning(msg)
       def info(self, msg): self._logger.info(msg)

       @abstractmethod
       def _define_output_feature_names(self, input_feature_names): ...
       @abstractmethod
       def _construct_nodes(self, x): ...
   ```
4. `KM3NeTNodesAsTimeSeries` inherits from this new standalone `NodeDefinition`

**Test:**
```python
from nuT_local.nuT_data_representation import KM3NeTNodesAsTimeSeries
import torch
node_def = KM3NeTNodesAsTimeSeries(max_hits=300, trig_name="trig", unique=False)
x = torch.randn(500, 12)
out = node_def(x)
assert out.shape[0] <= 300 and out.shape[1] == 12
```

---

### Option B: Keep `KM3NeTHitsSequence` with standalone stubs

**Rationale:** Preserves the full data pipeline for users who need detector standardization and perturbation without graphnet.

**Changes to `nuT_data_representation.py`:**
1. Remove all 3 graphnet imports (lines 10-12)
2. Add standalone `NodeDefinition` base class (same as Option A above)
3. Add a standalone `Detector` base class (~30 lines):
   ```python
   class Detector(nn.Module):
       """Standalone detector stub for feature standardization."""
       geometry_table_path = ""
       xyz = []
       string_id_column = ""
       sensor_id_column = ""

       def __init__(self):
           super().__init__()

       @abstractmethod
       def feature_map(self) -> Dict[str, Callable]:
           """Return dict mapping feature names to standardization functions."""
           ...

       def forward(self, input_features, input_feature_names):
           return self._standardize(input_features, input_feature_names)

       def _standardize(self, input_features, input_feature_names):
           feature_map = self.feature_map()
           for i, name in enumerate(input_feature_names):
               if name in feature_map:
                   input_features[:, i] = feature_map[name](input_features[:, i])
           return input_features

       @staticmethod
       def _identity(x): return x
   ```
4. Add a standalone `GraphDefinition` base class (~40 lines) that:
   - Accepts `detector`, `node_definition`, `input_feature_names`, `dtype`, `perturbation_dict`, `seed`
   - In `forward()`: applies detector standardization → calls node_definition → returns Data
   - Handles perturbation via Gaussian noise (from numpy)
   - Does NOT implement inactive sensors, sensor masking, string masking, truth labels (graphnet-specific features)
5. `KM3NeTHitsSequence` inherits from the new standalone `GraphDefinition`

**Test:**
```python
from nuT_local.nuT_data_representation import KM3NeTNodesAsTimeSeries, KM3NeTHitsSequence
import torch

# Test NodeDefinition standalone
node_def = KM3NeTNodesAsTimeSeries(max_hits=300, trig_name="trig", unique=False)
x = torch.randn(500, 12)
out = node_def(x)
assert out.shape[0] <= 300 and out.shape[1] == 12

# Test GraphDefinition with a concrete Detector
# (requires implementing a concrete detector subclass for testing)
```

---

## Step 6: Update `__init__.py`

**Changes:**
- If Option A (Step 5): Remove `KM3NeTHitsSequence` from exports
- If Option B (Step 5): Also export the standalone `Detector` base class
- Verify all remaining exports work without graphnet

**Test:** `python -c "from nuT_local import nuT, nuT_PROMETHEUS, KM3NeTNodesAsTimeSeries, Encoder_block"`

---

## Step 7: Update `test_benchmark_models.py`

**Changes:**
- Tests currently import from `nuT.nuT_model`. Since `__init__.py` already re-exports from `nuT_model_no_graphnet`, change imports to use the package root: `from nuT import nuT_PROMETHEUS`
- Remove any remaining torch_geometric dependencies if present

**Test:** `pytest test_benchmark_models.py -v -s`

---

## Verification

After all steps, confirm:
1. `pip uninstall graphnet` (or test in a clean venv without graphnet)
2. `python -c "from nuT_local import nuT, nuT_PROMETHEUS; print('OK')"` succeeds
3. `pytest nuT_local/test_benchmark_models.py -v -s` passes
4. Model forward pass produces correct output shapes

## Remaining dependencies after extraction
- `torch` (PyTorch)
- `numpy` (used in data representation)
- No graphnet, no pytorch_lightning, no torch_geometric
