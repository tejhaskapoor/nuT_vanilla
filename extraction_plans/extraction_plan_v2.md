# Plan: Extract nuT from GraphNet (v2)

## Context
The nuT transformer model (`nuT_local/`) is built on top of the graphnet framework. The goal is to make nuT **fully standalone** — including training — so it runs without graphnet installed. This requires replacing not only the model base classes but also graphnet's `StandardModel` training pipeline, task definitions, loss functions, and data loading.

The training script `prometheus_train.py` currently uses graphnet's `StandardModel`, which asserts `isinstance(backbone, Model)` — so simply changing the base class breaks training. The solution is to replace `StandardModel` with a standalone PyTorch Lightning module.

## Files in scope
- `nuT_local/nuT_components_layers.py` — Encoder_block
- `nuT_local/nuT_components_embedding.py` — Feature/positional/pairwise processing + mask classes
- `nuT_local/nuT_model_no_graphnet.py` — Main model classes (nuT, nuT_PROMETHEUS)
- `nuT_local/nuT_data_representation.py` — Data pipeline
- `nuT_local/__init__.py` — Package exports
- `nuT_local/test_benchmark_models.py` — Tests
- `prometheus_train.py` — Training script (needs rewrite)
- **NEW files to create:** `nuT_local/nuT_training.py` (standalone StandardModel + tasks + losses)

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
import torch
from nuT_local.nuT_components_embedding import CausalityMask, EuclideanMask, IdsMask
m = CausalityMask(1.33); assert m(torch.randn(2, 10, 4)).shape == (2, 10, 10)
m = EuclideanMask(50); assert m(torch.randn(2, 10, 3)).shape == (2, 10, 10)
m = IdsMask(); assert m(torch.randn(2, 10)).shape == (2, 10, 10)
```

---

## Step 3: Inline `array_to_sequence` into `nuT_model_no_graphnet.py`

**Changes:**
- Add missing imports at the top: `from typing import Tuple, Any` (add `Tuple` and `Any` to the existing typing import on line 5)
- Copy the `array_to_sequence` function from `src/graphnet/models/utils.py` (lines 69-110) as a module-level function. It's ~15 lines of pure PyTorch:
  ```python
  def array_to_sequence(
      x: Tensor,
      batch_idx: LongTensor,
      padding_value: Any = 0,
      excluding_value: Any = torch.inf,
  ) -> Tuple[Tensor, Tensor, Tensor]:
      if torch.any(torch.eq(x, excluding_value)):
          raise ValueError(...)
      _, seq_length = torch.unique(batch_idx, return_counts=True)
      x_list = torch.split(x, seq_length.tolist())
      x = torch.nn.utils.rnn.pad_sequence(
          x_list, batch_first=True, padding_value=excluding_value
      )
      mask = torch.ne(x[:, :, 1], excluding_value)
      x[~mask] = padding_value
      return x, mask, seq_length
  ```
- Also add `from torch import LongTensor` (or use `torch.LongTensor` inline)
- Remove `from graphnet.models.utils import array_to_sequence` (line 25)
- Remove unused `from torch_geometric.utils import to_dense_batch` (line 27)

**Test** (run as a script, not with relative imports):
```python
import torch
import sys; sys.path.insert(0, '/path/to/graphnet')
from nuT_local.nuT_model_no_graphnet import array_to_sequence
x = torch.randn(20, 5)
batch = torch.tensor([0]*7 + [1]*6 + [2]*7)
padded, mask, lengths = array_to_sequence(x, batch)
assert padded.shape == (3, 7, 5)
assert mask.shape == (3, 7)
print("OK")
```
**Note:** The "relative import with no known parent package" error occurs when running the file directly (`python nuT_model_no_graphnet.py`). Always test by importing from the package (`from nuT_local.nuT_model_no_graphnet import ...`) or add the parent dir to `sys.path`.

---

## Step 4: Replace `Model` base class with `nn.Module` in `nuT_model_no_graphnet.py`

**Changes:**
- Remove `from graphnet.models import Model` (line 8)
- Change `class nuT(Model)` → `class nuT(nn.Module)` (line 32)
- Change `class nuT_PROMETHEUS(Model)` → `class nuT_PROMETHEUS(nn.Module)` (line 223)
- Change `forward(self, data: Data)` type hint to `forward(self, data)` on both classes (lines 138, 326)
- Remove `from torch_geometric.data import Data` (line 28)

After this step, `nuT_model_no_graphnet.py` has **zero** graphnet/torch_geometric imports.

**⚠️ This will break `prometheus_train.py`** because `StandardModel` asserts `isinstance(backbone, Model)`. That's expected — Step 6 replaces `StandardModel` with a standalone training module.

**Test:** There are two ways to verify Step 4 works:

**Option 1: Run existing pytest** (but first fix the import in test file — line 99 imports `from nuT.nuT_model` which still uses graphnet's Model). To test the no-graphnet version specifically, temporarily change line 99 to:
```python
from nuT.nuT_model_no_graphnet import nuT_PROMETHEUS
```
Then run from the **graphnet root directory** (the parent of `nuT_local/`):
```bash
cd /path/to/graphnet
pytest nuT_local/test_benchmark_models.py::TestNuT_PROMETHEUS::test_forward_pass -v -s
```

**Option 2: Quick inline test** — create a small test script `test_step4.py` in the graphnet root:
```python
import torch
import sys
from nuT_local import nuT_PROMETHEUS

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
```
Run: `python test_step4.py`

**Important note:** The existing `create_mock_data()` in `test_benchmark_models.py` creates `data.x` with shape `[batch_size, seq_length, n_features]` (3D), but `array_to_sequence` expects `[n, d]` (2D flat). The existing tests were written for a model variant that accepts pre-batched data. The test script above uses the correct flat format.

---

## Step 5: Decouple `nuT_data_representation.py` from graphnet

Remove all 3 graphnet imports (lines 10-12). Two options — choose at implementation time.

### Option A: Remove `KM3NeTHitsSequence`, keep only `KM3NeTNodesAsTimeSeries`

**Changes:**
1. Remove all 3 graphnet imports (lines 10-12)
2. Remove `KM3NeTHitsSequence` class entirely (lines 270-309)
3. Add these imports at the top of the file (merge with existing imports):
   ```python
   import logging
   from abc import abstractmethod
   import torch.nn as nn
   ```
4. Add standalone `NodeDefinition` base class before `KM3NeTNodesAsTimeSeries`. Full code:

```python
class NodeDefinition(nn.Module):
    """Standalone base class for defining graph nodes (replaces graphnet's NodeDefinition).

    Subclasses must implement:
        - _define_output_feature_names(input_feature_names) -> List[str]
        - _construct_nodes(x: torch.Tensor) -> torch.Tensor
    """

    def __init__(
        self, input_feature_names: Optional[List[str]] = None
    ) -> None:
        """Construct NodeDefinition.

        Args:
            input_feature_names: Column names for input features.
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        if input_feature_names is not None:
            self.set_output_feature_names(
                input_feature_names=input_feature_names
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Construct nodes from raw node features.

        Args:
            x: node features with shape [num_pulses, d].

        Returns:
            Processed node features.
        """
        return self._construct_nodes(x=x)

    @property
    def _output_feature_names(self) -> List[str]:
        """Return output feature names."""
        if not hasattr(self, '_hidden_output_feature_names'):
            raise AttributeError(
                f"{self.__class__.__name__} was instantiated without "
                f"`input_feature_names`. Please instantiate with "
                f"`input_feature_names` or call set_output_feature_names()."
            )
        return self._hidden_output_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return len(self._output_feature_names)

    def set_number_of_inputs(self, input_feature_names: List[str]) -> None:
        """Set number of inputs expected by node definition."""
        assert isinstance(input_feature_names, list)
        self.nb_inputs = len(input_feature_names)

    def set_output_feature_names(self, input_feature_names: List[str]) -> None:
        """Set output feature names as a member variable."""
        self._hidden_output_feature_names = self._define_output_feature_names(
            input_feature_names
        )

    # Logging helpers (replace graphnet's Logger mixin)
    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self._logger.warning(msg)

    def info(self, msg: str) -> None:
        """Log an info message."""
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._logger.error(msg)

    @abstractmethod
    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Construct names of output columns."""
        ...

    @abstractmethod
    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Construct nodes from raw node features x."""
        ...
```

### Option B: Keep `KM3NeTHitsSequence` with standalone stubs

Same as Option A, plus add standalone `Detector` and `GraphDefinition` stub classes. See `extraction_plan_v1.md` for full details.

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

## Step 6: Create standalone training module `nuT_local/nuT_training.py`

This is the **key new step** — replacing graphnet's `StandardModel`, tasks, and loss functions.

### 6a: Standalone loss functions

Reimplement the 3 loss functions used in `prometheus_train.py`. Each is simple (~10-20 lines):

```python
class LossFunction(nn.Module):
    """Base loss that supports per-event weights."""
    def forward(self, prediction, target, weights=None):
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        return torch.mean(elements)

class BinaryCrossEntropyWithLogitsLoss(LossFunction):
    def _forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean(dim=-1)

class LogCoshLoss(LossFunction):
    def _forward(self, pred, target):
        diff = pred - target
        return (torch.log(torch.cosh(diff))).mean(dim=-1)

class VonMisesFisher3DLoss(LossFunction):
    def _forward(self, pred, target):
        # pred: [N, 4] = [dir_x, dir_y, dir_z, kappa]
        # target: [N, 3] = [dir_x, dir_y, dir_z]
        kappa = pred[:, 3]
        # ... Bessel function log normalization ...
        # ... dot product between predicted and target directions ...
        # (copy from graphnet's implementation)
```

### 6b: Standalone task heads

Each task is a linear head + output transform:

```python
class Task(nn.Module):
    def __init__(self, hidden_size, target_labels, loss_function,
                 transform_prediction_and_target=None, transform_inference=None):
        super().__init__()
        self.target_labels = [target_labels] if isinstance(target_labels, str) else target_labels
        self.loss_function = loss_function
        self._transform_target = transform_prediction_and_target
        self._transform_inference = transform_inference
        # Subclasses set self.head = nn.Linear(hidden_size, nb_outputs)

class EnergyReconstruction(Task):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return F.softplus(self.head(x)) + 1e-6  # ensure positive

class DirectionReconstructionWithKappa(Task):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.head = nn.Linear(hidden_size, 3)  # direction
        self.kappa_head = nn.Linear(hidden_size, 1)  # uncertainty
    def forward(self, x):
        direction = self.head(x)
        direction = direction / direction.norm(dim=-1, keepdim=True)  # normalize
        kappa = F.softplus(self.kappa_head(x)) + 1e-6
        return torch.cat([direction, kappa], dim=-1)

class BinaryClassificationTaskLogits(Task):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.head(x)  # raw logits
```

### 6c: Standalone `NuTStandardModel` (replaces graphnet's `StandardModel`)

A PyTorch Lightning module that orchestrates backbone + tasks:

```python
class NuTStandardModel(pl.LightningModule):
    def __init__(self, backbone, tasks, optimizer_class, optimizer_kwargs,
                 scheduler_class=None, scheduler_kwargs=None, scheduler_config=None):
        super().__init__()
        self.backbone = backbone
        self.tasks = nn.ModuleList(tasks) if isinstance(tasks, list) else nn.ModuleList([tasks])
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}

    def forward(self, data):
        x = self.backbone(data)  # [B, hidden_dim]
        predictions = [task(x) for task in self.tasks]
        return predictions

    def shared_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        total_loss = torch.tensor(0.0, device=self.device)
        for task, pred in zip(self.tasks, predictions):
            target = torch.stack([batch[label] for label in task.target_labels], dim=1)
            if task._transform_target:
                target = task._transform_target(target)
                pred_for_loss = task._transform_target(pred)  # apply same transform
            else:
                pred_for_loss = pred
            loss = task.loss_function(pred_for_loss, target)
            total_loss = total_loss + loss
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer_class(self.parameters(), **self._optimizer_kwargs)
        if self._scheduler_class:
            scheduler = self._scheduler_class(optimizer, **self._scheduler_kwargs)
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler, **self._scheduler_config
            }}
        return optimizer
```

**Key file:** `nuT_local/nuT_training.py` — contains all of the above (~200 lines total).

**Test:**
```python
from nuT_local.nuT_training import NuTStandardModel, LogCoshLoss, EnergyReconstruction
from nuT_local import nuT_PROMETHEUS
backbone = nuT_PROMETHEUS(**DEFAULT_CONFIG)
task = EnergyReconstruction(hidden_size=256, target_labels='energy', loss_function=LogCoshLoss())
model = NuTStandardModel(backbone=backbone, tasks=[task], optimizer_class=torch.optim.AdamW, optimizer_kwargs={'lr': 1e-3})
# Verify model.forward() works with mock data
```

---

## Step 7: Data loading — keep graphnet for now

**Decision:** Keep using graphnet's `SQLiteDataset`, `make_train_validation_dataloader`, `KM3NeTHitsSequence`, and `Detector` for data loading only. Data loading is a separate concern from the model and can be decoupled in a future phase.

This means `prometheus_train.py` will still import from graphnet for:
- `graphnet.training.utils.make_train_validation_dataloader`
- `graphnet.models.detector.detector.Detector` (for `PrometheusDetector`)
- `graphnet.constants` (for geometry table paths)

But the **model, tasks, losses, and training loop** will be fully standalone.

**No code changes needed** — just keep the existing data loading imports in `prometheus_train.py`.

---

## Step 8: Rewrite `prometheus_train.py` to use standalone pipeline

Replace all graphnet imports with standalone equivalents:

| Old (graphnet) | New (standalone) |
|---|---|
| `from graphnet.models import StandardModel` | `from nuT_local.nuT_training import NuTStandardModel` |
| `from graphnet.models.task.reconstruction import ...` | `from nuT_local.nuT_training import EnergyReconstruction, ...` |
| `from graphnet.training.loss_functions import ...` | `from nuT_local.nuT_training import LogCoshLoss, ...` |
| `from graphnet.models.detector.detector import Detector` | `from nuT_local.nuT_data_representation import Detector` (if Option B Step 5) |
| `from graphnet.training.callbacks import ProgressBar` | Use `pl.callbacks.TQDMProgressBar` directly |
| `from graphnet.utilities.logging import Logger` | Use Python `logging` module |
| `from graphnet.utilities.argparse import ArgumentParser` | Use `argparse.ArgumentParser` |
| `from graphnet.training.labels import Label` | Inline the `Track` label class (it's simple) |
| `from graphnet.constants import ...` | Hardcode the geometry path or use a config var |

Replace `model = StandardModel(...)` with `model = NuTStandardModel(...)`.

Replace `model.fit(...)` with:
```python
trainer = pl.Trainer(
    max_epochs=..., accelerator='gpu', devices=[0],
    callbacks=callbacks, logger=mlflow_logger, ...
)
trainer.fit(model, training_dataloader, validation_dataloader)
```

**Test:** Run the full training script on a small dataset.

---

## Step 9: Update `__init__.py` and `test_benchmark_models.py`

**`__init__.py`:**
- Remove `KM3NeTHitsSequence` from exports (if Option A Step 5)
- Add exports from `nuT_training.py`

**`test_benchmark_models.py`:**
- Update imports to use package root
- Add tests for `NuTStandardModel`

**Test:** `pytest nuT_local/test_benchmark_models.py -v -s`

---

## Verification

After all steps:
1. All nuT_local imports work without graphnet
2. `pytest nuT_local/test_benchmark_models.py -v -s` passes
3. `prometheus_train.py` runs a training loop (at least `fast_dev_run=True`)
4. Model forward pass produces correct output shapes

## Remaining dependencies after extraction

**nuT model package (`nuT_local/`):**
- `torch` (PyTorch) — core dependency
- `pytorch_lightning` (for training loop — this is NOT graphnet)
- `numpy` (data representation)
- **No graphnet, no torch_geometric**

**Training script (`prometheus_train.py`):**
- All of the above, plus:
- `graphnet` — **only for data loading** (`SQLiteDataset`, `make_train_validation_dataloader`, `Detector`, geometry constants)
- `torch_geometric` — required by graphnet's data loading
- This can be decoupled in a future phase
