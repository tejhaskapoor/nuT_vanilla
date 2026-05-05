# nuT_no_graphnet — Cleanup & Refactor Summary

**Date:** 2026-04-02

---

## 1. GraphNet Remnants Removed

| File | What was removed / renamed |
|---|---|
| `constants.py` | `GRAPHNET_ROOT_DIR` → `NUT_ROOT_DIR`; opening docstrings referencing graphnet replaced; unused `PRETRAINED_MODEL_DIR` (pointed to non-existent `src/graphnet/models/pretrained`) removed |
| `nuT_detector.py` | Class `ORCA115_graphnet` → `ORCA115_legacy`; duplicate shadowed `feature_map()` block (and associated unused methods) removed |
| `nuT_model_no_graphnet.py` | All commented-out graphnet/PyG import lines deleted (`##from graphnet.models.gnn.gnn import GNN`, `##from graphnet.models.utils import array_to_sequence`, `## from torch_geometric.utils import to_dense_batch`, `## from torch_geometric.data import Data`, `##class nuT(GNN)`) |
| `nuT_data_representation.py` | "in GraphNeT" removed from module docstring; `## Node definition dependency removed...` dead comment removed |
| `nuT_components_embedding.py` | Four `## class Foo(LightningModule):` dead commented-out class headers removed |
| `prometheus_train_no_graphnet.py` | "still uses graphnet's GraphDefinition for data loading" comment updated to accurately describe the current pipeline |

---

## 2. Code Consolidation

### `nuT_model_no_graphnet.py`
`nuT` and `nuT_PROMETHEUS` were >90% identical. Merged into a single `nuT` class with a new `detector_type` parameter:

```python
nuT(..., detector_type="KM3NeT")   # default — KM3NeT layout
nuT(..., detector_type="Prometheus")
```

`detector_type` controls:
- Which metadata columns are stripped before the transformer (`id_cols_to_remove`)
- Which position-coordinate key names are used (`pos_x/y/z` vs `sensor_pos_x/y/z`)

`nuT_PROMETHEUS` is kept as a one-line backward-compatible alias:
```python
def nuT_PROMETHEUS(*args, **kwargs):
    kwargs.setdefault("detector_type", "Prometheus")
    ...
    return nuT(*args, **kwargs)
```

The detector-specific configuration is centralised in a `_DETECTOR_CONFIGS` dict at the top of the file, making it easy to add new detector types.

---

## 3. Bug Fixes

| File | Bug | Fix |
|---|---|---|
| `nuT_components_embedding.py` | `emb_dim[0]` / `emb_dim[1]` in the `"Kaggle"` embedding branch indexed the wrong variable | Changed to `emb_dims[0]` / `emb_dims[1]` |

---

## 4. GPU / Performance Additions

### `prometheus_train_no_graphnet.py`
- Added `torch.set_float32_matmul_precision('high')` after the cudnn reproducibility flags. This enables TF32 for matrix multiplications on Ampere+ GPUs (A100, H100, RTX 3090+), giving ~2× throughput with negligible accuracy loss for neural network training.
- Replaced three `print()` calls (flash attention kernel check) with `logger.info()`.
- Added explanatory comment above `torch.compile(mode='reduce-overhead')`.

### `configs/config_prometheus_energy.yaml`
- Replaced invalid `gpus: 4` key (not read by the training script) with the correct `accelerator: "gpu"` and `devices: [...]` keys that `pl.Trainer` and the training script actually consume.

---

## 5. Readability & Documentation

### Physics constants documented
- `self.c = 299792458.0` — speed of light in vacuum (m/s)
- `scaling_t = 1.e-9` — converts nanoseconds to seconds for causality mask
- `self.v = self.c / self.refractive_index` — Cherenkov photon speed in medium
- Detector normalisation constants annotated (e.g. `/ 2500.0  # max event window ~2500 ns`, `/ 256.0  # ToT is 8-bit`)

### New / improved docstrings
- `Detector` base class — standardization contract explained
- `nuT` class — full parameter-by-parameter docstring including `detector_type`
- `array_to_sequence()` — input/output tensor shapes documented
- `PairwiseProcessing` — Conv2d architecture and tensor shapes explained
- `LogCMK` — mathematical definition of log C_m(κ) and rationale for float64 intermediate computation
- `KM3NeTNodesAsTimeSeries` — three pulse-selection modes documented in class docstring
- `KM3NeTHitsSequence` — pipeline steps enumerated in `forward()` docstring

### Type hints fixed
- `labels.py`: all `__call__` return types corrected from `torch.tensor` (constructor) to `torch.Tensor` (type)

### Dead comments cleaned up
- Commented-out alternative classification logic in `labels.py` replaced with concise PDG-code comments
- `# TK: added brackets...` style personal notes removed or replaced with proper inline comments

### `dataloader.py`
- Added `RuntimeError` if the truth table query returns 0 rows (clear error message instead of silent empty dataset)
- `pin_memory=torch.cuda.is_available()` annotated as "page-locked memory for faster GPU transfers"

### `configs/config_prometheus_energy.yaml`
- All config keys annotated with inline comments explaining units, valid values, and effect on training

---

## 6. Verification

Tests run from the repo root with `PYTHONPATH=. python nuT_no_graphnet/tests/test_no_graphnet_dep.py`:

```
✓ All imports succeeded (no graphnet/PyG required)
✓ Labels work with dict batches
✓ batch_size computed correctly from dict 'n_pulses'
✓ compute_loss dict-merge works correctly for list-of-dicts batches
✓ nuT_PROMETHEUS.forward works with dict batch
✓ NuTStandardModel forward + compute_loss works end-to-end
✓ No graphnet or torch_geometric in loaded modules
All tests passed!
```

No Python source files contain a live `import graphnet` or `import torch_geometric` statement.
