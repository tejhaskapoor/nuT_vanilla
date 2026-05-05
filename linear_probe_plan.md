# Linear Probing Analysis for nuT_vanilla Block Representations

## Context

The nuT_vanilla model stacks `no_blocks` identical `Encoder_block` modules sequentially. After training on neutrino energy regression (100k events, LogCosh loss in log10 space), we want to understand **how much task-relevant information is encoded at each depth**. Linear probing is the standard tool: freeze the pretrained backbone, extract the CLS token embedding after each block, train a lightweight linear head on it, and compare performance across depths. This reveals whether representations improve gradually, plateau, or whether early blocks already capture most of the signal.

---

## Part 1: Conceptual Design

### What to probe and why

The CLS token accumulates global event information through attention at every block. Probing it after block `k` tells us: *"how much does a linear model know about target Y, given only the representation available at depth k?"*

**Primary probe target (matches training task)**
- `log10(initial_state_energy)` — regression, using MAE and R². This is the direct task metric; a good probe = the representation has compressed energy information linearly.

**Secondary probe targets (physics insight)**
- `log10(total_charge)` — simplest proxy for energy; if early blocks already nail this, they are tracking charge aggregation.
- `vertex_z` or `vertex_r` — spatial origin of the neutrino interaction; tests whether geometry is learnt.
- `is_track` (binary) — event topology; tests whether particle type separates at each depth.
- `n_pulses` (log-scaled) — event multiplicity; a simple structural feature that should be easy to decode.

These secondaries are already in the data files or trivially derivable. Choose whichever truth columns your HDF5 files contain beyond `initial_state_energy`.

### Metric to report per probe

| Target type | Metric |
|---|---|
| Regression (energy, charge) | R² and MAE in log10 space |
| Classification (track/shower) | AUC |

Plot metric vs block index (0-indexed, plus a "block −1" = after positional encoding, before any transformer block).

### Interpretation guide

| Pattern | Meaning |
|---|---|
| Monotone increase, saturates near last block | Information accumulates steadily; deeper = better |
| Sharp jump at block k then plateau | Block k is the critical transformation layer |
| Near-zero for all secondary targets | Model specialised purely for primary task |
| Primary ≫ secondary even at block 0 | Initial embedding already captures task-relevant stats |
| Secondary improves faster than primary | Model builds intermediate physics features before compressing to energy |

### Comparison baselines to include

1. **Random init**: same model architecture, weights never trained. Gives a floor — how much linear probing of a random CLS token achieves.
2. **Input features directly**: a linear model trained on per-event aggregates of the raw features (mean, sum of charge). Gives an upper bound for what is linearly accessible without any transformer.
3. **Full model output** (last block): the actual training performance — should match or be close to your reported 100k run results.

---

## Part 2: Implementation Design (minimal code changes)

### Strategy: standalone probe script using PyTorch hooks

No changes to `nuT_model_no_graphnet.py`, `model_components.py`, or `training.py`. A new self-contained script `linear_probe_analysis.py` does:

1. Load trained checkpoint with existing config loading code.
2. Register `forward_hook` on each `model.blocks[i]` to capture the CLS token output (index 0 of the sequence).
3. Run inference over a held-out subset of the 100k dataset → collect `(block_idx → [N, model_dim])` representation matrices.
4. For each block, train a small linear model (sklearn `Ridge` or `LinearRegression` / `LogisticRegression`) on 80% and evaluate on 20%.
5. Plot results.

### Hook registration (key code idea)

```python
activations = {}  # block_idx -> list of CLS tensors

def make_hook(block_idx):
    def hook(module, input, output):
        # output shape: [B, L, model_dim]; CLS token is position 0
        activations[block_idx].append(output[:, 0, :].detach().cpu())
    return hook

handles = []
for i, block in enumerate(model.blocks):
    activations[i] = []
    handles.append(block.register_forward_hook(make_hook(i)))

# Run forward pass (inference only, no grad)
model.eval()
with torch.no_grad():
    for batch in dataloader:
        _ = model(batch)  # hooks fire, CLS tokens accumulate

# Remove hooks when done
for h in handles:
    h.remove()
```

After the loop, `activations[i]` is a list of tensors; `torch.cat(activations[i])` gives `[N, model_dim]`.

### Also capture "block −1" (post-embedding, pre-block)

Hook on `model.pos_enc` output (or `model.processing` if no pos_enc) to get the initial CLS token before any attention.

### Linear probe training

```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
import numpy as np

def probe_regression(X, y, test_frac=0.2):
    n = len(y)
    split = int(n * (1 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    clf = Ridge(alpha=1.0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return r2_score(y_te, y_pred), np.mean(np.abs(y_pred - y_te))
```

No GPU needed — sklearn fits on `[N, 128]` matrices instantly.

### Output: one summary plot

- X-axis: block index (−1 = embedding, 0..n_blocks−1)
- Y-axis left: R² for primary (energy) and secondaries
- Y-axis right (twin): MAE in log10(E)
- Dashed horizontal lines: random-init baseline, raw-feature baseline

---

## Critical Files

| File | Role | Change needed |
|---|---|---|
| `nuT_model_no_graphnet.py` | nuT_vanilla definition | **None** |
| `model_components.py` | Encoder_block, FeaturesProcessing | **None** |
| `training.py` | Task/loss definitions | **None** |
| `dataloader.py` | Data loading | **None** (reuse as-is) |
| `inf_scripts/pone-pro-infer.py` | Inference script to adapt | Reference for config/checkpoint loading |
| `configs/pone-pro-energy-config.yaml` | Config with `no_blocks`, `model_dim` | Read only |
| `linear_probe_analysis.py` | **New standalone script** | Create |

---

## Verification Plan

1. **Sanity check**: probe at last block should reproduce ≈ the same R² / MAE as your reported 100k training run inference results.
2. **Monotonicity check**: R² should be non-decreasing (or roughly so) with depth — if it jumps up then drops, investigate hook placement.
3. **Random baseline**: R² for untrained model should be ≈ 0; if it is high, the probe is leaking information via label statistics.
4. **Speed check**: extraction of all block representations for 10k events should take < 2 min on CPU.
