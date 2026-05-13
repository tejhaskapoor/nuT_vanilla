"""Data representation module for nuT: hit sequence construction and preprocessing."""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, default_rng

from torch import Tensor, LongTensor

from .detector import Detector


def array_to_sequence(
    x: Tensor,
    batch_idx: LongTensor,
    padding_value: Any = 0,
    excluding_value: Any = torch.inf,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a flat hit tensor to a padded batch sequence.

    Transforms a flat ``[N, d]`` tensor (all events concatenated) into a
    zero-padded ``[B, L, d]`` tensor suitable for batched transformer
    processing, where ``B`` is the batch size and ``L`` is the longest
    sequence in the batch.

    Args:
        x: Flat tensor ``[N, d]`` — all hits from all events in the batch.
        batch_idx: LongTensor ``[N]`` mapping each row in ``x`` to its event
            index (0-indexed). Equivalent to ``torch_geometric.data.Batch.batch``.
        padding_value: Value used to fill padded positions (default: 0).
        excluding_value: Sentinel value that must NOT appear in ``x``; used
            internally during padding (default: ``torch.inf``).

    Returns:
        x: Padded sequence ``[B, L, d]``.
        mask: Boolean tensor ``[B, L]``; ``True`` for real hits, ``False`` for padding.
        seq_length: IntTensor ``[B]`` with the number of hits per event.

    Raises:
        ValueError: If ``x`` already contains ``excluding_value``.
    """
    if torch.any(torch.eq(x, excluding_value)):
        raise ValueError(
            f"Input tensor `x` contains at least one element equal to the "
            f"sentinel value {excluding_value}. Choose a different excluding_value."
        )

    _, seq_length = torch.unique(batch_idx, return_counts=True)
    x_list = torch.split(x, seq_length.tolist())

    # Pad to the longest sequence in the batch using a sentinel, then swap
    x = torch.nn.utils.rnn.pad_sequence(
        x_list, batch_first=True, padding_value=excluding_value
    )
    # mask[b, i] == True if position i in event b is a real hit
    mask = torch.ne(x[:, :, 1], excluding_value)
    x[~mask] = padding_value
    return x, mask, seq_length


def unique(x, dim=None):
    """Return unique rows of x and the indices of their first occurrence.

    This is a stable version of torch.unique that also returns the indices
    of the first occurrence of each unique element (not supported natively
    until recent PyTorch versions).

    Reference: https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    Example::

        unique(tensor([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [1, 2, 5]
        ]), dim=0)
        => (tensor([[1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5]]),
            tensor([0, 1, 3]))
    """
    unique_vals, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim
    )
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique_vals, inverse.new_empty(unique_vals.size(0)).scatter_(0, inverse, perm)


class KM3NeTNodesAsTimeSeries(nn.Module):
    """Converts raw KM3NeT pulse data into a time-sorted hit sequence.

    Pulse selection modes (selected automatically based on constructor args):

    1. **Triggered only** — ``trig_name`` not in feature list:
       All pulses are time-sorted and truncated to ``max_hits``.

    2. **Triggered + noise, no subsampling** — ``trig_name`` provided,
       ``unique=False``: Triggered pulses are taken first (up to ``max_hits``),
       then noise pulses are randomly sampled to fill the remainder.

    3. **First-hit per PMT + triggered + noise** — ``trig_name`` provided,
       ``unique=True``, and all of ``du_id``/``dom_id``/``channel_id`` present:
       First hit per PMT is selected (time-sorted), then triggered, then noise.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_hits: int = 300,
        trig_name: Optional[str] = "trig",
        unique: Optional[bool] = False,
    ) -> None:
        """Construct KM3NeTNodesAsTimeSeries.

        Args:
            input_feature_names: Column names for input features.
            max_hits: Maximum number of hits to keep per event.
            trig_name: Name of the trigger flag column. If None or not
                found in ``input_feature_names``, all pulses are treated
                as triggered.
            unique: If True and all PMT-id columns are present, keep only
                the first hit per PMT channel before further subsampling.
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        if input_feature_names is None:
            input_feature_names = [
                "pos_x", "pos_y", "pos_z",
                "dir_x", "dir_y", "dir_z",
                "t", "tot",
                "du_id", "dom_id", "channel_id",
                "trig",
            ]

        self.all_features = input_feature_names
        self.output_feature_names = input_feature_names
        # Column names used together to uniquely identify a single PMT channel
        self.ids = ['du_id', 'dom_id', 'channel_id']

        if trig_name not in input_feature_names:
            self._logger.warning(
                f"trig_name '{trig_name}' not found in input_feature_names. "
                f"Assuming only triggered pulses — no subsampling."
            )
            trig_name = None
        else:
            self._logger.info(
                f"trig_name '{trig_name}' found — both triggered and noise "
                f"pulses will be handled."
            )

        if unique:
            self._logger.info("Checking if first-hit-per-PMT selection is possible...")
            if all(id_col in input_feature_names for id_col in self.ids):
                self._logger.info(
                    "All PMT-id columns found. First-hit selection enabled."
                )
                self.first_hit_selection = True
            else:
                self._logger.warning(
                    "Not all PMT-id columns found. First-hit selection disabled."
                )
                self.first_hit_selection = False
        else:
            self._logger.info("First-hit-per-PMT selection disabled.")
            self.first_hit_selection = False

        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.input_feature_names = input_feature_names
        self.n_features = len(self.all_features)
        self.max_length = max_hits
        self.trig_name = trig_name

    @property
    def nb_outputs(self) -> int:
        """Return number of output feature columns."""
        return len(self.output_feature_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process raw hit features into a time-sorted, subsampled sequence.

        Args:
            x: Raw hit features, shape ``[num_pulses, n_features]``.

        Returns:
            Processed hit tensor, shape ``[<=max_hits, n_features]``.
        """
        event_length = x.shape[0]
        ids = torch.arange(event_length)

        # Reorder columns to match self.all_features ordering
        graph = torch.zeros([event_length, self.n_features])
        for idx, feature in enumerate(self.all_features):
            graph[:event_length, idx] = x[ids, self.feature_indexes[feature]]

        # Sort hits by arrival time (ascending)
        graph_time_sorted = graph[
            torch.argsort(graph[:, self.feature_indexes['t']])
        ]

        # Optionally keep only the first hit per PMT channel
        if self.first_hit_selection:
            graph_time_sorted = self._unique_selection(graph_time_sorted)

        event_length = graph_time_sorted.shape[0]
        return self._hits_sampler(graph_time_sorted, event_length)

    def _unique_selection(self, x: torch.Tensor) -> torch.Tensor:
        """Keep the earliest hit per (DU, DOM, channel) PMT combination."""
        pmt_ids = x[:, [
            self.feature_indexes['du_id'],
            self.feature_indexes['dom_id'],
            self.feature_indexes['channel_id']
        ]]
        _, first_hit_indices = unique(pmt_ids, dim=0)
        x_unique = x[first_hit_indices]
        # Re-sort by time after unique selection
        return x_unique[torch.argsort(x_unique[:, self.feature_indexes['t']])]

    def _hits_sampler(self, x: torch.Tensor, event_length: int) -> torch.Tensor:
        """Subsample hits: triggered pulses first, then random noise fill."""
        if self.trig_name is not None:
            ids = torch.arange(event_length)
            trig_idx = self.feature_indexes[self.trig_name]

            # Separate triggered and noise hits
            trigger_mask = torch.nonzero(x[:, trig_idx] != 0.0).squeeze(1)
            noise_mask = torch.nonzero(x[:, trig_idx] == 0.0).squeeze(1)

            # Take all triggered hits (up to max_hits)
            ids_trig = ids[trigger_mask][:min(self.max_length, len(trigger_mask))]

            if noise_mask.shape[0] > 0:
                # Randomly sample noise hits to fill remaining budget
                n_noise = min(self.max_length - len(ids_trig), len(noise_mask))
                rand_idx = torch.randint(low=0, high=len(noise_mask), size=(n_noise,))
                ids_noise = ids[noise_mask][rand_idx]
                ids = torch.cat([ids_trig, ids_noise])
            else:
                ids = ids_trig
        else:
            # No trigger column: random subsample if event exceeds budget
            if event_length < self.max_length:
                ids = torch.arange(event_length)
            else:
                ids = torch.randperm(event_length)

        ids = ids[:min(self.max_length, event_length)]
        # Return hits sorted by time
        return x[ids.sort().values]


class KM3NeTHitsSequence(nn.Module):
    """Full data pipeline: detector standardization → hit sequence construction.

    Wraps a ``Detector`` (for feature normalization/standardization) and a
    ``KM3NeTNodesAsTimeSeries`` (for hit selection and time sorting) into a
    single ``nn.Module`` used by the dataloader.
    """

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional["KM3NeTNodesAsTimeSeries"] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
    ) -> None:
        """Construct KM3NeTHitsSequence.

        Args:
            detector: Detector instance handling feature standardization.
            node_definition: Hit sequence builder; defaults to
                ``KM3NeTNodesAsTimeSeries()``.
            input_feature_names: Names of input feature columns. If None,
                uses all features defined in ``detector.feature_map()``.
            dtype: Tensor dtype for node features.
            perturbation_dict: Maps feature name → perturbation std dev for
                data augmentation. E.g. ``{'t': 1, 'charge': 0.25}``.
            seed: RNG seed (int) or numpy Generator for perturbations.
        """
        super().__init__()
        self._detector = detector
        self._node_definition = (
            node_definition if node_definition is not None
            else KM3NeTNodesAsTimeSeries()
        )
        self.dtype = dtype
        self._perturbation_dict = perturbation_dict

        if input_feature_names is None:
            input_feature_names = list(self._detector.feature_map().keys())
        self._input_feature_names = input_feature_names

        # Precompute column indices for features that will be perturbed
        if isinstance(self._perturbation_dict, dict):
            self._perturbation_cols = [
                self._input_feature_names.index(key)
                for key in self._perturbation_dict.keys()
            ]

        # Set up RNG for perturbations
        if seed is not None:
            if isinstance(seed, int):
                self.rng = default_rng(seed)
            elif isinstance(seed, Generator):
                self.rng = seed
            else:
                raise ValueError("seed must be an int or a numpy Generator.")
        else:
            self.rng = default_rng()

    def forward(
        self,
        input_features,
        input_feature_names,
        truth_dicts=None,
        loss_weight_column=None,
        loss_weight=None,
        loss_weight_default_value=None,
        data_path=None,
        custom_label_functions=None,
    ):
        """Run the full preprocessing pipeline for one event.

        Steps:
          1. Perturb features (data augmentation, if configured)
          2. Convert to tensor
          3. Detector standardization (normalisation per feature)
          4. Hit selection and time sorting
          5. Enforce output dtype

        Returns a dict with ``"x"`` (hit tensor) and ``"n_pulses"`` (int),
        plus any truth columns from ``truth_dicts``.
        """
        input_features = self._perturb_input(input_features)
        if isinstance(input_features, torch.Tensor):
            input_features = input_features.detach().clone().to(self.dtype)
        else:
            input_features = torch.tensor(input_features, dtype=self.dtype)
        input_features = self._detector(input_features, input_feature_names)
        input_features = self._node_definition(input_features)
        input_features = input_features.type(self.dtype)

        result = {
            "x": input_features,
            "n_pulses": torch.tensor(len(input_features), dtype=torch.int32),
        }
        if truth_dicts is not None:
            for truth_dict in truth_dicts:
                for key, val in truth_dict.items():
                    result[key] = val
        return result

    def _perturb_input(self, input_features):
        """Add Gaussian noise to selected feature columns (data augmentation)."""
        if isinstance(self._perturbation_dict, dict):
            stds = np.array(list(self._perturbation_dict.values()), dtype=float)
            perturbed = self.rng.normal(
                loc=input_features[:, self._perturbation_cols],
                scale=stds,
            )
            input_features[:, self._perturbation_cols] = torch.from_numpy(perturbed).float()
        return input_features
