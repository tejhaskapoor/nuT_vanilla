"""Standalone training module for nuT models.

Replaces graphnet's StandardModel, tasks, and loss functions.
No graphnet imports required.
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
import pytorch_lightning as pl


# ============================================================================
# Utility
# ============================================================================

def eps_like(tensor: Tensor) -> float:
    """Return eps matching tensor's dtype."""
    return torch.finfo(tensor.dtype).eps


# ============================================================================
# Loss Functions
# ============================================================================

class LossFunction(nn.Module):
    """Base class for loss functions with optional per-event weights."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        ...


class LogCoshLoss(LossFunction):
    """Log-cosh loss. Acts like x^2 for small x; like |x| for large x."""

    @staticmethod
    def _log_cosh(x: Tensor) -> Tensor:
        return x + F.softplus(-2.0 * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return self._log_cosh(diff)


class BinaryCrossEntropyWithLogitsLoss(LossFunction):
    """Binary cross entropy from logits."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )


class LogCMK(torch.autograd.Function):
    """Numerically stable computation of log C_m(κ), the vMF normalisation constant.

    C_m(κ) = κ^(m/2-1) / ((2π)^(m/2) · I_{m/2-1}(κ))
    where I_ν is the modified Bessel function of the first kind.

    For large κ the Bessel function overflows in float32, so this uses a
    custom autograd Function that evaluates in float64 and casts back.

    MIT License — Copyright (c) 2019 Max Ryabinin
    Source: https://github.com/mryab/vmf_loss/blob/master/losses.py
    """

    @staticmethod
    def forward(ctx: Any, m: int, kappa: Tensor) -> Tensor:
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            scipy.special.iv(m / 2.0, kappa)
            / scipy.special.iv(m / 2.0 - 1, kappa)
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisher3DLoss(LossFunction):
    """Von Mises-Fisher loss for 3D direction reconstruction."""

    @classmethod
    def log_cmk_exact(cls, m: int, kappa: Tensor) -> Tensor:
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(cls, m: int, kappa: Tensor) -> Tensor:
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa ** 2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(cls, m: int, kappa: Tensor, kappa_switch: float = 100.0) -> Tensor:
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(m, kappa_switch)
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """prediction: [N, 4] = [dir_x, dir_y, dir_z, kappa], target: [N, 3]."""
        target = target.reshape(-1, 3)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]

        # Von Mises-Fisher loss
        m = target.size()[1]  # = 3
        k = torch.norm(p, dim=1)
        dotprod = torch.sum(p * target, dim=1)
        return -self.log_cmk(m, k) - dotprod


# ============================================================================
# Task Heads
# ============================================================================

class Task(nn.Module):
    """Base class for task heads.

    A task maps backbone output [N, hidden_size] -> task-specific predictions.
    Contains a linear projection (prediction head) + output transform.
    """

    def __init__(
        self,
        hidden_size: int,
        loss_function: LossFunction,
        *,
        target_labels: Optional[Union[str, List[str]]] = None,
        prediction_labels: Optional[Union[str, List[str]]] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        loss_weight: Optional[str] = None,
    ):
        super().__init__()
        self._loss_function = loss_function
        self._loss_weight = loss_weight
        self._inference = False

        # Target/prediction labels
        if target_labels is None:
            target_labels = self.default_target_labels
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        self._target_labels = target_labels

        if prediction_labels is None:
            prediction_labels = getattr(self, 'default_prediction_labels',
                                        [f"{t}_pred" for t in target_labels])
        if isinstance(prediction_labels, str):
            prediction_labels = [prediction_labels]
        self._prediction_labels = prediction_labels

        # Transforms
        self._transform_prediction_training: Callable = lambda x: x
        self._transform_prediction_inference: Callable = lambda x: x
        self._transform_target: Callable = lambda x: x

        if transform_prediction_and_target is not None:
            self._transform_prediction_training = transform_prediction_and_target
            self._transform_target = transform_prediction_and_target
        else:
            if transform_target is not None:
                self._transform_target = transform_target
            if transform_inference is not None:
                self._transform_prediction_inference = transform_inference

        # Linear projection: hidden_size -> nb_inputs
        self._affine = Linear(hidden_size, self.nb_inputs)

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Number of output features for the linear head."""
        ...

    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor:
        """Task-specific output transform after linear projection."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        self._regularisation_loss = 0
        x = self._affine(x)
        x = self._forward(x)
        if self._inference:
            return self._transform_prediction_inference(x)
        return self._transform_prediction_training(x)

    def compute_loss(self, pred: Tensor, data: dict) -> Tensor:
        """Compute loss given predictions and a dict of target tensors."""
        target = torch.stack(
            [data[label] for label in self._target_labels], dim=1
        )
        target = self._transform_target(target)
        weights = data.get(self._loss_weight) if self._loss_weight else None
        return self._loss_function(pred, target, weights=weights)

    def inference(self) -> None:
        self._inference = True

    def train_eval(self) -> None:
        self._inference = False


class EnergyReconstruction(Task):
    """Reconstructs energy using softplus."""

    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return F.softplus(x, beta=0.05) + eps_like(x)


class DirectionReconstructionWithKappa(Task):
    """Reconstructs 3D direction with kappa (uncertainty) from vMF distribution."""

    default_target_labels = ["direction"]
    default_prediction_labels = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa"]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class BinaryClassificationTask(Task):
    """Binary classification with sigmoid output."""

    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)


class BinaryClassificationTaskLogits(Task):
    """Binary classification returning raw logits."""

    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return x


# ============================================================================
# Standalone Training Model (replaces graphnet's StandardModel)
# ============================================================================

class NuTStandardModel(pl.LightningModule):
    """Standalone training model combining backbone + task heads.

    Replaces graphnet's StandardModel. Uses PyTorch Lightning directly.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tasks: Union[Task, List[Task]],
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self._tasks = ModuleList(tasks if isinstance(tasks, list) else [tasks])
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}

    @property
    def target_labels(self) -> List[str]:
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        return [label for task in self._tasks for label in task._prediction_labels]

    def forward(self, data) -> List[Tensor]:
        """Forward pass: backbone -> tasks."""
        if isinstance(data, list):
            x_list = [self.backbone(d) for d in data]
            x = torch.cat(x_list, dim=0)
        else:
            x = self.backbone(data)
        return [task(x) for task in self._tasks]

    def compute_loss(self, preds: List[Tensor], data) -> Tensor:
        """Compute and sum losses across tasks."""
        # Merge data from list of Data objects into a single dict
        if isinstance(data, list):
            data_merged = {}
            target_labels = list(set(self.target_labels))
            for label in target_labels:
                data_merged[label] = torch.cat(
                    [d[label] for d in data], dim=0
                )
            for task in self._tasks:
                if task._loss_weight is not None:
                    data_merged[task._loss_weight] = torch.cat(
                        [d[task._loss_weight] for d in data], dim=0
                    )
        else:
            # Single Data object — access attributes as dict
            data_merged = data

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        return torch.sum(torch.stack(losses))

    def shared_step(self, batch, batch_idx: int) -> Tensor:
        if not isinstance(batch, list):
            batch = [batch]
        preds = self(batch)
        return self.compute_loss(preds, batch)

    def training_step(self, train_batch, batch_idx: int) -> Tensor:
        if not isinstance(train_batch, list):
            train_batch = [train_batch]
        loss = self.shared_step(train_batch, batch_idx)
        # batch_size from n_pulses tensor length (number of events in the batch)
        batch_size = train_batch[0]["n_pulses"].shape[0]
        self.log("train_loss", loss, batch_size=batch_size,
                 prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx: int) -> Tensor:
        if not isinstance(val_batch, list):
            val_batch = [val_batch]
        loss = self.shared_step(val_batch, batch_idx)
        batch_size = val_batch[0]["n_pulses"].shape[0]
        self.log("val_loss", loss, batch_size=batch_size,
                 prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                **self._scheduler_config,
            }
        return config

    def inference(self) -> None:
        """Activate inference mode on all tasks."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "NuTStandardModel":
        """Override to sync task train/inference state."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self
