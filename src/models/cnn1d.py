"""Sequence-aware 1D-CNN.

Convolves across the sequence-of-flows axis (not across features), so it
learns local temporal patterns across nearby flows within a window.

Padding note: left-padding produced by src/data/sequences.py::build_sequences
is zero-valued. This model does not mask out padded positions internally
(no attention mask, no pack_padded_sequence) -- the convolution simply sees
zero vectors at padded positions like any other input. This is a
deliberate simplification for a laptop-scale project, stated here rather
than silently assumed, and its effect (if any) should be checked in the
error analysis by comparing performance on early-in-scenario flows, which
have the most padding, against later flows, which have none.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from src.models.torch_common import TorchClassifierWrapper


class CNN1D(nn.Module):
    """Stack of 1D convolutions over the sequence axis, then a linear head."""

    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        num_channels: list[int],
        kernel_size: int,
        n_classes: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_features
        for out_channels in num_channels:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits: (batch, n_classes)."""
        x = x.permute(0, 2, 1)  # -> (batch, n_features, seq_len) for Conv1d
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # -> (batch, channels)
        return self.head(x)


def build_cnn1d_wrapper(
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    class_weights: Dict[int, float] | None = None,
    sequence_length: int | None = None,
) -> TorchClassifierWrapper:
    """Build a 1D-CNN wrapped in the common torch training/inference interface.

    Args:
        hyperparameters: The "hyperparameters" block from
            configs/models/cnn1d.yaml.
        n_features: Number of features per flow (per timestep).
        n_classes: Number of target classes.
        class_weights: Optional per-class loss weights, from training split.
        sequence_length: Overrides hyperparameters["sequence_length"] when
            given, so the E05 ablation script can sweep this without
            editing the YAML file per run.

    Returns:
        A TorchClassifierWrapper for sequence input of shape
        (n, sequence_length, n_features).
    """
    seq_len = sequence_length or hyperparameters["sequence_length"]
    model = CNN1D(
        n_features=n_features,
        sequence_length=seq_len,
        num_channels=hyperparameters["num_channels"],
        kernel_size=hyperparameters["kernel_size"],
        n_classes=n_classes,
        dropout=hyperparameters.get("dropout", 0.2),
    )
    return TorchClassifierWrapper(model, hyperparameters, n_classes, class_weights)
