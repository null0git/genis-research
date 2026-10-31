"""Flow-independent MLP: the deep-learning counterpart to RF/XGBoost, and
the flow-independent comparison point for the sequence-aware 1D-CNN/GRU
models (see Section 5 of STAGE2_PROPOSAL_AND_METHODOLOGY.md for why an MLP
specifically is needed as this comparison point).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from src.models.torch_common import TorchClassifierWrapper


class MLP(nn.Module):
    """Simple feedforward network: Linear -> ReLU -> Dropout stack."""

    def __init__(self, n_features: int, hidden_sizes: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) -> logits: (batch, n_classes)."""
        return self.net(x)


def build_mlp_wrapper(
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    class_weights: Dict[int, float] | None = None,
) -> TorchClassifierWrapper:
    """Build an MLP wrapped in the common torch training/inference interface.

    Args:
        hyperparameters: The "hyperparameters" block from
            configs/models/mlp.yaml.
        n_features: Number of input features.
        n_classes: Number of target classes.
        class_weights: Optional per-class weights for the loss function,
            computed on the training split only.

    Returns:
        A TorchClassifierWrapper ready for .fit(X_train, y_train, X_val, y_val).
    """
    model = MLP(
        n_features=n_features,
        hidden_sizes=hyperparameters["hidden_sizes"],
        n_classes=n_classes,
        dropout=hyperparameters.get("dropout", 0.2),
    )
    return TorchClassifierWrapper(model, hyperparameters, n_classes, class_weights)
