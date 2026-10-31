"""Sequence-aware GRU.

GRU is used instead of LSTM specifically so this project does not read as
"fixing" the only existing GeNIS LSTM baseline (which was never given a
real sequence in the first place, see Related Work) -- it is a different
recurrent architecture applied to a genuinely new input (multi-flow
windows), not a variant of prior work on the same input.

Same padding caveat as cnn1d.py: left-padding is zero-valued and not
masked out of the recurrence (no pack_padded_sequence). This means the GRU
processes padded positions like real (zero) input at the start of a
sequence; the effect of this is checked qualitatively in the error
analysis, not assumed to be negligible.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from src.models.torch_common import TorchClassifierWrapper


class GRUClassifier(nn.Module):
    """GRU over the sequence axis, using the final hidden state for classification."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        n_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * directions, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits: (batch, n_classes)."""
        _, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, batch, hidden_size); take the
        # last layer's final hidden state(s).
        if h_n.shape[0] > 1:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1) if h_n.shape[0] >= 2 else h_n[-1]
        else:
            last = h_n[-1]
        return self.head(self.dropout(last))


def build_gru_wrapper(
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    class_weights: Dict[int, float] | None = None,
    sequence_length: int | None = None,
) -> TorchClassifierWrapper:
    """Build a GRU wrapped in the common torch training/inference interface.

    Args:
        hyperparameters: The "hyperparameters" block from
            configs/models/gru.yaml.
        n_features: Number of features per flow (per timestep).
        n_classes: Number of target classes.
        class_weights: Optional per-class loss weights, from training split.
        sequence_length: Present for interface parity with cnn1d's builder;
            GRU itself does not need a fixed sequence length at construction
            time, but the E05 ablation script still passes it for symmetry
            and clarity in logs.

    Returns:
        A TorchClassifierWrapper for sequence input of shape
        (n, sequence_length, n_features).
    """
    model = GRUClassifier(
        n_features=n_features,
        hidden_size=hyperparameters["hidden_size"],
        num_layers=hyperparameters.get("num_layers", 1),
        bidirectional=hyperparameters.get("bidirectional", False),
        n_classes=n_classes,
        dropout=hyperparameters.get("dropout", 0.2),
    )
    return TorchClassifierWrapper(model, hyperparameters, n_classes, class_weights)
