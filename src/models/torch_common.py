"""Shared training/inference loop for the torch-based models (MLP, 1D-CNN, GRU).

Each specific model (src/models/mlp.py, cnn1d.py, gru.py) defines only its
network architecture (an nn.Module) and hands it to TorchClassifierWrapper,
which implements the common training loop: batching, class-weighted loss,
early stopping on validation macro-F1, device placement, and a
scikit-learn-like fit/predict/predict_proba interface so
src/training/train.py can treat every model family uniformly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TorchClassifierWrapper:
    """Wraps an nn.Module with a scikit-learn-like fit/predict interface.

    Attributes:
        model: The underlying nn.Module (built by the caller).
        device: "cuda" if available, else "cpu".
    """

    def __init__(
        self,
        model: nn.Module,
        hyperparameters: Dict[str, Any],
        n_classes: int,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        self.model = model
        self.hp = hyperparameters
        self.n_classes = n_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if class_weights is not None:
            weight_tensor = torch.zeros(n_classes, dtype=torch.float32)
            for cls_idx, w in class_weights.items():
                weight_tensor[int(cls_idx)] = w
            weight_tensor = weight_tensor.to(self.device)
        else:
            weight_tensor = None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters.get("weight_decay", 0.0),
        )
        self.history: Dict[str, list] = {"train_loss": [], "val_macro_f1": []}

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.hp["batch_size"], shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TorchClassifierWrapper":
        """Train with early stopping on validation macro-F1.

        Args:
            X_train: Training features. Shape (n, n_features) for MLP, or
                (n, seq_len, n_features) for sequence models.
            y_train: Integer-encoded training labels, shape (n,).
            X_val: Validation features, same shape convention as X_train.
            y_val: Validation labels.

        Returns:
            self, with self.model set to the best-validation-F1 weights
            seen during training (early stopping restores these, not the
            final epoch's weights).
        """
        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        best_val_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0
        patience = self.hp.get("early_stopping_patience", 5)
        max_epochs = self.hp["max_epochs"]

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)

            val_preds = self.predict(X_val)
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

            self.history["train_loss"].append(mean_loss)
            self.history["val_macro_f1"].append(val_f1)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_macro_f1=%.4f",
                epoch + 1,
                max_epochs,
                mean_loss,
                val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no val improvement for %d epochs).",
                        epoch + 1,
                        patience,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for X.

        Args:
            X: Feature array, same shape convention used in fit().

        Returns:
            Array of shape (n, n_classes) of softmax probabilities.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the argmax class prediction for X."""
        return np.argmax(self.predict_proba(X), axis=1)

    def measure_inference_latency(
        self, X: np.ndarray, batch_size: int = 1, n_repeats: int = 100
    ) -> Dict[str, float]:
        """Measure per-sample inference latency at a given batch size.

        Args:
            X: Feature array to draw inference batches from.
            batch_size: Batch size to measure at (1 for single-flow
                real-time latency, per RQ3).
            n_repeats: Number of batches to time and average over.

        Returns:
            Dict with mean_latency_ms and std_latency_ms per batch.
        """
        self.model.eval()
        n = min(batch_size, X.shape[0])
        sample = torch.tensor(X[:n], dtype=torch.float32).to(self.device)

        # Warm-up run, excluded from timing (first-call overhead is not
        # representative of steady-state inference cost).
        with torch.no_grad():
            _ = self.model(sample)

        timings = []
        with torch.no_grad():
            for _ in range(n_repeats):
                start = time.perf_counter()
                _ = self.model(sample)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

        return {
            "mean_latency_ms": float(np.mean(timings)),
            "std_latency_ms": float(np.std(timings)),
            "batch_size": batch_size,
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
