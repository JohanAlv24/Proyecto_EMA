"""
VAE.py — Variational Autoencoder for UCR time-series archive.

Importable API
--------------
    from VAE import TimeSeriesVAE, VAETrainer, build_global_ucr_matrix, TimeSeriesVAEConfig

Quick start
-----------
    from VAE import TimeSeriesVAEConfig, TimeSeriesVAE, VAETrainer, build_global_ucr_matrix

    cfg   = TimeSeriesVAEConfig(input_length=256, latent_dim=32)
    model = TimeSeriesVAE(cfg)

    X, y  = build_global_ucr_matrix(["ArrowHead", "ECG200"], target_length=256)

    trainer = VAETrainer(model)
    trainer.fit(X, epochs=50)

    import torch
    x_batch = torch.from_numpy(X[:8])          # (8, 256)
    mu, sigma = model.encode(x_batch)          # (8, 32) each
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


__all__ = [
    "TimeSeriesVAEConfig",
    "TimeSeriesVAE",
    "VAETrainer",
    "load_ucr_dataset",
    "build_global_ucr_matrix",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesVAEConfig:
    """Hyperparameters for :class:`TimeSeriesVAE`.

    Args:
        input_length: Fixed series length after padding / truncation.
                      Must be divisible by 8.
        latent_dim:   Dimensionality of the latent space (size of mu / sigma).
        beta:         KL weight in the ELBO loss. 1.0 = standard VAE.
    """
    input_length: int = 256
    latent_dim: int = 32
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.input_length % 8 != 0:
            raise ValueError(
                f"input_length must be divisible by 8, got {self.input_length}"
            )
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")


# ---------------------------------------------------------------------------
# UCR data utilities
# ---------------------------------------------------------------------------

def load_ucr_dataset(
    name: str,
    archive_path: str = "UCRArchive_2018",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single UCR dataset by name.

    Args:
        name:         Dataset folder name, e.g. ``"ArrowHead"``.
        archive_path: Path to the UCRArchive_2018 root folder.

    Returns:
        ``(X_train, y_train, X_test, y_test)`` — ``X_*`` are float32 arrays of
        shape ``(n_samples, series_length)``; ``y_*`` are 1-D label arrays.

    Raises:
        FileNotFoundError: If the expected TSV files do not exist.
    """
    base = os.path.join(archive_path, name, name)
    train_path = base + "_TRAIN.tsv"
    test_path = base + "_TEST.tsv"

    for p in (train_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Dataset file not found: {p}")

    def _read(path: str) -> Tuple[np.ndarray, np.ndarray]:
        arr = pd.read_csv(path, sep="\t", header=None).values.astype(np.float32)
        labels = arr[:, 0]
        series = arr[:, 1:]
        return series, labels

    X_train, y_train = _read(train_path)
    X_test, y_test = _read(test_path)
    return X_train, y_train, X_test, y_test


def _zscore(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each series independently along the time axis."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def _pad_or_truncate(X: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros on the right or truncate each row to *target_length*."""
    n, current = X.shape
    if current == target_length:
        return X
    if current > target_length:
        return X[:, :target_length]
    pad = np.zeros((n, target_length - current), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def _preprocess(X: np.ndarray, target_length: int) -> np.ndarray:
    """Z-score then pad / truncate to *target_length*."""
    return _pad_or_truncate(_zscore(X), target_length)


def build_global_ucr_matrix(
    datasets: List[str],
    archive_path: str = "UCRArchive_2018",
    target_length: int = 256,
    use_test: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate multiple UCR datasets into a single normalized matrix.

    Each dataset is Z-score normalized independently before concatenation.
    Series are padded or truncated to *target_length*.

    Args:
        datasets:      List of UCR dataset names to include.
        archive_path:  Path to the UCRArchive_2018 root folder.
        target_length: Fixed output series length. Must be divisible by 8.
        use_test:      If True, also include test splits. Default False.

    Returns:
        ``(X, y)`` — float32 array ``(total_samples, target_length)`` and
        matching label array.

    Raises:
        ValueError: If *datasets* is empty or *target_length* is invalid.
    """
    if not datasets:
        raise ValueError("datasets list must not be empty")
    if target_length % 8 != 0:
        raise ValueError(
            f"target_length must be divisible by 8, got {target_length}"
        )

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for name in datasets:
        X_tr, y_tr, X_te, y_te = load_ucr_dataset(name, archive_path)
        X_proc = _preprocess(X_tr, target_length)
        y_proc = y_tr
        if use_test:
            X_proc = np.concatenate([X_proc, _preprocess(X_te, target_length)], axis=0)
            y_proc = np.concatenate([y_proc, y_te], axis=0)
        X_parts.append(X_proc)
        y_parts.append(y_proc)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimeSeriesVAE(nn.Module):
    """1-D Convolutional Variational Autoencoder for time series.

    The encoder maps each input series to a Gaussian distribution in latent
    space defined by mean **mu** and standard deviation **sigma**.
    A sample **z** is drawn via the reparameterization trick during training.

    Architecture (default input_length=256, latent_dim=32)
    -------------------------------------------------------
    Encoder
        Conv1d(1→32, k=3, s=2) ─ReLU─ (B,32,128)
        Conv1d(32→64, k=3, s=2) ─ReLU─ (B,64,64)
        Conv1d(64→128, k=3, s=2) ─ReLU─ (B,128,32)
        Flatten → (B,4096)
        Linear → mu  (B,32)
        Linear → logvar  (B,32)   [sigma = exp(0.5·logvar)]

    Decoder
        Linear(32→4096) → reshape (B,128,32)
        ConvTranspose1d(128→64, k=4, s=2) ─ReLU─ (B,64,64)
        ConvTranspose1d(64→32, k=4, s=2) ─ReLU─ (B,32,128)
        ConvTranspose1d(32→1, k=4, s=2) → (B,1,256)

    Args:
        config: :class:`TimeSeriesVAEConfig` instance.
    """

    def __init__(self, config: Optional[TimeSeriesVAEConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = TimeSeriesVAEConfig()
        self.config = config
        L = config.input_length
        D = config.latent_dim
        self._flat_size: int = 128 * (L // 8)

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self._flat_size, D)
        self.fc_logvar = nn.Linear(self._flat_size, D)

        # Decoder
        self.fc_decode = nn.Linear(D, self._flat_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_raw(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) given input x of shape (B,1,L)."""
        h = self.encoder_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of time series into latent distribution parameters.

        Args:
            x: Float tensor of shape ``(batch, 1, input_length)`` or
               ``(batch, input_length)`` — the channel dim is added automatically.

        Returns:
            ``(mu, sigma)`` — both ``(batch, latent_dim)``.
            *mu*    is the latent mean.
            *sigma* is the latent standard deviation  (``exp(0.5 · logvar)``).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mu, logvar = self._encode_raw(x)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to a reconstructed time series.

        Args:
            z: Tensor ``(batch, latent_dim)``.

        Returns:
            Reconstructed series ``(batch, 1, input_length)``.
        """
        h = self.fc_decode(z).view(z.size(0), 128, -1)
        return self.decoder_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: ``(batch, 1, input_length)`` or ``(batch, input_length)``.

        Returns:
            ``(x_recon, mu, sigma, z)``

            - *x_recon* ``(batch, 1, input_length)`` — reconstructed series.
            - *mu*      ``(batch, latent_dim)``       — latent mean.
            - *sigma*   ``(batch, latent_dim)``       — latent standard deviation.
            - *z*       ``(batch, latent_dim)``       — sampled latent vector.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mu, logvar = self._encode_raw(x)
        sigma = torch.exp(0.5 * logvar)
        z = self._reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, sigma, z

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        beta: Optional[float] = None,
    ) -> torch.Tensor:
        """ELBO loss = MSE reconstruction + beta * KL divergence.

        KL(N(mu,sigma) || N(0,1)) = 0.5 · sum(sigma² + mu² - 1 - log sigma²)

        Args:
            x:       Original input ``(batch, 1, input_length)``.
            x_recon: Reconstructed input, same shape as *x*.
            mu:      Latent mean ``(batch, latent_dim)``.
            sigma:   Latent std  ``(batch, latent_dim)``.
            beta:    KL weight. Defaults to ``config.beta``.

        Returns:
            Scalar loss tensor.
        """
        if beta is None:
            beta = self.config.beta
        recon = F.mse_loss(x_recon, x, reduction="sum")
        kl = 0.5 * torch.sum(
            sigma.pow(2) + mu.pow(2) - 1.0 - 2.0 * torch.log(sigma + 1e-8)
        )
        return recon + beta * kl


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VAETrainer:
    """Train, evaluate, save, and load a :class:`TimeSeriesVAE`.

    Args:
        model:         A :class:`TimeSeriesVAE` instance.
        learning_rate: Adam learning rate. Default ``1e-3``.
        device:        ``"cpu"``, ``"cuda"``, or ``"auto"`` (auto-detect).
    """

    def __init__(
        self,
        model: TimeSeriesVAE,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one full training epoch over *loader*.

        Returns:
            Mean per-sample loss for this epoch.
        """
        self.model.train()
        total_loss, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(self.device)
            if batch.dim() == 2:
                batch = batch.unsqueeze(1)
            x_recon, mu, sigma, _ = self.model(batch)
            loss = self.model.loss_function(batch, x_recon, mu, sigma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += batch.size(0)
        return total_loss / max(n, 1)

    def evaluate(self, loader: DataLoader) -> float:
        """Compute mean per-sample loss on *loader* without weight updates.

        Returns:
            Mean per-sample loss.
        """
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if batch.dim() == 2:
                    batch = batch.unsqueeze(1)
                x_recon, mu, sigma, _ = self.model(batch)
                loss = self.model.loss_function(batch, x_recon, mu, sigma)
                total_loss += loss.item()
                n += batch.size(0)
        return total_loss / max(n, 1)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.1,
        verbose: bool = True,
        early_stopping_patience: int = 0,
    ) -> Dict[str, list]:
        """Train the VAE on a numpy array of pre-processed time series.

        Args:
            X:                       Float32 array ``(n_samples, series_length)``.
                                     *series_length* must equal ``config.input_length``.
            epochs:                  Number of training epochs.
            batch_size:              Mini-batch size.
            val_split:               Fraction held out for validation. 0 = no val.
            verbose:                 Print epoch summaries when True.
            early_stopping_patience: Stop if val loss does not improve for this many
                                     consecutive epochs. 0 = disabled.

        Returns:
            Dict with ``"train_loss"`` and ``"val_loss"`` lists (one entry per epoch).

        Raises:
            ValueError: If X series length does not match ``config.input_length``.
        """
        if X.ndim != 2 or X.shape[1] != self.model.config.input_length:
            raise ValueError(
                f"X must have shape (n_samples, {self.model.config.input_length}), "
                f"got {X.shape}. Pre-process with build_global_ucr_matrix first."
            )

        X_t = torch.from_numpy(X.astype(np.float32))
        n_val = max(1, int(len(X_t) * val_split)) if val_split > 0.0 else 0
        n_train = len(X_t) - n_val
        if n_train < 1:
            raise ValueError("Not enough samples for training after val split.")

        idx = torch.randperm(len(X_t))
        X_train, X_val = X_t[idx[:n_train]], X_t[idx[n_train:]] if n_val > 0 else None

        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=batch_size, shuffle=True
        )
        val_loader = (
            DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
            if X_val is not None
            else None
        )

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_count = 0

        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(train_loader)
            history["train_loss"].append(tr)
            val_str = ""

            if val_loader is not None:
                vl = self.evaluate(val_loader)
                history["val_loss"].append(vl)
                val_str = f"  val_loss={vl:.4f}"

                if early_stopping_patience > 0:
                    if vl < best_val:
                        best_val, patience_count = vl, 0
                    else:
                        patience_count += 1
                    if patience_count >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}.")
                        break

            if verbose:
                print(f"Epoch [{epoch:>4}/{epochs}]  train_loss={tr:.4f}{val_str}")

        return history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to *path* (.pt file).

        Args:
            path: Destination file path (directories are created if needed).
        """
        dir_ = os.path.dirname(os.path.abspath(path))
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        cfg = self.model.config
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_length": cfg.input_length,
                    "latent_dim": cfg.latent_dim,
                    "beta": cfg.beta,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint saved with :meth:`save`.

        Args:
            path: Path to the .pt checkpoint file.

        Raises:
            ValueError: If the checkpoint config does not match the current model.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        saved = checkpoint["config"]  # plain dict
        current = self.model.config
        if (
            saved["input_length"] != current.input_length
            or saved["latent_dim"] != current.latent_dim
        ):
            raise ValueError(
                f"Checkpoint config {saved} does not match current model config "
                f"(input_length={current.input_length}, latent_dim={current.latent_dim})."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
