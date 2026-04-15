"""
VAE.py — Variational Autoencoder for UCR time-series archive.

This module provides a VAE implementation aligned with:
"Latent Network Construction for Univariate Time Series Based on
Variational Auto-Encode" (Sun et al.).

Design choices to match the paper as closely as possible:
- Encoder/decoder are symmetric 1-layer LSTMs.
- Hidden size is 96 in both encoder and decoder.
- Latent dimensions (Z_mu and Z_sigma) are 20.
- Adam optimizer learning rate defaults to 0.0005.
- ECG5000 helper split: 4500 samples for VAE training and 500 for latent-network construction.
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
    "preprocess_ucr_series",
    "build_global_ucr_matrix",
    "build_ecg5000_paper_split",
    "mean_absolute_percentage_error",
    "mape_original_scale",
    "train_single_dataset"
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesVAEConfig:
    """Hyperparameters for :class:`TimeSeriesVAE`.

    Args:
        input_length: Fixed series length after padding / truncation.
                      Paper ECG5000 uses 140.
        latent_dim:   Dimensionality of latent vectors (Z_mu and Z_sigma).
        hidden_size:  Number of LSTM hidden units in both encoder and decoder.
        num_layers:   Number of LSTM layers. Paper uses 1.
        beta:         KL weight in ELBO loss. 1.0 = standard VAE.
    """

    input_length: int = 140
    latent_dim: int = 20
    hidden_size: int = 96
    num_layers: int = 1
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.input_length < 2:
            raise ValueError("input_length must be >= 2")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
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
        name:         Dataset folder name, e.g. "ArrowHead".
        archive_path: Path to the UCRArchive_2018 root folder.

    Returns:
        (X_train, y_train, X_test, y_test) where X_* are float32 arrays of
        shape (n_samples, series_length), and y_* are 1-D labels.
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


def _get_series_statistics(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for each series before normalization.
    
    Returns:
        (means, stds) where means and stds have shape (n_samples, 1).
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _unzscore(X_normalized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse Z-score normalization to recover original scale.
    
    Args:
        X_normalized: Normalized series of shape (n_samples, series_length).
        mean: Per-series means of shape (n_samples, 1).
        std: Per-series stds of shape (n_samples, 1).
    
    Returns:
        Series in original scale.
    """
    return X_normalized * std + mean


def _pad_or_truncate(X: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros on the right or truncate each row to target_length."""

    n, current = X.shape
    if current == target_length:
        return X
    if current > target_length:
        return X[:, :target_length]
    pad = np.zeros((n, target_length - current), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def _preprocess(X: np.ndarray, target_length: int) -> np.ndarray:
    """Z-score then pad / truncate to target_length."""

    return _pad_or_truncate(_zscore(X), target_length)


def preprocess_ucr_series(X: np.ndarray, target_length: int) -> np.ndarray:
    """Public wrapper for time-series preprocessing used across scripts."""

    return _preprocess(X, target_length)


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute Symmetric MAPE (sMAPE %) using per-series averaging.
    
    sMAPE formula: 2 * |y_true - y_pred| / (|y_true| + |y_pred|)
    Range: [0, 200%] — stable near zero, no denominador explosion.
    
    Each series gets one sMAPE value by averaging along time, and the final value
    is the mean across all series.
    """

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true_arr.shape} and {y_pred_arr.shape}"
        )
    if y_true_arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays (n_series, series_length), got {y_true_arr.ndim}-D"
        )
    if y_true_arr.shape[0] == 0:
        return float("nan")

    numerator =  np.abs(y_true_arr - y_pred_arr)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + epsilon
    per_point = numerator / denominator
    per_series = per_point.mean(axis=1)
    return float(per_series.mean() * 100.0)


def mape_original_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute MAPE (Mean Absolute Percentage Error) in original scale.
    
    MAPE formula: 100 * mean(|y_true - y_pred| / (|y_true| + epsilon))
    Stable for values near zero with epsilon regularization.
    
    Args:
        y_true: Ground truth in original scale, shape (n_series, series_length).
        y_pred: Predictions in original scale, shape (n_series, series_length).
        epsilon: Small constant to avoid division by zero.
    
    Returns:
        MAPE percentage value.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true_arr.shape} and {y_pred_arr.shape}"
        )
    if y_true_arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays (n_series, series_length), got {y_true_arr.ndim}-D"
        )
    if y_true_arr.shape[0] == 0:
        return float("nan")
    
    numerator = np.abs(y_true_arr - y_pred_arr)
    denominator = np.abs(y_true_arr) + epsilon
    per_point = numerator / denominator
    per_series = per_point.mean(axis=1)
    return float(per_series.mean() * 100.0)


def build_global_ucr_matrix(
    datasets: List[str],
    archive_path: str = "UCRArchive_2018",
    target_length: int = 140,
    use_test: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate multiple UCR datasets into one normalized matrix.

    Each dataset is Z-score normalized independently before concatenation.
    Series are padded or truncated to target_length.

    Args:
        datasets:      List of UCR dataset names to include.
        archive_path:  Path to the UCRArchive_2018 root folder.
        target_length: Fixed output length for each series.
        use_test:      If True, include test split in addition to train.

    Returns:
        (X, y): X float32 array (total_samples, target_length), and labels y.
    """

    if not datasets:
        raise ValueError("datasets list must not be empty")
    if target_length < 2:
        raise ValueError("target_length must be >= 2")

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


def build_ecg5000_paper_split(
    archive_path: str = "UCRArchive_2018",
    target_length: int = 140,
    train_size: int = 4500,
    latent_network_size: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ECG5000 split following Sun et al. experiment protocol.

    Protocol:
    - Merge ECG5000 train and test splits (total 5000 samples).
    - Randomly select 4500 samples for VAE training.
    - Use the remaining 500 to construct/evaluate latent-space network.

    Returns:
        (X_vae_train, y_vae_train, X_latent_net, y_latent_net)
    """

    X_tr, y_tr, X_te, y_te = load_ucr_dataset("ECG5000", archive_path)
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    X_all = _preprocess(X_all, target_length).astype(np.float32)

    total_required = train_size + latent_network_size
    if len(X_all) < total_required:
        raise ValueError(
            f"ECG5000 has {len(X_all)} samples, but {total_required} are required"
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    X_vae = X_all[:train_size]
    y_vae = y_all[:train_size]
    X_latent = X_all[train_size:total_required]
    y_latent = y_all[train_size:total_required]
    return X_vae, y_vae, X_latent, y_latent


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TimeSeriesVAE(nn.Module):
    """LSTM-based Variational Autoencoder for univariate time series.

    Paper-like architecture:
    - Encoder: 1-layer LSTM with 96 hidden units.
    - Decoder: 1-layer LSTM with 96 hidden units.
    - Latent vectors Z_mu and Z_sigma of dimension 20.

    Input shape accepted by public methods:
    - (batch, input_length) or (batch, 1, input_length)

    Internal sequence format for LSTM:
    - (batch, seq_len, 1)
    """

    def __init__(self, config: Optional[TimeSeriesVAEConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = TimeSeriesVAEConfig()
        self.config = config

        D = config.latent_dim
        H = config.hidden_size
        N = config.num_layers

        # Encoder: x_t (scalar) -> hidden sequence
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=H,
            num_layers=N,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(H, D)
        self.fc_logvar = nn.Linear(H, D)

        # Decoder: z (repeated across time) -> hidden sequence -> x_hat_t
        self.decoder_lstm = nn.LSTM(
            input_size=D,
            hidden_size=H,
            num_layers=N,
            batch_first=True,
        )
        self.z_to_h0 = nn.Linear(D, H)
        self.z_to_c0 = nn.Linear(D, H)
        self.output_layer = nn.Linear(H, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_lstm_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to (B, L, 1) for LSTM processing."""

        if x.dim() == 2:
            # (B, L) -> (B, L, 1)
            return x.unsqueeze(-1)
        if x.dim() == 3 and x.size(1) == 1:
            # (B, 1, L) -> (B, L, 1)
            return x.transpose(1, 2)
        if x.dim() == 3 and x.size(-1) == 1:
            return x
        raise ValueError(
            "Input must have shape (batch, input_length) or (batch, 1, input_length)"
        )

    def _encode_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) given input x in any accepted public shape."""

        x_seq = self._to_lstm_input(x)
        _, (h_n, _) = self.encoder_lstm(x_seq)
        h_last = h_n[-1]  # (B, H), one layer by default
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch into latent distribution parameters.

        Returns:
            (mu, sigma), both of shape (batch, latent_dim).
        """

        mu, logvar = self._encode_raw(x)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into reconstructed time series.

        Args:
            z: Tensor (batch, latent_dim).

        Returns:
            Reconstructed series with shape (batch, 1, input_length).
        """

        B = z.size(0)
        L = self.config.input_length
        N = self.config.num_layers
        H = self.config.hidden_size

        z_seq = z.unsqueeze(1).repeat(1, L, 1)  # (B, L, D)

        h0 = torch.tanh(self.z_to_h0(z)).unsqueeze(0).repeat(N, 1, 1).contiguous()
        c0 = torch.tanh(self.z_to_c0(z)).unsqueeze(0).repeat(N, 1, 1).contiguous()
        h0 = h0.view(N, B, H)
        c0 = c0.view(N, B, H)

        dec_out, _ = self.decoder_lstm(z_seq, (h0, c0))
        x_recon_seq = self.output_layer(dec_out)  # (B, L, 1)
        return x_recon_seq.transpose(1, 2)  # (B, 1, L)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns:
            (x_recon, mu, sigma, z)
        """

        x_seq = self._to_lstm_input(x)
        mu, logvar = self._encode_raw(x_seq)
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
        """ELBO loss = MSE reconstruction + beta * KL divergence."""

        if beta is None:
            beta = self.config.beta

        if x.dim() == 2:
            x = x.unsqueeze(1)
        recon = F.mse_loss(x_recon, x, reduction="sum")
        kl = 0.5 * torch.sum(
            sigma.pow(2) + mu.pow(2) - 1.0 - 2.0 * torch.log(sigma + 1e-8)
        )
        return recon + beta * kl


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class VAETrainer:
    """Train, evaluate, save, and load a :class:`TimeSeriesVAE`."""

    def __init__(
        self,
        model: TimeSeriesVAE,
        learning_rate: float = 5e-4,
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

    def train_epoch(self, loader: DataLoader, beta: float = 1.0) -> float:
        """Run one full training epoch over loader and return mean sample loss.
        
        Args:
            loader: DataLoader with training batches.
            beta: KL divergence weight multiplier (for warm-up schedule).
        """

        self.model.train()
        total_loss, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(self.device)
            if batch.dim() == 2:
                batch = batch.unsqueeze(1)
            x_recon, mu, sigma, _ = self.model(batch)
            loss = self.model.loss_function(batch, x_recon, mu, sigma, beta=beta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += batch.size(0)
        return total_loss / max(n, 1)

    def evaluate(self, loader: DataLoader) -> float:
        """Compute mean per-sample loss on loader without weight updates."""

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
        kl_warmup_epochs: int = 25,
    ) -> Dict[str, list]:
        """Train the VAE on a numpy array of pre-processed time series.
        
        Args:
            X: Training data (n_samples, input_length).
            epochs: Number of training epochs.
            batch_size: Batch size for training and validation.
            val_split: Fraction of data to use for validation.
            verbose: Whether to print training progress.
            early_stopping_patience: Number of epochs with no improvement to stop. 0 disables.
            kl_warmup_epochs: Number of epochs to linearly warm up KL term (default 25).
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
        X_train = X_t[idx[:n_train]]
        X_val = X_t[idx[n_train:]] if n_val > 0 else None

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
            # Compute KL warm-up schedule: β grows linearly from 0 to 1 over kl_warmup_epochs
            beta = min(1.0, epoch / max(1, kl_warmup_epochs))
            
            tr = self.train_epoch(train_loader, beta=beta)
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
                print(f"Epoch [{epoch:>4}/{epochs}]  train_loss={tr:.4f}{val_str}  (beta={beta:.4f})")

        return history

    def reconstruct(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Reconstruct a 2-D matrix of time series and return numpy output."""

        expected_len = self.model.config.input_length
        if X.ndim != 2 or X.shape[1] != expected_len:
            raise ValueError(
                f"X must have shape (n_samples, {expected_len}), got {X.shape}"
            )

        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        recon_parts: List[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if batch.dim() == 2:
                    batch = batch.unsqueeze(1)
                x_recon, _, _, _ = self.model(batch)
                recon_parts.append(x_recon.squeeze(1).cpu().numpy())

        if not recon_parts:
            return np.empty((0, expected_len), dtype=np.float32)
        return np.concatenate(recon_parts, axis=0).astype(np.float32)

    def reconstruction_mape(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        epsilon: float = 1e-8,
    ) -> float:
        """Compute MAPE (%) between X and its VAE reconstruction."""

        X_recon = self.reconstruct(X, batch_size=batch_size)
        return mean_absolute_percentage_error(X, X_recon, epsilon=epsilon)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to path (.pt file)."""

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
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "beta": cfg.beta,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint saved with save()."""

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        saved = checkpoint["config"]
        current = self.model.config
        if (
            saved["input_length"] != current.input_length
            or saved["latent_dim"] != current.latent_dim
            or saved.get("hidden_size", current.hidden_size) != current.hidden_size
            or saved.get("num_layers", current.num_layers) != current.num_layers
        ):
            raise ValueError(
                f"Checkpoint config {saved} does not match current model config "
                f"(input_length={current.input_length}, latent_dim={current.latent_dim}, "
                f"hidden_size={current.hidden_size}, num_layers={current.num_layers})."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)


# ---------------------------------------------------------------------------
# Entrenamiento de un único dataset UCR
# ---------------------------------------------------------------------------


def train_single_dataset(
    name: str,
    archive_path: str,
    config: Dict[str, object],
    device: str,
    seed: int,
) -> Dict[str, object]:
    """Entrena un VAE sobre un único dataset UCR y devuelve sus métricas en escala original."""
    import torch

    X_tr, _, X_te, _ = load_ucr_dataset(name, archive_path)
    natural_length = int(X_tr.shape[1])
    target_length = int(config["target_length"])
    series_length = min(target_length, natural_length) if target_length > 0 else natural_length

    # Truncate/pad to series_length BEFORE capturing statistics
    X_tr_padded = _pad_or_truncate(X_tr, series_length)
    X_te_padded = _pad_or_truncate(X_te, series_length)

    # Capture mean and std for each series in original scale BEFORE normalization
    means_tr, stds_tr = _get_series_statistics(X_tr_padded)
    means_te, stds_te = _get_series_statistics(X_te_padded)

    # Now normalize
    X_tr_orig_padded = X_tr_padded.copy()  # Keep copy for later denormalization
    X_te_orig_padded = X_te_padded.copy()
    X_tr_norm = (X_tr_padded - means_tr) / stds_tr
    X_te_norm = (X_te_padded - means_te) / stds_te

    use_paper_split = bool(config["paper_ecg5000_split"]) and name == "ECG5000"
    if use_paper_split:
        paper_len = min(series_length, 140)
        X_train_arr, _, _, _ = build_ecg5000_paper_split(
            archive_path=archive_path,
            target_length=paper_len,
            seed=seed,
        )
        # For paper split, recalculate stats for the actual training set
        X_tr_eval = X_tr_norm[:, :paper_len]
        X_te_eval = X_te_norm[:, :paper_len]
        means_tr = means_tr[:, :paper_len] if means_tr.shape[1] > paper_len else means_tr
        stds_tr = stds_tr[:, :paper_len] if stds_tr.shape[1] > paper_len else stds_tr
        means_te = means_te[:, :paper_len] if means_te.shape[1] > paper_len else means_te
        stds_te = stds_te[:, :paper_len] if stds_te.shape[1] > paper_len else stds_te
        series_length = paper_len
    else:
        X_tr_eval = X_tr_norm
        X_te_eval = X_te_norm
        if bool(config["use_test"]):
            X_train_arr = np.concatenate([X_tr_eval, X_te_eval], axis=0)
        else:
            X_train_arr = X_tr_eval

    print(
        f"  Longitud de serie: {series_length} (natural: {natural_length})"
        f"  |  Train: {len(X_train_arr)}  |  Test: {len(X_te_eval)}"
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = TimeSeriesVAEConfig(
        input_length=series_length,
        latent_dim=int(config["latent_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        beta=float(config["beta"]),
    )
    model = TimeSeriesVAE(cfg)
    trainer = VAETrainer(
        model=model,
        learning_rate=float(config["learning_rate"]),
        device=device,
    )

    history = trainer.fit(
        X_train_arr,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        val_split=float(config["val_split"]),
        verbose=True,
        early_stopping_patience=int(config["early_stopping_patience"]),
        kl_warmup_epochs=25,  # Default warm-up schedule
    )

    batch = int(config["batch_size"])
    
    # Reconstruct in normalized space
    X_recon_tr_norm = trainer.reconstruct(X_tr_eval, batch_size=batch)
    X_recon_te_norm = trainer.reconstruct(X_te_eval, batch_size=batch)
    X_total_norm = np.concatenate([X_tr_eval, X_te_eval], axis=0)
    X_recon_total_norm = trainer.reconstruct(X_total_norm, batch_size=batch)

    # Denormalize reconstructions and originals to original scale
    X_tr_orig = X_tr_orig_padded  # Original scale, padded
    X_te_orig = X_te_orig_padded
    X_recon_tr_orig = _unzscore(X_recon_tr_norm, means_tr, stds_tr)
    X_recon_te_orig = _unzscore(X_recon_te_norm, means_te, stds_te)
    X_total_orig = np.concatenate([X_tr_orig, X_te_orig], axis=0)
    X_recon_total_orig = np.concatenate([X_recon_tr_orig, X_recon_te_orig], axis=0)

    # Calculate MAPE in original scale
    mape_train = mape_original_scale(X_tr_orig, X_recon_tr_orig)
    mape_test = mape_original_scale(X_te_orig, X_recon_te_orig)
    mape_total = mape_original_scale(X_total_orig, X_recon_total_orig)

    last_train = history["train_loss"][-1] if history["train_loss"] else float("nan")
    last_val = history["val_loss"][-1] if history["val_loss"] else float("nan")

    print(
        f"  train_loss: {last_train:.6f}"
        f"  |  MAPE (original scale)  Train: {mape_train:.4f}%  Test: {mape_test:.4f}%  Total: {mape_total:.4f}%"
    )

    if use_paper_split:
        print("  Nota: split del paper activo -- 4500 muestras VAE, 500 red latente.")

    save_model = str(config["save_model"])
    if save_model:
        if save_model.lower().endswith(".pt"):
            saved_path = save_model[:-3] + f"_{name}.pt"
        else:
            saved_path = os.path.join(save_model, f"vae_{name}.pt")
        trainer.save(saved_path)
        print(f"  Modelo guardado en: {saved_path}")

    return {
        "name": name,
        "n_train": int(len(X_tr_eval)),
        "n_test": int(len(X_te_eval)),
        "series_length": series_length,
        "mape_train": mape_train,
        "mape_test": mape_test,
        "mape_total": mape_total,
        "train_loss": last_train,
        "val_loss": last_val,
    }
