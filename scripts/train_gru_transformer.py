"""
train_gru_transformer.py — GRU-Transformer neural decoder for mice LFP data.

Purpose
-------
Trains an 8-way stimulus classifier on bandpower features extracted from
Neuropixels LFP recordings (Allen Brain Observatory, Visual Coding dataset).
The model combines a Transformer encoder (for cross-time-bin attention) with
a GRU (for sequential temporal modelling), followed by a small MLP classifier.

Inputs (read from data_dir, configured in config.yaml)
-------
  bp_time_features_rich532_{session_id}.npy  — shape (N, K=10, D=532)
  bp_time_labels_rich532_{session_id}.npy    — shape (N,)  string labels

Outputs (written to data_dir)
-------
  <checkpoint_name>.pt   — best checkpoint (by test accuracy)
  <log_filename>         — epoch-by-epoch training log

Usage
-----
  python scripts/train_gru_transformer.py --config config.yaml
  python scripts/train_gru_transformer.py --config config.yaml --data-dir /my/data
  python scripts/train_gru_transformer.py --config config.yaml --dry-run
"""

import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRU-Transformer decoder on mice LFP bandpower features."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override the data_dir specified in the config file."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse config and print a summary, then exit without training."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config & reproducibility
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_batch(
    x: torch.Tensor,
    noise_std: float = 0.05,
    time_jitter_prob: float = 0.2,
) -> torch.Tensor:
    """
    Apply stochastic augmentation to a batch of LFP feature sequences.

    Two independent augmentations are applied:
    - Additive Gaussian noise: encourages the model to ignore small
      spectral fluctuations that are not stimulus-locked.
    - Random circular time shift (±1 bin): teaches the model to be
      robust to small temporal misalignments in the trial window.

    Args:
        x: Float tensor of shape (B, K, D) on the target device.
        noise_std: Standard deviation of the additive Gaussian noise.
        time_jitter_prob: Probability that any single sample receives
            a ±1 circular shift along the time dimension.

    Returns:
        Augmented tensor of the same shape as x.
    """
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    if time_jitter_prob > 0:
        B, K, _ = x.shape
        shifts = torch.zeros(B, dtype=torch.long, device=x.device)
        mask = torch.rand(B, device=x.device) < time_jitter_prob
        shifts[mask] = torch.randint_like(shifts[mask], low=0, high=2) * 2 - 1
        time_idx = torch.arange(K, device=x.device)
        x = torch.stack(
            [x[b, torch.roll(time_idx, shifts=shifts[b].item()), :] for b in range(B)],
            dim=0,
        )

    return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRUTransformer(nn.Module):
    """
    GRU-Transformer encoder for time-series LFP bandpower classification.

    Architecture:
        1. Linear projection:   D_in  → d_model
        2. LayerNorm
        3. TransformerEncoder:  1 layer, 4-head self-attention
           (captures cross-time-bin interactions)
        4. GRU:                 2 layers, hidden_size = d_model
           (captures sequential temporal dynamics)
        5. Classifier MLP:      d_model → d_model/2 → n_classes
           (with LayerNorm + Dropout)

    The Transformer runs first to build a context-aware representation of
    the full trial, then the GRU processes the enriched sequence and its
    final hidden state is fed to the classifier.

    Args:
        in_dim:    Input feature dimension D (= channels × frequency bands).
        d_model:   Internal hidden dimension used throughout the network.
        n_heads:   Number of self-attention heads in the Transformer.
        n_transformer_layers: Number of stacked Transformer encoder layers.
        n_gru_layers:         Number of stacked GRU layers.
        n_classes: Number of stimulus categories to classify.
        dropout:   Dropout probability applied in Transformer and classifier.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_heads: int,
        n_transformer_layers: int,
        n_gru_layers: int,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            batch_first=True,
            activation="relu",
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Inter-layer GRU dropout is intentionally omitted: with only 2 layers
        # the regularisation is handled by the classifier Dropout instead.
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, K, D_in).
        Returns:
            Logits tensor of shape (B, n_classes).
        """
        h = self.input_proj(x)   # (B, K, d_model)
        h = self.in_norm(h)
        h = self.transformer(h)  # (B, K, d_model)
        out, _ = self.gru(h)     # (B, K, d_model)
        h_last = out[:, -1, :]   # (B, d_model) — last time-bin hidden state
        return self.classifier(h_last)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_data(cfg: dict, data_dir: str):
    """
    Load and concatenate bandpower feature arrays from all sessions.

    Returns:
        X: np.ndarray of shape (N_total, K, D)
        y: np.ndarray of shape (N_total,) with string stimulus labels
    """
    suffix = cfg["data"]["feature_suffix"]
    Xs, ys = [], []
    for sid in cfg["data"]["sessions"]:
        Xs.append(np.load(os.path.join(data_dir, f"bp_time_features_{suffix}_{sid}.npy")))
        ys.append(np.load(
            os.path.join(data_dir, f"bp_time_labels_{suffix}_{sid}.npy"),
            allow_pickle=True,
        ))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def balance_classes(X: np.ndarray, y: np.ndarray, seed: int):
    """
    Downsample all classes to the size of the smallest class.

    Only called when cfg['data']['do_balance'] is True.
    When disabled, class imbalance is handled via weighted loss instead.
    """
    labels_series = pd.Series(y)
    min_n = labels_series.value_counts().min()
    logging.info(f"Balancing classes to {min_n} samples each.")
    rng = np.random.default_rng(seed=seed)
    keep_indices = []
    for _, idxs in labels_series.groupby(labels_series).groups.items():
        idxs = np.array(list(idxs))
        if len(idxs) > min_n:
            idxs = rng.choice(idxs, size=min_n, replace=False)
        keep_indices.append(idxs)
    keep_indices = np.concatenate(keep_indices)
    rng.shuffle(keep_indices)
    logging.info(f"Balanced counts:\n{pd.Series(y[keep_indices]).value_counts()}")
    return X[keep_indices], y[keep_indices]


def preprocess(X_train: np.ndarray, X_test: np.ndarray):
    """
    Z-score normalisation computed on the training set and applied to both splits.

    The statistics (mean, std) are returned so they can be saved in the
    checkpoint — required to normalise new data at inference time.

    Returns:
        X_train_norm, X_test_norm, train_mean, train_std
    """
    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        (X_train - train_mean) / train_std,
        (X_test - train_mean) / train_std,
        train_mean,
        train_std,
    )


def build_loaders(
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    X_test: np.ndarray,
    y_test_enc: np.ndarray,
    batch_size: int,
):
    """Convert numpy arrays to PyTorch tensors and wrap in DataLoaders."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    aug_cfg: dict,
    grad_clip: float = 1.0,
) -> float:
    """Run one full training epoch and return the mean cross-entropy loss."""
    model.train()
    loss_sum = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = augment_batch(
            X_batch,
            noise_std=aug_cfg["noise_std"],
            time_jitter_prob=aug_cfg["time_jitter_prob"],
        )
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        loss_sum += loss.item() * X_batch.size(0)
    return loss_sum / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: np.ndarray,
) -> dict:
    """
    Run evaluation on a DataLoader and return a results dictionary.

    Returns:
        dict with keys: accuracy, f1_macro, confusion_matrix,
                        per_class_accuracy, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(classes, per_class_acc)),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI override for data_dir
    data_dir = args.data_dir if args.data_dir else cfg["data"]["data_dir"]

    # Logging: write to file in data_dir and to stdout
    log_path = os.path.join(data_dir, cfg["output"]["log_filename"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Config: {args.config}")
    logging.info(f"Data dir: {data_dir}")

    if args.dry_run:
        logging.info("=== DRY RUN — configuration summary ===")
        logging.info(f"  Sessions:  {cfg['data']['sessions']}")
        logging.info(f"  Model:     d_model={cfg['model']['d_model']}, "
                     f"n_heads={cfg['model']['n_heads']}, "
                     f"n_transformer_layers={cfg['model']['n_transformer_layers']}, "
                     f"n_gru_layers={cfg['model']['n_gru_layers']}")
        logging.info(f"  Training:  epochs={cfg['training']['num_epochs']}, "
                     f"lr={cfg['training']['learning_rate']}, "
                     f"batch={cfg['training']['batch_size']}")
        logging.info("Exiting (--dry-run).")
        return

    seed = cfg["data"]["random_seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---- Data loading -------------------------------------------------------
    X, y = load_data(cfg, data_dir)
    logging.info(f"Combined shape: {X.shape}")
    logging.info(f"Class counts:\n{pd.Series(y).value_counts()}")

    # Keep only classes present in the data (handles partial sessions gracefully)
    present_classes = pd.Series(y).value_counts().index.tolist()
    mask = np.isin(y, present_classes)
    X, y = X[mask], y[mask]

    if cfg["data"]["do_balance"]:
        X, y = balance_classes(X, y, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=seed,
    )
    logging.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ---- Label encoding -----------------------------------------------------
    y_train_enc, classes = pd.factorize(y_train)
    label_to_int = {label: i for i, label in enumerate(classes)}
    y_test_enc = pd.Series(y_test).map(label_to_int).values
    n_classes = len(classes)
    logging.info(f"Classes ({n_classes}): {list(classes)}")

    # ---- Normalisation ------------------------------------------------------
    X_train, X_test, train_mean, train_std = preprocess(X_train, X_test)
    logging.info(f"NaN in X_train: {np.isnan(X_train).any()} | "
                 f"Inf in X_train: {np.isinf(X_train).any()}")

    # ---- Class-weighted loss ------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train_enc,
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logging.info(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=cfg["training"]["label_smoothing"],
    )

    # ---- DataLoaders --------------------------------------------------------
    train_loader, test_loader = build_loaders(
        X_train, y_train_enc, X_test, y_test_enc,
        batch_size=cfg["training"]["batch_size"],
    )

    # ---- Model --------------------------------------------------------------
    K, D_in = X_train.shape[1], X_train.shape[2]
    model = GRUTransformer(
        in_dim=D_in,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_transformer_layers=cfg["model"]["n_transformer_layers"],
        n_gru_layers=cfg["model"]["n_gru_layers"],
        n_classes=n_classes,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- Training loop ------------------------------------------------------
    best_acc = 0.0
    best_state = None

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            aug_cfg=cfg["augmentation"],
            grad_clip=cfg["training"]["grad_clip_norm"],
        )
        logging.info(f"Epoch {epoch + 1:3d} | train loss = {train_loss:.4f}")

        results = evaluate(model, test_loader, device, classes)
        logging.info(
            f"Epoch {epoch + 1:3d} | test acc = {results['accuracy']:.3f} "
            f"| F1 macro = {results['f1_macro']:.3f}"
        )
        logging.info(f"Confusion matrix:\n{results['confusion_matrix']}")
        for cls_name, acc in results["per_class_accuracy"].items():
            logging.info(f"  {cls_name}: {acc:.3f}")

        if results["accuracy"] > best_acc:
            best_acc = results["accuracy"]
            best_state = model.state_dict()

    logging.info(f"Best test accuracy: {best_acc:.4f}")

    # ---- Save checkpoint ----------------------------------------------------
    checkpoint_path = os.path.join(data_dir, cfg["output"]["checkpoint_name"])
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "in_dim": D_in,
            "K": K,
            # Normalisation statistics needed to preprocess new data at inference
            "train_mean": train_mean,
            "train_std": train_std,
        },
        checkpoint_path,
    )
    logging.info(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
