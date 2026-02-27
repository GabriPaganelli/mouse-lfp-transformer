"""
train_linear_baseline.py — Linear and shallow-MLP baselines for mice LFP decoding.

Purpose
-------
Trains a linear (multinomial logistic regression) or 1-hidden-layer MLP classifier
on the same bandpower features used by the GRU-Transformer, providing a lower-bound
reference for the time-series model's performance.

This baseline is used in the paper to demonstrate the benefit of the temporal
architecture: the GRU-Transformer is compared against both the pure linear model
and the shallow MLP on the 8-class stimulus classification task.

Key differences from the GRU-Transformer training script:
  - Input is flattened: (K, D) → (K*D,) — no temporal structure
  - No data augmentation
  - Model is either a single Linear layer or a 1-hidden-layer MLP

Inputs (read from data_dir, configured in config.yaml)
-------
  bp_time_features_rich532_{session_id}.npy  — shape (N, K=10, D=532)
  bp_time_labels_rich532_{session_id}.npy    — shape (N,)  string labels

Outputs (written to data_dir)
-------
  linear_baseline_best.pt   — best checkpoint (by test accuracy)
  linear_baseline.log       — epoch-by-epoch training log

Usage
-----
  python scripts/train_linear_baseline.py --config config.yaml
  python scripts/train_linear_baseline.py --config config.yaml --model mlp
  python scripts/train_linear_baseline.py --config config.yaml --data-dir /my/data
"""

import argparse
import logging
import os
import random
from typing import Optional

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
        description="Train a linear or MLP baseline on mice LFP bandpower features."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the shared YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data_dir from config."
    )
    parser.add_argument(
        "--model", choices=["linear", "mlp"], default="mlp",
        help="'linear' for multinomial logistic regression; "
             "'mlp' for a 1-hidden-layer MLP (default: mlp)."
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
# Model
# ---------------------------------------------------------------------------

class LinearBaseline(nn.Module):
    """
    Flat linear classifier for LFP bandpower features.

    Two variants:
      - Linear (hidden_dim=None): multinomial logistic regression.
        A single affine layer; no non-linearity.
      - MLP   (hidden_dim=512):   one hidden layer with ReLU and Dropout.
        Adds non-linear capacity while still ignoring temporal order.

    In both cases the input sequence (K time bins × D features) is flattened
    to a single vector of size K*D before classification, so the model has
    no access to temporal structure — unlike the GRU-Transformer.

    Args:
        in_dim:     Flattened input dimension (K * D).
        n_classes:  Number of stimulus categories.
        hidden_dim: Hidden layer size for the MLP variant; None for pure linear.
        dropout:    Dropout probability applied after the hidden layer (MLP only).
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, K*D) — already flattened.
        Returns:
            Logits tensor of shape (B, n_classes).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Data pipeline  (shared logic with train_gru_transformer.py)
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


def preprocess(X_train: np.ndarray, X_test: np.ndarray):
    """
    Z-score normalisation on training statistics, applied to both splits.

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


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one full training epoch and return the mean cross-entropy loss."""
    model.train()
    loss_sum = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
        dict with keys: accuracy, f1_macro, confusion_matrix, per_class_accuracy
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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_dir = args.data_dir if args.data_dir else cfg["data"]["data_dir"]

    log_path = os.path.join(data_dir, "linear_baseline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Config: {args.config} | Model variant: {args.model}")
    logging.info(f"Data dir: {data_dir}")

    seed = cfg["data"]["random_seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---- Data loading -------------------------------------------------------
    X, y = load_data(cfg, data_dir)
    logging.info(f"Combined shape: {X.shape}")
    logging.info(f"Class counts:\n{pd.Series(y).value_counts()}")

    present_classes = pd.Series(y).value_counts().index.tolist()
    mask = np.isin(y, present_classes)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=seed,
    )

    # ---- Label encoding -----------------------------------------------------
    y_train_enc, classes = pd.factorize(y_train)
    label_to_int = {label: i for i, label in enumerate(classes)}
    y_test_enc = pd.Series(y_test).map(label_to_int).values
    n_classes = len(classes)
    logging.info(f"Classes ({n_classes}): {list(classes)}")

    # ---- Normalisation and flattening ---------------------------------------
    X_train, X_test, train_mean, train_std = preprocess(X_train, X_test)
    logging.info(f"NaN in X_train: {np.isnan(X_train).any()} | "
                 f"Inf: {np.isinf(X_train).any()}")

    # Flatten (K, D) -> (K*D,) — no temporal structure exposed to the model
    N_train, K, D_in = X_train.shape
    flat_dim = K * D_in
    X_train_flat = X_train.reshape(N_train, flat_dim)
    X_test_flat = X_test.reshape(X_test.shape[0], flat_dim)

    # ---- Class-weighted loss ------------------------------------------------
    # We use the same class-weighted + label-smoothed loss as the GRU-Transformer
    # to ensure a rigorous, apples-to-apples comparison. Without class weights,
    # the baseline on this highly imbalanced dataset would trivially learn to
    # predict the majority classes, making the comparison uninformative.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train_enc,
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=cfg["training"]["label_smoothing"],
    )

    # ---- DataLoaders --------------------------------------------------------
    X_train_t = torch.tensor(X_train_flat, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_flat, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=cfg["training"]["batch_size"],
    )

    # ---- Model --------------------------------------------------------------
    hidden_dim = 512 if args.model == "mlp" else None
    model = LinearBaseline(
        in_dim=flat_dim,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    logging.info(f"Model: {'1-hidden-layer MLP' if hidden_dim else 'Logistic Regression'} "
                 f"| input dim: {flat_dim}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- Training loop ------------------------------------------------------
    best_acc = 0.0
    best_state = None
    # Baseline converges faster; 60 epochs are sufficient
    num_epochs = 60

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
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
    checkpoint_path = os.path.join(data_dir, "linear_baseline_best.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "model_variant": args.model,
            "classes": classes,
            "flat_dim": flat_dim,
            "K": K,
            "D_in": D_in,
            "train_mean": train_mean,
            "train_std": train_std,
        },
        checkpoint_path,
    )
    logging.info(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
