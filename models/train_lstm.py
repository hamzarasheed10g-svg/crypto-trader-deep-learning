"""LSTM training procedure (Section 1.5.2 of the methodology).

Implements:
- Forward propagation + MSE/BCE loss (Section 1.5.2 "Training Phase")
- Backpropagation Through Time via ``loss.backward()`` (Werbos, 1990)
- Adam optimizer (Kingma & Ba, 2015)
- Validation monitoring with early stopping (Prechelt, 1998)
- Final test evaluation (MSE, RMSE, MAE, direction accuracy)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm import (
    LSTMForecaster,
    DualHeadLSTMForecaster,
    build_model_from_config,
)
from utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "cuda":
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    direction_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"mse": mse, "rmse": rmse, "mae": mae, "direction_accuracy": direction_acc}


def classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
    """Binary classification metrics. ``y_pred_logits`` are raw logits.

    Includes a "confidence-weighted accuracy" — the directional accuracy
    computed only on predictions where the model is highly confident
    (|prob - 0.5| > 0.1, i.e. p_up >= 0.6 or p_up <= 0.4). This is the
    metric that matters for trading: even a model with 53% overall accuracy
    can be profitable if its high-confidence subset is 60%+ accurate, because
    those are the bars it actually opens positions on.
    """
    prob = 1.0 / (1.0 + np.exp(-y_pred_logits))  # sigmoid
    pred = (prob > 0.5).astype(np.float32)
    acc = float(np.mean(pred == y_true))
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    tn = float(np.sum((pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    balanced_acc = 0.5 * (recall + specificity)

    # Confidence-weighted accuracy
    high_conf_mask = np.abs(prob - 0.5) > 0.1
    if high_conf_mask.any():
        high_conf_acc = float(np.mean(pred[high_conf_mask] == y_true[high_conf_mask]))
        high_conf_frac = float(high_conf_mask.mean())
    else:
        high_conf_acc = 0.0
        high_conf_frac = 0.0
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "high_conf_acc": high_conf_acc,
        "high_conf_frac": high_conf_frac,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    train_loss: list
    val_loss: list
    val_metrics: list
    best_epoch: int
    best_val_loss: float


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().unsqueeze(-1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def _make_dual_loader(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_dir: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Loader yielding ``(x, y_reg, y_dir)`` triples for the dual-head trainer."""
    Xt = torch.from_numpy(X).float()
    yrt = torch.from_numpy(y_reg).float().unsqueeze(-1)
    ydt = torch.from_numpy(y_dir).float().unsqueeze(-1)
    return DataLoader(TensorDataset(Xt, yrt, ydt), batch_size=batch_size, shuffle=shuffle)


def train_lstm(
    cfg,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    save_path: str | Path,
) -> Tuple[LSTMForecaster, TrainHistory, str]:
    """Train an LSTM and persist the best-val-loss checkpoint to ``save_path``.

    Returns the model loaded with best weights, training history, and loss kind.
    """
    device = resolve_device(cfg.device)
    log.info("Training LSTM on %s", device)

    model, loss_kind = build_model_from_config(cfg, input_size=X_train.shape[-1])
    model.to(device)

    if loss_kind == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lstm.training.learning_rate,
        weight_decay=cfg.lstm.training.weight_decay,
    )

    scheduler = None
    sched_name = cfg.lstm.training.scheduler
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lstm.training.epochs)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    elif sched_name in ("none", None):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler {sched_name!r}")

    train_loader = _make_loader(X_train, y_train, cfg.lstm.training.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, cfg.lstm.training.batch_size, shuffle=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainHistory(train_loss=[], val_loss=[], val_metrics=[], best_epoch=-1, best_val_loss=float("inf"))
    patience = cfg.lstm.training.early_stopping_patience
    bad_epochs = 0

    for epoch in range(1, cfg.lstm.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if cfg.lstm.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.lstm.training.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # ---- validation ----
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                preds = model(xb)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        preds_np = np.concatenate(all_preds).reshape(-1)
        targets_np = np.concatenate(all_targets).reshape(-1)
        if loss_kind == "bce":
            val_loss = float(nn.BCEWithLogitsLoss()(torch.tensor(preds_np), torch.tensor(targets_np)).item())
            metrics = classification_metrics(targets_np, preds_np)
        else:
            val_loss = float(np.mean((preds_np - targets_np) ** 2))
            metrics = regression_metrics(targets_np, preds_np)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_metrics.append(metrics)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        log.info("Epoch %3d | train_loss=%.6f val_loss=%.6f metrics=%s",
                 epoch, train_loss, val_loss, {k: round(v, 5) for k, v in metrics.items()})

        if val_loss < history.best_val_loss - 1e-9:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"input_size": X_train.shape[-1],
                               "hidden_size": cfg.lstm.hidden_size,
                               "num_layers": cfg.lstm.num_layers,
                               "dropout": cfg.lstm.dropout,
                               "bidirectional": cfg.lstm.bidirectional,
                               "prediction_target": cfg.lstm.prediction_target,
                               "loss_kind": loss_kind},
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    log.info("Best epoch: %d (val_loss=%.6f) saved to %s", history.best_epoch, history.best_val_loss, save_path)

    # Load best weights
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history, loss_kind


def train_lstm_dual(
    cfg,
    X_train: np.ndarray, y_train: np.ndarray, y_train_dir: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, y_val_dir: np.ndarray,
    save_path: str | Path,
) -> Tuple["DualHeadLSTMForecaster", TrainHistory]:
    """Train a dual-head LSTM (regression + direction) jointly.

    Loss::

        L = alpha_reg * MSE(r_hat, r_true) + alpha_dir * BCE(d_hat_logit, d_true)

    The two alphas are read from ``cfg.lstm.loss.alpha_regression`` and
    ``cfg.lstm.loss.alpha_direction``. Validation loss used for model selection
    is the same combined loss.

    Persisted checkpoint format::

        {
            "model_state_dict": ...,
            "config": {..., "prediction_target": "dual", "loss_kind": "dual"},
            "epoch": int,
            "val_loss": float,
        }

    The classification head's metrics (accuracy, balanced accuracy, F1,
    high-confidence accuracy) are tracked alongside the regression metrics
    so that ``lstm_history.json`` shows both signals improving over training.
    """
    device = resolve_device(cfg.device)
    log.info("Training DUAL-HEAD LSTM on %s", device)

    # Build the dual-head model directly (build_model_from_config returns
    # (model, "dual") for prediction_target == "dual"; we use it but ignore
    # the loss_kind because we manage two losses here).
    model, loss_kind = build_model_from_config(cfg, input_size=X_train.shape[-1])
    assert loss_kind == "dual", f"train_lstm_dual called with non-dual config (got {loss_kind!r})"
    assert isinstance(model, DualHeadLSTMForecaster)
    model.to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    alpha_reg = float(cfg.lstm.loss.alpha_regression)
    alpha_dir = float(cfg.lstm.loss.alpha_direction)
    log.info("Loss weights: alpha_reg=%.3f alpha_dir=%.3f", alpha_reg, alpha_dir)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lstm.training.learning_rate,
        weight_decay=cfg.lstm.training.weight_decay,
    )

    scheduler = None
    sched_name = cfg.lstm.training.scheduler
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lstm.training.epochs)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    elif sched_name in ("none", None):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler {sched_name!r}")

    train_loader = _make_dual_loader(X_train, y_train, y_train_dir,
                                     cfg.lstm.training.batch_size, shuffle=True)
    val_loader = _make_dual_loader(X_val, y_val, y_val_dir,
                                   cfg.lstm.training.batch_size, shuffle=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainHistory(train_loss=[], val_loss=[], val_metrics=[],
                           best_epoch=-1, best_val_loss=float("inf"))
    patience = cfg.lstm.training.early_stopping_patience
    bad_epochs = 0

    for epoch in range(1, cfg.lstm.training.epochs + 1):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_dir_loss = 0.0
        n_batches = 0
        for xb, yr, yd in train_loader:
            xb = xb.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            yd = yd.to(device, non_blocking=True)
            optimizer.zero_grad()
            reg_pred, dir_logit = model(xb)
            l_reg = mse(reg_pred, yr)
            l_dir = bce(dir_logit, yd)
            loss = alpha_reg * l_reg + alpha_dir * l_dir
            loss.backward()
            if cfg.lstm.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.lstm.training.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_reg_loss += l_reg.item()
            epoch_dir_loss += l_dir.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_reg_loss = epoch_reg_loss / max(n_batches, 1)
        train_dir_loss = epoch_dir_loss / max(n_batches, 1)

        # ---- validation ----
        model.eval()
        all_reg, all_dir_logit, all_yr, all_yd = [], [], [], []
        with torch.no_grad():
            for xb, yr, yd in val_loader:
                xb = xb.to(device); yr = yr.to(device); yd = yd.to(device)
                reg_pred, dir_logit = model(xb)
                all_reg.append(reg_pred.cpu().numpy())
                all_dir_logit.append(dir_logit.cpu().numpy())
                all_yr.append(yr.cpu().numpy())
                all_yd.append(yd.cpu().numpy())
        reg_np = np.concatenate(all_reg).reshape(-1)
        dirlogit_np = np.concatenate(all_dir_logit).reshape(-1)
        yr_np = np.concatenate(all_yr).reshape(-1)
        yd_np = np.concatenate(all_yd).reshape(-1)

        val_reg_loss = float(np.mean((reg_np - yr_np) ** 2))
        val_dir_loss = float(nn.BCEWithLogitsLoss()(
            torch.tensor(dirlogit_np), torch.tensor(yd_np)
        ).item())
        val_loss = alpha_reg * val_reg_loss + alpha_dir * val_dir_loss

        reg_metrics = regression_metrics(yr_np, reg_np)
        cls_metrics = classification_metrics(yd_np, dirlogit_np)
        # Merge with prefix so it's clear which head each metric came from
        metrics = {**{f"reg_{k}": v for k, v in reg_metrics.items()},
                   **{f"dir_{k}": v for k, v in cls_metrics.items()},
                   "val_reg_loss": val_reg_loss,
                   "val_dir_loss": val_dir_loss}

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_metrics.append(metrics)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        log.info(
            "Epoch %3d | train_loss=%.6f (reg=%.6f dir=%.6f) | "
            "val_loss=%.6f | dir_acc=%.3f hi_conf=%.3f@%.2f | reg_dir_acc=%.3f",
            epoch, train_loss, train_reg_loss, train_dir_loss,
            val_loss,
            cls_metrics["accuracy"], cls_metrics["high_conf_acc"],
            cls_metrics["high_conf_frac"],
            reg_metrics["direction_accuracy"],
        )

        if val_loss < history.best_val_loss - 1e-9:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_size": X_train.shape[-1],
                        "hidden_size": cfg.lstm.hidden_size,
                        "num_layers": cfg.lstm.num_layers,
                        "dropout": cfg.lstm.dropout,
                        "bidirectional": cfg.lstm.bidirectional,
                        "prediction_target": "dual",
                        "loss_kind": "dual",
                        "alpha_regression": alpha_reg,
                        "alpha_direction": alpha_dir,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch, patience)
                break

    log.info("Best epoch: %d (val_loss=%.6f) saved to %s",
             history.best_epoch, history.best_val_loss, save_path)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history


def evaluate_lstm_dual(
    model: "DualHeadLSTMForecaster",
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_dir: np.ndarray,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Final test-set evaluation of a dual-head LSTM.

    Returns a flat dict mixing regression and classification metrics with
    ``reg_`` and ``dir_`` prefixes so downstream JSON consumers can pick
    whichever they need.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).float().to(device)
        reg, dir_logit = model(Xt)
        reg_np = reg.cpu().numpy().reshape(-1)
        dir_logit_np = dir_logit.cpu().numpy().reshape(-1)
    reg_metrics = regression_metrics(y_test, reg_np)
    cls_metrics = classification_metrics(y_test_dir, dir_logit_np)
    return {**{f"reg_{k}": v for k, v in reg_metrics.items()},
            **{f"dir_{k}": v for k, v in cls_metrics.items()}}


def evaluate_lstm(
    model: LSTMForecaster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    loss_kind: str,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Final test-set evaluation (single-head models only).

    For dual-head models, use ``evaluate_lstm_dual`` which takes both targets.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).float().to(device)
        preds = model(Xt).cpu().numpy().reshape(-1)
    if loss_kind == "bce":
        return classification_metrics(y_test, preds)
    return regression_metrics(y_test, preds)


def load_lstm_checkpoint(path: str | Path, device: torch.device | None = None) -> Tuple[nn.Module, dict]:
    """Reconstruct an LSTM model from a checkpoint saved by ``train_lstm`` or ``train_lstm_dual``.

    For dual-head checkpoints, returns the underlying :class:`DualHeadLSTMForecaster`
    wrapped in :class:`DualHeadShim` so callers that do ``model(x)`` and expect
    a single regression tensor (the env, the PPO observation builder) keep working.
    To access *both* heads, use ``model.predict_both(x)`` on the shim, or call
    the dual model directly via ``model.dual(x)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    loss_kind = cfg.get("loss_kind", "mse")

    if loss_kind == "dual" or cfg.get("prediction_target") == "dual":
        from models.lstm import DualHeadShim
        dual = DualHeadLSTMForecaster(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )
        dual.load_state_dict(ckpt["model_state_dict"])
        dual.to(device)
        dual.eval()
        model: nn.Module = DualHeadShim(dual)
        model.to(device)
        return model, cfg

    model = LSTMForecaster(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
        output_size=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg
