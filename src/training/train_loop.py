from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.utils.metrics import compute_eer, compute_t_dcf


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    checkpoint_path: Optional[str] = None
    log_dir: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_train_val_split(dataset, val_fraction: float = 0.2, seed: int = 42) -> Tuple[Subset, Subset]:
    if not hasattr(dataset, "labels"):
        raise ValueError("Dataset must expose a labels attribute for stratified splitting")

    labels = np.asarray(dataset.labels)
    if labels.size < 2:
        raise ValueError("At least two samples are required to create train and validation splits")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_value in np.unique(labels):
        class_indices = np.where(labels == label_value)[0]
        rng.shuffle(class_indices)

        if class_indices.size == 1:
            train_indices.extend(class_indices.tolist())
            continue

        val_count = max(1, int(round(class_indices.size * val_fraction)))
        if val_count >= class_indices.size:
            val_count = class_indices.size - 1

        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    if not train_indices or not val_indices:
        indices = np.arange(labels.size)
        rng.shuffle(indices)
        val_count = max(1, int(round(labels.size * val_fraction)))
        if val_count >= labels.size:
            val_count = labels.size - 1
        val_indices = indices[:val_count].tolist()
        train_indices = indices[val_count:].tolist()

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_two_stream_loaders(
    dataset,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_subset, val_subset = stratified_train_val_split(dataset, val_fraction=val_fraction, seed=seed)

    train_batch_size = min(batch_size, len(train_subset))
    if train_batch_size < 2:
        raise ValueError("Training split must contain at least two samples to support batch normalization")

    val_batch_size = min(batch_size, max(1, len(val_subset)))

    train_loader = DataLoader(
        train_subset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _move_batch_to_device(batch, device: torch.device):
    if len(batch) != 3:
        raise ValueError("Two-stream training expects batches of (audio, stats, labels)")
    audio, stats, labels = batch
    return audio.to(device), stats.to(device), labels.to(device)


def _collect_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 1]


def _safe_binary_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)

    metrics = {
        "eer": float("nan"),
        "eer_threshold": float("nan"),
        "t_dcf": float("nan"),
        "t_dcf_threshold": float("nan"),
    }

    if y_true_array.size < 2 or np.unique(y_true_array).size < 2:
        return metrics

    eer, eer_threshold = compute_eer(y_true_array, y_score_array)
    t_dcf, t_dcf_threshold = compute_t_dcf(y_true_array, y_score_array)

    metrics["eer"] = float(eer)
    metrics["eer_threshold"] = float(eer_threshold)
    metrics["t_dcf"] = float(t_dcf)
    metrics["t_dcf_threshold"] = float(t_dcf_threshold)
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_examples = 0
    y_true: List[int] = []
    y_score: List[float] = []

    for batch in loader:
        audio, stats, labels = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(audio, stats)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_score.extend(_collect_scores(logits.detach()).cpu().tolist())

    metrics = _safe_binary_metrics(y_true, y_score)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["samples"] = float(total_examples)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    y_true: List[int] = []
    y_score: List[float] = []

    for batch in loader:
        audio, stats, labels = _move_batch_to_device(batch, device)
        logits = model(audio, stats)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_score.extend(_collect_scores(logits).cpu().tolist())

    metrics = _safe_binary_metrics(y_true, y_score)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["samples"] = float(total_examples)
    return metrics


def fit_two_stream_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: Optional[torch.device] = None,
    logger=None,
    tb_logger=None,
) -> List[Dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_val_eer = math.inf

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=config.max_grad_norm,
        )
        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_eer": float(train_metrics["eer"]),
            "train_t_dcf": float(train_metrics["t_dcf"]),
            "val_loss": float(val_metrics["loss"]),
            "val_eer": float(val_metrics["eer"]),
            "val_t_dcf": float(val_metrics["t_dcf"]),
        }
        history.append(epoch_metrics)

        if logger is not None:
            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f train_eer=%.4f val_eer=%.4f train_tdcf=%.4f val_tdcf=%.4f",
                epoch,
                epoch_metrics["train_loss"],
                epoch_metrics["val_loss"],
                epoch_metrics["train_eer"],
                epoch_metrics["val_eer"],
                epoch_metrics["train_t_dcf"],
                epoch_metrics["val_t_dcf"],
            )

        if tb_logger is not None:
            tb_logger.log_scalar("loss/train", epoch_metrics["train_loss"], epoch)
            tb_logger.log_scalar("loss/val", epoch_metrics["val_loss"], epoch)
            tb_logger.log_scalar("eer/train", epoch_metrics["train_eer"], epoch)
            tb_logger.log_scalar("eer/val", epoch_metrics["val_eer"], epoch)
            tb_logger.log_scalar("tdcf/train", epoch_metrics["train_t_dcf"], epoch)
            tb_logger.log_scalar("tdcf/val", epoch_metrics["val_t_dcf"], epoch)

        if checkpoint_path is not None and not math.isnan(epoch_metrics["val_eer"]) and epoch_metrics["val_eer"] < best_val_eer:
            best_val_eer = epoch_metrics["val_eer"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                    "metrics": epoch_metrics,
                },
                checkpoint_path,
            )

    return history
