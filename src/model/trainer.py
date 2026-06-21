"""
trainer.py

Manages model training loops, validation loops, per-task ROC-AUC evaluation,
learning rate scheduling, early stopping, and checkpoint saving.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

from src.model.loss import MaskedBCEWithLogitsLoss


class Trainer:
    """
    Training manager for Tox21 GNN model.

    Args:
        model: The GNN model (nn.Module).
        train_graphs: List of PyG Data objects for training.
        val_graphs: List of PyG Data objects for validation.
        lr: Learning rate (default: 1e-3).
        batch_size: Batch size for DataLoader (default: 64).
        epochs: Maximum training epochs (default: 100).
        patience: Early stopping patience (default: 15).
        save_path: Path to save best model checkpoint.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        train_graphs: list,
        val_graphs: list,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        save_path: str = "saved_models/gnn_tox21.pt",
        device: str = None,
        pos_weight: torch.Tensor = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.patience = patience
        self.save_path = save_path

        # DataLoaders
        self.train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Loss, Optimizer, Scheduler
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
            
        self.criterion = MaskedBCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_accs = []
        self.val_precs = []
        self.val_recs = []
        self.val_f1s = []

    def _train_one_epoch(self) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch)
            loss = self.criterion(logits, batch.y.view(-1, 12))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple:
        """Run validation. Returns (avg_val_loss, mean_roc_auc, per_task_aucs)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            targets = batch.y.view(-1, 12)

            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = targets.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(labels)

        avg_loss = total_loss / max(num_batches, 1)

        # Compute per-task ROC-AUC
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        per_task_auc = []
        per_task_acc = []
        per_task_prec = []
        per_task_rec = []
        per_task_f1 = []
        self.current_thresholds = []
        for task_idx in range(12):
            y_true = all_targets[:, task_idx]
            y_score = all_preds[:, task_idx]

            # Filter out NaN labels
            valid_mask = ~np.isnan(y_true)
            y_true_valid = y_true[valid_mask]
            y_score_valid = y_score[valid_mask]

            # ROC-AUC requires both classes to be present
            if len(np.unique(y_true_valid)) < 2:
                per_task_auc.append(float("nan"))
                per_task_acc.append(float("nan"))
                per_task_prec.append(float("nan"))
                per_task_rec.append(float("nan"))
                per_task_f1.append(float("nan"))
                self.current_thresholds.append(0.5)
            else:
                precisions, recalls, thresholds = precision_recall_curve(y_true_valid, y_score_valid)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
                
                y_pred_valid = (y_score_valid > best_thresh).astype(int)

                auc = roc_auc_score(y_true_valid, y_score_valid)
                acc = accuracy_score(y_true_valid, y_pred_valid)
                prec = precision_score(y_true_valid, y_pred_valid, zero_division=0)
                rec = recall_score(y_true_valid, y_pred_valid, zero_division=0)
                f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)
                
                per_task_auc.append(auc)
                per_task_acc.append(acc)
                per_task_prec.append(prec)
                per_task_rec.append(rec)
                per_task_f1.append(f1)
                self.current_thresholds.append(best_thresh)

        # Mean AUC across tasks (ignoring NaN tasks)
        valid_aucs = [a for a in per_task_auc if not np.isnan(a)]
        valid_accs = [a for a in per_task_acc if not np.isnan(a)]
        valid_precs = [a for a in per_task_prec if not np.isnan(a)]
        valid_recs = [a for a in per_task_rec if not np.isnan(a)]
        valid_f1s = [a for a in per_task_f1 if not np.isnan(a)]
        
        mean_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else 0.0
        mean_acc = np.mean(valid_accs) if len(valid_accs) > 0 else 0.0
        mean_prec = np.mean(valid_precs) if len(valid_precs) > 0 else 0.0
        mean_rec = np.mean(valid_recs) if len(valid_recs) > 0 else 0.0
        mean_f1 = np.mean(valid_f1s) if len(valid_f1s) > 0 else 0.0

        return avg_loss, mean_auc, mean_acc, mean_prec, mean_rec, mean_f1

    def train(self) -> dict:
        """
        Full training loop with early stopping and checkpointing.

        Returns:
            Dictionary with training history.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        print(f"Training on {self.device} for up to {self.epochs} epochs...")
        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val AUC':>9} | {'Val F1':>9} | {'LR':>10}")
        print("-" * 72)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, val_auc, val_acc, val_prec, val_rec, val_f1 = self._validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_accs.append(val_acc)
            self.val_precs.append(val_prec)
            self.val_recs.append(val_rec)
            self.val_f1s.append(val_f1)

            # Step the LR scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:>6} | {train_loss:>11.4f} | {val_loss:>9.4f} | {val_auc:>9.4f} | {val_f1:>9.4f} | {current_lr:>10.6f}"
            )

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_thresholds = list(self.current_thresholds)
                self._save_checkpoint(epoch, val_loss, val_auc)
                print(f"         > Best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(f"\n! Early stopping triggered after {epoch} epochs (patience={self.patience})")
                break

        print(f"\n* Training complete. Best val_loss={best_val_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_aucs": self.val_aucs,
            "val_accs": self.val_accs,
            "val_precs": self.val_precs,
            "val_recs": self.val_recs,
            "val_f1s": self.val_f1s,
            "best_thresholds": self.best_thresholds,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, val_auc: float):
        """Save model weights and metadata."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "best_thresholds": self.best_thresholds,
            },
            self.save_path,
        )
