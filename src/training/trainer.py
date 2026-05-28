"""
Trainer — training loop, validation, checkpointing, and logging.

Supports LSTM (stateful) and Transformer (stateless) models generically.
"""

import os
import json
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training, validation, checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[Dict] = None,
        output_dir: str = "outputs",
        device: str = "cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.config = config or {}
        self.device = torch.device(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad token

        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.epoch = 0

        self.model.to(self.device)

        logger.info(
            "Trainer initialized: model=%s, device=%s, params=%d",
            type(model).__name__,
            device,
            sum(p.numel() for p in model.parameters()),
        )

    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # LSTM stateful initialization
        hidden = None
        is_lstm = hasattr(self.model, "init_hidden")

        loop = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if is_lstm:
                # Initialize hidden for each gradient step
                hidden = self.model.init_hidden(inputs.size(0), self.device)

            self.optimizer.zero_grad()

            if is_lstm:
                logits, hidden = self.model(inputs, hidden)
                # Detach hidden to avoid backprop through entire sequence
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                logits = self.model(inputs)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            # Gradient clipping
            grad_clip = self.config.get("training", {}).get("grad_clip", 5.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns average loss."""
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if hasattr(self.model, "init_hidden"):
                hidden = self.model.init_hidden(inputs.size(0), self.device)
                logits, hidden = self.model(inputs, hidden)
            else:
                logits = self.model(inputs)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches else 0.0

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

        if is_best:
            best_path = os.path.join(os.path.dirname(path), "best.pt")
            torch.save(checkpoint, best_path)
            logger.info("Best model updated: %s", best_path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", {})
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("Loaded checkpoint: %s (epoch %d)", path, self.epoch)
        return checkpoint

    def train(self, num_epochs: int) -> Dict:
        """Full training loop."""
        logger.info("Starting training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                log_msg += " ⭐ (best)"
            logger.info(log_msg)

            # Checkpointing
            interval = self.config.get("training", {}).get("checkpoint_interval", 10)
            if (epoch + 1) % interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"epoch_{epoch+1}.pt"
                )
                is_best = val_loss <= self.best_val_loss if self.val_loader else False
                self.save_checkpoint(checkpoint_path, is_best=is_best)

            # Save training history
            with open(os.path.join(self.log_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        # Final checkpoint
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, "final.pt"),
            is_best=(val_loss <= self.best_val_loss) if self.val_loader else False,
        )

        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
        return self.history
