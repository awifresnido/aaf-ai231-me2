"""
VCM Trainer - Hardware-Aware Training with Mixed Precision
Works on both Jetson Orin Nano and DGX A100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestration for VCM with hardware awareness and mixed-precision support"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        class_weights: Optional[torch.Tensor] = None,
        command_indices: Optional[list] = None,
        select_metric: str = "val_loss",
    ):
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.command_indices = command_indices
        self.select_metric = select_metric
        self.best_val_loss = float('inf')
        self.best_metric = float('inf') if select_metric == 'val_loss' else float('-inf')
        self.best_epoch = None
        self.global_steps = 0
        self.optimizer = None
        self.scheduler = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        Train for one epoch with gradient accumulation support
        
        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (mel_specs, labels) in enumerate(train_loader):
            mel_specs = mel_specs.to(self.device)
            labels = labels.to(self.device)
            
            # ===== FORWARD PASS WITH MIXED PRECISION =====
            if self.mixed_precision:
                with autocast():
                    logits = self.model(mel_specs)
                    loss = self.criterion(logits, labels)
                    # Scale loss by gradient accumulation steps
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(mel_specs)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # ===== GRADIENT ACCUMULATION =====
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                self.global_steps += 1
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Periodic logging
            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}, Acc: {correct/total:.4f}"
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Optional[float]]:
        """
        Validate model with mixed precision support.

        Returns:
            (average_loss, accuracy, command_macro_f1) where command_macro_f1 is
            macro-F1 over the command leaf classes (excl. UNKNOWN/SILENCE) if
            self.command_indices is set, else None.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs = mel_specs.to(self.device)
                labels = labels.to(self.device)

                if self.mixed_precision:
                    with autocast():
                        logits = self.model(mel_specs)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(mel_specs)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.append(predicted.cpu())
                all_trues.append(labels.cpu())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        command_macro_f1 = None
        if self.command_indices:
            from sklearn.metrics import f1_score
            yt = torch.cat(all_trues).tolist()
            yp = torch.cat(all_preds).tolist()
            present = sorted(set(yt) & set(self.command_indices))
            if present:
                command_macro_f1 = float(f1_score(yt, yp, labels=present, average="macro", zero_division=0))

        return avg_loss, accuracy, command_macro_f1
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        scheduler_type: str = "cosine",
        early_stopping_patience: int = 5,
        save_dir: str = "checkpoints",
        weight_decay: float = 1e-5,
    ) -> Dict:
        """
        Full training loop with early stopping and checkpointing
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs to train
            lr: Learning rate
            scheduler_type: "cosine" or "plateau"
            early_stopping_patience: Early stopping patience
            save_dir: Directory to save checkpoints
        
        Returns:
            Training history dictionary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        else:  # plateau
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Training")
        logger.info(f"Epochs: {epochs}, LR: {lr}, Scheduler: {scheduler_type}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        logger.info(f"{'='*70}\n")
        
        no_improve_count = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_acc, cmd_f1 = self.validate(val_loader)
            select_value = val_loss if self.select_metric == 'val_loss' else (cmd_f1 if cmd_f1 is not None else 0.0)
            
            # Learning rate step
            if scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            # Log history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"cmd_f1: {cmd_f1:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save checkpoint if best validation loss / selection metric
            if self.select_metric == 'val_loss':
                improved = select_value < self.best_metric
            else:
                improved = select_value > self.best_metric
            if improved:
                self.best_metric = select_value
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                no_improve_count = 0
                checkpoint_path = save_dir / f"checkpoint_epoch{epoch:03d}_valloss{val_loss:.4f}.pt"
                self.save_checkpoint(checkpoint_path)
                self.save_checkpoint(save_dir / "best.pt")  # stable path for warm start / export
                logger.info(f"✓ Saved checkpoint to {checkpoint_path} (and best.pt)")
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch} epochs (patience={early_stopping_patience})")
                break
        
        logger.info("\n✓ Training completed!")
        return self.training_history
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.training_history,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('history', self.training_history)
        logger.info(f"✓ Loaded checkpoint from {path}")
    
    def save_history(self, path: Path) -> None:
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"✓ Saved training history to {path}")
