"""
training/multi_head_trainer.py

Trainer for the Multi-Head ConvNeXt architecture.
Handles dictionary outputs and multi-task loss weighting.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.trainer import EarlyStopping, get_device
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)

class MultiHeadTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterions: Dict[str, nn.Module],
        loss_weights: Dict[str, float],
        mode: str = "multi_head",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.criterions = criterions
        self.loss_weights = loss_weights
        self.mode = mode
        self.device = device or get_device()
        self.ckpt_dir = Path(checkpoint_dir) / mode
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = Path(log_dir) / mode
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.model.to(self.device)
        self._amp_enabled = (self.device.type in ("mps", "cuda"))
        self._amp_dtype = amp_dtype

        logger.info(
            "MultiHeadTrainer ready | device=%s | amp=%s | checkpoints=%s",
            self.device, self._amp_enabled, self.ckpt_dir,
        )

    def _compute_loss(self, logits_dict: dict, labels_dict: dict):
        # Head 1 (Binary BCEWithLogitsLoss)
        loss_h1 = self.criterions['h1'](logits_dict['normal_abnormal'], labels_dict['normal_abnormal'].float())
        
        # Head 2 (Pathology Routing Multi-class)
        # Filter out normal images for H2 (label == -1)
        valid_h2 = labels_dict['pathology'] != -1
        if valid_h2.sum() > 0:
            loss_h2 = self.criterions['h2'](logits_dict['pathology'][valid_h2], labels_dict['pathology'][valid_h2])
        else:
            loss_h2 = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Head 3 (Severity - Multi-Label BCEWithLogitsLoss)
        loss_h3 = 0.0
        for key in ['macular', 'diabetic', 'vascular', 'fluid', 'structural']:
            if isinstance(self.criterions['h3'], dict):
                loss_h3 += self.criterions['h3'][key](logits_dict['severity'][key], labels_dict['severity'][key].float())
            else:
                loss_h3 += self.criterions['h3'](logits_dict['severity'][key], labels_dict['severity'][key].float())
            
        total_loss = (self.loss_weights['h1'] * loss_h1 + 
                      self.loss_weights['h2'] * loss_h2 + 
                      self.loss_weights['h3'] * loss_h3)
                      
        return total_loss, loss_h1, loss_h2, loss_h3

    def _train_epoch(self, loader, optimizer, max_norm: float = 5.0, smoke_test: bool = False):
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)

        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) if not isinstance(v, dict) 
                      else {k2: v2.to(self.device, non_blocking=True) for k2, v2 in v.items()}
                      for k, v in labels.items()}

            optimizer.zero_grad(set_to_none=True)

            with _amp_ctx:
                logits = self.model(images)
                loss, _, _, _ = self._compute_loss(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            optimizer.step()
            total_loss += loss.item()
            
            if smoke_test and batch_idx >= 2:
                break

        return total_loss / max(1, n_batches if not smoke_test else 3)

    @torch.no_grad()
    def _val_epoch(self, loader, smoke_test: bool = False):
        self.model.eval()
        total_loss = 0.0
        h2_preds = []
        h2_targets = []
        n_batches = len(loader)

        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) if not isinstance(v, dict) 
                      else {k2: v2.to(self.device, non_blocking=True) for k2, v2 in v.items()}
                      for k, v in labels.items()}

            with _amp_ctx:
                logits = self.model(images)
                loss, _, _, _ = self._compute_loss(logits, labels)

            total_loss += loss.item()
            
            # Extract H2 metrics for selection
            valid_h2 = labels['pathology'] != -1
            if valid_h2.sum() > 0:
                h2_targets.extend(labels['pathology'][valid_h2].cpu().numpy())
                h2_preds.extend(logits['pathology'][valid_h2].argmax(dim=1).cpu().numpy())
                
            if smoke_test and batch_idx >= 2:
                break

        mean_loss = total_loss / max(1, n_batches if not smoke_test else 3)
        
        if len(h2_targets) > 0:
            from sklearn.metrics import recall_score
            h2_macro_f1 = f1_score(h2_targets, h2_preds, average='macro', zero_division=0)
            h2_recall = recall_score(h2_targets, h2_preds, average='macro', zero_division=0)
            h2_acc = accuracy_score(h2_targets, h2_preds)
        else:
            h2_macro_f1, h2_recall, h2_acc = 0.0, 0.0, 0.0
            
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            
        return mean_loss, h2_macro_f1, h2_recall, h2_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_epochs: int = 3,
        warmup_lr: float = 1e-3,
        finetune_epochs: int = 45,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        fold_id: int = 0,
        smoke_test: bool = False,
        resume_path: str = None
    ) -> Dict:
        global_step = fold_id * 100_000
        best_val_loss = float("inf")
        best_val_macro_f1 = 0.0
        best_metrics = {}
        
        start_epoch_warmup = 0
        start_epoch_ft = 0
        phase = 'warmup'
        
        if resume_path and os.path.exists(resume_path):
            logger.info(f"Loading checkpoint from {resume_path}...")
            import traceback
            try:
                ckpt = torch.load(resume_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                phase = ckpt.get('phase', 'warmup')
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                best_val_macro_f1 = ckpt.get('best_val_macro_f1', 0.0)
                
                saved_epoch = ckpt.get('epoch', -1)
                if phase == 'warmup':
                    start_epoch_warmup = saved_epoch + 1
                else:
                    start_epoch_ft = saved_epoch + 1 - warmup_epochs
                    
                logger.info(f"Resumed from {phase} at absolute epoch {saved_epoch + 1}.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                traceback.print_exc()

        # ── PHASE 1: WARM-UP ──
        if phase == 'warmup' and start_epoch_warmup < warmup_epochs:
            logger.info(f"PHASE 1 — Warm-up | {warmup_epochs} epochs | backbone FROZEN")
            self.model.freeze_backbone()
            optimizer_warmup = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=warmup_lr,
                weight_decay=weight_decay,
            )
            if resume_path and 'optimizer_state_dict' in ckpt and phase == 'warmup':
                optimizer_warmup.load_state_dict(ckpt['optimizer_state_dict'])

            for epoch in range(start_epoch_warmup, warmup_epochs):
                train_loss = self._train_epoch(train_loader, optimizer_warmup, smoke_test=smoke_test)
                val_loss, h2_f1, h2_recall, h2_acc = self._val_epoch(val_loader, smoke_test=smoke_test)
                global_step += 1
                
                logger.info(f"Ep {epoch:3d} [warmup|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lr {warmup_lr:.2e}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_warmup, epoch=epoch, phase='warmup', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1)
                if smoke_test: break
            
            phase = 'finetune'

        # ── PHASE 2: FINE-TUNING ──
        if phase == 'finetune' and start_epoch_ft < finetune_epochs:
            logger.info(f"PHASE 2 — Fine-tuning | max {finetune_epochs} epochs | backbone UNFROZEN")
            self.model.unfreeze_backbone()
            optimizer_ft = torch.optim.AdamW(
                self.model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer_ft, T_0=10, T_mult=2, eta_min=1e-6,
            )
            if resume_path and 'optimizer_state_dict' in ckpt and ckpt.get('phase') == 'finetune':
                optimizer_ft.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            early_stopper = EarlyStopping(patience=patience, mode="min")
            early_stopper.best_value = best_val_loss

            for epoch in range(start_epoch_ft, finetune_epochs):
                train_loss = self._train_epoch(train_loader, optimizer_ft, smoke_test=smoke_test)
                val_loss, h2_f1, h2_recall, h2_acc = self._val_epoch(val_loader, smoke_test=smoke_test)
                scheduler.step()
                
                current_lr = scheduler.get_last_lr()[0]
                global_step += 1
                abs_epoch = warmup_epochs + epoch
                
                self.writer.add_scalar(f"fold{fold_id}/loss/train", train_loss, global_step)
                self.writer.add_scalar(f"fold{fold_id}/loss/val", val_loss, global_step)
                self.writer.add_scalar(f"fold{fold_id}/metrics/h2_macro_f1", h2_f1, global_step)

                logger.info(f"Ep {abs_epoch:3d} [finetune|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lr {current_lr:.2e}")

                if h2_f1 > best_val_macro_f1:
                    best_val_macro_f1 = h2_f1
                    best_metrics = {"val_loss": val_loss, "h2_macro_f1": h2_f1, "epoch": abs_epoch}
                    self._save_checkpoint(f"fold{fold_id}_best_model.pth", optimizer=optimizer_ft, scheduler=scheduler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1)
                    logger.info(f"  ✓ New best — H2 macro_f1={h2_f1:.4f}")

                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_ft, scheduler=scheduler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1)

                if early_stopper.step(val_loss):
                    logger.info(f"Early stopping at epoch {abs_epoch}")
                    break
                    
                if smoke_test: break

        return best_metrics

    def _save_checkpoint(self, filename: str, optimizer=None, scheduler=None, epoch=None, phase=None, best_val_loss=None, best_val_macro_f1=None) -> None:
        path = self.ckpt_dir / filename
        state = {
            "model_state_dict": self.model.state_dict(),
            "mode": self.mode,
        }
        if optimizer: state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler: state["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None: state["epoch"] = epoch
        if phase is not None: state["phase"] = phase
        if best_val_loss is not None: state["best_val_loss"] = best_val_loss
        if best_val_macro_f1 is not None: state["best_val_macro_f1"] = best_val_macro_f1
        torch.save(state, path)
