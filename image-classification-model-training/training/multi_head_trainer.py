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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.trainer import EarlyStopping
from utils.device import get_device, get_raw_model, ComputeManager
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score

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
        compute_manager: Optional[ComputeManager] = None,
        amp_dtype: torch.dtype = torch.float16,
        metric_extractors: Optional[Dict[str, callable]] = None,
    ) -> None:
        self.criterions = criterions
        self.loss_weights = loss_weights
        self.mode = mode
        self.compute_manager = compute_manager or ComputeManager(device=device)
        self.device = self.compute_manager.device
        self.model = self.compute_manager.prepare_model(model)
        
        # Default strategy for Multi-Class classification (H2 Pathology)
        self.metric_extractors = metric_extractors or {
            'h2': lambda logits: torch.argmax(logits, dim=1)
        }
        self.ckpt_dir = Path(checkpoint_dir) / mode
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = Path(log_dir) / mode
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self._amp_enabled = (self.device.type in ["cuda", "mps"])
        self._amp_dtype = amp_dtype
        
        # Initialize GradScaler for mixed precision (CUDA only) to prevent FP16 gradient overflow
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))

        logger.info(
            "MultiHeadTrainer ready | device=%s | amp=%s (%s) | checkpoints=%s",
            self.device, self._amp_enabled, self._amp_dtype, self.ckpt_dir,
        )

    def _compute_loss(self, logits_dict: dict, labels_dict: dict):
        # Head 1 (Binary BCEWithLogitsLoss)
        loss_h1 = self.criterions['h1'](logits_dict['normal_abnormal'], labels_dict['normal_abnormal'].float())
        
        # Head 2 (Granular Pathology - Asymmetric Loss)
        # Hierarchical Loss Masking: Only calculate H2 loss for Abnormal samples (h1 label == 1)
        valid_h2_mask = (labels_dict['normal_abnormal'] == 1).view(-1)
        
        if valid_h2_mask.sum() > 0:
            target_logits = logits_dict['pathology'][valid_h2_mask]
            target_labels = labels_dict['pathology'][valid_h2_mask]
            loss_h2 = self.criterions['h2'](target_logits, target_labels)
        else:
            loss_h2 = torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = self.loss_weights['h1'] * loss_h1 + self.loss_weights['h2'] * loss_h2
                      
        return total_loss, loss_h1, loss_h2, torch.tensor(0.0, device=self.device)

    def _train_epoch(
        self,
        loader,
        optimizer,
        max_norm: float = 5.0,
        smoke_test: bool = False,
        accum_steps: int = 1,
        save_steps: int = 2250,
        fold_id: int = 0,
        epoch: int = 0,
        phase: str = "finetune",
        best_val_loss: float = float("inf"),
        best_val_macro_f1: float = 0.0,
        hf_repo: Optional[str] = None,
    ):
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)
        start_time = time.time()

        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) if not isinstance(v, dict) 
                      else {k2: v2.to(self.device, non_blocking=True) for k2, v2 in v.items()}
                      for k, v in labels.items()}

            with _amp_ctx:
                logits = self.model(images)
                loss, _, _, _ = self._compute_loss(logits, labels)
                # Scale loss for gradient accumulation
                accum_loss = loss / accum_steps

            # Backward pass
            self.scaler.scale(accum_loss).backward()

            # Perform optimizer step every accum_steps or on the last batch
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            del logits, accum_loss
            self.compute_manager.flush_cache(batch_idx=batch_idx)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - start_time
                sec_per_batch = elapsed / (batch_idx + 1)
                logger.info(f"   [Train] Batch {batch_idx + 1}/{n_batches} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s ({sec_per_batch:.2f}s/batch)")

            # Mid-Epoch Checkpoint Saving
            if save_steps > 0 and (batch_idx + 1) % save_steps == 0 and (batch_idx + 1) < n_batches:
                self._save_checkpoint(
                    f"fold{fold_id}_last_model.pth",
                    optimizer=optimizer,
                    epoch=epoch,
                    phase=phase,
                    batch_idx=batch_idx,
                    best_val_loss=best_val_loss,
                    best_val_macro_f1=best_val_macro_f1,
                    hf_repo=hf_repo,
                )
                logger.info(f"   ✓ Mid-epoch checkpoint saved at batch {batch_idx + 1}/{n_batches} (fold{fold_id}_last_model.pth)")
            
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
            
            # Extract H2 metrics via Injected Strategy (SOLID Dependency Inversion)
            valid_h2_mask = (labels['normal_abnormal'] == 1).view(-1)
            if valid_h2_mask.sum() > 0:
                batch_targets = labels['pathology'][valid_h2_mask]
                batch_logits = logits['pathology'][valid_h2_mask]
                
                # Keep targets and logits 1D/2D as they naturally are.
                # Just flatten them into the list.
                h2_targets.extend(batch_targets.cpu().numpy().tolist())
                
                batch_preds = self.metric_extractors['h2'](batch_logits)
                h2_preds.extend(batch_preds.cpu().numpy().tolist())

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
                logger.info(f"   [Val] Batch {batch_idx + 1}/{n_batches} | Loss: {loss.item():.4f}")
                
            if smoke_test and batch_idx >= 2:
                break

        mean_loss = total_loss / max(1, n_batches if not smoke_test else 3)
        
        if len(h2_targets) > 0:
            from sklearn.metrics import recall_score
            # Convert flat lists back to 1D numpy arrays
            h2_targets_np = np.array(h2_targets)
            h2_preds_np = np.array(h2_preds)
            
            h2_macro_f1 = f1_score(h2_targets_np, h2_preds_np, average='macro', zero_division=0)
            h2_recall = recall_score(h2_targets_np, h2_preds_np, average='macro', zero_division=0)
            h2_acc = accuracy_score(h2_targets_np, h2_preds_np)
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
        resume_path: str = None,
        hf_repo: str = None,
        accum_steps: int = 1,
        save_steps: int = 2250
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
        if phase == 'warmup' and warmup_epochs > 0:
            logger.info(f"PHASE 1 — Warm-up | {warmup_epochs} epochs | backbone FROZEN")
            model_to_freeze = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_freeze.freeze_backbone()
            optimizer_warmup = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=warmup_lr,
                weight_decay=weight_decay,
            )
            if resume_path and 'optimizer_state_dict' in ckpt and phase == 'warmup':
                optimizer_warmup.load_state_dict(ckpt['optimizer_state_dict'])

            for epoch in range(start_epoch_warmup, warmup_epochs):
                ep_start = time.time()
                train_loss = self._train_epoch(
                    train_loader,
                    optimizer_warmup,
                    smoke_test=smoke_test,
                    accum_steps=accum_steps,
                    save_steps=save_steps,
                    fold_id=fold_id,
                    epoch=epoch,
                    phase='warmup',
                    best_val_loss=best_val_loss,
                    best_val_macro_f1=best_val_macro_f1,
                    hf_repo=hf_repo
                )
                val_loss, h2_f1, h2_recall, h2_acc = self._val_epoch(val_loader, smoke_test=smoke_test)
                global_step += 1
                ep_duration = time.time() - ep_start
                
                logger.info(f"Ep {epoch:3d} [warmup|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lr {warmup_lr:.2e} | time {ep_duration:.1f}s")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_warmup, epoch=epoch, phase='warmup', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo)
                if smoke_test: break
            
            phase = 'finetune'

        # ── PHASE 2: FINE-TUNING ──
        if phase == 'finetune' and finetune_epochs > 0:
            logger.info(f"PHASE 2 — Fine-tuning | max {finetune_epochs} epochs | backbone UNFROZEN")
            model_to_unfreeze = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_unfreeze.unfreeze_backbone()
            optimizer_ft = torch.optim.AdamW(
                model_to_unfreeze.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr),
                weight_decay=weight_decay,
            )
            
            # Port the state from optimizer_warmup to prevent Adam momentum shock on the head
            if 'optimizer_warmup' in locals():
                for p, state in optimizer_warmup.state.items():
                    optimizer_ft.state[p] = state
                    
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
                ep_start = time.time()
                abs_epoch = warmup_epochs + epoch
                train_loss = self._train_epoch(
                    train_loader,
                    optimizer_ft,
                    smoke_test=smoke_test,
                    accum_steps=accum_steps,
                    save_steps=save_steps,
                    fold_id=fold_id,
                    epoch=abs_epoch,
                    phase='finetune',
                    best_val_loss=best_val_loss,
                    best_val_macro_f1=best_val_macro_f1,
                    hf_repo=hf_repo
                )
                val_loss, h2_f1, h2_recall, h2_acc = self._val_epoch(val_loader, smoke_test=smoke_test)
                scheduler.step()
                
                current_lr = scheduler.get_last_lr()[0]
                global_step += 1
                ep_duration = time.time() - ep_start
                
                self.writer.add_scalar(f"fold{fold_id}/loss/train", train_loss, global_step)
                self.writer.add_scalar(f"fold{fold_id}/loss/val", val_loss, global_step)
                self.writer.add_scalar(f"fold{fold_id}/metrics/h2_macro_f1", h2_f1, global_step)

                logger.info(f"Ep {abs_epoch:3d} [finetune|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lr {current_lr:.2e} | time {ep_duration:.1f}s")

                if h2_f1 > best_val_macro_f1:
                    best_val_macro_f1 = h2_f1
                    best_metrics = {"val_loss": val_loss, "h2_macro_f1": h2_f1, "epoch": abs_epoch}
                    self._save_checkpoint(f"fold{fold_id}_best_model.pth", optimizer=optimizer_ft, scheduler=scheduler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo)
                    logger.info(f"  ✓ New best — H2 macro_f1={h2_f1:.4f}")

                # Always save last (rolling) and a numbered epoch snapshot so no epoch is ever lost
                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_ft, scheduler=scheduler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo)
                self._save_checkpoint(f"fold{fold_id}_epoch_{abs_epoch:03d}.pth", optimizer=optimizer_ft, scheduler=scheduler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1)
                logger.info(f"  Saved fold{fold_id}_epoch_{abs_epoch:03d}.pth  (f1={h2_f1:.4f})")

                if early_stopper.step(val_loss):
                    logger.info(f"Early stopping at epoch {abs_epoch}")
                    break
                    
                if smoke_test: break

        return best_metrics

    def _save_checkpoint(self, filename: str, optimizer=None, scheduler=None, epoch=None, phase=None, batch_idx=None, best_val_loss=None, best_val_macro_f1=None, hf_repo: str = None) -> None:
        path = self.ckpt_dir / filename
        tmp_path = path.with_suffix(".pth.tmp")  # atomic write: temp → rename
        state = {
            "model_state_dict": self.model.state_dict(),
            "mode": self.mode,
        }
        if optimizer: state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler: state["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None: state["epoch"] = epoch
        if phase is not None: state["phase"] = phase
        if batch_idx is not None: state["batch_idx"] = batch_idx
        if best_val_loss is not None: state["best_val_loss"] = best_val_loss
        if best_val_macro_f1 is not None: state["best_val_macro_f1"] = best_val_macro_f1
        torch.save(state, tmp_path)        # write to temp first
        os.replace(tmp_path, path)         # atomic rename — SIGINT during write can't corrupt final .pth

        # Real-time Cloud Backup to Hugging Face Hub (prevents data loss on Kaggle/Colab timeout)
        hf_token = os.environ.get("HF_TOKEN")
        target_repo = hf_repo or os.environ.get("HF_REPO_ID")
        if target_repo and hf_token and ("best" in filename or "last" in filename):
            clean_repo = target_repo.replace("https://huggingface.co/", "").strip("/")
            primary_type = "model"
            if clean_repo.startswith("spaces/"):
                clean_repo = clean_repo.replace("spaces/", "")
                primary_type = "space"
            elif clean_repo.startswith("datasets/"):
                clean_repo = clean_repo.replace("datasets/", "")
                primary_type = "dataset"
            elif "dataset" in os.environ.get("HF_REPO_TYPE", "").lower():
                primary_type = "dataset"
            elif "space" in os.environ.get("HF_REPO_TYPE", "").lower():
                primary_type = "space"

            candidate_types = [primary_type] + [t for t in ["model", "dataset", "space"] if t != primary_type]
            
            uploaded = False
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                for r_type in candidate_types:
                    try:
                        api.upload_file(
                            path_or_fileobj=str(path),
                            path_in_repo=filename,
                            repo_id=clean_repo,
                            token=hf_token,
                            repo_type=r_type
                        )
                        logger.info(f"   ☁ Checkpoint '{filename}' successfully backed up to HuggingFace ({clean_repo} | type={r_type})")
                        uploaded = True
                        break
                    except Exception:
                        continue
                if not uploaded:
                    logger.warning(f"   ⚠ Could not upload checkpoint to HuggingFace ({clean_repo}) across model/dataset/space types.")
            except Exception as e:
                logger.warning(f"   ⚠ Could not initialize HuggingFace upload: {e}")
