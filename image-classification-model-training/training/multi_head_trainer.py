"""
training/multi_head_trainer.py

Trainer for the Multi-Head ConvNeXt architecture.
Handles dictionary outputs, multi-task loss weighting, per-class telemetry,
and out-of-fold cross-validation consistency tracking.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from training.trainer import EarlyStopping
from training.losses import FocalLoss
from utils.device import get_raw_model, ComputeManager

logger = logging.getLogger(__name__)

DEFAULT_PATHOLOGY_CLASSES = [
    "CNV", "DRUSEN", "AMD", "General_AMD", "DME", "DR", "MH", "RVO", "RAO", "CSR", "ERM", "VID"
]

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
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.criterions = criterions
        self.loss_weights = loss_weights
        self.mode = mode
        self.compute_manager = compute_manager or ComputeManager(device=device)
        self.device = self.compute_manager.device
        self.model = self.compute_manager.prepare_model(model)
        self.class_names = class_names or DEFAULT_PATHOLOGY_CLASSES
        
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

    def _move_labels_to_device(self, labels: dict) -> dict:
        """Move label tensors (or nested dictionary of labels) to the target compute device."""
        return {
            k: v.to(self.device, non_blocking=True) if not isinstance(v, dict)
            else {k2: v2.to(self.device, non_blocking=True) for k2, v2 in v.items()}
            for k, v in labels.items()
        }

    def _compute_loss(self, logits_dict: dict, labels_dict: dict):
        """Compute multi-task weighted loss with FP32 stability casting and H2 hierarchical masking."""
        # Cast logits to float32 BEFORE loss to prevent NaN from FP16 overflow.
        logits_dict = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in logits_dict.items()}

        # Head 1 (Binary BCEWithLogitsLoss)
        loss_h1 = self.criterions['h1'](logits_dict['normal_abnormal'], labels_dict['normal_abnormal'].float())
        
        # Head 2 (Granular Pathology - Asymmetric Loss)
        # Hierarchical Loss Masking: Only calculate H2 loss for Abnormal samples (h1 label == 1) with valid pathology class (>= 0)
        num_h2_classes = logits_dict['pathology'].size(-1)
        valid_h2_mask = ((labels_dict['normal_abnormal'] == 1).view(-1)) & (labels_dict['pathology'] >= 0) & (labels_dict['pathology'] < num_h2_classes)
        
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
        max_norm: float = 3.0,
        smoke_test: bool = False,
        accum_steps: int = 1,
        save_steps: int = 2250,
        fold_id: int = 0,
        epoch: int = 0,
        phase: str = "finetune",
        best_val_loss: float = float("inf"),
        best_val_macro_f1: float = 0.0,
        hf_repo: Optional[str] = None,
        start_batch: int = 0,
    ) -> float:
        """Run one single training epoch with gradient accumulation and AMP scaling."""
        self.model.train()
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        processed_batches = 0
        n_batches = len(loader)
        start_time = time.time()

        _amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx < start_batch:
                continue

            images = images.to(self.device, non_blocking=True)
            labels = self._move_labels_to_device(labels)

            with _amp_ctx:
                logits = self.model(images)
                loss, _, _, _ = self._compute_loss(logits, labels)
                accum_loss = loss / accum_steps

            # Backward pass
            self.scaler.scale(accum_loss).backward()

            # Step optimizer every accum_steps or on the last batch
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            processed_batches += 1
            total_loss += loss.item()

            del logits, accum_loss
            self.compute_manager.flush_cache(batch_idx=batch_idx)
            
            if self.compute_manager.is_main_process and ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches):
                elapsed = time.time() - start_time
                sec_per_batch = elapsed / (batch_idx + 1)
                logger.info(f"   [Train] Batch {batch_idx + 1}/{n_batches} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s ({sec_per_batch:.2f}s/batch)")

            # Mid-Epoch Checkpoint Saving (Rank 0 only)
            if self.compute_manager.is_main_process and save_steps > 0 and (batch_idx + 1) % save_steps == 0 and (batch_idx + 1) < n_batches:
                self._save_checkpoint(
                    f"fold{fold_id}_last_model.pth",
                    optimizer=optimizer,
                    scaler=self.scaler,
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

        return total_loss / max(1, processed_batches if not smoke_test else min(3, processed_batches))

    @torch.no_grad()
    def _val_epoch(self, loader, smoke_test: bool = False) -> Tuple[float, float, float, float, dict]:
        """Run validation loop and compute loss, macro F1, recall, accuracy, and per-class metrics."""
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
            labels = self._move_labels_to_device(labels)

            with _amp_ctx:
                logits = self.model(images)
                loss, _, _, _ = self._compute_loss(logits, labels)

            total_loss += loss.item()
            
            # Extract H2 metrics via Injected Strategy (SOLID Dependency Inversion)
            num_h2_classes = logits['pathology'].size(-1)
            valid_h2_mask = ((labels['normal_abnormal'] == 1).view(-1)) & (labels['pathology'] >= 0) & (labels['pathology'] < num_h2_classes)
            if valid_h2_mask.sum() > 0:
                batch_targets = labels['pathology'][valid_h2_mask]
                batch_logits = logits['pathology'][valid_h2_mask].float()
                
                h2_targets.extend(batch_targets.cpu().numpy().tolist())
                batch_preds = self.metric_extractors['h2'](batch_logits)
                h2_preds.extend(batch_preds.cpu().numpy().tolist())

            if self.compute_manager.is_main_process and ((batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches):
                logger.info(f"   [Val] Batch {batch_idx + 1}/{n_batches} | Loss: {loss.item():.4f}")
                
            if smoke_test and batch_idx >= 2:
                break

        mean_loss = total_loss / max(1, n_batches if not smoke_test else 3)
        
        if len(h2_targets) > 0:
            h2_targets_np = np.array(h2_targets)
            h2_preds_np = np.array(h2_preds)
            
            h2_macro_f1 = float(f1_score(h2_targets_np, h2_preds_np, average='macro', zero_division=0))
            h2_recall = float(recall_score(h2_targets_np, h2_preds_np, average='macro', zero_division=0))
            h2_acc = float(accuracy_score(h2_targets_np, h2_preds_np))
            num_classes = logits['pathology'].size(-1) if 'logits' in locals() else len(self.class_names)
        else:
            h2_macro_f1, h2_recall, h2_acc = 0.0, 0.0, 0.0
            num_classes = len(self.class_names)
            
        per_class_metrics = self._compute_per_class_metrics(h2_targets, h2_preds, num_classes)

        if self.device.type == 'mps':
            torch.mps.empty_cache()
            
        return mean_loss, h2_macro_f1, h2_recall, h2_acc, per_class_metrics

    def _compute_per_class_metrics(self, h2_targets: list, h2_preds: list, num_classes: int) -> dict:
        """Compute per-class F1, Precision, Recall, and Support dictionary."""
        per_class_metrics = {}
        if len(h2_targets) > 0:
            h2_targets_np = np.array(h2_targets)
            h2_preds_np = np.array(h2_preds)

            c_names = self.class_names if len(self.class_names) == num_classes else [f"Class_{i}" for i in range(num_classes)]

            per_class_f1 = f1_score(h2_targets_np, h2_preds_np, labels=list(range(num_classes)), average=None, zero_division=0)
            per_class_prec = precision_score(h2_targets_np, h2_preds_np, labels=list(range(num_classes)), average=None, zero_division=0)
            per_class_rec = recall_score(h2_targets_np, h2_preds_np, labels=list(range(num_classes)), average=None, zero_division=0)

            for idx, name in enumerate(c_names):
                supp = int(np.sum(h2_targets_np == idx))
                per_class_metrics[name] = {
                    "f1": float(per_class_f1[idx]),
                    "precision": float(per_class_prec[idx]),
                    "recall": float(per_class_rec[idx]),
                    "support": supp,
                }
        else:
            for name in self.class_names:
                per_class_metrics[name] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}
        return per_class_metrics

    def _record_epoch_metrics(
        self, fold_id: int, global_step: int, train_loss: float, val_loss: float, h2_f1: float, per_class_metrics: dict
    ) -> None:
        """Record epoch metrics and per-class F1 scores into TensorBoard."""
        if not self.compute_manager.is_main_process:
            return
        self.writer.add_scalar(f"fold{fold_id}/loss/train", train_loss, global_step)
        self.writer.add_scalar(f"fold{fold_id}/loss/val", val_loss, global_step)
        self.writer.add_scalar(f"fold{fold_id}/metrics/h2_macro_f1", h2_f1, global_step)
        for c_name, m in per_class_metrics.items():
            self.writer.add_scalar(f"fold{fold_id}/per_class_f1/{c_name}", m["f1"], global_step)

    def _log_per_class_breakdown(self, fold_id: int, epoch: int, phase: str, per_class_metrics: dict) -> None:
        """Log formatted per-class metrics breakdown table to logger."""
        if not self.compute_manager.is_main_process or not per_class_metrics:
            return
        
        dead_classes = [c for c, m in per_class_metrics.items() if m["f1"] == 0.0]
        active_classes = [c for c, m in per_class_metrics.items() if m["f1"] > 0.0]

        logger.info(f"--- Per-Class Validation Breakdown (Fold {fold_id} | Ep {epoch:3d} | {phase}) ---")
        logger.info(f"{'Class Name':<18} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Support':<8} | Status")
        logger.info("-" * 80)
        for c_name, m in per_class_metrics.items():
            status = "DEAD (0 F1)" if m["f1"] == 0.0 else "Active"
            logger.info(f"{c_name:<18} | {m['f1']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['support']:<8d} | {status}")
        logger.info("-" * 80)
        dead_str = ", ".join(dead_classes) if dead_classes else "None"
        logger.info(f"Active Classes: {len(active_classes)}/{len(per_class_metrics)} | Dead ({len(dead_classes)}): {dead_str}\n")

    def update_oof_summary(self, fold_id: int, epoch: int, best_macro_f1: float, best_per_class: dict) -> None:
        """Save best per-class metrics to oof_per_class_summary.json and trigger consistency log."""
        oof_json = self.ckpt_dir / "oof_per_class_summary.json"
        data = {}
        if oof_json.exists():
            try:
                with open(oof_json, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
                
        data[f"fold_{fold_id}"] = {
            "best_epoch": epoch,
            "best_macro_f1": float(best_macro_f1),
            "per_class": best_per_class
        }
        
        if self.compute_manager.is_main_process:
            with open(oof_json, "w") as f:
                json.dump(data, f, indent=2)
                
            self._log_oof_consistency(data)

    def _log_oof_consistency(self, data: dict) -> None:
        """Log out-of-fold consistency comparison table across completed folds."""
        if not self.compute_manager.is_main_process or not data:
            return

        completed_folds = sorted(data.keys())
        sample_fold = completed_folds[0]
        class_names = list(data[sample_fold]["per_class"].keys())

        logger.info("\n==================== OUT-OF-FOLD (OOF) CLASS CONSISTENCY SUMMARY ====================")
        fold_headers = " | ".join([f"{f.upper():<7}" for f in completed_folds])
        header = f"{'Class Name':<18} | {fold_headers} | Mean F1  | Consistency Analysis"
        logger.info(header)
        logger.info("-" * len(header))

        rows_df = []
        for cname in class_names:
            f1s = []
            fold_str_list = []
            row_dict = {"class_name": cname}
            for f_key in completed_folds:
                f1_val = data[f_key]["per_class"].get(cname, {}).get("f1", 0.0)
                f1s.append(f1_val)
                fold_str_list.append(f"{f1_val:<7.4f}")
                row_dict[f_key] = f1_val

            mean_f1 = float(np.mean(f1s)) if f1s else 0.0
            row_dict["mean_f1"] = mean_f1

            if len(completed_folds) == 1:
                status = "DEAD (Fold 0)" if mean_f1 == 0.0 else f"Active (F1: {mean_f1:.4f})"
            else:
                if all(val == 0.0 for val in f1s):
                    status = "CONSISTENTLY DEAD (All Folds 0.0)"
                elif any(val == 0.0 for val in f1s) and any(val > 0.0 for val in f1s):
                    dead_f = [f for f, v in zip(completed_folds, f1s) if v == 0.0]
                    status = f"FOLD-SPECIFIC FAILURE (Dead in {', '.join(dead_f)})"
                elif mean_f1 >= 0.70:
                    status = "CONSISTENTLY HIGH (F1 >= 0.70)"
                else:
                    status = "MODERATE / VARIABLE"

            row_dict["consistency_status"] = status
            rows_df.append(row_dict)

            fold_cols_str = " | ".join(fold_str_list)
            logger.info(f"{cname:<18} | {fold_cols_str} | {mean_f1:<8.4f} | {status}")

        logger.info("====================================================================================================\n")

        try:
            df = pd.DataFrame(rows_df)
            csv_path = self.ckpt_dir / "oof_cross_fold_class_summary.csv"
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.debug(f"Could not write oof CSV: {e}")

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
        """Run full two-phase training protocol (Head Warm-up -> Backbone Fine-tuning)."""
        global_step = fold_id * 100_000
        best_val_loss = float("inf")
        best_val_macro_f1 = 0.0
        best_metrics = {}
        
        start_epoch_warmup = 0
        start_epoch_ft = 0
        start_batch_warmup = 0
        start_batch_ft = 0
        phase = 'warmup'
        ckpt = None

        # Strict Fail-Fast Validation: Compute fold-specific class weights from unique training patients
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "compute_class_weights"):
            try:
                fold_h2_alpha = train_loader.dataset.compute_class_weights("h2").to(self.device)
                if isinstance(self.criterions.get('h2'), FocalLoss):
                    self.criterions['h2'].alpha = fold_h2_alpha
                    if self.compute_manager.is_main_process:
                        class_names = train_loader.dataset.get_class_names("h2")
                        logger.info(f"=== Fold {fold_id} Patient-Based FocalLoss Alpha Weights (Training Set Only) ===")
                        for idx, (c_name, w_val) in enumerate(zip(class_names, fold_h2_alpha.tolist())):
                            logger.info(f"  {c_name:<15} : {w_val:.2f}")
                        logger.info("==========================================================================")
                else:
                    raise RuntimeError(f"Strict Assertion Failure: H2 criterion is not FocalLoss (got {type(self.criterions.get('h2'))})")
            except Exception as exc:
                if self.compute_manager.is_main_process:
                    logger.error(f"CRITICAL ERROR: Failed to compute fold-specific FocalLoss class weights: {exc}")
                raise RuntimeError(f"Aborting training run due to critical loss configuration failure: {exc}") from exc

        actual_resume_path = resume_path
        if resume_path and not os.path.exists(resume_path):
            try:
                from huggingface_hub import hf_hub_download
                repo_id = hf_repo or os.environ.get("HF_REPO_ID") or "NMundhra/OCT-Classification-Model"
                clean_repo = repo_id.replace("https://huggingface.co/", "").strip("/")
                target_filename = os.path.basename(resume_path) if "/" in resume_path else (resume_path if resume_path.endswith(".pth") else "fold0_best_model.pth")
                if self.compute_manager.is_main_process:
                    logger.info(f"Downloading resume checkpoint '{target_filename}' directly from Hugging Face repository '{clean_repo}'...")
                actual_resume_path = hf_hub_download(
                    repo_id=clean_repo,
                    filename=target_filename,
                    token=os.environ.get("HF_TOKEN")
                )
            except Exception as e_hf:
                if self.compute_manager.is_main_process:
                    logger.warning(f"Could not download resume checkpoint from Hugging Face ({resume_path}): {e_hf}")

        if actual_resume_path and os.path.exists(actual_resume_path):
            if self.compute_manager.is_main_process:
                logger.info(f"Loading checkpoint from {actual_resume_path}...")
            import traceback
            try:
                ckpt = torch.load(actual_resume_path, map_location=self.device)
                get_raw_model(self.model).load_state_dict(ckpt['model_state_dict'])
                phase = ckpt.get('phase', 'warmup')
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                best_val_macro_f1 = ckpt.get('best_val_macro_f1', 0.0)
                
                saved_epoch = ckpt.get('epoch', -1)
                saved_batch_idx = ckpt.get('batch_idx')
                if 'scaler_state_dict' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                if phase == 'warmup':
                    if saved_batch_idx is None:
                        start_epoch_warmup = saved_epoch + 1
                    else:
                        start_epoch_warmup = saved_epoch
                        start_batch_warmup = saved_batch_idx + 1
                else:
                    if saved_batch_idx is None:
                        start_epoch_ft = saved_epoch + 1 - warmup_epochs
                    else:
                        start_epoch_ft = saved_epoch - warmup_epochs
                        start_batch_ft = saved_batch_idx + 1
                    
                if self.compute_manager.is_main_process:
                    logger.info(f"Resumed from {phase} at absolute epoch {saved_epoch + 1}.")
            except Exception as e:
                if self.compute_manager.is_main_process:
                    logger.warning(f"Failed to load checkpoint: {e}")
                    traceback.print_exc()

        # ── PHASE 1: WARM-UP ──
        if phase == 'warmup' and warmup_epochs > 0:
            if self.compute_manager.is_main_process:
                logger.info(f"PHASE 1 — Warm-up | {warmup_epochs} epochs | backbone FROZEN")
            model_to_freeze = get_raw_model(self.model)
            model_to_freeze.freeze_backbone()
            optimizer_warmup = torch.optim.AdamW(
                model_to_freeze.get_param_groups(backbone_lr=warmup_lr, head_lr=warmup_lr, weight_decay=weight_decay),
            )
            if resume_path and 'optimizer_state_dict' in ckpt and phase == 'warmup':
                try:
                    optimizer_warmup.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    if self.compute_manager.is_main_process:
                        logger.warning(f"Could not restore warmup optimizer state: {e}")

            for epoch in range(start_epoch_warmup, warmup_epochs):
                current_start_batch = start_batch_warmup if epoch == start_epoch_warmup else 0
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
                    hf_repo=hf_repo,
                    start_batch=current_start_batch,
                )
                val_loss, h2_f1, h2_recall, h2_acc, per_class_metrics = self._val_epoch(val_loader, smoke_test=smoke_test)
                global_step += 1
                ep_duration = time.time() - ep_start
                
                group_lrs = [f"{group['lr']:.2e}" for group in optimizer_warmup.param_groups]
                lrs_str = ", ".join(group_lrs)
                self._record_epoch_metrics(fold_id, global_step, train_loss, val_loss, h2_f1, per_class_metrics)
                if self.compute_manager.is_main_process:
                    logger.info(f"Ep {epoch:3d} [warmup|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lrs [{lrs_str}] | time {ep_duration:.1f}s")

                if h2_f1 > best_val_macro_f1:
                    best_val_macro_f1 = h2_f1
                    best_metrics = {"val_loss": val_loss, "h2_macro_f1": h2_f1, "epoch": epoch, "per_class": per_class_metrics}
                    self._save_checkpoint(f"fold{fold_id}_best_model.pth", optimizer=optimizer_warmup, scaler=self.scaler, epoch=epoch, phase='warmup', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo, per_class=per_class_metrics)
                    if self.compute_manager.is_main_process:
                        logger.info(f"  ✓ New best — H2 macro_f1={h2_f1:.4f}")
                        self._log_per_class_breakdown(fold_id, epoch, "warmup", per_class_metrics)
                    self.update_oof_summary(fold_id=fold_id, epoch=epoch, best_macro_f1=h2_f1, best_per_class=per_class_metrics)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_warmup, scaler=self.scaler, epoch=epoch, phase='warmup', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo, per_class=per_class_metrics)
                if smoke_test: break
            
            phase = 'finetune'

        # ── PHASE 2: FINE-TUNING ──
        if phase == 'finetune' and finetune_epochs > 0:
            if self.compute_manager.is_main_process:
                logger.info(f"PHASE 2 — Gradual Fine-tuning | max {finetune_epochs} epochs | Stage 3 Bottleneck -> Full Backbone")
            model_to_unfreeze = get_raw_model(self.model)
            # Stage 2A: Start by unfreezing Stage 3 (deepest bottleneck) only to preserve early edge/texture filters
            model_to_unfreeze.unfreeze_stage3_only()
            optimizer_ft = torch.optim.AdamW(
                model_to_unfreeze.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay, early_backbone_factor=0.1),
            )
            
            # Port state from optimizer_warmup to prevent Adam momentum shock
            if 'optimizer_warmup' in locals():
                for p, state in optimizer_warmup.state.items():
                    optimizer_ft.state[p] = state
                    
            import math
            lr_lambda = lambda ep: 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * min(ep, finetune_epochs) / max(1, finetune_epochs)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lr_lambda)
            if resume_path and 'optimizer_state_dict' in ckpt and ckpt.get('phase') == 'finetune':
                try:
                    optimizer_ft.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    if self.compute_manager.is_main_process:
                        logger.warning(f"Could not restore finetune optimizer state: {e}")
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            # Change 3: Early stopping on macro-F1 (mode=max, patience=3) beginning after warmup
            early_stopper = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")
            early_stopper.best_value = best_val_macro_f1 if best_val_macro_f1 > 0 else None

            for epoch in range(start_epoch_ft, finetune_epochs):
                if epoch == 4:
                    if self.compute_manager.is_main_process:
                        logger.info("Unfreezing full backbone gradually (early stages stem/0/1/2 at 0.1x LR)...")
                    model_to_unfreeze.unfreeze_full_backbone()
                    old_state = optimizer_ft.state
                    optimizer_ft = torch.optim.AdamW(
                        model_to_unfreeze.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay, early_backbone_factor=0.1),
                    )
                    optimizer_ft.state.update(old_state)

                current_start_batch = start_batch_ft if epoch == start_epoch_ft else 0
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
                    hf_repo=hf_repo,
                    start_batch=current_start_batch,
                )
                val_loss, h2_f1, h2_recall, h2_acc, per_class_metrics = self._val_epoch(val_loader, smoke_test=smoke_test)
                scheduler.step()
                
                group_lrs = [f"{group['lr']:.2e}" for group in optimizer_ft.param_groups]
                lrs_str = ", ".join(group_lrs)
                global_step += 1
                ep_duration = time.time() - ep_start
                
                self._record_epoch_metrics(fold_id, global_step, train_loss, val_loss, h2_f1, per_class_metrics)
                if self.compute_manager.is_main_process:
                    logger.info(f"Ep {abs_epoch:3d} [finetune|fold{fold_id}] loss {train_loss:.4f}/{val_loss:.4f} | H2 F1 {h2_f1:.4f} | H2 Rec {h2_recall:.4f} | lrs [{lrs_str}] | time {ep_duration:.1f}s")

                # Save best macro-F1 checkpoints
                if h2_f1 > best_val_macro_f1:
                    best_val_macro_f1 = h2_f1
                    best_metrics = {"val_loss": val_loss, "h2_macro_f1": h2_f1, "epoch": abs_epoch, "per_class": per_class_metrics}
                    self._save_checkpoint(f"fold{fold_id}_best_model.pth", optimizer=optimizer_ft, scheduler=scheduler, scaler=self.scaler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo, per_class=per_class_metrics)
                    self._save_checkpoint(f"fold{fold_id}_best_macro_f1.pth", optimizer=optimizer_ft, scheduler=scheduler, scaler=self.scaler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, per_class=per_class_metrics)
                    if self.compute_manager.is_main_process:
                        logger.info(f"  ✓ New best macro-F1 — H2 macro_f1={h2_f1:.4f}")
                        self._log_per_class_breakdown(fold_id, abs_epoch, "finetune", per_class_metrics)
                    self.update_oof_summary(fold_id=fold_id, epoch=abs_epoch, best_macro_f1=h2_f1, best_per_class=per_class_metrics)

                # Save best validation loss checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(f"fold{fold_id}_best_val_loss.pth", optimizer=optimizer_ft, scheduler=scheduler, scaler=self.scaler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, per_class=per_class_metrics)
                    if self.compute_manager.is_main_process:
                        logger.info(f"  ✓ New best val_loss — val_loss={val_loss:.4f}")

                # Always save last (rolling) and a numbered epoch snapshot so no epoch is ever lost
                self._save_checkpoint(f"fold{fold_id}_last_model.pth", optimizer=optimizer_ft, scheduler=scheduler, scaler=self.scaler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, hf_repo=hf_repo, per_class=per_class_metrics)
                self._save_checkpoint(f"fold{fold_id}_epoch_{abs_epoch:03d}.pth", optimizer=optimizer_ft, scheduler=scheduler, scaler=self.scaler, epoch=abs_epoch, phase='finetune', best_val_loss=best_val_loss, best_val_macro_f1=best_val_macro_f1, per_class=per_class_metrics)
                if self.compute_manager.is_main_process:
                    logger.info(f"  Saved fold{fold_id}_epoch_{abs_epoch:03d}.pth  (f1={h2_f1:.4f})")

                if early_stopper.step(h2_f1):
                    if self.compute_manager.is_main_process:
                        logger.info(f"Early stopping triggered on macro-F1 at epoch {abs_epoch} (patience={patience})")
                    break
                    
                if smoke_test: break

        return best_metrics

    def _save_checkpoint(
        self,
        filename: str,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=None,
        phase=None,
        batch_idx=None,
        best_val_loss=None,
        best_val_macro_f1=None,
        hf_repo: Optional[str] = None,
        per_class: Optional[dict] = None,
    ) -> None:
        """Atomically save checkpoint dictionary to disk and attempt HuggingFace backup."""
        if not self.compute_manager.is_main_process:
            return

        path = self.ckpt_dir / filename
        tmp_path = path.with_suffix(".pth.tmp")  # atomic write: temp → rename
        state = {
            "model_state_dict": get_raw_model(self.model).state_dict(),
            "mode": self.mode,
        }
        if optimizer: state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler: state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None: state["scaler_state_dict"] = scaler.state_dict()
        if epoch is not None: state["epoch"] = epoch
        if phase is not None: state["phase"] = phase
        if batch_idx is not None: state["batch_idx"] = batch_idx
        if best_val_loss is not None: state["best_val_loss"] = best_val_loss
        if best_val_macro_f1 is not None: state["best_val_macro_f1"] = best_val_macro_f1
        if per_class is not None: state["per_class"] = per_class

        torch.save(state, tmp_path)        # write to temp first
        os.replace(tmp_path, path)         # atomic rename — SIGINT during write can't corrupt final .pth

        self._upload_checkpoint_to_hf(path, filename, hf_repo)

    def _upload_checkpoint_to_hf(self, path: Path, filename: str, hf_repo: Optional[str]) -> None:
        """Helper to upload checkpoint asynchronously to HuggingFace Hub."""
        hf_token = os.environ.get("HF_TOKEN")
        target_repo = hf_repo or os.environ.get("HF_REPO_ID")

        if not hf_token:
            try:
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass

        if not target_repo:
            try:
                from kaggle_secrets import UserSecretsClient
                target_repo = UserSecretsClient().get_secret("HF_REPO_ID")
            except Exception:
                pass

        # Limit HuggingFace uploads strictly to best model checkpoints (e.g. fold0_best_model.pth)
        if not (target_repo and hf_token and filename.endswith("_best_model.pth")):
            return

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

        last_error = None
        try:
            import logging
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            from huggingface_hub import HfApi
            api = HfApi()
            for r_type in candidate_types:
                try:
                    try:
                        api.create_repo(repo_id=clean_repo, token=hf_token, repo_type=r_type, exist_ok=True)
                    except Exception as e_create:
                        logger.debug(f"HF create_repo ({r_type}) note: {e_create}")

                    api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo=filename,
                        repo_id=clean_repo,
                        token=hf_token,
                        repo_type=r_type
                    )
                    logger.info(f"   ☁ Checkpoint '{filename}' successfully backed up to HuggingFace ({clean_repo} | type={r_type})")
                    return
                except Exception as e_up:
                    last_error = e_up
                    continue
            logger.warning(f"   ⚠ Could not upload checkpoint to HuggingFace ({clean_repo}): {last_error}")
        except Exception as e:
            logger.warning(f"   ⚠ Could not initialize HuggingFace upload: {e}")
