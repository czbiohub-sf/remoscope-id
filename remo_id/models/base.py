"""Shared base class for all remo-id models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class BaseModel(pl.LightningModule):
    """Shared base for all remo-id models.

    Handles loss setup, LR scheduling, optimizer configuration,
    and training/validation/test steps via _forward_step().

    Subclasses must implement _forward_step(batch) -> (loss, logits, labels).
    """

    def _setup_loss(self, task_type, num_classes, class_weights=None):
        """Configure loss function based on task type.

        Args:
            task_type: 'binary' or 'multiclass'
            num_classes: Number of classes
            class_weights: Tensor of per-class weights from batch composition.
                For binary: scalar tensor used as BCEWithLogitsLoss pos_weight.
                For multiclass: 1D tensor used as CrossEntropyLoss weight.
        """
        self.task_type = task_type
        self.num_classes = num_classes
        if task_type == "multiclass":
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:  # binary
            pos_weight = class_weights if class_weights is not None else None
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _setup_lr(self, lr, lr_schedule, warmup_epochs, max_epochs, weight_decay=0.01):
        """Store LR scheduling parameters."""
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def _compute_loss(self, logits, labels):
        """Task-type-aware loss computation.

        Handles shape alignment: logits may be (N,) or (N,1) for binary.
        For multiclass with 2D float labels, treats labels as a soft proportion
        vector and uses soft-target cross-entropy (LLP training).
        """
        if self.task_type == "multiclass":
            # Soft proportion vector: (B, C) float target for LLP
            if labels.dim() == 2 and labels.dtype.is_floating_point:
                weight = getattr(self.loss, "weight", None)
                return F.cross_entropy(logits, labels, weight=weight)
            return self.loss(logits, labels.long())
        else:
            # Binary: match logits shape — squeeze or unsqueeze labels accordingly
            if logits.dim() > 1 and logits.shape[-1] == 1:
                return self.loss(logits, labels.float().unsqueeze(1))
            else:
                return self.loss(logits, labels.float())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[self.warmup_epochs])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer

    def _forward_step(self, batch):
        """Returns (loss, logits, labels). Must be implemented by subclasses."""
        raise NotImplementedError

    def _soft_metrics(self, logits, labels):
        """Compute proportion_l1 and top1_match metrics for multiclass runs.

        Works for both hard labels (1D int tensor) and soft proportion vectors
        (2D float tensor). Hard labels are one-hot encoded before the comparison
        so the metric values are comparable across hard/soft training modes.
        """
        probs = F.softmax(logits, dim=-1)
        if labels.dim() == 1:
            target = F.one_hot(labels.long(), self.num_classes).float()
        else:
            target = labels.float()
        l1 = (probs - target).abs().mean()
        top1 = (probs.argmax(-1) == target.argmax(-1)).float().mean()
        return l1, top1

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._forward_step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.task_type == "multiclass":
            l1, top1 = self._soft_metrics(logits, labels)
            self.log('train/proportion_l1', l1, on_step=False, on_epoch=True)
            self.log('train/top1_match', top1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._forward_step(batch)
        self.log('validation/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.task_type == "multiclass":
            l1, top1 = self._soft_metrics(logits, labels)
            self.log('validation/proportion_l1', l1, on_step=False, on_epoch=True, sync_dist=True)
            self.log('validation/top1_match', top1, on_step=False, on_epoch=True, sync_dist=True)
        self._log_validation_metrics(logits, labels)
        return loss

    def _log_validation_metrics(self, logits, labels):
        """Log task-appropriate validation metrics."""
        if self.task_type == "binary":
            probs = torch.sigmoid(logits.squeeze(-1) if logits.dim() > 1 else logits)
            preds = (probs >= 0.5).float()
            bag_labels = labels.float()
            acc = (preds == bag_labels).float().mean()
            self.log('validation/accuracy', acc, on_step=False, on_epoch=True, sync_dist=True)
            tp = ((preds == 1) & (bag_labels == 1)).sum().float()
            fp = ((preds == 1) & (bag_labels == 0)).sum().float()
            fn = ((preds == 0) & (bag_labels == 1)).sum().float()
            self.log('validation/precision', tp / (tp + fp + 1e-8), on_step=False, on_epoch=True, sync_dist=True)
            self.log('validation/recall', tp / (tp + fn + 1e-8), on_step=False, on_epoch=True, sync_dist=True)
            tn = ((preds == 0) & (bag_labels == 0)).sum().float()
            self.log('validation/specificity', tn / (tn + fp + 1e-8), on_step=False, on_epoch=True, sync_dist=True)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            self.log('validation/f1', f1, on_step=False, on_epoch=True, sync_dist=True)
        elif self.task_type == "multiclass":
            preds = logits.argmax(dim=-1)
            # Soft proportion labels → hard class indices via argmax
            hard_labels = labels.argmax(dim=-1) if labels.dim() == 2 else labels
            acc = (preds == hard_labels).float().mean()
            self.log('validation/accuracy', acc, on_step=False, on_epoch=True, sync_dist=True)

            for c in range(self.num_classes):
                mask = hard_labels == c
                if mask.sum() > 0:
                    class_acc = (preds[mask] == c).float().mean()
                    self.log(f'validation/accuracy_class{c}', class_acc, on_step=False, on_epoch=True, sync_dist=True)

            # F1-macro from per-class precision/recall
            f1_sum = 0.0
            for c in range(self.num_classes):
                tp = ((preds == c) & (hard_labels == c)).sum().float()
                fp = ((preds == c) & (hard_labels != c)).sum().float()
                fn = ((preds != c) & (hard_labels == c)).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1_sum += 2 * prec * rec / (prec + rec + 1e-8)
            self.log('validation/f1_macro', f1_sum / self.num_classes, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Print train/val loss to stdout for SLURM log tracking."""
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get('train/loss')
        val_loss = metrics.get('validation/loss')
        if train_loss is not None and val_loss is not None:
            parts = [f"Epoch {self.current_epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}"]
            val_top1 = metrics.get('validation/top1_match')
            if val_top1 is not None:
                parts.append(f"val_top1={val_top1:.4f}")
            val_acc = metrics.get('validation/accuracy')
            if val_acc is not None:
                parts.append(f"val_acc={val_acc:.4f}")
            val_f1 = metrics.get('validation/f1') or metrics.get('validation/f1_macro')
            if val_f1 is not None:
                parts.append(f"val_f1={val_f1:.4f}")
            print(" | ".join(parts), flush=True)

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._forward_step(batch)
        self.log('test/loss', loss)
        return loss
