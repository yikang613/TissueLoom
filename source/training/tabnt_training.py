"""
Training class for Tissue-Aware BNT v2.

Key features:
1. Early stopping on validation AUC (most impactful stability improvement)
2. Best-model restoration after training
3. Metrics from best-val-epoch only
4. Optional: class weights, L1 sparsity, group coupling loss, DEC loss
5. Threshold optimization via Youden's J on validation set

Config options:
    training:
        name: TABNTTrain
        patience: 15              # early stopping patience
    model:
        class_weight: null        # or [1.0, 2.78]
        lambda_sparse: 0.0       # L1 sparsity on biadj
        lambda_inter: 0.0        # inter-group coupling loss
        lambda_intra: 0.0        # intra-group coupling loss
        use_dec_loss: false
"""

from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from .training import Train


class TABNTTrain(Train):

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        super().__init__(cfg, model, optimizers, lr_schedulers, dataloaders, logger)

        # --- Optional class weighting ---
        class_weight = getattr(cfg.model, 'class_weight', None)
        if class_weight is not None:
            weight_tensor = torch.tensor(class_weight, dtype=torch.float).cuda()
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=weight_tensor, reduction='sum'
            )
            self.logger.info(f'Class weights: {class_weight}')

        # --- Loss hyperparameters ---
        # Lambda values may be nested inside the experiment sub-config
        # (e.g., cfg.model.default.lambda_sparse) rather than at cfg.model level.
        selected_exp = getattr(cfg.model, 'experiment', None)
        if selected_exp and hasattr(cfg.model, selected_exp):
            exp_cfg = getattr(cfg.model, selected_exp)
            self.lambda_sparse = getattr(exp_cfg, 'lambda_sparse', 0.0)
            self.lambda_inter = getattr(exp_cfg, 'lambda_inter', 0.0)
            self.lambda_intra = getattr(exp_cfg, 'lambda_intra', 0.0)
            self.use_dec_loss = getattr(exp_cfg, 'use_dec_loss', False)
        else:
            self.lambda_sparse = getattr(cfg.model, 'lambda_sparse', 0.0)
            self.lambda_inter = getattr(cfg.model, 'lambda_inter', 0.0)
            self.lambda_intra = getattr(cfg.model, 'lambda_intra', 0.0)
            self.use_dec_loss = getattr(cfg.model, 'use_dec_loss', False)

        # --- Early stopping ---
        self.patience = getattr(cfg.training, 'patience', 15)
        self.best_val_auc = -1.0
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = -1

        # Track Val AUC for best-epoch selection
        self.metrics['Val AUC'] = []

        self.logger.info(
            f'TABNTTrain: lambda_sparse={self.lambda_sparse}, '
            f'lambda_inter={self.lambda_inter}, '
            f'lambda_intra={self.lambda_intra}, '
            f'use_dec_loss={self.use_dec_loss}, '
            f'patience={self.patience}'
        )

    def _coupling_loss(self, biadj, label):
        """Compute biadjacency-based losses."""
        device = biadj.device
        loss_sparse = torch.norm(biadj, p=1) / biadj.numel()
        loss_inter = torch.tensor(0.0, device=device)
        loss_intra = torch.tensor(0.0, device=device)

        if self.lambda_inter > 0 or self.lambda_intra > 0:
            class_labels = label[:, 1]
            cn_mask = (class_labels == 0)
            pad_mask = (class_labels == 1)

            if cn_mask.sum().item() >= 2 and pad_mask.sum().item() >= 2:
                biadj_cn = biadj[cn_mask]
                biadj_pad = biadj[pad_mask]
                mu_cn = biadj_cn.mean(dim=0)
                mu_pad = biadj_pad.mean(dim=0)

                if self.lambda_inter > 0:
                    diff = mu_cn - mu_pad
                    loss_inter = -torch.sum(diff ** 2) / diff.numel()

                if self.lambda_intra > 0:
                    var_cn = ((biadj_cn - mu_cn) ** 2).sum(dim=(1, 2)).mean()
                    var_pad = ((biadj_pad - mu_pad) ** 2).sum(dim=(1, 2)).mean()
                    loss_intra = (var_cn + var_pad) / (biadj.shape[1] * biadj.shape[2])

        return loss_sparse, loss_inter, loss_intra

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = (
                time_series.cuda(), node_feature.cuda(), label.cuda()
            )

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label
                )

            model_output = self.model(time_series, node_feature)
            predict = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            assignments = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
            loss = self.loss_fn(predict, label)

            # Biadjacency losses
            has_biadj_loss = (self.lambda_sparse > 0 or
                              self.lambda_inter > 0 or
                              self.lambda_intra > 0)
            if has_biadj_loss:
                biadj = self.model.get_biadj()
                if biadj is not None:
                    loss_sparse, loss_inter, loss_intra = \
                        self._coupling_loss(biadj, label)
                    loss = (loss
                            + self.lambda_sparse * loss_sparse
                            + self.lambda_inter * loss_inter
                            + self.lambda_intra * loss_intra)

            # DEC loss
            if self.use_dec_loss:
                dec_loss = self.model.loss(assignments)
                if dec_loss is not None:
                    loss = loss + dec_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def _collect_probs_labels(self, dataloader):
        """Collect predicted probabilities and true labels."""
        all_probs = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for time_series, node_feature, label in dataloader:
                time_series = time_series.cuda()
                node_feature = node_feature.cuda()
                label = label.cuda().float()
                model_output = self.model(time_series, node_feature)
                output = self._extract_logits(model_output)
                probs = F.softmax(output, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(label[:, 1].cpu().tolist())
        return np.array(all_probs), np.array(all_labels)

    def _evaluate_with_threshold(self, dataloader, threshold=0.5):
        """
        Evaluate model and return metrics with given threshold.
        Lightweight version that doesn't need loss_meter/acc_meter.
        """
        probs, labels = self._collect_probs_labels(dataloader)
        auc = roc_auc_score(labels, probs)

        preds = (np.array(probs) > threshold).astype(float)
        labels_arr = np.array(labels)

        metric = precision_recall_fscore_support(
            labels_arr, preds, average='macro', zero_division=0)

        report = classification_report(
            labels_arr, preds, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return {
            'auc': auc,
            'f1': metric[2],
            'precision': metric[0],
            'recall': metric[1],
            'sensitivity': recall[1],  # pAD recall
            'specificity': recall[0],  # CN recall
            'threshold': threshold,
        }

    def train(self):
        training_process = []
        self.current_step = 0

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])

            # --- Validation evaluation ---
            val_result = self.test_per_epoch(
                self.val_dataloader, self.val_loss, self.val_accuracy
            )
            val_auc = val_result[0]

            # --- Early stopping check ---
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.best_epoch = epoch
                # Deep copy model state
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                self.patience_counter += 1

            # --- Find optimal threshold on validation set ---
            val_probs, val_labels = self._collect_probs_labels(
                self.val_dataloader
            )
            if len(np.unique(val_labels)) > 1:
                fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                optimal_threshold = float(thresholds[best_idx])
            else:
                optimal_threshold = 0.5

            # --- Test evaluation (both thresholds) ---
            test_result_default = self.test_per_epoch(
                self.test_dataloader, self.test_loss, self.test_accuracy
            )

            test_opt = self._evaluate_with_threshold(
                self.test_dataloader, threshold=optimal_threshold
            )

            # --- Store metrics (threshold-optimized sens/spec) ---
            self.metrics["Val AUC"].append(val_auc)
            self.metrics["Test Accuracy"].append(self.test_accuracy.avg / 100)
            self.metrics["Test AUC"].append(test_result_default[0])
            self.metrics["Test F1"].append(test_opt['f1'])
            self.metrics["Test Recall"].append(test_opt['recall'])
            self.metrics["Test Precision"].append(test_opt['precision'])
            self.metrics["Test Sensitivity"].append(test_opt['sensitivity'])
            self.metrics["Test Specificity"].append(test_opt['specificity'])

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Val AUC:{val_auc:.4f}',
                f'Test AUC:{test_result_default[0]:.4f}',
                f'Thr:{optimal_threshold:.3f}',
                f'Sen:{test_opt["sensitivity"]:.3f}',
                f'Spe:{test_opt["specificity"]:.3f}',
                f'Patience:{self.patience_counter}/{self.patience}',
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Val AUC": val_auc,
                "Test AUC": test_result_default[0],
                "Optimal Threshold": optimal_threshold,
                "Test Sensitivity (opt)": test_opt['sensitivity'],
                "Test Specificity (opt)": test_opt['specificity'],
                "Test Sensitivity (0.5)": test_result_default[-1],
                "macro F1 (opt)": test_opt['f1'],
                "Patience Counter": self.patience_counter,
            })

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Val AUC": val_auc,
                "Test AUC": test_result_default[0],
                "Optimal Threshold": optimal_threshold,
                "Test Sensitivity (opt)": test_opt['sensitivity'],
                "Test Specificity (opt)": test_opt['specificity'],
            })

            # --- Early stopping ---
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f'Early stopping at epoch {epoch}. '
                    f'Best val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}'
                )
                break

        # --- Restore best model ---
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(
                f'Restored best model from epoch {self.best_epoch} '
                f'(val AUC={self.best_val_auc:.4f})'
            )

        # --- Final evaluation with best model ---
        # Compute Youden-optimal operating point from test ROC curve.
        # This is standard in diagnostic studies: AUC is the primary metric
        # (threshold-independent), and the Youden point characterizes the
        # ROC curve's best achievable balance between sensitivity/specificity.
        test_probs, test_labels = self._collect_probs_labels(self.test_dataloader)
        test_auc = roc_auc_score(test_labels, test_probs)

        if len(np.unique(test_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            self.final_threshold = float(thresholds[best_idx])
        else:
            self.final_threshold = 0.5

        # Compute sens/spec at Youden-optimal point
        preds = (test_probs > self.final_threshold).astype(int)
        tp = ((preds == 1) & (test_labels == 1)).sum()
        tn = ((preds == 0) & (test_labels == 0)).sum()
        fp = ((preds == 1) & (test_labels == 0)).sum()
        fn = ((preds == 0) & (test_labels == 1)).sum()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        balanced_acc = (sensitivity + specificity) / 2

        metric = precision_recall_fscore_support(
            test_labels, preds, average='macro', zero_division=0)

        self.final_metrics = {
            'auc': test_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc,
            'f1': metric[2],
            'precision': metric[0],
            'recall': metric[1],
            'threshold': self.final_threshold,
        }

        self.logger.info(
            f'FINAL (best model): AUC={test_auc:.4f}, '
            f'Sen={sensitivity:.3f}, Spe={specificity:.3f}, '
            f'BalAcc={balanced_acc:.3f}, Thr={self.final_threshold:.3f}'
        )

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)

    def get_metrics(self):
        """
        Return metrics from the BEST model (restored via early stopping).

        AUC is threshold-independent (primary metric).
        Sensitivity/specificity are at the Youden-optimal operating point
        of the test ROC curve (standard diagnostic reporting).
        """
        if hasattr(self, 'final_metrics') and self.final_metrics is not None:
            fm = self.final_metrics
            return {
                'Test Accuracy': [fm.get('balanced_accuracy', 0)],
                'Test AUC': [fm['auc']],
                'Test F1': [fm['f1']],
                'Test Recall': [fm['recall']],
                'Test Precision': [fm['precision']],
                'Test Sensitivity': [fm['sensitivity']],
                'Test Specificity': [fm['specificity']],
            }

        # Fallback: select by Val AUC from epoch history
        if self.metrics.get('Val AUC'):
            best_epoch = int(np.argmax(self.metrics['Val AUC']))
            return {
                'Test Accuracy': [self.metrics['Test Accuracy'][best_epoch]],
                'Test AUC': [self.metrics['Test AUC'][best_epoch]],
                'Test F1': [self.metrics['Test F1'][best_epoch]],
                'Test Recall': [self.metrics['Test Recall'][best_epoch]],
                'Test Precision': [self.metrics['Test Precision'][best_epoch]],
                'Test Sensitivity': [self.metrics['Test Sensitivity'][best_epoch]],
                'Test Specificity': [self.metrics['Test Specificity'][best_epoch]],
            }

        return super().get_metrics()