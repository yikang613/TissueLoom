from datetime import datetime
import wandb
import hydra
import numpy as np
import torch
import json
import os
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from sklearn.metrics import roc_auc_score
from datetime import datetime

# Optional seed fixing
try:
    from .utils.seed import seed_everything
    _HAS_SEED = True
except ImportError:
    _HAS_SEED = False


def resolve_model_name(model_cfg: DictConfig) -> str:
    if hasattr(model_cfg, 'name'):
        return model_cfg.name
    selected_experiment = None
    if hasattr(model_cfg, 'experiment'):
        selected_experiment = model_cfg.experiment
    elif hasattr(model_cfg, 'exp_name'):
        selected_experiment = model_cfg.exp_name
    if selected_experiment and hasattr(model_cfg, selected_experiment):
        experiment_cfg = getattr(model_cfg, selected_experiment)
        if hasattr(experiment_cfg, 'name'):
            return experiment_cfg.name
    return "unknown_model"


def resolve_selected_experiment(model_cfg: DictConfig):
    if hasattr(model_cfg, 'experiment'):
        return model_cfg.experiment
    if hasattr(model_cfg, 'exp_name'):
        return model_cfg.exp_name
    return None


def _extract_scalar(val):
    """Extract scalar from metrics (may be list or scalar)."""
    if isinstance(val, (list, np.ndarray)):
        return float(val[0]) if len(val) > 0 else 0.0
    return float(val)


def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)

    model_name = resolve_model_name(cfg.model)
    selected_experiment = resolve_selected_experiment(cfg.model)
    is_deviation_experiment = (
        model_name == 'TissueAwareBNT_v2'
        and selected_experiment in ('exp14_deviation', 'exp15_deviation_only')
    )

    if is_deviation_experiment:
        from source.utils.healthy_baseline import (
            setup_healthy_baseline_from_dataloader,
            verify_baseline_set,
        )
        setup_healthy_baseline_from_dataloader(
            model, dataloaders[0], device='cuda'
        )
        if not verify_baseline_set(model):
            raise RuntimeError(
                f"Healthy baseline was not set for experiment='{selected_experiment}'."
            )

    is_prior_experiment = (
        model_name == 'TissueAwareBNT_v2'
        and getattr(cfg.model.get(selected_experiment, cfg.model), 'use_prior', False)
    )
    if is_prior_experiment:
        from source.utils.healthy_baseline import (
            setup_functional_prior_from_dataloader,
            verify_baseline_set,
        )
        setup_functional_prior_from_dataloader(
            model, dataloaders[0], device='cuda'
        )
        if not verify_baseline_set(model):
            raise RuntimeError("Functional prior not set but use_prior=True.")

    is_zone_model = model_name in ('TissueAwareBNT_v3', 'TissueAwareBNT_final')
    if is_zone_model:
        from source.utils.healthy_baseline_v3 import (
            setup_v3_for_fold,
            verify_v3_setup,
        )
        model_device = next(model.parameters()).device
        setup_v3_for_fold(
            model, dataloaders[0], device=model_device,
            zone_method='topk', zone_k=15,
        )
        if not verify_v3_setup(model):
            raise RuntimeError("TA-BNT zone/prior setup failed.")
        logger.info(f'[{model_name}] setup_v3_for_fold completed')

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    training.train()

    # --- Optional: save checkpoint + extract features for interpretability ---
    # Activate with: +save_checkpoint=true
    save_ckpt = getattr(cfg, 'save_checkpoint', False)
    if save_ckpt:
        import json
        try:
            from hydra.utils import get_original_cwd
            ckpt_dir = os.path.join(get_original_cwd(), 'model_features')
        except Exception:
            ckpt_dir = os.path.join(os.getcwd(), 'model_features')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model state dict (best model already restored by early stopping)
        ckpt_path = os.path.join(ckpt_dir, 'ta_bnt_checkpoint.pt')
        torch.save(training.model.state_dict(), ckpt_path)
        logger.info(f'[Checkpoint] Saved to {ckpt_path}')

        # Extract tissue-pair bias
        inner = training.model.model if hasattr(training.model, 'model') else training.model
        if hasattr(inner, 'attention') and hasattr(inner.attention, 'tissue_pair_bias'):
            bias = inner.attention.tissue_pair_bias.detach().cpu().numpy()
            np.save(os.path.join(ckpt_dir, 'tissue_pair_bias.npy'), bias)
            logger.info(f'[Checkpoint] Tissue-pair bias: '
                        f'GM->GM={bias[0,0]:+.4f}, GM->WM={bias[0,1]:+.4f}, '
                        f'WM->GM={bias[1,0]:+.4f}, WM->WM={bias[1,1]:+.4f}')

        # Save zones and B_prior (needed to reload model for extraction)
        if hasattr(inner, '_zones') and inner._zones is not None:
            zones_serializable = {k: {'gm': v['gm'], 'wm': v['wm']}
                                  for k, v in inner._zones.items()}
            with open(os.path.join(ckpt_dir, 'zones.json'), 'w') as f:
                json.dump(zones_serializable, f, indent=2)
        if hasattr(inner, 'bipartite') and hasattr(inner.bipartite, 'B_prior'):
            np.save(os.path.join(ckpt_dir, 'B_prior.npy'),
                    inner.bipartite.B_prior.cpu().numpy())

        # Extract test set predictions + affinity matrices
        import torch.nn.functional as F_ckpt
        training.model.eval()
        all_probs = []
        all_labels = []
        all_biadj = []

        with torch.no_grad():
            for time_series, node_feature, label in dataloaders[2]:  # test loader
                time_series = time_series.cuda()
                node_feature = node_feature.cuda()
                output = training.model(time_series, node_feature)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                probs = F_ckpt.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(label[:, 1].cpu().tolist())

                # Get affinity matrix
                if hasattr(training.model, 'get_biadj'):
                    biadj = training.model.get_biadj()
                    if biadj is not None:
                        all_biadj.append(biadj.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        if all_biadj:
            all_biadj = np.concatenate(all_biadj, axis=0)
            np.save(os.path.join(ckpt_dir, 'test_affinity_matrices.npy'), all_biadj)

        np.savez(os.path.join(ckpt_dir, 'test_predictions.npz'),
                 probs=all_probs, labels=all_labels)

        logger.info(f'[Checkpoint] Test predictions: {len(all_probs)} subjects, '
                    f'AUC={roc_auc_score(all_labels, all_probs):.4f}')
        if all_biadj is not None and len(all_biadj) > 0:
            logger.info(f'[Checkpoint] Affinity matrices: {all_biadj.shape}')

    return training



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    all_metrics = {
        'accuracy': [],
        'AUC': [],
        'F1-score': [],
        'sensitivity': [],
        'specificity': [],
    }

    model_name = resolve_model_name(cfg.model)
    kfold_cfg = getattr(cfg.dataset, 'k_fold', None)
    kfold_enabled = bool(getattr(kfold_cfg, 'enabled', False))
    run_count = int(getattr(kfold_cfg, 'n_splits', 5)) if kfold_enabled else cfg.repeat_time
    group_name = f"{cfg.dataset.name}_{model_name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"

    base_seed = getattr(cfg, 'seed', None)

    # Per-fold results for saving
    fold_results = []

    for run_idx in range(run_count):
        if base_seed is not None and _HAS_SEED:
            run_seed = base_seed + run_idx
            seed_everything(run_seed, deterministic=True)
            print(f"\n[Seed] Run {run_idx+1}/{run_count}, seed={run_seed}")

        if kfold_enabled:
            with open_dict(cfg):
                cfg.dataset.k_fold.current_fold = run_idx
                if base_seed is not None:
                    cfg.dataset.k_fold.split_seed = base_seed
            run_name = f"fold_{run_idx+1}_of_{run_count}"
        else:
            run_name = None

        run = wandb.init(
            project=cfg.project,
            reinit=True,
            group=f"{group_name}",
            tags=[f"{cfg.dataset.name}"],
            mode="offline",
            name=run_name,
        )
        training = model_training(cfg)
        metrics = training.get_metrics()

        auc = _extract_scalar(metrics['Test AUC'])
        acc = _extract_scalar(metrics['Test Accuracy'])
        f1 = _extract_scalar(metrics['Test F1'])
        sens = _extract_scalar(metrics['Test Sensitivity'])
        spec = _extract_scalar(metrics['Test Specificity'])

        # --- Save per-fold test predictions for interpretability ---
        # Activate with: ++save_predictions=true
        save_preds = getattr(cfg, 'save_predictions', False)
        # Hydra may pass "true" as string instead of boolean
        if isinstance(save_preds, str):
            save_preds = save_preds.lower() in ('true', '1', 'yes')
        print(f"[DEBUG] save_predictions={save_preds}, "
              f"has _collect_probs_labels={hasattr(training, '_collect_probs_labels')}, "
              f"has test_dataloader={hasattr(training, 'test_dataloader')}")

        if save_preds:
            try:
                from hydra.utils import get_original_cwd
                pred_dir = os.path.join(get_original_cwd(), 'subject_predictions')
            except Exception:
                pred_dir = os.path.join(os.getcwd(), 'subject_predictions')
            os.makedirs(pred_dir, exist_ok=True)
            print(f"[DEBUG] pred_dir={pred_dir}")

            if hasattr(training, '_collect_probs_labels') and hasattr(training, 'test_dataloader'):
                test_probs, test_labels = training._collect_probs_labels(
                    training.test_dataloader)

                pred_file = os.path.join(pred_dir,
                    f"{model_name}_{cfg.dataset.name}_{getattr(cfg.dataset, 'input_type', 'default')}.csv")
                write_header = not os.path.exists(pred_file)
                with open(pred_file, 'a') as pf:
                    if write_header:
                        pf.write("seed,fold,subject_idx,prob,label\n")
                    for idx, (p, l) in enumerate(zip(test_probs, test_labels)):
                        pf.write(f"{base_seed},{run_idx},{idx},{p:.6f},{int(l)}\n")
                print(f"[Saved] {len(test_probs)} predictions to {pred_file}")
            else:
                print(f"[WARNING] Cannot save predictions: "
                      f"_collect_probs_labels={hasattr(training, '_collect_probs_labels')}, "
                      f"test_dataloader={hasattr(training, 'test_dataloader')}")

        all_metrics['accuracy'].append(acc)
        all_metrics['AUC'].append(auc)
        all_metrics['F1-score'].append(f1)
        all_metrics['sensitivity'].append(sens)
        all_metrics['specificity'].append(spec)

        fold_results.append({
            'fold': run_idx,
            'seed': base_seed + run_idx if base_seed is not None else None,
            'AUC': auc,
            'Accuracy': acc,
            'F1': f1,
            'Sensitivity': sens,
            'Specificity': spec,
        })

        run.finish()

    # Calculate and print mean and standard deviation
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    # ── Auto-save results to JSON ──
    dataset_name = cfg.dataset.name
    input_type = getattr(cfg.dataset, 'input_type', 'default')
    experiment = resolve_selected_experiment(cfg.model) or 'default'

    save_data = {
        'model_name': model_name,
        'experiment': experiment,
        'dataset': dataset_name,
        'input_type': input_type,
        'base_seed': base_seed,
        'n_folds': run_count,
        'kfold_enabled': kfold_enabled,
        'summary': {
            'AUC_mean': float(np.mean(all_metrics['AUC'])),
            'AUC_std': float(np.std(all_metrics['AUC'])),
            'Accuracy_mean': float(np.mean(all_metrics['accuracy'])),
            'Accuracy_std': float(np.std(all_metrics['accuracy'])),
            'F1_mean': float(np.mean(all_metrics['F1-score'])),
            'F1_std': float(np.std(all_metrics['F1-score'])),
            'Sensitivity_mean': float(np.mean(all_metrics['sensitivity'])),
            'Sensitivity_std': float(np.std(all_metrics['sensitivity'])),
            'Specificity_mean': float(np.mean(all_metrics['specificity'])),
            'Specificity_std': float(np.std(all_metrics['specificity'])),
        },
        'per_fold': fold_results,
    }

    # Save to results directory (use Hydra's ORIGINAL cwd, not the
    # per-run output directory, so all seeds accumulate in one place)
    try:
        from hydra.utils import get_original_cwd
        results_dir = os.path.join(get_original_cwd(), 'saved_results')
    except Exception:
        results_dir = os.path.join(os.getcwd(), 'saved_results')
    os.makedirs(results_dir, exist_ok=True)

    # Optional suffix that lets ablation variants of the same model class
    # write to distinct CSVs (e.g., TissueAwareBNT_final_no_bias_*).
    name_suffix = getattr(cfg, 'model_name_suffix', '') or ''
    if name_suffix and not name_suffix.startswith('_'):
        name_suffix = '_' + name_suffix
    tagged_model_name = f"{model_name}{name_suffix}"

    seed_str = f"_seed{base_seed}" if base_seed is not None else ""
    filename = f"{tagged_model_name}_{dataset_name}_{input_type}{seed_str}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Saved] {filepath}")

    # Also append to a master CSV for easy aggregation across seeds
    csv_path = os.path.join(results_dir,
                            f"{tagged_model_name}_{dataset_name}_{input_type}_all.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write("seed,fold,AUC,Accuracy,F1,Sensitivity,Specificity\n")
        for fr in fold_results:
            f.write(f"{fr['seed']},{fr['fold']},"
                    f"{fr['AUC']:.6f},{fr['Accuracy']:.6f},"
                    f"{fr['F1']:.6f},{fr['Sensitivity']:.6f},"
                    f"{fr['Specificity']:.6f}\n")
    print(f"[Appended] {csv_path}")


if __name__ == '__main__':
    main()