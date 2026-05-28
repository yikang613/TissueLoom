import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import torch
import wandb
import numpy as np
import random


# ----------------------------
# Utilities
# ----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (optional; may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_label_container(ds):
    """
    Try common attribute names used by PyTorch datasets.
    Returns (attr_name, labels_obj) or (None, None).
    """
    for name in ["labels", "targets", "y", "Y"]:
        if hasattr(ds, name):
            obj = getattr(ds, name)
            return name, obj
    return None, None


def _as_numpy_1d(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    # list or other sequence
    return np.asarray(x).reshape(-1)


def _write_back(ds, attr_name, new_labels_np):
    """
    Write labels back preserving original container type.
    """
    old = getattr(ds, attr_name)
    if torch.is_tensor(old):
        new_t = torch.as_tensor(new_labels_np, dtype=old.dtype, device=old.device)
        setattr(ds, attr_name, new_t)
    elif isinstance(old, np.ndarray):
        setattr(ds, attr_name, new_labels_np.astype(old.dtype, copy=False))
    else:
        # list or something else
        setattr(ds, attr_name, new_labels_np.tolist())


def permute_labels_in_dataloader(dataloader, rng: np.random.Generator):
    """
    Permute labels stored in the underlying dataset object.
    This is the key fix vs shuffling batch tensors.
    """
    ds = dataloader.dataset
    attr, lab_obj = _get_label_container(ds)
    if attr is None:
        raise RuntimeError(
            "Could not find dataset labels attribute. "
            "Expected dataset to have one of: labels/targets/y/Y"
        )

    labs = _as_numpy_1d(lab_obj)
    perm = rng.permutation(len(labs))
    labs_perm = labs[perm]
    _write_back(ds, attr, labs_perm)
    return attr, labs  # return original labels for restoration


def restore_labels_in_dataloader(dataloader, attr_name, original_labels_np):
    ds = dataloader.dataset
    _write_back(ds, attr_name, original_labels_np)


def build_training_objects(cfg: DictConfig):
    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    return training


def train_and_get_test_metrics(training):
    training.train()
    metrics = training.get_metrics()

    # Make this robust to key naming
    # (adjust if your project uses different keys)
    out = {}
    for k in ["Test Accuracy", "Test AUC", "Test F1"]:
        if k not in metrics:
            raise KeyError(f"Metric '{k}' not found in training.get_metrics(). Available: {list(metrics.keys())}")
        out[k] = float(metrics[k])
    return out


# ----------------------------
# Main permutation test
# ----------------------------
def run_permutation_test(cfg: DictConfig, num_permutations=20, base_seed=1234):
    """
    Proper permutation test:
      1) Fit/eval once with true labels -> observed AUC
      2) Repeat with permuted labels -> null distribution
      3) p-value = (1 + count(null_auc >= obs_auc)) / (1 + n_perm_kept)
    """

    print(f"[PermutationTest] Starting. num_permutations={num_permutations}")

    # ------------------
    # 1) Observed run
    # ------------------
    set_all_seeds(base_seed)
    training_obs = build_training_objects(cfg)
    obs_metrics = train_and_get_test_metrics(training_obs)
    obs_auc = obs_metrics["Test AUC"]
    print(f"[PermutationTest] OBSERVED: Acc={obs_metrics['Test Accuracy']:.4f} "
          f"AUC={obs_auc:.4f} F1={obs_metrics['Test F1']:.4f}")

    # ------------------
    # 2) Permutation runs
    # ------------------
    perm_acc, perm_auc, perm_f1 = [], [], []
    rng = np.random.default_rng(base_seed)

    for p in range(num_permutations):
        # New seed per permutation for training stochasticity (keeps runs independent)
        seed_p = base_seed + 1000 + p
        set_all_seeds(seed_p)

        training = build_training_objects(cfg)

        # Permute TRAIN labels at dataset level (not batch tensors)
        # This keeps your loaders unchanged, but enforces the null hypothesis.
        try:
            attr_name, orig_labels = permute_labels_in_dataloader(training.train_dataloader, rng)
        except Exception as e:
            print(f"[PermutationTest] Perm {p+1}: failed to permute labels: {e}")
            continue

        try:
            m = train_and_get_test_metrics(training)
            perm_acc.append(m["Test Accuracy"])
            perm_auc.append(m["Test AUC"])
            perm_f1.append(m["Test F1"])
            print(f"[PermutationTest] Perm {p+1}/{num_permutations}: "
                  f"Acc={m['Test Accuracy']:.4f} AUC={m['Test AUC']:.4f} F1={m['Test F1']:.4f}")
        except Exception as e:
            print(f"[PermutationTest] Perm {p+1}: training failed: {e}")
        finally:
            # Always restore original labels so each permutation is clean
            restore_labels_in_dataloader(training.train_dataloader, attr_name, orig_labels)

    perm_auc = np.asarray(perm_auc, dtype=float)

    if perm_auc.size == 0:
        print("[PermutationTest] No successful permutations. Cannot compute p-value.")
        return

    # p-value: add-one smoothing (recommended)
    p_value_auc = (1.0 + float(np.sum(perm_auc >= obs_auc))) / (1.0 + float(perm_auc.size))

    print("\n[PermutationTest] NULL DISTRIBUTION (permuted labels)")
    print(f"  kept_permutations: {perm_auc.size}")
    print(f"  AUC: {perm_auc.mean():.4f} ± {perm_auc.std(ddof=1) if perm_auc.size > 1 else 0.0:.4f}")
    print(f"  Acc: {np.mean(perm_acc):.4f} ± {np.std(perm_acc, ddof=1) if len(perm_acc) > 1 else 0.0:.4f}")
    print(f"  F1 : {np.mean(perm_f1):.4f} ± {np.std(perm_f1, ddof=1) if len(perm_f1) > 1 else 0.0:.4f}")

    print("\n[PermutationTest] P-VALUE (AUC)")
    print(f"  observed_auc: {obs_auc:.4f}")
    print(f"  p_value_auc:  {p_value_auc:.6f}")

    if p_value_auc < 0.05:
        print("[PermutationTest] ✅ Significant: AUC is above the permuted-label null (p < 0.05).")
    else:
        print("[PermutationTest] ❌ Not significant: cannot reject null (p >= 0.05).")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Make runs easier to track; Hydra will change working dir by default. :contentReference[oaicite:2]{index=2}
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
        cfg.wandb_mode = "disabled"

    wandb.init(mode="disabled")

    # You can change the number here
    run_permutation_test(cfg, num_permutations=20, base_seed=1234)


if __name__ == '__main__':
    main()
