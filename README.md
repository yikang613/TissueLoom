# TissueLoom

A **tissue-aware brain-network framework** for whole-brain functional connectivity analysis across the Alzheimer's disease (AD) continuum. TissueLoom jointly represents gray-matter (Schaefer-200) and white-matter (JHU/Eve, 48 bundles) regions in a unified 248-node graph and introduces three coupled mechanisms:

1. **Tissue-conditioned self-attention** — injects gray/white tissue identity into the attention scores.
2. **Bidirectional cross-tissue affinity propagation** — models GM↔WM coupling with a data-driven healthy-coupling prior.
3. **Anatomy-informed circuit-dictionary readout** — aggregates node representations over canonical white-matter tract circuits.

> **Status:** Manuscript in preparation.


## Installation

```bash
pip install -r requirements.txt
```

## Repository layout

```
source/
├── __main__.py            # entry point (Hydra-configured)
├── conf/                  # Hydra configs (dataset / model / training / optimizer / preprocess)
│   ├── dataset/           # AD, EMCI, MCI, LMCI, pAD (+ ABCD/ABIDE from upstream)
│   └── model/             # ta_bnt_final_configs (TissueLoom) + included baselines
├── models/
│   ├── tissueformer/      # TissueLoom model — our contribution (ta_bnt_final.py = final model)
│   ├── BNT/               # upstream Brain Network Transformer base
│   └── brainnetcnn.py  fbnetgen.py  transformer.py
├── dataset/               # dataset loaders (no data included)
├── training/              # training loops (incl. tabnt_training.py)
├── components/            # logger, optimizer, LR scheduler
└── utils/                 # healthy-coupling prior, metrics, seeding, etc.
```

The model itself is `source/models/tissueformer/ta_bnt_final.py`, configured via `source/conf/model/ta_bnt_final_configs.yaml`. The `tissueformer/` directory and the `ta_bnt_*` file names are kept from the project's earlier name so that existing configs and checkpoints keep working.

## Data

**No data is included in this repository.** The study uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), which is governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/) and cannot be redistributed. Apply for access through ADNI.

The dataset config files (`source/conf/dataset/*.yaml`) use placeholder paths (e.g. `/path/to/adni_fc_data`) — point these to your own functional-connectivity files after obtaining the data.

For each binary task, the configured directory must contain a `CN/` folder and the corresponding case folder (`pAD/`, `EMCI/`, `MCI/`, `LMCI/`, or `AD/`). Each participant-level MATLAB file must provide `SCH2_time` with shape `[200, T]` and `Eve_time` with shape `[48, T]`.

## Quick start

```bash
# Train TissueLoom on one of the binary AD-continuum tasks
# (pAD / EMCI / MCI / LMCI / AD, each vs cognitively normal controls)
python -m source dataset=pAD model=ta_bnt_final_configs
```

Override any Hydra config field on the command line (e.g. `dataset.batch_size=32`).

For the repeated nested evaluation used by the paper pipeline, run ten independently shuffled five-fold outer evaluations. Each outer training partition receives its own inner validation split for early stopping and threshold selection; the outer test fold is evaluated only once after the best validation model is restored.

```bash
python -m source \
  dataset=pAD \
  model=ta_bnt_final_configs \
  model.experiment=default \
  training.train=TABNTTrain \
  preprocess=mixup \
  dataset.k_fold.enabled=true \
  dataset.k_fold.n_splits=5 \
  eval_mode=nested_cv \
  n_repeats=10 \
  seed=42
```

The command creates 50 training runs and records the repeat index, outer fold, split seeds, training seed, per-fold metrics, and per-repeat summaries in `saved_results/`.

## Tests

Dependency-free checks cover the 10×5 run plan and Python source integrity:

```bash
python -m unittest discover -s tests -v
```

An end-to-end test is not included because ADNI-derived inputs cannot be redistributed. The full training command must still be validated in the paper's CUDA environment with authorized data.

## Acknowledgements

Built on the Brain Network Transformer codebase
([Wayfear/BrainNetworkTransformer](https://github.com/Wayfear/BrainNetworkTransformer), NeurIPS 2022) — see `README_upstream_BNT.md` for the original README. The BNT clustering components use [pt-dec](https://github.com/vlukiyanov/pt-dec).

Reference baseline implementations used during research include [Com-BrainTF](https://github.com/ubc-tea/Com-BrainTF), [DHGFormer](https://github.com/iMoonLab/DHGFormer), [ALTER/LRBGT](https://github.com/yushuowiki/ALTER), and [BrainNetMLP](https://github.com/JayceonHo/BrainNetMLP). Their source code is not redistributed in the current repository; please use the official repositories under their respective terms.

Data: Alzheimer's Disease Neuroimaging Initiative (ADNI). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for software notices and license texts.

## Citation

If you use TissueLoom, please cite the manuscript (in preparation).
