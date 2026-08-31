# TissueLoom

A **tissue-aware brain-network framework** for whole-brain functional connectivity analysis across the Alzheimer's disease (AD) continuum. TissueLoom jointly represents gray-matter (Schaefer-200) and white-matter (JHU/Eve, 48 bundles) regions in a unified 248-node graph and introduces three coupled mechanisms:

1. **Tissue-conditioned self-attention** — injects gray/white tissue identity into the attention scores.
2. **Bidirectional cross-tissue affinity propagation** — models GM↔WM coupling with a data-driven healthy-coupling prior.
3. **Anatomy-informed circuit-dictionary readout** — aggregates node representations over canonical white-matter tract circuits.


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
│   └── model/             # ta_bnt_final_configs (TissueLoom) + baselines
├── models/
│   ├── tissueformer/      # TissueLoom model — our contribution (ta_bnt_final.py = final model)
│   ├── BNT/               # upstream Brain Network Transformer base (also used by baselines)
│   ├── ComBrainTF/  DHGFormer/  LRBGT/   # transformer baselines
│   └── brainnetcnn.py  brainnetmlp.py  fbnetgen.py  transformer.py
├── dataset/               # dataset loaders (no data included)
├── training/              # training loops (incl. tabnt_training.py)
├── components/            # logger, optimizer, LR scheduler
└── utils/                 # healthy-coupling prior, metrics, seeding, etc.
```

The model itself is `source/models/tissueformer/ta_bnt_final.py`, configured via `source/conf/model/ta_bnt_final_configs.yaml`. The `tissueformer/` directory and the `ta_bnt_*` file names are kept from the project's earlier name so that existing configs and checkpoints keep working.

## Data

**No data is included in this repository.** The study uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), which is governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/) and cannot be redistributed. Apply for access through ADNI.

The dataset config files (`source/conf/dataset/*.yaml`) use placeholder paths (e.g. `/path/to/adni_fc_data`) — point these to your own functional-connectivity files after obtaining the data.

## Quick start

```bash
# Train TissueLoom on one of the binary AD-continuum tasks
# (pAD / EMCI / MCI / LMCI / AD, each vs cognitively normal controls)
python -m source dataset=pAD model=ta_bnt_final_configs
```

Override any Hydra config field on the command line (e.g. `dataset.batch_size=32`).

## Acknowledgements

Built on the Brain Network Transformer codebase
([Wayfear/BrainNetworkTransformer](https://github.com/Wayfear/BrainNetworkTransformer), NeurIPS 2022) — see `README_upstream_BNT.md` for the original README. Data: Alzheimer's Disease Neuroimaging Initiative (ADNI).

## Citation

If you use TissueLoom, please cite the manuscript (in preparation).
