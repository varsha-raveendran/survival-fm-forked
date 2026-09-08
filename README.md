# Adapting Tabular Foundation Models for Time-to-Event Prediction

[![Paper](https://img.shields.io/badge/paper-AIiH%202026-green)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

SurvFM is a framework for transferring pretrained tabular foundation models to censored time-to-event prediction.

The repository implements and compares three adaptation interfaces:

- **Zero-shot reformulation** using the native TabFM classification interface over discretized time horizons.
- **Classification adaptation** using censoring-aware temporal expansion and a trainable classification head.
- **Survival-head adaptation** using CoxPH, MTLR, or DeepHit over frozen, context-conditioned TabFM representations.

The currently supported tabular foundation model backbones are **TabPFN**, **TabDPT**, and **TabICL**. The code supports both single-risk and competing-risk prediction and includes loaders for the public datasets used in the benchmark, including SurvSet.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/kaylode/survival-fm.git
cd survival-fm
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

## Quick Start

### Load a SurvSet dataset

SurvFM includes a wrapper around the [SurvSet](https://github.com/ErikinBC/SurvSet) collection.

```python
from sklearn.model_selection import train_test_split

from survpfn.dataloaders.data_utils import load_survset_dataset
from survpfn.dataloaders.data_utils.survset import list_survset_datasets

print(list_survset_datasets())
df, duration_col, event_col = load_survset_dataset("GBSG2")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[event_col],
)

print(df.shape)
print(duration_col, event_col)
```


## Low-Level Model API

The registry above is useful for reproducing benchmark configurations. The underlying classes can also be instantiated directly when finer control over context size, temporal discretization, head type, ensembling, or optimization is required.

### Zero-shot prediction

Zero-shot prediction directly reuses the pretrained TabFM classification interface. No survival head or backbone parameters are optimized.

```python
from survpfn.models.shared.zeroshot import ZeroShotSurvivalPredictor

model = ZeroShotSurvivalPredictor(
    backbone="tabpfn", # "tabdpt", "tabicl"
    method="per_bin",
    n_bins=20,
    context_size=256,
    max_context_size=256,
    use_time_bin_encoder=True, # enables the structured temporal features used by the time-bin-encoded zero-shot configuration.
    n_estimators=1,
    device="cuda:0",
)

model.fit(
    train_df,
    duration_col=duration_col,
    event_col=event_col,
)

survival_df = model.predict_survival(test_df)

print(survival_df.shape)
print(survival_df.head())
```


### Classification-head adaptation

For direct control over censoring-aware classification adaptation:

```python
import numpy as np

from survpfn.models.tabpfn import TabPFNSurvPHFinetune

feature_cols = [
    c for c in train_df.columns
    if c not in {duration_col, event_col}
]

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
T_train = train_df[duration_col].to_numpy()
E_train = train_df[event_col].to_numpy()

X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

model = TabPFNSurvPHFinetune(
    num_durations=20,
    context_size=256,
    batch_size=128,
    device="cuda:0",
    freeze_backbone=True,
)

model.fit(
    X_train,
    T_train,
    E_train,
)

survival_df = model.predict_survival_df(
    X_test,
    n_ensemble=5,
)

print(survival_df.shape)
```

This interface internally expands each subject over discrete prediction horizons and trains a censoring-aware classification head while retaining the pretrained TabFM backbone.

### CoxPH, MTLR, and DeepHit survival heads

The survival-head API exposes the downstream objective through `head_type`.

```python
import numpy as np

from survpfn.models.tabpfn import TabPFNSurvPH

feature_cols = [
    c for c in train_df.columns
    if c not in {duration_col, event_col}
]

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
T_train = train_df[duration_col].to_numpy()
E_train = train_df[event_col].to_numpy()

X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

model = TabPFNSurvPH(.        # "TabDPTSurvPH", "TabICLSurvPH"
    head_type="cox",          # "cox", "mtlr", or "deephit"
    freeze_tabpfn=True,
    num_durations=20,
    input_dim=len(feature_cols),
    context_size=256,
    device="cuda:0",
    epochs=50,
    batch_size=128,
)

model.fit(
    X_train,
    T_train,
    E_train,
)

survival_df = model.predict_survival_df(
    X_test,
    n_ensemble=5,
)

print(survival_df.shape)
```

All three retain the pretrained backbone when the corresponding freeze option is enabled and train the downstream survival head over context-conditioned representations.


## Project Structure

```text
survival-fm/
|-- README.md                  # Project overview and usage notes
|-- pyproject.toml             # Package metadata and dependencies
|-- uv.lock                    # Reproducible uv environment
|-- docs/                      # Additional documentation
|-- checkpoints/               # Local model checkpoints
|-- survpfn/
|   |-- configs/               # Model and tuning configuration
|   |-- dataloaders/           # Public, SurvSet, EHR, and CR loaders
|   |-- metrics/               # Survival and competing-risk metrics
|   |-- models/
|   |   |-- tabpfn/            # TabPFN adapters
|   |   |-- tabdpt/            # TabDPT adapters
|   |   |-- tabicl/            # TabICL adapters
|   |   |-- shared/            # Shared losses, binning, and training logic
|   |   |-- sr_models/         # Single-risk baselines
|   |   `-- cr_models/         # Competing-risk baselines
|   |-- scripts/               # Benchmark and analysis scripts
|   `-- utils/                 # Configuration and utilities
|-- tests/
```

## Citation

If you use the MTLR-based TabFM adaptation introduced in our earlier work, please cite:
The broader adaptation-interface benchmark, including CoxPH, DeepHit, classification adaptation, structured time-bin encoding, context-resampled training, and competing-risk experiments, is described in our [latest manuscript](https://arxiv.org/abs/2609.04901).

```bibtex
@inproceedings{pham2026tabular,
  title={Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation},
  author={Pham, Minh-Khoi and Cotugno, Luca and S{\^\i}rbu, Alina and Mai, Tai Tan and Crane, Martin and Bezbradica, Marija},
  booktitle={International Conference on AI in Healthcare},
  pages={315--328},
  year={2026},
  organization={Springer},
  doi={10.1007/978-3-032-35387-0_23}
}

@misc{pham2026adaptationinterfacesincontexttabular,
      title={Adaptation Interfaces for In-Context Tabular Foundation Models in Time-to-Event Prediction}, 
      author={Minh-Khoi Pham and Luca Cotugno and Dan Cernei and Alina Sirbu and Stefano Masi and Giuseppe Prencipe and Alessandro Pingitore and Patrizia Landi and Working Group on Uric Acid and Cardiovascular Risk of the Italian Society of Hypertension and Tai Tan Mai and Martin Crane and Marija Bezbradica},
      year={2026},
      eprint={2609.04901},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.04901}, 
}
```

