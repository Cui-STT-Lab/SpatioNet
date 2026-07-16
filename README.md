# SpatioNet

**SpatioNet** is a reference-free deconvolution framework for spatial
transcriptomics (ST). It estimates spatially resolved cell-type proportions
and transcriptional profiles directly from spatial gene expression, without
requiring an external single-cell RNA-seq reference.

SpatioNet couples a latent Dirichlet allocation (LDA) topic model with two
graph priors:

- a **gene network** (e.g. STRING-refined edges) that shapes topic–gene
  profiles, and
- a **spatial network** that encourages neighboring spots to share coherent
  cell-type compositions while preserving local tissue structure.

**SpatioNet is developed and maintained by Phuong Vo and Yuehua Cui.**

For bugs, questions, or feedback, please open an
[issue](https://github.com/Cui-STT-Lab/SpatioNet/issues).

## Reference

Vo, P. and Y. Cui. SpatioNet integrates spatial and gene-network
regularization for reference-free deconvolution of spatial transcriptomics.

## Highlights

- **Reference-free spatial deconvolution** — estimates latent cell-type
  proportions without an external scRNA-seq atlas.
- **Gene-network-guided topics** — incorporates gene–gene network structure
  to improve biological interpretability of topic–gene profiles (β).
- **Spatially regularized topic weights** — encourages neighboring spots to
  have coherent compositions (γ / θ-like weights) while retaining local
  heterogeneity.
- **Downstream-ready outputs** — returns topic weights and transcriptional
  matrices for visualization, clustering, marker interpretation, and
  spatial analysis.

## Method overview

![SpatioNet method overview](SpatioNet_overview_image.png)

SpatioNet fits an LDA model and iteratively updates Dirichlet priors using
the spatial difference matrices (document–topic prior ξ) and the gene
network (topic–word prior η), then warm-starts LDA with the refined
priors.

## Installation

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/Cui-STT-Lab/SpatioNet.git
cd SpatioNet
pip install -r requirements.txt
```

Then either work from the repository root (so `spationet` is importable) or
add the repo to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/SpatioNet:${PYTHONPATH}"
```

## Quick start: MPOA

The repository includes preprocessed MERFISH **MPOA** inputs under `data/`
and example outputs under `example/output/mpoa/`.

### 1. Load expression and coordinates

```python
import pickle
import pandas as pd
from spationet.model.model import train

with open("data/raw/MPOA_feat.pkl", "rb") as f:
    feat = pickle.load(f)

pos = pd.read_csv("data/raw/mpoa_pos.csv", index_col=0)
```

### 2. Load gene network and spatial difference matrices

```python
with open("data/STRING_processed/mpoa_gene_network.pkl", "rb") as f:
    M = pickle.load(f)

weight = pd.read_csv(
    "data/STRING_processed/mpoa_gene_network.csv"
)["abs_corr"].to_numpy()

with open("data/spatial_processed/mpoa_diff.pkl", "rb") as f:
    diff_matrix = pickle.load(f)
```

### 3. Train SpatioNet

```python
n_topics = 12

lda = train(
    sample_features=feat,
    difference_matrices=diff_matrix,
    difference_penalty=10,
    M=M,
    weight=weight,
    n_topics=n_topics,
    n_iters=2,
    max_lda_iter=100,
    max_admm_iter=15,
    n_parallel_processes=8,
    save=True,
    output_dir="example/output/mpoa",
)
```

### 4. Extract results

When `save=True`, `train()` writes model and matrix files under
`output_dir`. You can also extract matrices from the returned object:

```python
import os

beta = lda.components_.copy()
gamma = lda._unnormalized_transform(feat)

columns = [f"Topic-{i}" for i in range(n_topics)]
gamma_df = pd.DataFrame(gamma, index=feat.index, columns=columns)
beta_df = pd.DataFrame(lda.components_, columns=feat.columns, index=columns)
lda.topic_weights = gamma_df

path = "example/output/mpoa"
os.makedirs(path, exist_ok=True)

with open(f"{path}/model_topics={n_topics}.pkl", "wb") as f:
    pickle.dump(lda, f)

gamma_df.to_csv(f"{path}/gamma_topics={n_topics}.csv")
beta_df.to_csv(f"{path}/beta_topics={n_topics}.csv")
```

A notebook walkthrough is in `tests/test_on_mpoa.ipynb`.

## Example outputs

### Cell-type composition (θ / topic weights)

![Spatial topic-weight PCC](example/output/mpoa/theta_PCC.png)

### Cell-type gene profiles (β)

![Topic–gene PCC](example/output/mpoa/beta_PCC.png)

### Spatial cell-type distribution across bregma sections

![Bregma overview](figures/bregma.png)

## Output files

Typical files written under `output_dir` (and/or by the example above):

| File | Description |
|------|-------------|
| `model_topics=<K>.pkl` | fitted SpatioNet / LDA model |
| `gamma_topics=<K>.csv` | spatial topic weights per spot |
| `beta_topics=<K>.csv` | topic–gene contribution matrix |
| `theta_PCC.png`, `beta_PCC.png` | example evaluation figures |

## Project structure

| Path | Description |
|------|-------------|
| `spationet/model/` | LDA training, ADMM / primal–dual solvers |
| `spationet/network/` | gene- and spatial-graph construction; prior updates |
| `spationet/evaluation/` | evaluation helpers |
| `spationet/utils/` | shared utilities |
| `data/raw/` | MPOA feature matrix, corpus, and coordinates |
| `data/STRING_processed/` | refined gene network |
| `data/spatial_processed/` | spatial difference matrices |
| `example/output/mpoa/` | example fitted outputs and figures |
| `figures/` | README figures |
| `tests/` | MPOA notebook / tests |

## Dependencies

See `requirements.txt` for pinned versions. Core packages include NumPy,
SciPy, pandas, scikit-learn, matplotlib, seaborn, multiprocess, and tqdm.

## Authors

| Role | Name |
|------|------|
| Authors / maintainers | Phuong Vo, Yuehua Cui |
