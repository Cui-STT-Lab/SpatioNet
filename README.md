# Leveraging gene networks for spatially-informed reference-free deconvolution in spatial transcriptomics with SpatioNet

SpatioNet is a spatial transcriptomics method that leverages gene-network priors and spatial regularization to perform reference-free deconvolution. The framework uses spatially-aware topic modeling to recover spatial topic weights and gene-topic profiles without requiring an external single-cell reference.

## Installation

Install the repository in editable mode:

```bash
git clone https://github.com/Cui-STT-Lab/SpatioNet.git
cd SpatioNet
pip install -e .
```

## Dependencies

```python
import pandas as pd
import numpy as np
import os
import logging
```

## Run SpatioNet with MPOA Data

### Data Loading and Preprocessing

The SpatioNet workflow begins by loading processed spatial transcriptomics features and gene-network priors.

```python
from spationet.model.model import train

PATH_TO_DATA = '../data/'

with open(f'{PATH_TO_DATA}/raw/mpoa/MPOA_feat.pkl', 'rb') as f:
    feat = pickle.load(f)

with open(f'{PATH_TO_DATA}/STRING_processed/mpoa/mpoa_gene_network.pkl', 'rb') as f:
    M = pickle.load(f)

with open(f'{PATH_TO_DATA}/spatial_processed/mpoa_diff.pkl', 'rb') as f:
    diff_matrix = pickle.load(f)

weight = pd.read_csv(f'{PATH_TO_DATA}/STRING_processed/mpoa/mpoa_gene_network.csv')['abs_corr'].to_numpy()
```

### Model Training

Train SpatioNet with spatial and gene-network regularization:

```python
n_topics = 12

lda = train(
    sample_features=feat,
    difference_matrices=diff_matrix,
    M=M,
    weight=weight,
    n_topics=n_topics,
    n_iters=3,
    max_lda_iter=100,
    max_admm_iter=15,
    n_parallel_processes=8,
    save=True,
    output_dir='/Users/phuong/Library/CloudStorage/OneDrive-MichiganStateUniversity/2. SpatioNet/example/output/mpoa',
)
```

### Save Outputs

SpatioNet saves the trained model, topic weights, and topic-word matrix into the output directory.

```python
PATH_TO_MODELS = '../example/output/mpoa'

# model: saved as model_topics={n_topics}.pkl
# gamma: saved as gamma_topics={n_topics}.csv
# beta: saved as beta_topics={n_topics}.csv
```

### Results Extraction and Evaluation

SpatialCD extracts deconvolution results, computes evaluation metrics, and saves them to the defined output path:

```python
PATH_TO_MODELS = '../example/output/mpoa/'
save_results(spatialcd_model, n_topics, n_neighbors, corpus, PATH_TO_MODELS)
```

The output figures can be viewed in the repository or output directory:

![Spatial plot](example/output/mpoa/theta_PCC.png)

![Heatmap plot](figures/bregma.png)

## Key Features

- **Gene-network priors**: Uses refined gene-gene relationships to regularize topic-word priors
- **Spatial smoothing**: Applies ADMM-based spatial regularization on spot/topic priors
- **Reference-free deconvolution**: Does not require external cell-type reference data
- **Parallel training**: Supports multiprocessing for faster inference and training
- **Exportable outputs**: Saves gamma, beta, and model files for downstream analysis

## Output Files

SpatioNet generates the following outputs:

- **`model_topics=<n_topics>.pkl`**: Pickled LDA model with learned components
- **`gamma_topics=<n_topics>.csv`**: Spatial topic weights per spot/pixel
- **`beta_topics=<n_topics>.csv`**: Topic-word matrix describing gene contributions for each topic

## Parameters

- `n_topics`: Number of latent topics
- `difference_penalty`: Spatial regularization strength for topic priors
- `max_admm_iter`: Number of ADMM iterations for prior updates
- `n_parallel_processes`: Number of parallel workers for inference/training
- `sample_features`: Spot-by-gene count or feature matrix
- `difference_matrices`: Spatial difference matrices for each sample
- `M`: Gene-network difference matrix
- `weight`: Gene-network edge weights

## Citation

If you use SpatioNet in your research, please cite:

```
Vo, Phuong, and Yuehua Cui. (2026) Leveraging gene networks for spatially-informed reference-free deconvolution in spatial transcriptomics with SpatioNet.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub or contact the development team.
