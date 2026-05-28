"""Training routine for the SpatioNet model.

This module defines the main ``train`` function for SpatioNet.

Model workflow
--------------
1. Fit an initial LDA model.
2. Update the document-topic Dirichlet prior xi using spatial difference matrices.
3. Warm-start LDA using the updated xi.
4. Update the topic-word Dirichlet prior ni using the gene network.
5. Warm-start LDA using the updated ni.
6. Return the fitted LDA object and optionally save gamma, beta, and model files.

Expected inputs
---------------
sample_features:
    pandas DataFrame with rows as spatial spots/pixels and columns as genes.
    is the sample ID and index[i][1] is the spot/pixel ID.

difference_matrices:
    dict mapping each sample ID to its spatial difference matrix.

M:
    gene-network matrix used in the ni update.

weight:
    edge weight vector, usually the abs_corr column from gene_network.csv.

Example
-------
from spationet.model.model_train import train

model = train(
    sample_features=feat,
    difference_matrices=diff_matrix,
    M=M,
    weight=weight,
    n_topics=12,
    n_parallel_processes=8,
    output_dir="./results/",
    save=True
)
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .online_lda import LatentDirichletAllocation
from ..network.update_priors import _update_xis, _update_nis


def _make_output_path(
    output_dir: str,
    prefix: str,
    n_outer_iters: int,
    n_iters: int,
    difference_penalty: float,
    n_topics: int,
    extension: str,
) -> str:
    """Create a standardized output file path."""
    filename = (
        f"{prefix}"
        f"_topics={n_topics}"
        f".{extension}"
    )
    return os.path.join(output_dir, filename)


def _extract_spot_ids(index: pd.Index) -> list:
    """Extract spot/pixel IDs from a tuple-like index.

    If the index is not tuple-like, the original index values are returned.
    """
    spot_ids = []
    for idx in index:
        try:
            spot_ids.append(idx[1])
        except Exception:
            spot_ids.append(idx)
    return spot_ids


def train(
    sample_features: pd.DataFrame,
    difference_matrices: Dict[Any, Any],
    M: Any,
    weight: np.ndarray,
    n_topics: int,
    difference_penalty: float = 0.25,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    max_lda_iter: int = 100,
    max_admm_iter: int = 15,
    n_outer_iters: int = 2,
    n_iters: int = 3,
    n_parallel_processes: int = 1,
    verbosity: int = 1,
    primal_dual_mu: float = 2.0,
    admm_rho: float = 1.0,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
    random_state: int = 0,
    evaluate_every: int = 5,
    mean_change_tol: float = 1e-6,
    doc_topic_prior_init: Optional[float] = None,
    topic_word_prior_init: float = 1.0,
    warm_start_method: str = "partial_fit",
    output_dir: str = "./",
    save: bool = True,
) -> LatentDirichletAllocation:
    """Train a SpatioNet model.

    Parameters
    ----------
    sample_features
        Spot-by-gene count matrix. Rows are spatial spots/pixels and columns are genes.
    difference_matrices
        Dictionary of sample-specific spatial difference matrices.
    M
        Gene-network matrix used in the ni/topic-word prior update.
    weight
        Edge weights for the gene network, usually refined edge correlations.
    n_topics
        Number of latent topics.
    difference_penalty
        Spatial regularization penalty for xi update.
    max_primal_dual_iter
        Maximum primal-dual iterations in ADMM.
    max_dirichlet_iter
        Maximum Dirichlet iterations in ADMM.
    max_dirichlet_ls_iter
        Maximum line-search iterations for Dirichlet update.
    max_lda_iter
        Maximum LDA iterations for the initial LDA model.
    max_admm_iter
        Maximum ADMM outer iterations.
    n_outer_iters
        Label used in output filenames. Kept for compatibility with experiments.
    n_iters
        Number of xi-update iterations and number of ni-update iterations.
    n_parallel_processes
        Number of parallel processes used in prior updates and LDA.
    verbosity
        Verbosity level for logging and ADMM.
    primal_dual_mu
        Primal-dual mu parameter.
    admm_rho
        ADMM rho parameter.
    primal_tol
        Primal tolerance for ADMM convergence.
    threshold
        Optional threshold used by the ADMM solver.
    random_state
        Random seed for LDA.
    evaluate_every
        LDA evaluation frequency.
    mean_change_tol
        LDA mean change tolerance.
    doc_topic_prior_init
        Initial document-topic prior. If None, defaults to 50 / n_topics.
    topic_word_prior_init
        Initial topic-word prior.
    warm_start_method
        Either "partial_fit" or "fit". "partial_fit" is recommended because it
        better preserves warm-started components.
    output_dir
        Directory where gamma, beta, and model files are saved.
    save
        If True, save gamma CSV, beta CSV, and fitted model pickle.

    Returns
    -------
    LatentDirichletAllocation
        Fitted LDA model with ``topic_weights`` attached.
    """
    if not isinstance(sample_features, pd.DataFrame):
        raise TypeError("sample_features must be a pandas DataFrame.")

    if n_topics <= 0:
        raise ValueError("n_topics must be positive.")

    if n_iters < 0:
        raise ValueError("n_iters must be non-negative.")

    if difference_penalty <= 0:
        raise ValueError("difference_penalty must be positive.")

    if warm_start_method not in {"partial_fit", "fit"}:
        raise ValueError("warm_start_method must be either 'partial_fit' or 'fit'.")

    if doc_topic_prior_init is None:
        doc_topic_prior_init = 50 / n_topics

    if save:
        os.makedirs(output_dir, exist_ok=True)

    X = sample_features.values

    print(">>> First LDA")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        doc_topic_prior=doc_topic_prior_init,
        topic_word_prior=topic_word_prior_init,
        max_iter=max_lda_iter,
        evaluate_every=evaluate_every,
        mean_change_tol=mean_change_tol,
        random_state=random_state,
        n_jobs=n_parallel_processes,
    )

    lda.fit(X)
    gamma = lda._unnormalized_transform(X)
    beta = lda.components_.copy()

    for outer_iter in range(n_outer_iters):
        # print(f'\n===== Outer refinement iteration {outer_iter + 1}/{n_outer_iters} =====')

        # ------------------------------------------------------------
        # Stage 1: spatial xi/document-topic prior update
        # ------------------------------------------------------------
        for i in range(n_iters):
            logging.info(">>> Starting xi iteration %s", i + 1)
            print(f">>> Update Xis iteration {i + 1}")

            xis = _update_xis(
                sample_features=sample_features,
                difference_matrices=difference_matrices,
                difference_penalty=difference_penalty,
                gamma=gamma,
                n_parallel_processes=n_parallel_processes,
                max_iter=max_admm_iter,
                max_primal_dual_iter=max_primal_dual_iter,
                max_dirichlet_iter=max_dirichlet_iter,
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                verbosity=verbosity,
                primal_dual_mu=primal_dual_mu,
                admm_rho=admm_rho,
                primal_tol=primal_tol,
                threshold=threshold,
            )

            print(f">>> Warm Update LDA after xi refinement {i + 1}")

            lda.set_components(beta)
            lda.doc_topic_prior = xis

            if warm_start_method == "partial_fit":
                lda.partial_fit(X)
            else:
                lda.fit(X)

            delta = np.linalg.norm(lda.components_ - beta) / max(np.linalg.norm(beta), 1e-12)
            print(f"    >> LDA beta change after xi refinement: {delta:.6f}")

            beta = lda.components_.copy()
            gamma = lda._unnormalized_transform(X)

        # ------------------------------------------------------------
        # Stage 2: gene-network ni/topic-word prior update
        # ------------------------------------------------------------
        for i in range(n_iters):
            logging.info(">>> Starting ni iteration %s", i + 1)
            print(f">>> Update Nis iteration {i + 1}")

            nis = _update_nis(
                beta=beta,
                M=M,
                weight=weight,
                sample_id="all",
                max_iter=max_admm_iter,
                max_primal_dual_iter=max_primal_dual_iter,
                max_dirichlet_iter=max_dirichlet_iter,
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                verbosity=verbosity,
                rho=admm_rho,
                mu=primal_dual_mu,
                primal_tol=primal_tol,
                threshold=threshold,
            )

            print(f">>> Warm Update LDA after ni refinement {i + 1}")

            lda.set_components(beta)
            lda.topic_word_prior = nis

            if warm_start_method == "partial_fit":
                lda.partial_fit(X)
            else:
                lda.fit(X)

            delta = np.linalg.norm(lda.components_ - beta) / max(np.linalg.norm(beta), 1e-12)
            print(f"    >> LDA beta change after ni refinement: {delta:.6f}")

            beta = lda.components_.copy()
            

    # ------------------------------------------------------------
    # Final topic weights and outputs
    # ------------------------------------------------------------
    print(">>> Getting final topic weights")
    final_gamma = lda.fit_transform(X)

    columns = [f"Topic-{i}" for i in range(n_topics)]
    gamma_df = pd.DataFrame(
        final_gamma,
        index=_extract_spot_ids(sample_features.index),
        columns=columns,
    )

    beta_df = pd.DataFrame(
        lda.components_,
        columns=sample_features.columns,
        index=columns,
    )

    lda.topic_weights = gamma_df

    if save:
        gamma_path = _make_output_path(
            output_dir=output_dir,
            prefix="gamma",
            n_topics=n_topics,
            extension="csv",
        )
        beta_path = _make_output_path(
            output_dir=output_dir,
            prefix="beta",
            n_topics=n_topics,
            extension="csv",
        )
        model_path = _make_output_path(
            output_dir=output_dir,
            prefix="model",
            n_topics=n_topics,
            extension="pkl",
        )

        gamma_df.to_csv(gamma_path, index=True)
        beta_df.to_csv(beta_path, index=True)

        with open(model_path, "wb") as f:
            pickle.dump(lda, f)

        print("gamma saved to:", gamma_path)
        print("beta saved to:", beta_path)
        print("model saved to:", model_path)

    try:
        logging.info(">>> Final perplexity: %s", lda.perplexity(X))
    except Exception:
        logging.info(">>> Final perplexity could not be computed.")

    return lda


def _topic_name(i: int) -> str:
    return f"Topic-{i}"


def infer(
    components: np.ndarray,
    sample_features: pd.DataFrame,
    difference_matrices: Dict[Any, Any],
    difference_penalty: float = 1.0,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    max_admm_iter: int = 15,
    n_parallel_processes: int = 1,
) -> LatentDirichletAllocation:
    """Run inference on a Spatial-LDA model.

    This runs only the ADMM updates to get spatially-regularized topic weights
    on ``sample_features``, allowing us to infer topic weights on new samples
    after training a model to get topic parameters.

    Parameters
    ----------
    components
        The components of the Spatial-LDA model (typically
        ``spatial_lda_model.components_``).
    sample_features
        Dataframe that contains neighborhood features of index cells indexed by
        (sample ID, cell ID).
    difference_matrices
        Difference matrix corresponding to the spatial regularization structure
        imposed on the samples.
    difference_penalty
        Penalty on topic priors of "adjacent" index cells.
    max_primal_dual_iter
        Maximum number of primal-dual iterations to run.
    max_dirichlet_iter
        Maximum number of Newton steps to take in computing updates for tau.
    max_dirichlet_ls_iter
        Maximum number of line-search steps to take in computing updates for tau.
    max_admm_iter
        Maximum number of ADMM iterations to run.
    n_parallel_processes
        Number of parallel processes to use.

    Returns
    -------
    LatentDirichletAllocation
        Spatial-LDA model with the same topic parameters as the original model
        but with new topic-weights corresponding to the provided
        ``sample_features`` and ``difference_matrices``.
    """
    start_time = time.time()
    logging.info('>>> Starting inference')

    n_topics = components.shape[0]
    complete_lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=0,
        n_jobs=n_parallel_processes,
        max_iter=2,
        doc_topic_prior=None,
    )
    complete_lda.set_components(components)

    gamma = complete_lda._unnormalized_transform(sample_features.values)
    xis = _update_xis(
        sample_features=sample_features,
        difference_matrices=difference_matrices,
        difference_penalty=difference_penalty,
        gamma=gamma,
        n_parallel_processes=n_parallel_processes,
        max_iter=max_admm_iter,
        max_primal_dual_iter=max_primal_dual_iter,
        max_dirichlet_iter=max_dirichlet_iter,
        max_dirichlet_ls_iter=max_dirichlet_ls_iter,
        verbosity=0,
        primal_dual_mu=2.0,
        admm_rho=0.1,
        primal_tol=1e-3,
        threshold=None,
    )

    complete_lda.doc_topic_prior_ = xis
    columns = [_topic_name(i) for i in range(n_topics)]
    topic_weights = pd.DataFrame(
        complete_lda.transform(sample_features.values),
        index=sample_features.index,
        columns=columns,
    )
    complete_lda.topic_weights = topic_weights

    logging.info('>>> Inference took %s seconds.', time.time() - start_time)
    return complete_lda

