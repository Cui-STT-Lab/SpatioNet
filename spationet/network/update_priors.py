"""Prior-update utilities for SpatioNet.

This module contains the two prior updates used by the SpatioNet training loop:

1. Spatial document-topic prior update: xi / alpha
2. Gene-network topic-word prior update: ni / eta

The main public helpers used by ``spationet.model.model.train`` are:

- ``_update_xis``: update xi/alpha for all spatial samples.
- ``_update_ni_weight``: update ni/eta using one gene-network matrix and edge weights.
- ``_update_nis``: thin wrapper around ``_update_ni_weight`` for compatibility with
  train() calls that use beta, M, and weight.
"""

from __future__ import annotations

from collections import OrderedDict
import itertools
import logging
from multiprocessing import Pool
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.special import digamma
from tqdm.auto import tqdm

from ..model import admm


# -----------------------------------------------------------------------------
# Spatial prior update: xi / alpha
# -----------------------------------------------------------------------------

def _update_xi(
    counts: np.ndarray,
    diff_matrix: Any,
    diff_penalty: float,
    sample_id: Any,
    verbosity: int = 0,
    max_iter: int = 15,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    rho: float = 1e-1,
    mu: float = 2.0,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Infer smoothed document-topic Dirichlet parameters for one sample.

    Parameters
    ----------
    counts
        Unnormalized document-topic counts for spots in one sample.
        Shape: n_spots_in_sample x n_topics.
    diff_matrix
        Spatial difference matrix for this sample.
    diff_penalty
        Spatial smoothing penalty. Larger values imply stronger smoothing. Default = 0.25.
    sample_id
        Sample identifier used for logging.
    """
    if diff_penalty <= 0:
        raise ValueError("diff_penalty must be positive.")

    if verbosity >= 1:
        logging.info(">>> Inferring xi/alpha for sample %s", sample_id)

    penalty_weight = 1.0 / diff_penalty
    cs = digamma(counts) - digamma(np.sum(counts, axis=1, keepdims=True))
    s = penalty_weight * np.ones(diff_matrix.shape[0])

    result = admm.admm(
        cs,
        diff_matrix,
        s,
        rho,
        verbosity=verbosity,
        mu=mu,
        primal_tol=primal_tol,
        max_dirichlet_iter=max_dirichlet_iter,
        max_dirichlet_ls_iter=max_dirichlet_ls_iter,
        max_primal_dual_iter=max_primal_dual_iter,
        max_iter=max_iter,
        threshold=threshold,
    )

    if verbosity >= 1:
        logging.info(">>> Done inferring xi/alpha for sample %s", sample_id)

    return result


# Backward-compatible alias.
_update_alpha = _update_xi


def _wrap_update_xi(inputs: Dict[str, Any]) -> np.ndarray:
    """Multiprocessing wrapper for _update_xi."""
    return _update_xi(**inputs)


def _get_sample_ids(index: pd.Index) -> np.ndarray:
    """Extract sample IDs from a MultiIndex-like index.

    The original SpatialCD/gSpatialCD workflow uses row indices such as
    (sample_id, pixel_id). This helper keeps the package behavior consistent.
    """
    return index.map(lambda x: x[0]).to_numpy()


def _update_xis(
    sample_features: pd.DataFrame,
    difference_matrices: Dict[Any, Any],
    difference_penalty: float,
    gamma: np.ndarray,
    n_parallel_processes: int,
    verbosity: int,
    primal_dual_mu: float = 2,
    admm_rho: float = 0.1,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    max_iter: int = 15,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Update xi/alpha for all samples using spatial difference matrices.

    Parameters
    ----------
    sample_features
        Spot-by-gene count matrix. Its index should contain sample IDs as the
        first element, for example ("Bregma-1", pixel_id).
    difference_matrices
        Dictionary mapping sample_id to the corresponding spatial difference
        matrix.
    difference_penalty
        Spatial smoothing penalty.
    gamma
        Unnormalized document-topic counts from LDA.
        Shape: n_spots x n_topics.
    """
    sample_idxs = _get_sample_ids(sample_features.index)
    unique_idxs = np.unique(sample_idxs)
    new_xis = np.zeros_like(gamma)

    if n_parallel_processes > 1:
        with Pool(n_parallel_processes) as pool:
            sample_masks = [sample_idxs == sample_idx for sample_idx in unique_idxs]
            sample_counts = [gamma[sample_mask, :] for sample_mask in sample_masks]
            sample_diff_matrices = [difference_matrices[sample_idx] for sample_idx in unique_idxs]
            diff_penalties = [difference_penalty for _ in unique_idxs]

            tasks = OrderedDict(
                (
                    ("counts", sample_counts),
                    ("diff_matrix", sample_diff_matrices),
                    ("diff_penalty", diff_penalties),
                    ("sample_id", unique_idxs),
                    ("max_iter", itertools.repeat(max_iter)),
                    ("max_primal_dual_iter", itertools.repeat(max_primal_dual_iter)),
                    ("max_dirichlet_iter", itertools.repeat(max_dirichlet_iter)),
                    ("max_dirichlet_ls_iter", itertools.repeat(max_dirichlet_ls_iter)),
                    # Logging can cause multiprocessing to hang.
                    ("verbosity", itertools.repeat(0)),
                    ("rho", itertools.repeat(admm_rho)),
                    ("mu", itertools.repeat(primal_dual_mu)),
                    ("primal_tol", itertools.repeat(primal_tol)),
                    ("threshold", itertools.repeat(threshold)),
                )
            )

            kw_tasks = [
                {k: v for k, v in zip(tasks.keys(), values)}
                for values in zip(*tasks.values())
            ]

            results = list(
                tqdm(
                    pool.imap(_wrap_update_xi, kw_tasks),
                    total=len(unique_idxs),
                    position=1,
                    desc="Update xi",
                )
            )

        new_xis = np.concatenate(results, axis=0)

    else:
        for sample_idx in unique_idxs:
            sample_mask = sample_idxs == sample_idx
            sample_counts = gamma[sample_mask, :]
            sample_diff_matrix = difference_matrices[sample_idx]

            new_xis[sample_mask, :] = _update_xi(
                counts=sample_counts,
                diff_matrix=sample_diff_matrix,
                diff_penalty=difference_penalty,
                sample_id=sample_idx,
                max_primal_dual_iter=max_primal_dual_iter,
                max_dirichlet_iter=max_dirichlet_iter,
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                max_iter=max_iter,
                verbosity=verbosity,
                rho=admm_rho,
                mu=primal_dual_mu,
                primal_tol=primal_tol,
                threshold=threshold,
            )

    return new_xis


# -----------------------------------------------------------------------------
# Gene-network prior update: ni / eta
# -----------------------------------------------------------------------------

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize rows safely so each row sums to one."""
    row_sums = np.sum(x, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return x / row_sums


def _update_ni(
    counts: np.ndarray,
    diff_matrix: Any,
    diff_penalty: float,
    sample_id: Any,
    verbosity: int = 0,
    max_iter: int = 15,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    rho: float = 1e-1,
    mu: float = 2.0,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Infer topic-word Dirichlet prior using an unweighted graph penalty."""
    if diff_penalty <= 0:
        raise ValueError("diff_penalty must be positive.")

    if verbosity >= 1:
        logging.info(">>> Inferring ni/eta for sample %s", sample_id)

    penalty_weight = 1.0 / diff_penalty
    cs = digamma(counts) - digamma(np.sum(counts, axis=1, keepdims=True))
    cst = cs.T
    s = penalty_weight * np.ones(diff_matrix.shape[0])

    result = admm.admm(
        cst,
        diff_matrix,
        s,
        rho,
        verbosity=verbosity,
        mu=mu,
        primal_tol=primal_tol,
        max_dirichlet_iter=max_dirichlet_iter,
        max_dirichlet_ls_iter=max_dirichlet_ls_iter,
        max_primal_dual_iter=max_primal_dual_iter,
        max_iter=max_iter,
        threshold=threshold,
    )

    result = _normalize_rows(result.T)

    if verbosity >= 1:
        logging.info(">>> Done inferring ni/eta for sample %s", sample_id)

    return result


def _update_ni_weight(
    counts: np.ndarray,
    diff_matrix: Any,
    weight: np.ndarray,
    sample_id: Any,
    verbosity: int = 1,
    max_iter: int = 15,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    rho: float = 1e-1,
    mu: float = 2.0,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Infer topic-word Dirichlet prior using weighted gene-network edges.

    Parameters
    ----------
    counts
        Topic-word count matrix, usually ``lda.components_``.
        Shape: n_topics x n_genes.
    diff_matrix
        Gene-network difference matrix M.
    weight
        Edge weights, default: use the abs_corr column from refined gene-network edges.
        Length should match the number of rows in ``diff_matrix``.
    sample_id
        Identifier used only for logging. For one global gene network, use "all".
    """
    if threshold is not None and not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1) if provided.")

    if verbosity >= 1:
        logging.info(">>> Inferring weighted ni/eta for sample %s", sample_id)

    cs = digamma(counts) - digamma(np.sum(counts, axis=1, keepdims=True))
    cst = cs.T

    result = admm.admm(
        cst,
        diff_matrix,
        weight,
        rho,
        verbosity=verbosity,
        mu=mu,
        primal_tol=primal_tol,
        max_dirichlet_iter=max_dirichlet_iter,
        max_dirichlet_ls_iter=max_dirichlet_ls_iter,
        max_primal_dual_iter=max_primal_dual_iter,
        max_iter=max_iter,
        threshold=threshold,
    )

    result = _normalize_rows(result.T)

    if verbosity >= 1:
        logging.info(">>> Done inferring weighted ni/eta for sample %s", sample_id)

    return result


def _update_nis(
    beta: np.ndarray,
    M: Any,
    weight: np.ndarray,
    sample_id: Any = "all",
    verbosity: int = 1,
    max_iter: int = 15,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    rho: float = 1e-1,
    mu: float = 2.0,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Compatibility wrapper used by train().

    This matches calls of the form:

    ``nis = _update_nis(beta=beta, M=M, weight=weight, sample_id="all")``

    and forwards them to ``_update_ni_weight``.
    """
    return _update_ni_weight(
        counts=beta,
        diff_matrix=M,
        weight=weight,
        sample_id=sample_id,
        verbosity=verbosity,
        max_iter=max_iter,
        max_primal_dual_iter=max_primal_dual_iter,
        max_dirichlet_iter=max_dirichlet_iter,
        max_dirichlet_ls_iter=max_dirichlet_ls_iter,
        rho=rho,
        mu=mu,
        primal_tol=primal_tol,
        threshold=threshold,
    )


# -----------------------------------------------------------------------------
# Optional multi-sample ni helpers retained for backward compatibility
# -----------------------------------------------------------------------------

def _wrap_update_ni(inputs: Dict[str, Any]) -> np.ndarray:
    """Multiprocessing wrapper for _update_ni."""
    return _update_ni(**inputs)


def _wrap_update_ni_weight(inputs: Dict[str, Any]) -> np.ndarray:
    """Multiprocessing wrapper for _update_ni_weight."""
    return _update_ni_weight(**inputs)


def _update_nis_by_sample(
    sample_features: pd.DataFrame,
    difference_matrices: Dict[Any, Any],
    difference_penalty: float,
    betas: np.ndarray,
    n_parallel_processes: int,
    verbosity: int,
    primal_dual_mu: float = 2,
    admm_rho: float = 0.1,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    max_iter: int = 15,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Backward-compatible multi-sample unweighted ni update.

    This is kept under a different name to avoid conflicting with the package
    train() wrapper ``_update_nis(beta, M, weight, ...)``.
    """
    sample_idxs = _get_sample_ids(sample_features.index)
    unique_idxs = np.unique(sample_idxs)
    new_nis = np.zeros_like(betas)

    if n_parallel_processes > 1:
        with Pool(n_parallel_processes) as pool:
            sample_masks = [sample_idxs == sample_idx for sample_idx in unique_idxs]
            sample_counts = [betas[sample_mask, :] for sample_mask in sample_masks]
            sample_diff_matrices = [difference_matrices[sample_idx] for sample_idx in unique_idxs]
            diff_penalties = [difference_penalty for _ in unique_idxs]

            tasks = OrderedDict(
                (
                    ("counts", sample_counts),
                    ("diff_matrix", sample_diff_matrices),
                    ("diff_penalty", diff_penalties),
                    ("sample_id", unique_idxs),
                    ("max_iter", itertools.repeat(max_iter)),
                    ("max_primal_dual_iter", itertools.repeat(max_primal_dual_iter)),
                    ("max_dirichlet_iter", itertools.repeat(max_dirichlet_iter)),
                    ("max_dirichlet_ls_iter", itertools.repeat(max_dirichlet_ls_iter)),
                    ("verbosity", itertools.repeat(0)),
                    ("rho", itertools.repeat(admm_rho)),
                    ("mu", itertools.repeat(primal_dual_mu)),
                    ("primal_tol", itertools.repeat(primal_tol)),
                    ("threshold", itertools.repeat(threshold)),
                )
            )

            kw_tasks = [
                {k: v for k, v in zip(tasks.keys(), values)}
                for values in zip(*tasks.values())
            ]

            results = list(
                tqdm(
                    pool.imap(_wrap_update_ni, kw_tasks),
                    total=len(unique_idxs),
                    position=1,
                    desc="Update ni",
                )
            )

        new_nis = np.concatenate(results, axis=0)

    else:
        for sample_idx in unique_idxs:
            sample_mask = sample_idxs == sample_idx
            sample_counts = betas[sample_mask, :]
            sample_diff_matrix = difference_matrices[sample_idx]

            new_nis[sample_mask, :] = _update_ni(
                counts=sample_counts,
                diff_matrix=sample_diff_matrix,
                diff_penalty=difference_penalty,
                sample_id=sample_idx,
                max_primal_dual_iter=max_primal_dual_iter,
                max_dirichlet_iter=max_dirichlet_iter,
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                max_iter=max_iter,
                verbosity=verbosity,
                rho=admm_rho,
                mu=primal_dual_mu,
                primal_tol=primal_tol,
                threshold=threshold,
            )

    return new_nis


def _update_nis_weight_by_sample(
    sample_features: pd.DataFrame,
    difference_matrices: Dict[Any, Any],
    betas: np.ndarray,
    weight: np.ndarray,
    n_parallel_processes: int,
    verbosity: int,
    primal_dual_mu: float = 2,
    admm_rho: float = 0.1,
    max_primal_dual_iter: int = 400,
    max_dirichlet_iter: int = 20,
    max_dirichlet_ls_iter: int = 10,
    max_iter: int = 15,
    primal_tol: float = 1e-3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Backward-compatible multi-sample weighted ni update.

    This fixes the old task-key typo by passing ``weight`` rather than
    ``distance`` into ``_update_ni_weight``.
    """
    sample_idxs = _get_sample_ids(sample_features.index)
    unique_idxs = np.unique(sample_idxs)
    new_nis = np.zeros_like(betas)

    if n_parallel_processes > 1:
        with Pool(n_parallel_processes) as pool:
            sample_masks = [sample_idxs == sample_idx for sample_idx in unique_idxs]
            sample_counts = [betas[sample_mask, :] for sample_mask in sample_masks]
            sample_diff_matrices = [difference_matrices[sample_idx] for sample_idx in unique_idxs]
            weights = [weight for _ in unique_idxs]

            tasks = OrderedDict(
                (
                    ("counts", sample_counts),
                    ("diff_matrix", sample_diff_matrices),
                    ("weight", weights),
                    ("sample_id", unique_idxs),
                    ("max_iter", itertools.repeat(max_iter)),
                    ("max_primal_dual_iter", itertools.repeat(max_primal_dual_iter)),
                    ("max_dirichlet_iter", itertools.repeat(max_dirichlet_iter)),
                    ("max_dirichlet_ls_iter", itertools.repeat(max_dirichlet_ls_iter)),
                    ("verbosity", itertools.repeat(0)),
                    ("rho", itertools.repeat(admm_rho)),
                    ("mu", itertools.repeat(primal_dual_mu)),
                    ("primal_tol", itertools.repeat(primal_tol)),
                    ("threshold", itertools.repeat(threshold)),
                )
            )

            kw_tasks = [
                {k: v for k, v in zip(tasks.keys(), values)}
                for values in zip(*tasks.values())
            ]

            results = list(
                tqdm(
                    pool.imap(_wrap_update_ni_weight, kw_tasks),
                    total=len(unique_idxs),
                    position=1,
                    desc="Update weighted ni",
                )
            )

        new_nis = np.concatenate(results, axis=0)

    else:
        for sample_idx in unique_idxs:
            sample_mask = sample_idxs == sample_idx
            sample_counts = betas[sample_mask, :]
            sample_diff_matrix = difference_matrices[sample_idx]

            new_nis[sample_mask, :] = _update_ni_weight(
                counts=sample_counts,
                diff_matrix=sample_diff_matrix,
                weight=weight,
                sample_id=sample_idx,
                max_primal_dual_iter=max_primal_dual_iter,
                max_dirichlet_iter=max_dirichlet_iter,
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                max_iter=max_iter,
                verbosity=verbosity,
                rho=admm_rho,
                mu=primal_dual_mu,
                primal_tol=primal_tol,
                threshold=threshold,
            )

    return new_nis
