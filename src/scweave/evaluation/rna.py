"""
Gene-expression evaluation for scWeave.

The metric reported for RNA prediction is the per-cell Spearman correlation (SCC)
between predicted and measured expression across genes.
"""

import warnings

import numpy as np
from scipy.stats import spearmanr


def spearman_per_cell(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-cell Spearman correlation between predicted and true gene expression.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_cells, n_genes)
        Measured gene expression.
    y_pred : np.ndarray, shape (n_cells, n_genes)
        Predicted gene expression.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        Spearman correlation for each cell (NaN for cells with no variance).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_cells = len(y_true)
    scores = np.full(n_cells, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_cells):
            scores[i] = spearmanr(y_true[i], y_pred[i])[0]
    return scores
