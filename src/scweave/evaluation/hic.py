"""
Single-cell Hi-C evaluation for scWeave.

Two metrics are reported for chromatin-structure prediction:

- ``hicrep_per_chromosome`` : HiCRep stratum-adjusted correlation on the
  pseudobulk (cell-averaged) contact maps, one score per chromosome. Measures how
  well the population-level structure is recovered.
- ``auroc_per_cell`` : area under the ROC curve for each individual cell, treating
  contact prediction as binary classification over all chromosomes. Measures how
  well single-cell structure is recovered.

Both metrics compare only the non-padded region of each contact map, detected from
the ground truth.
"""

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

# PearsonRConstantInputWarning moved across scipy versions.
try:
    from scipy.stats._stats import PearsonRConstantInputWarning
except ImportError:
    try:
        from scipy.stats import PearsonRConstantInputWarning
    except ImportError:  # pragma: no cover - newer scipy dropped the class
        PearsonRConstantInputWarning = RuntimeWarning


def hicrep_per_chromosome(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-chromosome HiCRep on pseudobulk contact maps.

    Cells are averaged into a single pseudobulk map per chromosome, then HiCRep is
    computed on the non-padded region of each chromosome.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_cells, n_chr, size, size)
        Ground truth (binarized) contact maps.
    y_pred : np.ndarray, shape (n_cells, n_chr, size, size)
        Predicted contact maps (probabilities).

    Returns
    -------
    np.ndarray, shape (n_chr,)
        HiCRep score for each chromosome.
    """
    true_pb = np.mean(y_true, axis=0)
    pred_pb = np.mean(y_pred, axis=0)
    n_chr = true_pb.shape[0]

    scores = np.empty(n_chr)
    for chr_idx in range(n_chr):
        true_2d = true_pb[chr_idx].astype(np.float32)
        pred_2d = pred_pb[chr_idx].astype(np.float32)

        r0, r1, c0, c1 = _find_data_region(true_2d)
        true_region = true_2d[r0:r1, c0:c1]
        pred_region = pred_2d[r0:r1, c0:c1]

        try:
            hicrep = _compute_hicrep(
                true_region, pred_region, max_bins=true_region.shape[0] - 1
            )
            if np.isnan(hicrep):
                hicrep = 0.0
        except Exception:
            hicrep = 0.0
        scores[chr_idx] = hicrep

    return scores


def auroc_per_cell(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-cell AUROC of binary contact prediction, pooled over all chromosomes.

    For each cell, the non-padded pixels of every chromosome are pooled and AUROC
    is computed between the binarized ground truth and the predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_cells, n_chr, size, size)
        Ground truth (binarized) contact maps.
    y_pred : np.ndarray, shape (n_cells, n_chr, size, size)
        Predicted contact probabilities.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        AUROC for each cell (NaN for cells whose ground truth has a single class).
    """
    n_cells = len(y_true)
    n_chr = y_true.shape[1]
    scores = np.full(n_cells, np.nan)

    for i in range(n_cells):
        true_list = []
        prob_list = []
        for chr_idx in range(n_chr):
            true_2d = y_true[i, chr_idx].astype(np.float32)
            prob_2d = y_pred[i, chr_idx].astype(np.float32)
            r0, r1, c0, c1 = _find_data_region(true_2d)
            true_list.append(true_2d[r0:r1, c0:c1].ravel())
            prob_list.append(prob_2d[r0:r1, c0:c1].ravel())

        y_t = np.concatenate(true_list)
        y_p = np.concatenate(prob_list)

        if np.std(y_t) == 0:
            continue
        try:
            scores[i] = roc_auc_score(y_t, y_p)
        except ValueError:
            scores[i] = np.nan

    return scores


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _find_data_region(matrix: np.ndarray) -> tuple:
    """Find the non-padded (non-zero) bounding box of a zero-padded contact map."""
    H, W = matrix.shape
    corners_zero = (
        matrix[0, 0] == 0
        and matrix[0, -1] == 0
        and matrix[-1, 0] == 0
        and matrix[-1, -1] == 0
    )
    if not corners_zero:
        return 0, H, 0, W

    nonzero_rows = np.where(matrix.sum(axis=1) > 0)[0]
    nonzero_cols = np.where(matrix.sum(axis=0) > 0)[0]
    if len(nonzero_rows) == 0 or len(nonzero_cols) == 0:
        return 0, H, 0, W

    return nonzero_rows[0], nonzero_rows[-1] + 1, nonzero_cols[0], nonzero_cols[-1] + 1


def _vstrans(d1: np.ndarray, d2: np.ndarray) -> float:
    """Variance-stabilizing transformation for HiCRep stratum weights."""
    ranks_1 = np.argsort(d1) + 1
    ranks_2 = np.argsort(d2) + 1
    nranks_1 = ranks_1 / max(ranks_1)
    nranks_2 = ranks_2 / max(ranks_2)
    nk = len(ranks_1)
    return np.sqrt(np.var(nranks_1 / nk) * np.var(nranks_2 / nk))


def _compute_hicrep(
    A: np.ndarray,
    B: np.ndarray,
    max_bins: int = 40,
    correlation_method: str = "PCC",
) -> float:
    """
    HiCRep stratum-adjusted correlation coefficient between two contact maps.

    A correlation is computed for each diagonal up to ``max_bins`` and combined as
    a variance-weighted sum.
    """
    if (len(A.shape) != 2) or (len(B.shape) != 2):
        raise ValueError("both input matrices must be 2D")
    if A.shape != B.shape:
        raise ValueError("matrices not of the same size")

    if max_bins < 0 or max_bins > int(A.shape[0] - 5):
        max_bins = int(A.shape[0] - 5)

    mat1 = csr_matrix(A)
    mat2 = csr_matrix(B)

    corr_diag = np.zeros(len(range(max_bins)))
    weight_diag = corr_diag.copy()

    for d in range(max_bins):
        d1 = np.asarray(mat1.diagonal(d)).flatten()
        d2 = np.asarray(mat2.diagonal(d)).flatten()
        mask = (~np.isnan(d1)) & (~np.isnan(d2))
        d1 = d1[mask]
        d2 = d2[mask]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PearsonRConstantInputWarning)
            if len(d1) > 1 and np.std(d1) > 0 and np.std(d2) > 0:
                if correlation_method == "PCC":
                    cor = pearsonr(d1, d2)[0]
                elif correlation_method == "SCC":
                    cor = spearmanr(d1, d2)[0]
                else:
                    raise ValueError(f"Invalid correlation method: {correlation_method}")
                corr_diag[d] = cor
            else:
                corr_diag[d] = np.nan

        if len(d1) > 0:
            weight_diag[d] = len(d1) * _vstrans(d1, d2)

    # Drop the self-correlation diagonal and any NaN strata.
    corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
    mask = ~np.isnan(corr_diag)
    corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]

    if weight_diag.sum() > 0:
        weight_diag /= weight_diag.sum()
    else:
        return np.nan

    return float(np.nansum(corr_diag * weight_diag))
