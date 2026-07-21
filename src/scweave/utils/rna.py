"""
RNA input preparation for scWeave.

Turns an AnnData object of single-cell gene expression into the array the model
expects: columns aligned to the model's gene set (in the model's gene order),
library-size normalized, and log-transformed. Genes the model was trained on but
absent from the AnnData are filled with zeros.
"""

from importlib import resources
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
from scipy import sparse as sp

ArrayLike = Union[np.ndarray, list]

_SPECIES_FILES = {
    "mouse": "mouse_genes.txt",
    "human": "human_genes.txt",
}


def load_gene_names(species: str) -> np.ndarray:
    """
    Load the model's gene set for a species.

    Parameters
    ----------
    species : {"mouse", "human"}
        Which packaged gene list to load. These are the Ensembl gene IDs, in the
        exact order the corresponding scWeave model was trained on.

    Returns
    -------
    np.ndarray
        1-D array of Ensembl gene IDs.
    """
    if species not in _SPECIES_FILES:
        raise ValueError(
            f"species must be one of {sorted(_SPECIES_FILES)}, got {species!r}"
        )
    text = resources.files("scweave.resources").joinpath(_SPECIES_FILES[species]).read_text()
    return np.array(text.split(), dtype=object)


def prepare_rna(
    adata: Union[ad.AnnData, str, Path],
    gene_names: Union[ArrayLike, str, Path],
    normalize_library_size: bool = True,
    target_sum: float = 1e4,
    apply_log: bool = True,
) -> np.ndarray:
    """
    Prepare single-cell RNA expression for scWeave inference.

    Aligns the AnnData to the model's gene set, normalizes by library size, and
    applies a log1p transform, matching the preprocessing used to train scWeave.

    Parameters
    ----------
    adata : anndata.AnnData, str, or Path
        Single-cell gene expression. Either an in-memory AnnData or a path to an
        ``.h5ad`` file. ``adata.var_names`` must be gene identifiers that match
        those in ``gene_names`` (e.g. Ensembl gene IDs).
    gene_names : array-like, str, or Path
        The model's gene set, in the model's gene order. May be:
        - an array/list of gene IDs,
        - one of ``"mouse"`` / ``"human"`` to use the packaged gene list, or
        - a path to a ``.txt`` (one gene per line) or ``.npy`` file.
    normalize_library_size : bool, default=True
        Scale each cell to sum to ``target_sum`` before log transform.
    target_sum : float, default=1e4
        Target library size for normalization.
    apply_log : bool, default=True
        Apply a ``log1p`` transform.

    Returns
    -------
    np.ndarray
        Expression matrix of shape ``(n_cells, len(gene_names))``, float32,
        ready to pass to :meth:`scweave.scWeave.predict_hic_from_rna` or
        :meth:`scweave.scWeave.match`. Model genes absent from ``adata`` are 0.
    """
    if isinstance(adata, (str, Path)):
        adata = ad.read_h5ad(adata)
    gene_names = _resolve_gene_names(gene_names)

    X = adata.X
    dataset_genes = np.asarray(adata.var_names, dtype=object)
    gene_idx_map = {g: i for i, g in enumerate(dataset_genes)}
    col_indices = np.fromiter(
        (gene_idx_map.get(g, -1) for g in gene_names),
        dtype=np.int64,
        count=len(gene_names),
    )
    valid_mask = col_indices >= 0
    valid_cols = col_indices[valid_mask]

    if sp.issparse(X):
        X_sub = X[:, valid_cols].toarray().astype(np.float32, copy=False)
    else:
        X_sub = np.asarray(X[:, valid_cols], dtype=np.float32)

    out = np.zeros((adata.n_obs, len(gene_names)), dtype=np.float32)
    out[:, valid_mask] = X_sub

    if normalize_library_size:
        library_sizes = out.sum(axis=1, keepdims=True)
        library_sizes = np.where(library_sizes == 0, 1.0, library_sizes)
        out *= np.float32(target_sum) / library_sizes.astype(np.float32)

    if apply_log:
        np.log1p(out, out=out)

    return out


def _resolve_gene_names(gene_names: Union[ArrayLike, str, Path]) -> np.ndarray:
    """Resolve gene_names from a species keyword, a file path, or an array."""
    if isinstance(gene_names, str) and gene_names in _SPECIES_FILES:
        return load_gene_names(gene_names)
    if isinstance(gene_names, (str, Path)):
        path = Path(gene_names)
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=True)
        return np.array(path.read_text().split(), dtype=object)
    return np.asarray(gene_names, dtype=object)
