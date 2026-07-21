"""
Dataset construction and loading for scWeave training.

``prepare_dataset`` turns one or more paired datasets (RNA AnnData + HiCFoundation
embeddings + contact matrices) into the on-disk split layout that
``MultimodalDataset`` reads:

    out_dir/
    ├── training_set/
    │   ├── train_data_rna.npy      (n_train, n_genes)
    │   ├── train_embeddings.npy    (n_train, n_chr, 14, 14, 1024)
    │   ├── train_matrices.npy      (n_train, n_chr, 224, 224)
    │   ├── val_data_rna.npy        (n_val, n_genes)
    │   ├── val_embeddings.npy
    │   ├── val_matrices.npy
    │   ├── gene_names.npy
    │   ├── chrom_order.npy
    │   └── split_info.npy
    └── test_sets/
        └── <name>/                 (one directory per input dataset)
            ├── test_data_rna.npy
            ├── test_embeddings.npy
            ├── test_matrices.npy
            ├── test_ids.npy
            └── split_info.npy

Train and validation splits are merged across datasets; the test split is kept
separate per dataset so each can be evaluated on its own.

Note on memory: splits are gathered fully in RAM before being written, and each
dataset's embeddings are loaded fully into RAM. The Hi-C embeddings are large
(order 100 GB for several thousand cells), so dataset preparation requires a
machine with substantial RAM and disk.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from scweave.utils.rna import _resolve_gene_names, prepare_rna


# ============================================================================
# Exceptions
# ============================================================================

class DatasetValidationError(ValueError):
    """Custom exception for dataset validation failures."""
    pass


# ============================================================================
# Helper Functions
# ============================================================================

def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    if isinstance(array, torch.Tensor):
        return array
    # Create writable copy to avoid memory sharing issues and warnings
    return torch.from_numpy(np.array(array, copy=True)).float()


# ============================================================================
# MultimodalDataset - Loads RNA + Hi-C embeddings + contact matrices
# ============================================================================

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal single-cell data: RNA expression and Hi-C.

    Efficiently loads three modalities from memory-mapped files:
    1. RNA expression matrix (cells × genes)
    2. Hi-C embeddings (cells × chromosomes × 14 × 14 × 1024)
    3. Hi-C contact matrices (cells × chromosomes × 224 × 224, binarized)

    Uses memory mapping to avoid loading entire dataset into RAM.

    Parameters
    ----------
    data_dir : Path
        Directory containing memmap files and metadata
    split : str, default='train'
        Data split to load: 'train', 'val', or 'test'

    Attributes
    ----------
    n_cells : int
        Number of cells in split
    n_genes : int
        Number of RNA genes
    embeddings_shape : tuple
        Dimensions of Hi-C embeddings per cell (chromosomes, 14, 14, 1024)
    matrices_shape : tuple
        Dimensions of Hi-C contact matrices per cell (chromosomes, 224, 224)

    Examples
    --------
    >>> dataset = MultimodalDataset(data_dir="data/splits", split="train")
    >>> sample = dataset[0]
    >>> print(sample['rna'].shape)         # (n_genes,)
    >>> print(sample['embeddings'].shape)  # (n_chr, 14, 14, 1024)
    >>> print(sample['matrices'].shape)    # (n_chr, 224, 224)
    >>> print(sample['label'])             # Cell type label
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        in_memory: bool = False,
    ):
        """
        Initialize multimodal dataset.

        Parameters
        ----------
        data_dir : Path
            Directory containing RNA, embeddings, and matrices memmap files
        split : str, default='train'
            Data split: 'train', 'val', or 'test'
        in_memory : bool, default=False
            If True, load the entire split fully into RAM as torch tensors.
            Matrices are pre-binarized once at load time so __getitem__ is
            pure tensor slicing with no per-sample allocation or casting.
            Requires enough RAM to hold RNA + embeddings + matrices for
            the split.

        Raises
        ------
        ValueError
            If split is invalid
        FileNotFoundError
            If data files don't exist
        DatasetValidationError
            If data is misaligned or invalid
        """
        self._data_dir = Path(data_dir)
        self._split = split
        self._in_memory = in_memory

        # Validate split name
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # Load metadata
        self._metadata = self._load_metadata()

        # Load labels
        labels_path = self._data_dir / f"{split}_labels.npy"
        self._labels = self._load_labels(labels_path if labels_path.exists() else None)

        if in_memory:
            self._load_in_memory()
        else:
            self._rna_data = self._load_memmap_file(
                f"{split}_data_rna.npy", 'rna_shape', "RNA data"
            )
            self._embeddings = self._load_memmap_file(
                f"{split}_embeddings.npy", 'embeddings_shape', "embeddings"
            )
            self._matrices = self._load_memmap_file(
                f"{split}_matrices.npy", 'matrices_shape', "matrices"
            )

        # Validate alignment
        self._validate()

    def _load_full_array(
        self,
        filename: str,
        shape_key: str,
        description: str,
    ) -> np.ndarray:
        """Load a data file fully into RAM (no memmap)."""
        file_path = self._data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{description} file not found: {file_path}")
        return np.load(file_path)

    def _load_in_memory(self) -> None:
        """Load RNA, embeddings, and matrices fully into RAM as torch tensors.

        Matrices are pre-binarized to float32 so the per-sample `(x > 0).float()`
        in __getitem__ becomes a no-op.
        """
        split = self._split

        rna = self._load_full_array(f"{split}_data_rna.npy", 'rna_shape', "RNA data")
        emb = self._load_full_array(f"{split}_embeddings.npy", 'embeddings_shape', "embeddings")
        mat = self._load_full_array(f"{split}_matrices.npy", 'matrices_shape', "matrices")

        # Convert to torch tensors. from_numpy shares memory; we own these arrays
        # so no copy is needed.
        self._rna_data = torch.from_numpy(np.ascontiguousarray(rna, dtype=np.float32))
        self._embeddings = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
        # Pre-binarize matrices once (not per-sample)
        mat_bin = (mat > 0).astype(np.float32)
        self._matrices = torch.from_numpy(mat_bin)

        del rna, emb, mat, mat_bin

    def _load_labels(self, labels_path: Optional[Path]) -> Optional[np.ndarray]:
        """Load labels from file if it exists."""
        if labels_path is None or not labels_path.exists():
            return None
        try:
            return np.load(labels_path, allow_pickle=True)
        except Exception as e:
            raise DatasetValidationError(f"Failed to load labels: {e}")

    def _get_label(self, idx: int) -> Optional[Any]:
        """Get label for sample at index."""
        if self._labels is None:
            return None
        return self._labels[idx]

    def _load_memmap_file(
        self,
        filename: str,
        shape_key: str,
        description: str,
    ) -> np.ndarray:
        """Load a .npy file in memory-mapped read mode."""
        file_path = self._data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{description} file not found: {file_path}")
        return np.load(file_path, mmap_mode='r')

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load split metadata (shapes) from split_info.npy.

        Returns
        -------
        Dict[str, Any]
            Metadata for this split with shape information
        """
        split_info_path = self._data_dir / "split_info.npy"
        if not split_info_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {split_info_path}. "
                f"Run prepare_dataset first."
            )

        try:
            split_info = np.load(split_info_path, allow_pickle=True).item()

            rna_path = self._data_dir / f"{self._split}_data_rna.npy"
            emb_path = self._data_dir / f"{self._split}_embeddings.npy"
            mat_path = self._data_dir / f"{self._split}_matrices.npy"

            if not all([rna_path.exists(), emb_path.exists(), mat_path.exists()]):
                raise FileNotFoundError(
                    f"Data files not found for split '{self._split}'"
                )

            n_genes = split_info['n_genes']
            n_chr = split_info['n_chromosomes']
            n_cells = split_info['n_train'] if self._split == 'train' else split_info['n_val']

            return {
                'rna_shape': (n_cells, n_genes),
                'embeddings_shape': (n_cells, n_chr, 14, 14, 1024),
                'matrices_shape': (n_cells, n_chr, 224, 224),
                'n_cells': n_cells,
            }
        except Exception as e:
            raise DatasetValidationError(f"Failed to load metadata from split_info: {e}")

    def _validate(self) -> None:
        """Validate data alignment across modalities."""
        if self._rna_data.shape[0] != self._embeddings.shape[0]:
            raise DatasetValidationError(
                f"RNA cells ({self._rna_data.shape[0]}) != "
                f"embeddings cells ({self._embeddings.shape[0]})"
            )

        if self._rna_data.shape[0] != self._matrices.shape[0]:
            raise DatasetValidationError(
                f"RNA cells ({self._rna_data.shape[0]}) != "
                f"matrices cells ({self._matrices.shape[0]})"
            )

        if self._labels is not None and len(self._labels) != self.n_cells:
            raise DatasetValidationError(
                f"Labels ({len(self._labels)}) != cells ({self.n_cells})"
            )

    def __len__(self) -> int:
        """Return number of cells."""
        return self.n_cells

    @property
    def n_cells(self) -> int:
        """Number of cells in split."""
        return self._rna_data.shape[0]

    @property
    def n_genes(self) -> int:
        """Number of RNA genes."""
        return self._rna_data.shape[1]

    @property
    def embeddings_shape(self) -> tuple:
        """Shape of HiC embeddings for one sample (excluding batch dim)."""
        return self._embeddings.shape[1:]

    @property
    def matrices_shape(self) -> tuple:
        """Shape of HiC matrices for one sample (excluding batch dim)."""
        return self._matrices.shape[1:]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample with all three modalities as tensors.

        Skips samples with NaN values by advancing to next valid sample.

        Parameters
        ----------
        idx : int
            Cell index in split

        Returns
        -------
        Dict[str, Any]
            Sample with keys:
            - 'rna': RNA expression, shape (n_genes,)
            - 'embeddings': Hi-C embeddings, shape (n_chr, 14, 14, 1024)
            - 'matrices': Hi-C contact (binarized), shape (n_chr, 224, 224), values in {0, 1}
            - 'label': Cell type label (None if not available)
        """
        if self._in_memory:
            # Everything is already a torch tensor in RAM; getitem is pure slicing.
            while idx < self.n_cells:
                rna = self._rna_data[idx]
                embeddings = self._embeddings[idx]
                matrices = self._matrices[idx]
                if not (torch.isnan(rna).any() or torch.isnan(embeddings).any()):
                    break
                idx += 1

            if idx >= self.n_cells:
                raise IndexError(f"No valid samples found from index {idx - self.n_cells + 1} onwards")

            return {
                'rna': rna,
                'embeddings': embeddings,
                'matrices': matrices,  # already pre-binarized float32
                'label': self._get_label(idx),
            }

        # Memmap path (lazy disk reads)
        while idx < self.n_cells:
            rna = self._rna_data[idx]
            embeddings = self._embeddings[idx]
            matrices = self._matrices[idx]

            if not (np.isnan(rna).any() or np.isnan(embeddings).any() or np.isnan(matrices).any()):
                break
            idx += 1

        if idx >= self.n_cells:
            raise IndexError(f"No valid samples found from index {idx - self.n_cells + 1} onwards")

        matrices = _to_tensor(matrices)
        matrices_binary = (matrices > 0).float()

        return {
            'rna': _to_tensor(rna),
            'embeddings': _to_tensor(embeddings),
            'matrices': matrices_binary,
            'label': self._get_label(idx),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'split': self._split,
            'embeddings_shape': self.embeddings_shape,
            'matrices_shape': self.matrices_shape,
            'has_labels': self._labels is not None,
            'data_dir': str(self._data_dir),
        }


# ============================================================================
# Dataset preparation: raw paired inputs -> on-disk split layout
# ============================================================================

def _load_hic_data(embeddings_path: Path, matrices_path: Path, ids_path: Path) -> Dict[str, Any]:
    """Load HiC embeddings, matrices, and cell ids fully into RAM."""
    embeddings = np.load(embeddings_path)
    matrices = np.load(matrices_path)
    ids = np.load(ids_path, allow_pickle=True)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    if matrices.dtype != np.float32:
        matrices = matrices.astype(np.float32, copy=False)

    return {
        'embeddings': embeddings,
        'matrices': matrices,
        'ids': np.asarray([str(x) for x in ids], dtype=object),
        'n_cells': embeddings.shape[0],
    }


def _align_rna_to_hic(X_rna: np.ndarray, adata_ids: np.ndarray, hic_ids: np.ndarray) -> np.ndarray:
    """Reorder RNA rows (in AnnData order) to match the Hi-C cell order."""
    id_to_idx = {cid: i for i, cid in enumerate(adata_ids)}
    reindex = np.fromiter(
        (id_to_idx.get(cid, -1) for cid in hic_ids),
        dtype=np.int64,
        count=len(hic_ids),
    )
    missing_mask = reindex < 0
    if missing_mask.any():
        missing = hic_ids[missing_mask]
        raise ValueError(
            f"{missing_mask.sum():,} Hi-C cell IDs not found in RNA data. "
            f"First 10 missing IDs: {list(missing[:10])}"
        )
    return X_rna[reindex]


def _create_splits(
    n_cells: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Create random train/val/test splits."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n_cells)
    rng.shuffle(indices)

    train_end = int(n_cells * train_ratio)
    val_end = train_end + int(n_cells * val_ratio)

    return {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:],
    }


def _gather_split(indices_list: List[tuple], processed_datasets: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Gather cells from multiple datasets into contiguous in-memory arrays."""
    n_cells = len(indices_list)
    n_genes = processed_datasets[0]['X_rna'].shape[1]
    emb_shape = processed_datasets[0]['embeddings'].shape[1:]
    mat_shape = processed_datasets[0]['matrices'].shape[1:]

    rna_out = np.empty((n_cells, n_genes), dtype=np.float32)
    emb_out = np.empty((n_cells,) + emb_shape, dtype=np.float32)
    mat_out = np.empty((n_cells,) + mat_shape, dtype=np.float32)
    ids_out = np.empty(n_cells, dtype=object)

    idx_arr = np.asarray(indices_list, dtype=np.int64)  # (N, 3): dataset_num, cell_idx, out_idx
    dataset_nums = idx_arr[:, 0]
    cell_indices = idx_arr[:, 1]
    out_positions_all = np.arange(n_cells, dtype=np.int64)

    for ds_num in np.unique(dataset_nums):
        mask = dataset_nums == ds_num
        src = cell_indices[mask]
        out = out_positions_all[mask]
        dataset = processed_datasets[ds_num]

        rna_out[out] = dataset['X_rna'][src]
        emb_out[out] = dataset['embeddings'][src]
        mat_out[out] = dataset['matrices'][src]
        ids_out[out] = dataset['ids'][src]

    return {
        'rna': rna_out,
        'emb': emb_out,
        'mat': mat_out,
        'ids': np.array(ids_out.tolist()),
    }


def _write_split(merged_dir: Path, split_name: str, indices_list: List[tuple], processed_datasets: List[Dict[str, Any]]) -> None:
    """Build a merged split in RAM, then save each array."""
    if len(indices_list) == 0:
        return
    split = _gather_split(indices_list, processed_datasets)
    np.save(merged_dir / f"{split_name}_data_rna.npy", split['rna'])
    np.save(merged_dir / f"{split_name}_embeddings.npy", split['emb'])
    np.save(merged_dir / f"{split_name}_matrices.npy", split['mat'])
    np.save(merged_dir / f"{split_name}_ids.npy", split['ids'])


def _save_test_set(test_dir: Path, name: str, test_indices: np.ndarray, dataset: Dict[str, Any]) -> None:
    """Save a single dataset's test split."""
    dataset_test_dir = test_dir / name
    dataset_test_dir.mkdir(parents=True, exist_ok=True)
    if len(test_indices) == 0:
        return

    src = np.asarray(test_indices, dtype=np.int64)
    np.save(dataset_test_dir / "test_data_rna.npy", dataset['X_rna'][src].astype(np.float32, copy=False))
    np.save(dataset_test_dir / "test_embeddings.npy", dataset['embeddings'][src].astype(np.float32, copy=False))
    np.save(dataset_test_dir / "test_matrices.npy", dataset['matrices'][src].astype(np.float32, copy=False))
    np.save(dataset_test_dir / "test_ids.npy", np.asarray(dataset['ids'][src]))

    split_info = {
        'n_train': 0,
        'n_val': len(src),
        'n_genes': dataset['X_rna'].shape[1],
        'n_chromosomes': dataset['embeddings'].shape[1],
        'datasets': [name],
    }
    np.save(dataset_test_dir / "split_info.npy", split_info)


def prepare_dataset(
    adatas: Sequence[Union[str, Path, ad.AnnData]],
    embeddings: Sequence[Union[str, Path]],
    matrices: Sequence[Union[str, Path]],
    ids: Sequence[Union[str, Path]],
    names: Sequence[str],
    gene_names: Union[str, Path, np.ndarray, list],
    out_dir: Union[str, Path],
    chrom_order: Optional[Sequence[str]] = None,
    normalize_library_size: bool = True,
    target_sum: float = 1e4,
    apply_log: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Build the scWeave training split layout from one or more paired datasets.

    For each dataset the RNA AnnData is aligned to the model's gene set and to the
    Hi-C cell order, then split into train/val/test. Train and validation splits
    are merged across datasets; each dataset's test split is written separately.

    Parameters
    ----------
    adatas : sequence of str, Path, or AnnData
        RNA expression, one per dataset. ``obs.index`` must contain cell IDs that
        match the corresponding ``ids`` file; ``var_names`` must match ``gene_names``.
    embeddings : sequence of str or Path
        Paths to HiCFoundation embeddings ``.npy`` files, shape
        (n_cells, n_chr, 14, 14, 1024), one per dataset.
    matrices : sequence of str or Path
        Paths to contact-matrix ``.npy`` files, shape (n_cells, n_chr, 224, 224),
        one per dataset.
    ids : sequence of str or Path
        Paths to cell-ID ``.npy`` files, one per dataset, matching the Hi-C cell order.
    names : sequence of str
        Dataset names, used for the per-dataset test directories.
    gene_names : str, Path, array, or list
        The model's gene set (``"mouse"``/``"human"``, an array, or a path).
    out_dir : str or Path
        Output directory; ``training_set/`` and ``test_sets/`` are written here.
    chrom_order : sequence of str, optional
        Chromosome names, in order. If None, defaults to ``chr1..chrN``.
    normalize_library_size, target_sum, apply_log : RNA preprocessing options,
        passed to :func:`scweave.prepare_rna` (defaults match training).
    train_ratio, val_ratio, test_ratio : float
        Split fractions; must sum to 1.
    seed : int
        Random seed for the splits.
    """
    n = len(adatas)
    if not (len(embeddings) == len(matrices) == len(ids) == len(names) == n):
        raise ValueError("adatas, embeddings, matrices, ids, and names must have the same length")

    gene_names = _resolve_gene_names(gene_names)
    out_dir = Path(out_dir)

    processed_datasets: List[Dict[str, Any]] = []
    n_chr = None
    for i in range(n):
        hic = _load_hic_data(Path(embeddings[i]), Path(matrices[i]), Path(ids[i]))

        adata = adatas[i] if isinstance(adatas[i], ad.AnnData) else ad.read_h5ad(adatas[i])
        X_rna = prepare_rna(
            adata,
            gene_names,
            normalize_library_size=normalize_library_size,
            target_sum=target_sum,
            apply_log=apply_log,
        )
        adata_ids = np.asarray([str(x) for x in adata.obs.index], dtype=object)
        X_rna = _align_rna_to_hic(X_rna, adata_ids, hic['ids'])

        if n_chr is None:
            n_chr = hic['embeddings'].shape[1]
        elif hic['embeddings'].shape[1] != n_chr:
            raise ValueError(
                f"Chromosome count mismatch: dataset '{names[i]}' has "
                f"{hic['embeddings'].shape[1]} chromosomes, expected {n_chr}"
            )

        splits = _create_splits(hic['n_cells'], train_ratio, val_ratio, test_ratio, seed)

        processed_datasets.append({
            'name': names[i],
            'X_rna': X_rna,
            'embeddings': hic['embeddings'],
            'matrices': hic['matrices'],
            'ids': hic['ids'],
            'splits': splits,
        })

    if chrom_order is None:
        chrom_order = np.array([f"chr{i + 1}" for i in range(n_chr)], dtype=object)
    else:
        chrom_order = np.asarray(chrom_order, dtype=object)

    # Build merged train/val index lists (dataset_num, cell_idx, out_idx).
    train_indices: List[tuple] = []
    val_indices: List[tuple] = []
    out_idx = 0
    for ds_num, dataset in enumerate(processed_datasets):
        for cell_idx in dataset['splits']['train']:
            train_indices.append((ds_num, int(cell_idx), out_idx))
            out_idx += 1
        for cell_idx in dataset['splits']['val']:
            val_indices.append((ds_num, int(cell_idx), out_idx))
            out_idx += 1

    # Save merged training/validation splits.
    training_dir = out_dir / "training_set"
    training_dir.mkdir(parents=True, exist_ok=True)
    _write_split(training_dir, "train", train_indices, processed_datasets)
    _write_split(training_dir, "val", val_indices, processed_datasets)

    np.save(training_dir / "gene_names.npy", np.asarray(gene_names))
    np.save(training_dir / "chrom_order.npy", chrom_order)
    np.save(training_dir / "split_info.npy", {
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_genes': len(gene_names),
        'n_chromosomes': int(n_chr),
        'datasets': list(names),
    })

    # Save per-dataset test splits.
    test_dir = out_dir / "test_sets"
    test_dir.mkdir(parents=True, exist_ok=True)
    for dataset in processed_datasets:
        _save_test_set(test_dir, dataset['name'], dataset['splits']['test'], dataset)
