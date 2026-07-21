"""
scWeave inference API.

A single, flat wrapper around a trained :class:`TranslatorModel` that exposes the
three things a user wants at inference time:

- ``predict_hic_from_rna`` : gene expression  -> chromatin contact maps
- ``predict_rna_from_hic`` : chromatin contact maps -> gene expression
- ``match``                : match cells across modalities in the shared latent space

The wrapper handles device placement, evaluation mode, batching over large inputs,
and numpy <-> tensor conversion, so the model can be used without touching
PyTorch Lightning internals.
"""

import inspect
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from scweave.models.translator import TranslatorModel

ArrayLike = Union[np.ndarray, torch.Tensor]


class scWeave:
    """
    High-level inference interface for a trained scWeave model.

    Parameters
    ----------
    model : TranslatorModel
        A trained translator model.
    device : str or torch.device, optional
        Device to run inference on. If ``None``, uses CUDA when available,
        otherwise CPU.

    Examples
    --------
    >>> model = scWeave.load("scweave.ckpt")
    >>> hic = model.predict_hic_from_rna(rna)               # (n, num_chr, 224, 224)
    >>> rna = model.predict_rna_from_hic(hic_embeddings)    # (n, n_genes)
    >>> similarity, matches = model.match(rna, hic_embeddings, direction="rna_to_hic")
    """

    def __init__(
        self,
        model: TranslatorModel,
        device: Union[str, torch.device, None] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device, None] = None,
    ) -> "scWeave":
        """
        Load a trained scWeave model from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a trained ``TranslatorModel`` checkpoint (.ckpt).
        device : str or torch.device, optional
            Device to load the model onto. If ``None``, uses CUDA when available.

        Returns
        -------
        scWeave
            An inference-ready model wrapper.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        hparams = dict(checkpoint.get("hyper_parameters", {}))
        # Keep only the arguments the current model accepts, so checkpoints saved
        # with extra (now-removed) hyperparameters still load.
        accepted = set(inspect.signature(TranslatorModel.__init__).parameters) - {"self"}
        model = TranslatorModel(**{k: v for k, v in hparams.items() if k in accepted})
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, device=device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_hic_from_rna(self, rna: ArrayLike, batch_size: int = 32) -> np.ndarray:
        """
        Predict single-cell Hi-C contact maps from gene expression.

        Parameters
        ----------
        rna : np.ndarray or torch.Tensor
            Gene expression, shape (n_cells, n_genes).
        batch_size : int, default=32
            Number of cells processed per forward pass.

        Returns
        -------
        np.ndarray
            Predicted contact probabilities in [0, 1],
            shape (n_cells, num_chr, 224, 224).
        """
        def step(batch: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.model.predict_hic_from_rna(batch))

        return self._run_batched(step, rna, batch_size)

    def predict_rna_from_hic(self, hic: ArrayLike, batch_size: int = 32) -> np.ndarray:
        """
        Predict gene expression from single-cell Hi-C embeddings.

        Parameters
        ----------
        hic : np.ndarray or torch.Tensor
            HiCFoundation embeddings, shape (n_cells, num_chr, 14, 14, 1024).
        batch_size : int, default=32
            Number of cells processed per forward pass.

        Returns
        -------
        np.ndarray
            Predicted gene expression, shape (n_cells, n_genes).
        """
        return self._run_batched(self.model.predict_rna_from_hic, hic, batch_size)

    # ------------------------------------------------------------------
    # Cross-modality matching
    # ------------------------------------------------------------------

    def match(
        self,
        rna: ArrayLike,
        hic: ArrayLike,
        direction: str = "rna_to_hic",
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match cells across modalities in the shared latent space.

        The query modality is translated into the shared space and compared, by
        cosine similarity, against the encoded representation of the other
        (reference) modality. Each query cell is matched to its most similar
        reference cell.

        Parameters
        ----------
        rna : np.ndarray or torch.Tensor
            Gene expression, shape (n_rna, n_genes).
        hic : np.ndarray or torch.Tensor
            HiCFoundation embeddings, shape (n_hic, num_chr, 14, 14, 1024).
        direction : {"rna_to_hic", "hic_to_rna"}, default="rna_to_hic"
            - ``"rna_to_hic"``: RNA cells are the query, Hi-C cells the reference;
              ``matches[i]`` is the index of the Hi-C cell matched to RNA cell ``i``.
            - ``"hic_to_rna"``: Hi-C cells are the query, RNA cells the reference;
              ``matches[i]`` is the index of the RNA cell matched to Hi-C cell ``i``.
        batch_size : int, default=32
            Number of cells encoded per forward pass.

        Returns
        -------
        similarity : np.ndarray
            Cosine similarity matrix, shape (n_query, n_reference).
        matches : np.ndarray
            Index of the matched reference cell for each query cell,
            shape (n_query,).
        """
        if direction == "rna_to_hic":
            query = self._run_batched(self.model.translate_rna_to_hic_latent, rna, batch_size)
            reference = self._run_batched(self.model.encode_hic, hic, batch_size)
        elif direction == "hic_to_rna":
            query = self._run_batched(self.model.translate_hic_to_rna_latent, hic, batch_size)
            reference = self._run_batched(self.model.encode_rna, rna, batch_size)
        else:
            raise ValueError(
                f"direction must be 'rna_to_hic' or 'hic_to_rna', got {direction!r}"
            )

        similarity = _cosine_similarity(query, reference)
        matches = similarity.argmax(axis=1)
        return similarity, matches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_batched(self, fn, x: ArrayLike, batch_size: int) -> np.ndarray:
        """Run ``fn`` over ``x`` in batches under no_grad and return a numpy array."""
        x = self._to_tensor(x)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = x[start : start + batch_size].to(self.device)
                outputs.append(fn(batch).cpu())
        return torch.cat(outputs, dim=0).numpy()

    def _to_tensor(self, x: ArrayLike) -> torch.Tensor:
        """Convert a numpy array or tensor to a float32 CPU tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.float()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between every row of ``a`` and every row of ``b``."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T
