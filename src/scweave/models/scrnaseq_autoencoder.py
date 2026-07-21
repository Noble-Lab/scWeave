"""
Masked Autoencoder for scRNA-seq data using PyTorch Lightning.

Masks equal counts of non-zero and zero genes by setting them to a
sentinel value (-1). The model learns to reconstruct all values including
the masked positions.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def _create_masked_input(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Create masked input by randomly masking mask_ratio fraction of the batch.

    Randomly masks mask_ratio of all non-zero values across the entire batch.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, n_features)
    mask_ratio : float
        Fraction of non-zero values to mask (0.0 to 1.0)

    Returns
    -------
    torch.Tensor
        Masked input with masked positions set to -1
    """
    x_masked = x.clone()

    # Get non-zero mask
    nonzero_mask = x > 0.0

    # Generate random values and mask non-zero positions
    rand_vals = torch.rand_like(x)
    # Only consider non-zero positions
    mask_decision = (rand_vals < mask_ratio) & nonzero_mask

    # Apply masking
    x_masked[mask_decision] = -1.0

    return x_masked


class scRNAseqAE(pl.LightningModule):
    """
    Masked Autoencoder for scRNA-seq data.

    Masks equal counts of non-zero and zero genes during training and
    reconstructs all values including the masked positions.

    Architecture:
    - Encoder: input_dim → hidden_dim → latent_dim
    - Decoder: latent_dim → hidden_dim → input_dim
    - Normalization: RMSNorm
    - Activation: SiLU

    Parameters
    ----------
    input_dim : int
        Number of genes (input features)
    mask_ratio : float, default=0.15
        Fraction of non-zero genes to mask during training (0-1)
    latent_dim : int, default=2048
        Latent space dimensionality
    hidden_dim : int, default=4096
        Hidden layer dimension
    learning_rate : float, default=1e-4
        Adam learning rate
    """

    def __init__(
        self,
        input_dim: int,
        mask_ratio: float = 0.15,
        latent_dim: int = 2048,
        hidden_dim: int = 4096,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate

        # Activation
        act = nn.SiLU()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            act,
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            act,
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation (no masking).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent of shape (batch_size, latent_dim)

        Returns
        -------
        torch.Tensor
            Reconstruction of shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masking.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Reconstruction of shape (batch_size, input_dim)
        """
        x_masked = _create_masked_input(x, mask_ratio=self.mask_ratio)
        z = self.encoder(x_masked)
        reconstruction = self.decoder(z)
        return reconstruction

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Batch data with key 'rna': RNA expression of shape (batch_size, n_genes)
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Training loss
        """
        x = batch['rna']
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Batch data with key 'rna': RNA expression of shape (batch_size, n_genes)
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Validation loss
        """
        x = batch['rna']
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return Adam(self.parameters(), lr=self.learning_rate)
