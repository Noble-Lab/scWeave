"""
Translator model for bidirectional cross-modality translation.

Couples an scRNA-seq autoencoder and an scHi-C autoencoder through two learnable
translator modules, and trains the whole model end-to-end from scratch. Once
trained it can translate in both directions (RNA -> HiC and HiC -> RNA) and
expose the shared latent representations used to match cells across modalities.
"""

from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .scrnaseq_autoencoder import scRNAseqAE
from .schic_autoencoder import scHiCAutoencoder
from .utils import _init_linear_weights


class TranslatorModule(nn.Module):
    """
    Translator module for cross-modality latent space transformation.

    Two-layer MLP with SiLU activation, RMSNorm, and residual connection.
    Architecture aligned with scRNA-seq and scHi-C autoencoders.

    Parameters
    ----------
    latent_dim : int, default=2048
        Latent space dimensionality (same for input and output)
    """

    def __init__(self, latent_dim: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim

        # Two-layer MLP with SiLU and RMSNorm
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.SiLU()
        self.norm1 = nn.RMSNorm(latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.norm2 = nn.RMSNorm(latent_dim)

        # Initialize weights
        self.apply(_init_linear_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Translate latent representation with residual connection.

        Parameters
        ----------
        z : torch.Tensor
            Input latent representation, shape (batch, latent_dim)

        Returns
        -------
        torch.Tensor
            Translated latent representation, shape (batch, latent_dim)
        """
        residual = z
        z = self.fc1(z)
        z = self.act(z)
        z = self.norm1(z)
        z = self.fc2(z)
        z = self.norm2(z)
        return z + residual  # Residual connection


class TranslatorModel(pl.LightningModule):
    """
    Translator model for bidirectional translation between RNA and HiC.

    Combines an scRNA-seq autoencoder and an scHi-C autoencoder with two learnable
    translator modules, trained end-to-end from scratch. After training the model
    can translate in either direction and expose the shared latent representations
    used to match cells across modalities.

    Architecture:
    ```
    RNA -> RNA_encoder -> RNA_latent -> RNA_decoder -> RNA_recon
                              |
                      RNA_to_HiC_translator
                              |
                        HiC_latent_pred -> HiC_decoder -> HiC_from_RNA

    HiC -> HiC_encoder -> HiC_latent -> HiC_decoder -> HiC_recon
                              |
                      HiC_to_RNA_translator
                              |
                        RNA_latent_pred -> RNA_decoder -> RNA_from_HiC
    ```

    Loss components:
    - RNA reconstruction: MSE(RNA_recon, RNA)
    - HiC reconstruction: BCE(HiC_recon, HiC_matrices)
    - RNA->HiC translation: BCE(HiC_from_RNA, HiC_matrices)
    - HiC->RNA translation: MSE(RNA_from_HiC, RNA)
    - Latent alignment (optional): soft nearest-neighbor loss in the shared space

    Parameters
    ----------
    rna_dim : int
        RNA feature dimension (number of genes)
    latent_dim : int, default=2048
        Shared latent space dimensionality for both modalities
    rna_hidden_dim : int, default=4096
        Hidden dimension for RNA autoencoder
    rna_mask_ratio : float, default=0.3
        Masking ratio for RNA autoencoder training
    hic_num_chr : int, default=20
        Number of chromosomes (scHi-C autoencoder)
    hic_patch_grid_size : int, default=14
        Patch grid size (scHi-C autoencoder)
    hic_input_embed_dim : int, default=1024
        Input embedding dimension from HiCFoundation
    hic_feature_proj_dim : int, default=256
        Feature projection dimension (scHi-C autoencoder)
    hic_chr_embed_dim : int, default=1024
        Chromosome embedding dimension (scHi-C autoencoder)
    hic_patch_embed_dim : int, default=256
        Patch embedding dimension (scHi-C autoencoder)
    hic_output_size : int, default=224
        Output matrix size (scHi-C autoencoder)
    hic_patch_size : int, default=16
        Patch size for Vision Transformer (scHi-C autoencoder)
    hic_vit_depth : int, default=8
        Number of transformer blocks (scHi-C autoencoder)
    hic_vit_heads : int, default=8
        Number of attention heads (scHi-C autoencoder)
    hic_chr_encoder_intermediate_dim : int, default=2048
        Chromosome encoder intermediate dimension
    hic_cell_encoder_intermediate_dim : int, default=4096
        Cell encoder intermediate dimension
    hic_cell_decoder_intermediate_dim : int, default=4096
        Cell decoder intermediate dimension
    hic_chr_decoder_intermediate_dim : int, default=2048
        Chromosome decoder intermediate dimension
    hic_pos_weight : float, default=3.0
        Positive class weight for BCE loss (scHi-C autoencoder)
    learning_rate : float, default=1e-4
        Adam learning rate
    rna_recon_weight : float, default=1.0
        Weight for RNA reconstruction loss
    hic_recon_weight : float, default=1.0
        Weight for HiC reconstruction loss
    rna_to_hic_weight : float, default=1.0
        Weight for RNA->HiC translation loss
    hic_to_rna_weight : float, default=1.0
        Weight for HiC->RNA translation loss
    latent_align_weight : float, default=0.0
        Weight for the latent alignment loss
    latent_align_temperature : float, default=0.1
        Temperature for the soft nearest-neighbor alignment loss

    Examples
    --------
    >>> model = TranslatorModel(rna_dim=5656, hic_num_chr=20)
    >>> rna = torch.randn(16, 5656)
    >>> hic_embeddings = torch.randn(16, 20, 14, 14, 1024)
    >>> rna_recon, rna_from_hic, hic_recon, hic_from_rna = model(rna, hic_embeddings)
    >>> hic_pred = model.predict_hic_from_rna(rna)
    >>> rna_pred = model.predict_rna_from_hic(hic_embeddings)
    """

    def __init__(
        self,
        rna_dim: int,
        latent_dim: int = 2048,
        # RNA Autoencoder parameters
        rna_hidden_dim: int = 4096,
        rna_mask_ratio: float = 0.3,
        # scHi-C Autoencoder parameters
        hic_num_chr: int = 20,
        hic_patch_grid_size: int = 14,
        hic_input_embed_dim: int = 1024,
        hic_feature_proj_dim: int = 256,
        hic_chr_embed_dim: int = 1024,
        hic_patch_embed_dim: int = 256,
        hic_output_size: int = 224,
        hic_patch_size: int = 16,
        hic_vit_depth: int = 8,
        hic_vit_heads: int = 8,
        hic_chr_encoder_intermediate_dim: int = 2048,
        hic_cell_encoder_intermediate_dim: int = 4096,
        hic_cell_decoder_intermediate_dim: int = 4096,
        hic_chr_decoder_intermediate_dim: int = 2048,
        hic_pos_weight: float = 3.0,
        # Translator parameters
        learning_rate: float = 1e-4,
        # Loss weights
        rna_recon_weight: float = 1.0,
        hic_recon_weight: float = 1.0,
        rna_to_hic_weight: float = 1.0,
        hic_to_rna_weight: float = 1.0,
        latent_align_weight: float = 0.0,
        latent_align_temperature: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.rna_recon_weight = rna_recon_weight
        self.hic_recon_weight = hic_recon_weight
        self.rna_to_hic_weight = rna_to_hic_weight
        self.hic_to_rna_weight = hic_to_rna_weight
        self.latent_align_weight = latent_align_weight
        self.latent_align_temperature = latent_align_temperature

        # RNA encoder/decoder
        self.rna_model = scRNAseqAE(
            input_dim=rna_dim,
            latent_dim=latent_dim,
            hidden_dim=rna_hidden_dim,
            mask_ratio=rna_mask_ratio,
            learning_rate=learning_rate,
        )

        # HiC encoder/decoder
        self.hic_model = scHiCAutoencoder(
            num_chr=hic_num_chr,
            patch_grid_size=hic_patch_grid_size,
            input_embed_dim=hic_input_embed_dim,
            feature_proj_dim=hic_feature_proj_dim,
            chr_embed_dim=hic_chr_embed_dim,
            cell_embed_dim=latent_dim,
            patch_embed_dim=hic_patch_embed_dim,
            output_size=hic_output_size,
            patch_size=hic_patch_size,
            vit_depth=hic_vit_depth,
            vit_heads=hic_vit_heads,
            chr_encoder_intermediate_dim=hic_chr_encoder_intermediate_dim,
            cell_encoder_intermediate_dim=hic_cell_encoder_intermediate_dim,
            cell_decoder_intermediate_dim=hic_cell_decoder_intermediate_dim,
            chr_decoder_intermediate_dim=hic_chr_decoder_intermediate_dim,
            learning_rate=learning_rate,
            pos_weight=hic_pos_weight,
        )

        # Translator modules for cross-modality
        self.rna_to_hic_translator = TranslatorModule(latent_dim=latent_dim)
        self.hic_to_rna_translator = TranslatorModule(latent_dim=latent_dim)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hic_pos_weight))

    def forward(
        self,
        rna: torch.Tensor,
        hic_embeddings: torch.Tensor,
        return_latents: bool = False,
    ) -> Tuple:
        """
        Forward pass for cross-modality translation.

        Parameters
        ----------
        rna : torch.Tensor
            RNA expression data, shape (batch_size, rna_dim)
        hic_embeddings : torch.Tensor
            HiC embeddings, shape (batch_size, num_chr, 14, 14, 1024)
        return_latents : bool, default=False
            If True, also return a dict of the four latent tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (rna_recon, rna_from_hic, hic_recon, hic_from_rna)
        If return_latents is True, a second element is appended:
            dict with keys: rna_latent, hic_latent, rna_to_hic_latent, hic_to_rna_latent
        """
        # Encode to latent space
        rna_latent = self.rna_model.encode(rna)
        hic_latent = self.hic_model.encode(hic_embeddings)

        # Translate latents
        rna_to_hic_latent = self.rna_to_hic_translator(rna_latent)
        hic_to_rna_latent = self.hic_to_rna_translator(hic_latent)

        # Decode: within-modality reconstruction
        rna_recon = self.rna_model.decode(rna_latent)
        hic_recon = self.hic_model.decode(hic_latent)

        # Decode: cross-modality translation
        rna_from_hic = self.rna_model.decode(hic_to_rna_latent)
        hic_from_rna = self.hic_model.decode(rna_to_hic_latent)

        outputs = (rna_recon, rna_from_hic, hic_recon, hic_from_rna)
        if return_latents:
            latents = {
                "rna_latent": rna_latent,
                "hic_latent": hic_latent,
                "rna_to_hic_latent": rna_to_hic_latent,
                "hic_to_rna_latent": hic_to_rna_latent,
            }
            return outputs, latents
        return outputs

    def _soft_nn_loss(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Soft nearest-neighbor loss between two paired latent matrices.

        For each cell i, computes a softmax over cosine similarities to all
        cells in B, then maximises the probability on the true pair B[i].
        Computed symmetrically (A->B and B->A) and averaged.

        Parameters
        ----------
        A, B : torch.Tensor, shape (batch, latent_dim)
            Paired latents — A[i] and B[i] are the same cell.

        Returns
        -------
        torch.Tensor
            Scalar loss (lower = better alignment).
        """
        A = nn.functional.normalize(A, dim=-1)
        B = nn.functional.normalize(B, dim=-1)
        logits = torch.mm(A, B.T) / self.latent_align_temperature  # (N, N)
        labels = torch.arange(len(A), device=A.device)
        loss_ab = nn.functional.cross_entropy(logits, labels)
        loss_ba = nn.functional.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2

    def _compute_losses(
        self,
        rna: torch.Tensor,
        rna_recon: torch.Tensor,
        rna_from_hic: torch.Tensor,
        hic_matrices: torch.Tensor,
        hic_recon: torch.Tensor,
        hic_from_rna: torch.Tensor,
        latents: Optional[dict] = None,
    ) -> dict:
        """
        Compute all loss components.

        Parameters
        ----------
        rna : torch.Tensor
            Ground truth RNA expression
        rna_recon : torch.Tensor
            Reconstructed RNA from RNA latent
        rna_from_hic : torch.Tensor
            RNA predicted from HiC (cross-modal translation)
        hic_matrices : torch.Tensor
            Ground truth HiC contact matrices
        hic_recon : torch.Tensor
            Reconstructed HiC from HiC latent
        hic_from_rna : torch.Tensor
            HiC predicted from RNA (cross-modal translation)

        Returns
        -------
        dict
            Dictionary with keys: 'rna_recon', 'hic_recon', 'rna_to_hic',
            'hic_to_rna', 'latent_align', 'total'
        """
        # 1. RNA reconstruction loss (MSE)
        rna_recon_loss = self.mse_loss(rna_recon, rna)

        # 2. HiC reconstruction loss (BCE)
        hic_recon_loss = self.bce_with_logits_loss(hic_recon, hic_matrices)

        # 3. Translation losses (compare decoded predictions to ground truth)
        # RNA -> HiC translation: compare predicted HiC matrices to ground truth
        rna_to_hic_loss = self.bce_with_logits_loss(hic_from_rna, hic_matrices)
        # HiC -> RNA translation: compare predicted RNA expression to ground truth
        hic_to_rna_loss = self.mse_loss(rna_from_hic, rna)

        # 4. Latent alignment loss (soft nearest-neighbor, optional)
        latent_align_loss = torch.tensor(0.0, device=rna.device)
        if self.latent_align_weight > 0.0 and latents is not None:
            loss_rna_to_hic = self._soft_nn_loss(latents["rna_to_hic_latent"], latents["hic_latent"])
            loss_hic_to_rna = self._soft_nn_loss(latents["hic_to_rna_latent"], latents["rna_latent"])
            latent_align_loss = (loss_rna_to_hic + loss_hic_to_rna) / 2

        # 5. Total loss (weighted sum of all components)
        total_loss = (
            self.rna_recon_weight * rna_recon_loss
            + self.hic_recon_weight * hic_recon_loss
            + self.rna_to_hic_weight * rna_to_hic_loss
            + self.hic_to_rna_weight * hic_to_rna_loss
            + self.latent_align_weight * latent_align_loss
        )

        return {
            'rna_recon': rna_recon_loss,
            'hic_recon': hic_recon_loss,
            'rna_to_hic': rna_to_hic_loss,
            'hic_to_rna': hic_to_rna_loss,
            'latent_align': latent_align_loss,
            'total': total_loss,
        }

    def training_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Batch data with keys 'rna', 'embeddings', 'matrices'
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Training loss
        """
        # Extract batch data (dict format, consistent with autoencoders)
        rna = batch['rna']
        hic_embeddings = batch['embeddings']
        hic_matrices = batch['matrices']

        # Forward pass
        (rna_recon, rna_from_hic, hic_recon, hic_from_rna), latents = self(
            rna, hic_embeddings, return_latents=True
        )

        # Compute all losses
        losses = self._compute_losses(
            rna, rna_recon, rna_from_hic,
            hic_matrices, hic_recon, hic_from_rna,
            latents=latents,
        )

        # Compute grouped losses for logging
        recon_loss = losses['rna_recon'] + losses['hic_recon']
        trans_loss = losses['rna_to_hic'] + losses['hic_to_rna']

        # Log losses: total + grouped components (sync_dist for correct mean under DDP)
        self.log("train_loss", losses['total'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon", recon_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_trans", trans_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_align", losses['latent_align'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return losses['total']

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Batch data with keys 'rna', 'embeddings', 'matrices'
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Validation loss
        """
        # Extract batch data (dict format, consistent with autoencoders)
        rna = batch['rna']
        hic_embeddings = batch['embeddings']
        hic_matrices = batch['matrices']

        # Forward pass
        (rna_recon, rna_from_hic, hic_recon, hic_from_rna), latents = self(
            rna, hic_embeddings, return_latents=True
        )

        # Compute all losses
        losses = self._compute_losses(
            rna, rna_recon, rna_from_hic,
            hic_matrices, hic_recon, hic_from_rna,
            latents=latents,
        )

        # Compute grouped losses for logging
        recon_loss = losses['rna_recon'] + losses['hic_recon']
        trans_loss = losses['rna_to_hic'] + losses['hic_to_rna']

        # Log losses: total + grouped components (sync_dist for correct mean under DDP)
        self.log("val_loss", losses['total'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recon", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_trans", trans_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_align", losses['latent_align'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return losses['total']

    # ========================================================================
    # User-facing API methods
    # ========================================================================

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Encode RNA expression to RNA latent space.

        Parameters
        ----------
        rna : torch.Tensor
            RNA expression data, shape (batch_size, rna_dim)

        Returns
        -------
        torch.Tensor
            RNA latent representation, shape (batch_size, latent_dim)
        """
        return self.rna_model.encode(rna)

    def encode_hic(self, hic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode HiC embeddings to HiC latent space.

        Parameters
        ----------
        hic_embeddings : torch.Tensor
            HiC embeddings, shape (batch_size, num_chr, 14, 14, 1024)

        Returns
        -------
        torch.Tensor
            HiC latent representation, shape (batch_size, latent_dim)
        """
        return self.hic_model.encode(hic_embeddings)

    def translate_rna_to_hic_latent(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Translate RNA expression to HiC latent space.

        Parameters
        ----------
        rna : torch.Tensor
            RNA expression data, shape (batch_size, rna_dim)

        Returns
        -------
        torch.Tensor
            Translated HiC latent representation, shape (batch_size, latent_dim)
        """
        rna_latent = self.encode_rna(rna)
        return self.rna_to_hic_translator(rna_latent)

    def translate_hic_to_rna_latent(self, hic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Translate HiC embeddings to RNA latent space.

        Parameters
        ----------
        hic_embeddings : torch.Tensor
            HiC embeddings, shape (batch_size, num_chr, 14, 14, 1024)

        Returns
        -------
        torch.Tensor
            Translated RNA latent representation, shape (batch_size, latent_dim)
        """
        hic_latent = self.encode_hic(hic_embeddings)
        return self.hic_to_rna_translator(hic_latent)

    def predict_hic_from_rna(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Predict HiC contact matrices from RNA expression.

        Full cross-modality pipeline: RNA -> RNA latent -> translated HiC latent -> HiC matrices

        Parameters
        ----------
        rna : torch.Tensor
            RNA expression data, shape (batch_size, rna_dim)

        Returns
        -------
        torch.Tensor
            Predicted HiC contact matrices, shape (batch_size, num_chr, 224, 224)
        """
        hic_latent_pred = self.translate_rna_to_hic_latent(rna)
        return self.hic_model.decode(hic_latent_pred)

    def predict_rna_from_hic(self, hic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict RNA expression from HiC embeddings.

        Full cross-modality pipeline: HiC -> HiC latent -> translated RNA latent -> RNA expression

        Parameters
        ----------
        hic_embeddings : torch.Tensor
            HiC embeddings, shape (batch_size, num_chr, 14, 14, 1024)

        Returns
        -------
        torch.Tensor
            Predicted RNA expression, shape (batch_size, rna_dim)
        """
        rna_latent_pred = self.translate_hic_to_rna_latent(hic_embeddings)
        return self.rna_model.decode(rna_latent_pred)

    def configure_optimizers(self):
        """Configure the Adam optimizer and ReduceLROnPlateau scheduler.

        Returns
        -------
        dict
            Optimizer and scheduler configuration
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
