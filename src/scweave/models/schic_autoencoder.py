"""
scHiC Autoencoder for HiC embedding dimensionality reduction.

Self-contained module providing an autoencoder architecture for learning compact
representations of high-dimensional scHiC embeddings (e.g., HiCFoundation
embeddings) and reconstructing Hi-C contact matrices.

Architecture:
- Encoder: Embeddings → Feature Projection → Chr Encoder → Cell Encoder → Cell Embedding
- Decoder: Cell Embedding → Cell Decoder → Chr Decoder → Vision Transformer → Matrices
"""

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from timm.models.vision_transformer import Block

from .utils import _init_linear_weights, init_position_embed


# ============================================================================
# ENCODER COMPONENTS
# ============================================================================


class FeatureProjector(nn.Module):
    """Projects high-dim features to lower dimension with single linear layer."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

        # Initialize weights
        _init_linear_weights(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features while preserving spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, H, W, in_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, H, W, out_dim)
        """
        batch_size, H, W = x.shape[:3]
        # Reshape to (batch*H*W, in_dim)
        x = x.reshape(batch_size * H * W, -1)
        # Apply projection
        x = self.fc(x)
        # Reshape back to (batch, H, W, out_dim)
        x = x.reshape(batch_size, H, W, -1)

        return x


class ChromosomeEncoder(nn.Module):
    """Encodes spatial patch grid to chromosome embedding via 2-layer MLP."""

    def __init__(
        self,
        patch_grid_size: int = 14,
        encoder_embed_dim: int = 256,
        output_dim: int = 1024,
        intermediate_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size
        self.encoder_embed_dim = encoder_embed_dim

        # 2-layer MLP: flatten all patches and project to chromosome embedding
        input_dim = self.num_patches * encoder_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, output_dim),
        )
        self.norm = nn.RMSNorm(output_dim)

        # Initialize weights
        self.fc.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial patches to chromosome embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, H, W, feat_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, output_dim)
        """
        batch_size, H, W, feat_dim = x.shape
        assert H == self.patch_grid_size and W == self.patch_grid_size, \
            f"Expected input shape ({batch_size}, {self.patch_grid_size}, {self.patch_grid_size}, *), got ({batch_size}, {H}, {W}, *)"
        assert feat_dim == self.encoder_embed_dim, \
            f"Expected feature dim {self.encoder_embed_dim}, got {feat_dim}"

        # Flatten spatial dims: (batch, H, W, feat_dim) -> (batch, H*W*feat_dim)
        x_flat = x.reshape(batch_size, -1)

        # Project to output dimension and normalize
        output = self.fc(x_flat)
        output = self.norm(output)

        return output


class CellEncoder(nn.Module):
    """Encodes chromosome embeddings to single cell embedding via 2-layer MLP."""

    def __init__(
        self,
        num_chr: int = 20,
        chr_dim: int = 1024,
        cell_dim: int = 2048,
        intermediate_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_chr * chr_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, cell_dim),
        )
        self.norm = nn.RMSNorm(cell_dim)

        # Initialize weights
        self.fc.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode chromosome embeddings to cell embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, num_chr, chr_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, cell_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        x = self.norm(x)
        return x


# ============================================================================
# DECODER COMPONENTS
# ============================================================================


class CellDecoder(nn.Module):
    """Decodes cell embedding back to chromosome embeddings via 2-layer MLP."""

    def __init__(
        self,
        num_chr: int = 20,
        chr_dim: int = 1024,
        cell_dim: int = 2048,
        intermediate_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.num_chr = num_chr
        self.chr_dim = chr_dim
        self.fc = nn.Sequential(
            nn.Linear(cell_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, num_chr * chr_dim),
        )
        self.norm = nn.RMSNorm(chr_dim)

        # Initialize weights
        self.fc.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode cell embedding to chromosome embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, cell_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, num_chr, chr_dim)
        """
        batch_size = x.shape[0]
        x = self.fc(x)
        x = x.reshape(batch_size, self.num_chr, self.chr_dim)
        # Apply RMSNorm across chr_dim for each chromosome
        x = self.norm(x)
        return x


class ChromosomeDecoder(nn.Module):
    """Decodes chromosome embedding to patch representations on spatial grid via 2-layer MLP."""

    def __init__(
        self,
        patch_grid_size: int = 14,
        input_dim: int = 1024,
        output_dim: int = 256,
        intermediate_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.patch_grid_size = patch_grid_size
        self.output_dim = output_dim
        self.num_patches = patch_grid_size * patch_grid_size

        # 2-layer MLP: project from chr embedding to patch tokens
        self.fc = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, self.num_patches * output_dim),
        )
        self.norm = nn.RMSNorm(output_dim)

        # Initialize weights
        self.fc.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode chromosome embedding to patch representations on spatial grid.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, input_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, patch_grid_size, patch_grid_size, output_dim)
        """
        batch_size = x.shape[0]

        # Project to patch tokens: (batch, input_dim) -> (batch, num_patches*output_dim)
        x = self.fc(x)
        x = x.reshape(batch_size, self.patch_grid_size, self.patch_grid_size, self.output_dim)
        # Normalize across output_dim for each patch
        x = self.norm(x)

        return x


class HiCMatrixDecoder(nn.Module):
    """Vision Transformer that decodes patch representations to 224x224 Hi-C matrices."""

    def __init__(
        self,
        patch_embed_dim: int = 256,
        output_size: int = 224,
        patch_size: int = 16,
        num_heads: int = 8,
        depth: int = 8,
    ) -> None:
        super().__init__()
        self.patch_embed_dim = patch_embed_dim
        self.output_size = output_size
        self.patch_size = patch_size
        self.num_patches = (output_size // patch_size) ** 2  # 196 for 224x224

        # Position embeddings for output patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, patch_embed_dim), requires_grad=False
        )
        self.pos_embed.data.copy_(
            init_position_embed(patch_embed_dim, (output_size // patch_size, output_size // patch_size), False)
        )

        # Transformer blocks with RMSNorm
        self.blocks = nn.ModuleList(
            [
                Block(
                    patch_embed_dim,
                    num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.RMSNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(patch_embed_dim)

        # Output projection to reconstruct patches (224x224 = 14*16 x 14*16)
        self.output_proj = nn.Linear(patch_embed_dim, patch_size**2)

    def freeze(self) -> None:
        """Freeze all transformer blocks and normalization layer."""
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all transformer blocks and normalization layer."""
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if vision transformer blocks are frozen."""
        for param in self.blocks.parameters():
            if param.requires_grad:
                return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process patch embeddings and reconstruct matrices.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, patch_grid_size, patch_grid_size, patch_embed_dim)

        Returns
        -------
        torch.Tensor
            Output of shape (batch, output_size, output_size)
        """
        batch_size, H, W, embed_dim = x.shape
        patch_grid_size = self.output_size // self.patch_size
        assert H == patch_grid_size and W == patch_grid_size, \
            f"Input spatial size ({H}, {W}) must match (output_size // patch_size) = ({patch_grid_size}, {patch_grid_size})"
        assert embed_dim == self.patch_embed_dim, \
            f"Input embedding dim {embed_dim} must match patch_embed_dim {self.patch_embed_dim}"

        # Flatten spatial to patch tokens: (batch, H, W, embed_dim) -> (batch, H*W, embed_dim)
        patches = x.reshape(batch_size, self.num_patches, self.patch_embed_dim)

        # Add positional embeddings
        patches = patches + self.pos_embed

        # Apply transformer blocks
        for blk in self.blocks:
            patches = blk(patches)
        patches = self.norm(patches)

        # Project to output dimension: (batch, num_patches, embed_dim) -> (batch, num_patches, patch_size^2)
        output = self.output_proj(patches)  # (batch, num_patches, patch_size^2)

        # Reshape to spatial: (batch, H*W, p*p) -> (batch, H, W, p, p)
        output = output.reshape(batch_size, patch_grid_size, patch_grid_size, self.patch_size, self.patch_size)

        # Rearrange and flatten to final size: (batch, H, W, p, p) -> (batch, H*p, W*p)
        output = output.permute(0, 1, 3, 2, 4)  # (batch, H, p, W, p)
        output = output.reshape(batch_size, patch_grid_size * self.patch_size, patch_grid_size * self.patch_size)

        return output


# ============================================================================
# LIGHTNING MODULE
# ============================================================================


class scHiCAutoencoder(pl.LightningModule):
    """
    PyTorch Lightning autoencoder for scHiC embedding reconstruction.

    Maps HiCFoundation embeddings to Hi-C contact matrices using a pipeline with:
    - Feature projection for dimensionality reduction
    - Per-chromosome encoding/decoding
    - Vision Transformer for matrix reconstruction
    - Binary cross-entropy loss with pos_weight for class imbalance

    Parameters
    ----------
    num_chr : int, default=20
        Number of chromosomes
    patch_grid_size : int, default=14
        Input patch grid size (e.g., 14x14 for HiCFoundation)
    input_embed_dim : int, default=1024
        Input embedding dimension from HiCFoundation (fixed)
    feature_proj_dim : int, default=256
        Dimensionality after feature projection
    chr_embed_dim : int, default=1024
        Chromosome embedding dimensionality
    cell_embed_dim : int, default=2048
        Cell-level embedding dimensionality (latent space)
    patch_embed_dim : int, default=256
        Patch embedding dimensionality for HiC Matrix Decoder
    output_size : int, default=224
        Output matrix size (224x224 for Hi-C, fixed)
    patch_size : int, default=16
        Patch size for HiC Matrix Decoder (fixed)
    vit_depth : int, default=8
        Number of transformer blocks
    vit_heads : int, default=8
        Number of attention heads
    chr_encoder_intermediate_dim : int, default=2048
        Intermediate dimension for chromosome encoder (2 × chr_embed_dim)
    cell_encoder_intermediate_dim : int, default=4096
        Intermediate dimension for cell encoder (2 × cell_embed_dim)
    cell_decoder_intermediate_dim : int, default=4096
        Intermediate dimension for cell decoder (2 × cell_embed_dim)
    chr_decoder_intermediate_dim : int, default=2048
        Intermediate dimension for chromosome decoder (2 × chr_embed_dim)
    learning_rate : float, default=1e-4
        Adam learning rate
    pos_weight : float, default=3.0
        BCEWithLogitsLoss pos_weight (penalty on false negatives)
    """

    def __init__(
        self,
        num_chr: int = 20,
        patch_grid_size: int = 14,
        input_embed_dim: int = 1024,
        feature_proj_dim: int = 256,
        chr_embed_dim: int = 1024,
        cell_embed_dim: int = 2048,
        patch_embed_dim: int = 256,
        output_size: int = 224,
        patch_size: int = 16,
        vit_depth: int = 8,
        vit_heads: int = 8,
        chr_encoder_intermediate_dim: int = 2048,
        cell_encoder_intermediate_dim: int = 4096,
        cell_decoder_intermediate_dim: int = 4096,
        chr_decoder_intermediate_dim: int = 2048,
        learning_rate: float = 1e-4,
        pos_weight: float = 3.0,
    ):
        """Initialize the scHiC autoencoder."""
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_chr = num_chr
        self.patch_grid_size = patch_grid_size

        # ENCODER: Input -> Feature Proj -> Chr Encoder -> Cell Encoder
        self.feature_proj = FeatureProjector(in_dim=input_embed_dim, out_dim=feature_proj_dim)
        self.chr_encoder = ChromosomeEncoder(
            patch_grid_size=patch_grid_size,
            encoder_embed_dim=feature_proj_dim,
            output_dim=chr_embed_dim,
            intermediate_dim=chr_encoder_intermediate_dim,
        )
        self.cell_encoder = CellEncoder(
            num_chr=num_chr,
            chr_dim=chr_embed_dim,
            cell_dim=cell_embed_dim,
            intermediate_dim=cell_encoder_intermediate_dim,
        )

        # DECODER: Cell Decoder -> Chr Decoder -> HiC Matrix Decoder
        self.cell_decoder = CellDecoder(
            num_chr=num_chr,
            chr_dim=chr_embed_dim,
            cell_dim=cell_embed_dim,
            intermediate_dim=cell_decoder_intermediate_dim,
        )
        self.chr_decoder = ChromosomeDecoder(
            patch_grid_size=patch_grid_size,
            input_dim=chr_embed_dim,
            output_dim=patch_embed_dim,
            intermediate_dim=chr_decoder_intermediate_dim,
        )
        self.hic_matrix_decoder = HiCMatrixDecoder(
            patch_embed_dim=patch_embed_dim,
            output_size=output_size,
            patch_size=patch_size,
            num_heads=vit_heads,
            depth=vit_depth,
        )

        # Binary cross-entropy loss with pos_weight for contact/no-contact classification
        # pos_weight: penalty multiplier for mispredicting contacts (1s)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode HiC embeddings to cell latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch, num_chr, 14, 14, 1024)

        Returns
        -------
        torch.Tensor
            Cell embedding of shape (batch, cell_embed_dim)
        """
        batch_size, num_chr, H, W, feat_dim = x.shape
        assert num_chr == self.num_chr, f"Expected {self.num_chr} chromosomes, got {num_chr}"
        assert H == self.patch_grid_size and W == self.patch_grid_size, \
            f"Expected patch grid size ({self.patch_grid_size}, {self.patch_grid_size}), got ({H}, {W})"

        # Step 1: Feature Projection
        x_flat = x.reshape(batch_size * num_chr, H, W, feat_dim)
        x_proj_flat = self.feature_proj(x_flat)
        x_proj = x_proj_flat.reshape(batch_size, num_chr, H, W, -1)

        # Step 2: Chromosome Encoding (per-chromosome)
        x_proj_flat = x_proj.reshape(batch_size * num_chr, H, W, -1)
        chr_embs_flat = self.chr_encoder(x_proj_flat)
        chr_embs = chr_embs_flat.reshape(batch_size, num_chr, -1)

        # Step 3: Cell Encoding
        cell_emb = self.cell_encoder(chr_embs)

        return cell_emb

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode cell latent representation to reconstructed matrices.

        Parameters
        ----------
        z : torch.Tensor
            Cell embedding of shape (batch, cell_embed_dim)

        Returns
        -------
        torch.Tensor
            Reconstructed matrices of shape (batch, num_chr, 224, 224)
        """
        batch_size = z.shape[0]
        H = self.patch_grid_size

        # Step 1: Cell Decoding
        chr_embs_recon = self.cell_decoder(z)

        # Step 2: Chromosome Decoding (per-chromosome)
        chr_embs_recon_flat = chr_embs_recon.reshape(batch_size * self.num_chr, -1)
        patch_emb_flat = self.chr_decoder(chr_embs_recon_flat)
        patch_embeddings = patch_emb_flat.reshape(batch_size, self.num_chr, H, H, -1)

        # Step 3: HiC Matrix Decoder (per-chromosome)
        patch_emb_flat = patch_embeddings.reshape(batch_size * self.num_chr, H, H, -1)
        reconstructed_flat = self.hic_matrix_decoder(patch_emb_flat)
        reconstructed = reconstructed_flat.reshape(batch_size, self.num_chr, *reconstructed_flat.shape[1:])

        return reconstructed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch, num_chr, 14, 14, 1024)

        Returns
        -------
        torch.Tensor
            Reconstructed contact matrices of shape (batch, num_chr, 224, 224)
        """
        cell_emb = self.encode(x)
        reconstructed = self.decode(cell_emb)
        return reconstructed

    def freeze_hic_matrix_decoder(self) -> None:
        """Freeze the HiCMatrixDecoder."""
        self.hic_matrix_decoder.freeze()

    def unfreeze_hic_matrix_decoder(self) -> None:
        """Unfreeze the HiCMatrixDecoder."""
        self.hic_matrix_decoder.unfreeze()

    def is_hic_matrix_decoder_frozen(self) -> bool:
        """Check if HiC matrix decoder is frozen."""
        return self.hic_matrix_decoder.is_frozen()

    def configure_optimizers(self):
        """Configure optimizer."""
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: dict, _):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Batch data with keys:
            - 'embeddings': Hi-C embeddings (batch, num_chr, 14, 14, 1024)
            - 'matrices': Hi-C contact matrices (batch, num_chr, 224, 224) binarized ground truth

        Returns
        -------
        torch.Tensor
            Loss value
        """
        embeddings, matrices = batch['embeddings'], batch['matrices']
        reconstructed = self.forward(embeddings)
        loss = self.loss_fn(reconstructed, matrices)

        # Log both step and epoch level (consistent with scRNA-seq)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, _):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Batch data with keys 'embeddings' and 'matrices'

        Returns
        -------
        torch.Tensor
            Validation loss
        """
        embeddings, matrices = batch['embeddings'], batch['matrices']
        reconstructed = self.forward(embeddings)
        loss = self.loss_fn(reconstructed, matrices)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
