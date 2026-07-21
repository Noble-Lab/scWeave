"""
Training entry point for scWeave.

``train_translator`` trains a :class:`~scweave.models.TranslatorModel` from scratch
on a prepared dataset and returns the path to the best checkpoint (which
:meth:`scweave.scWeave.load` can then load for inference).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from scweave.models import TranslatorModel


def train_translator(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    rna_dim: int,
    hic_num_chr: int = 20,
    latent_dim: int = 2048,
    # RNA autoencoder
    rna_hidden_dim: int = 4096,
    rna_mask_ratio: float = 0.3,
    # scHi-C autoencoder
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
    # optimization
    learning_rate: float = 1e-4,
    # loss weights
    rna_recon_weight: float = 1.0,
    hic_recon_weight: float = 1.0,
    rna_to_hic_weight: float = 1.0,
    hic_to_rna_weight: float = 1.0,
    latent_align_weight: float = 0.1,
    latent_align_temperature: float = 0.1,
    # training config
    max_epochs: int = 25,
    early_stopping_patience: Optional[int] = None,
    gradient_clip_val: float = 1.0,
    checkpoint_dir: Union[str, Path] = "experiments/checkpoints",
    log_dir: Union[str, Path] = "experiments/logs",
    experiment_name: str = "scweave",
    devices: int = 1,
    accelerator: str = "auto",
    strategy: Optional[str] = None,
    seed: int = 125,
) -> Dict[str, Any]:
    """
    Train a scWeave translator model from scratch.

    Parameters
    ----------
    train_dataloader, val_dataloader : DataLoader
        Loaders over a prepared dataset (see :func:`scweave.training.prepare_dataset`
        and :class:`~scweave.training.MultimodalDataset`). Each batch is a dict with
        keys ``'rna'``, ``'embeddings'``, and ``'matrices'``.
    rna_dim : int
        Number of genes (RNA feature dimension).
    hic_num_chr : int, default=20
        Number of chromosomes.
    latent_dim, rna_hidden_dim, rna_mask_ratio, hic_* :
        Model architecture hyperparameters (see :class:`~scweave.models.TranslatorModel`).
    learning_rate : float, default=1e-4
        Adam learning rate.
    rna_recon_weight, hic_recon_weight, rna_to_hic_weight, hic_to_rna_weight,
    latent_align_weight, latent_align_temperature :
        Loss weights and alignment temperature.
    max_epochs : int, default=25
        Maximum training epochs.
    early_stopping_patience : int, optional
        If set, stop after this many epochs without val_loss improvement.
    gradient_clip_val : float, default=1.0
        Gradient-norm clipping value.
    checkpoint_dir, log_dir : str or Path
        Where checkpoints and CSV logs are written.
    experiment_name : str, default="scweave"
        Name for the checkpoint/log subdirectory.
    devices : int, default=1
        Number of devices.
    accelerator : str, default="auto"
        Lightning accelerator ("auto", "gpu", "cpu", ...).
    strategy : str, optional
        Lightning strategy (e.g. "ddp" for multi-GPU). If None, Lightning chooses.
    seed : int, default=125
        Random seed.

    Returns
    -------
    dict
        ``{"best_checkpoint_path", "best_val_loss", "model", "log_dir"}``.
    """
    pl.seed_everything(seed, workers=True)

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = TranslatorModel(
        rna_dim=rna_dim,
        latent_dim=latent_dim,
        rna_hidden_dim=rna_hidden_dim,
        rna_mask_ratio=rna_mask_ratio,
        hic_num_chr=hic_num_chr,
        hic_patch_grid_size=hic_patch_grid_size,
        hic_input_embed_dim=hic_input_embed_dim,
        hic_feature_proj_dim=hic_feature_proj_dim,
        hic_chr_embed_dim=hic_chr_embed_dim,
        hic_patch_embed_dim=hic_patch_embed_dim,
        hic_output_size=hic_output_size,
        hic_patch_size=hic_patch_size,
        hic_vit_depth=hic_vit_depth,
        hic_vit_heads=hic_vit_heads,
        hic_chr_encoder_intermediate_dim=hic_chr_encoder_intermediate_dim,
        hic_cell_encoder_intermediate_dim=hic_cell_encoder_intermediate_dim,
        hic_cell_decoder_intermediate_dim=hic_cell_decoder_intermediate_dim,
        hic_chr_decoder_intermediate_dim=hic_chr_decoder_intermediate_dim,
        hic_pos_weight=hic_pos_weight,
        learning_rate=learning_rate,
        rna_recon_weight=rna_recon_weight,
        hic_recon_weight=hic_recon_weight,
        rna_to_hic_weight=rna_to_hic_weight,
        hic_to_rna_weight=hic_to_rna_weight,
        latent_align_weight=latent_align_weight,
        latent_align_temperature=latent_align_temperature,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / experiment_name,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    callbacks = [checkpoint_callback]
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=early_stopping_patience, mode="min", verbose=True)
        )

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=log_dir, name=experiment_name),
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_dataloader, val_dataloader)

    best_path = checkpoint_callback.best_model_path
    best_model = TranslatorModel.load_from_checkpoint(best_path)

    return {
        "best_checkpoint_path": best_path,
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "model": best_model,
        "log_dir": log_dir / experiment_name,
    }
