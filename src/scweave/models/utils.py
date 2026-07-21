"""
Utility functions for model initialization and positional embeddings.
"""

import numpy as np
import torch
import torch.nn as nn


def _init_linear_weights(module: nn.Module) -> None:
    """Initialize linear layer weights with Kaiming initialization for SiLU activation."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


# ============================================================================
# POSITIONAL EMBEDDING UTILITIES (from hicfoundation_decoder)
# ============================================================================


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Build 1D sin-cos positional embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    pos : np.ndarray
        1D array of positions

    Returns
    -------
    np.ndarray
        Positional embeddings of shape (len(pos), embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (embed_dim // 2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim//2)

    emb_sin = np.sin(out)  # (M, embed_dim//2)
    emb_cos = np.cos(out)  # (M, embed_dim//2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Build 2D sin-cos positional embeddings from a coordinate grid.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must be even)
    grid : np.ndarray
        Shape (2, 1, W, H) containing [grid_w, grid_h]

    Returns
    -------
    np.ndarray
        Positional embeddings of shape (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, embed_dim//2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, embed_dim//2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, embed_dim)
    return emb


def get_2d_sincos_pos_embed_rectangle(
    embed_dim: int, grid_size: tuple[int, int], cls_token: bool = False
) -> np.ndarray:
    """
    Build 2D sin-cos positional embeddings on a rectangular grid.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must be even)
    grid_size : tuple[int, int]
        (H, W) grid size
    cls_token : bool, default=False
        Prepend a zero vector for CLS if True

    Returns
    -------
    np.ndarray
        Positional embeddings of shape (H*W, embed_dim) or (1+H*W, embed_dim)
    """
    grid_size_h, grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_w, grid_size_h])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def init_position_embed(
    embed_dim: int, pos_embed_size: tuple[int, int], cls_token: bool
) -> torch.Tensor:
    """
    Build fixed 2D sin-cos positional embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    pos_embed_size : tuple[int, int]
        (h, w) grid size
    cls_token : bool
        Whether to include a class token position

    Returns
    -------
    torch.Tensor
        Positional embeddings of shape (1, L + (1 if cls_token else 0), embed_dim)
    """
    pe = get_2d_sincos_pos_embed_rectangle(embed_dim, pos_embed_size, cls_token)
    return torch.from_numpy(pe).float().unsqueeze(0)
