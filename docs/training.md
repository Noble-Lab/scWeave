# Training

Step-by-step instructions for training scWeave from scratch. The steps below
reproduce the Figure 2 (mouse) model. Training has three parts: obtaining the
data, building the training splits, and training the model.

## Preparing datasets from source

All single-cell RNA-seq and single-cell Hi-C data used in this manuscript is
publicly available.
The HiRES dataset is available at GEO GSE223917; the GAGE-seq dataset is available
at GEO GSE238001; the CHARM dataset is available at GEO GSE303006.

The Figure 2 model is trained on four mouse co-assay datasets:

| Assay | Species | Tissue | Cells | Cell types | Contacts | UMIs | GEO |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| HiRES | mouse | developing embryo | 7469 | 21 | 306,194 | 9,865 | GSE223917 |
| HiRES | mouse | brain cortex | 399 | 7 | 376,633 | 28,709 | GSE223917 |
| GAGE-seq | mouse | brain cortex | 3105 | 29 | 393,445 | 20,160 | GSE238001 |
| CHARM | mouse | brain cortex | 4265 | 20 | 485,033 | 8,959 | GSE303006 |

Cells, and the median numbers of Hi-C contacts and RNA-seq UMIs per cell.

Coming soon: instructions for turning the raw data into the per-dataset AnnData,
HiCFoundation embeddings, and contact matrices that `prepare_dataset` consumes.

## Building the training splits

`prepare_dataset` takes the paired datasets (RNA + Hi-C) and writes the on-disk
split layout that training reads. For each dataset it aligns the RNA to the
model's gene set and to the Hi-C cell order, then splits into train / val / test.
Train and validation are merged across datasets; each dataset's test split is kept
separate.

```python
from scweave.training import prepare_dataset

# the four mouse datasets used for the Figure 2 model
datasets = ["HiRES_embryo", "HiRES_brain", "GAGEseq_brain", "CHARM_brain"]

prepare_dataset(
    adatas=[f"{d}/rna.h5ad" for d in datasets],
    embeddings=[f"{d}/embeddings.npy" for d in datasets],
    matrices=[f"{d}/matrices.npy" for d in datasets],
    ids=[f"{d}/ids.npy" for d in datasets],
    names=datasets,
    gene_names="mouse",              # or "human", an array, or a path
    out_dir="data/prepared",
)
```

Each dataset provides four things, aligned by cell:
- **adata** — RNA expression; `obs.index` holds cell IDs, `var_names` holds gene IDs.
- **embeddings** — HiCFoundation embeddings, shape `(n_cells, n_chr, 14, 14, 1024)`.
- **matrices** — contact matrices, shape `(n_cells, n_chr, 224, 224)`.
- **ids** — the Hi-C cell IDs, matching the embedding/matrix order.

This writes `data/prepared/training_set/` (merged train + val) and
`data/prepared/test_sets/<name>/` (one per dataset).

> **Memory and disk.** `prepare_dataset` loads each dataset's embeddings fully
> into RAM and gathers each split in RAM before writing it out. Hi-C embeddings are
> large (on the order of 100 GB for a few thousand cells), so this step requires a
> machine with substantial RAM and free disk space.

## Training the model

Once the dataset is prepared, `train_translator` trains scWeave from scratch and
saves the best checkpoint (by validation loss). You build DataLoaders over the
prepared splits and pass them in.

```python
import torch
from torch.utils.data import DataLoader
from scweave.training import MultimodalDataset, train_translator

def collate(batch):
    keys = ("rna", "embeddings", "matrices")
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

train = MultimodalDataset("data/prepared/training_set", split="train")
val   = MultimodalDataset("data/prepared/training_set", split="val")
train_loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(val, batch_size=8, collate_fn=collate)

result = train_translator(
    train_loader,
    val_loader,
    rna_dim=train.n_genes,
    hic_num_chr=train.embeddings_shape[0],
    max_epochs=25,
    checkpoint_dir="experiments/checkpoints",
    log_dir="experiments/logs",
)
print(result["best_checkpoint_path"])
```

`train_translator` exposes the model hyperparameters and loss weights as arguments
(with defaults matching the paper), plus training options such as `max_epochs`,
`devices`, and `strategy` (e.g. `"ddp"` for multi-GPU). The returned checkpoint is
loaded for inference with `scWeave.load(...)`.
