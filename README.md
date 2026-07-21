# scWeave

**scWeave: A deep learning model that bidirectionally translates between gene expression and chromatin structure at single-cell resolution**

Chromatin structure and gene expression are intimately linked, yet characterizing how the two covary has proven challenging, primarily because the two modalities are rarely measured in the same cells.
Recently, single-cell co-assay protocols have enabled simultaneous profiling of both modalities within the same cells, but these experiments remain costly and technically challenging.
To better characterize the relationship between 3D chromatin architecture and gene expression and to enable cross-modality inference from single-modality measurements, we developed a model called scWeave that bidirectionally translates between gene expression (scRNA-seq) and 3D chromatin architecture (scHi-C) at single-cell resolution.
The scWeave model employs dual autoencoders to extract separate cell-level latent representations and learns to translate between these representations using dedicated translation modules.
We evaluate scWeave on six publicly available co-assay datasets spanning mouse embryonic development, mouse cortex, mouse olfactory epithelium, and human bone marrow.
On held-out mouse cells, scWeave outperforms a nearest-neighbor baseline and existing methods adapted to single-cell resolution, achieving a 57% improvement in median Spearman correlation when predicting gene expression from chromatin structure and an 18.8% improvement in median HiCRep similarity in the reverse direction relative to the next-best baseline.
We further show that scWeave learns cross-modally aligned latent representations at single-cell resolution, enabling cells profiled in one modality to be matched to their counterparts in the other.
Finally, scWeave generalizes to entirely held-out developmental timepoints in mouse olfactory epithelium and performs well on held-out human bone marrow cells despite limited human training data.
By predicting the unmeasured chromatin architecture or transcriptional state from a single measured modality, scWeave offers a route to extend the benefits of costly co-assays to the many cell types, developmental stages, and species that are currently profiled with only one modality.

<p align="center">
  <img src="assets/arch.png" alt="scWeave model architecture" width="900">
</p>

---

## Requirements
- Python >= 3.10
- A CUDA-capable GPU is recommended for both training and inference. 

All dependencies are installed automatically (see `pyproject.toml`).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Noble-Lab/scWeave.git
cd scWeave
```

### 2. Create an isolated environment with [uv](https://docs.astral.sh/uv/)

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install scWeave

```bash
uv pip install -e .
```

This pulls in all dependencies and exposes the `scweave` package.

### 4. Verify the installation

```bash
python -c "import scweave; print(scweave.__version__)"
```

---

## Model

scWeave is built from two autoencoders and two translator modules, trained
together end to end (see the architecture figure above).

**Dual autoencoders.** Each modality has its own autoencoder that compresses a
cell into a shared-size latent vector and reconstructs it. The RNA autoencoder
encodes gene expression into the latent space and decodes it back to expression.
The Hi-C autoencoder encodes a cell's chromatin structure through a hierarchy —
per-chromosome patch embeddings, then a chromosome-level encoder, then a
cell-level encoder — into the same latent space, and decodes it back to one
*cis* contact map per chromosome. Both autoencoders map into a latent space of
the same dimensionality, so a cell has a comparable representation regardless of
which modality it was measured in.

**Translator modules.** Two small residual MLPs bridge the two latent spaces:
one translates an RNA latent into the corresponding Hi-C latent, and the other
translates a Hi-C latent into the corresponding RNA latent. Translation therefore
happens entirely in the latent space, which keeps the modality-specific encoders
and decoders reusable in both directions.

**Bidirectional prediction and matching.** Predicting one modality from the other
chains an encoder, a translator, and a decoder: for example, RNA to Hi-C encodes
the gene expression, translates the RNA latent into a Hi-C latent, and decodes
that into contact maps. Because both modalities share the latent space, and
training additionally encourages a cell's two representations to align there,
the same latent space can be used to match cells across modalities.

**Training.** The whole model is trained from scratch, jointly optimizing
within-modality reconstruction, cross-modality translation in both directions,
and an alignment term that pulls each cell's RNA-derived and Hi-C-derived latents
together.

---

## Preparing inputs

scWeave ingests two single-cell modalities, one row per cell: gene expression
and HiCFoundation embeddings of chromatin structure.

### Gene expression (RNA)

`prepare_rna` turns an AnnData of single-cell counts into the array the model
expects. It aligns your cells to the model's gene set (in the model's gene
order), normalizes by library size, and applies a `log1p` transform, exactly as
scWeave was trained.

```python
from scweave import prepare_rna, load_gene_names

gene_names = load_gene_names("mouse")            # or "human"
rna = prepare_rna("expression.h5ad", gene_names) # (n_cells, n_genes)
```

`load_gene_names(species)` returns the Ensembl gene IDs, in the exact order, that
the model was trained on. These lists ship with
the package.

Notes:
- `adata` may be an in-memory `AnnData` or a path to an `.h5ad` file.
- `adata.var_names` must be gene identifiers matching `gene_names` (Ensembl gene
  IDs); genes the model expects but missing from your data are filled with zeros.
- `gene_names` also accepts a raw array, or a path to a `.txt` (one gene per
  line). 

### Chromatin structure (Hi-C)

Coming soon!

By the end of this section you have two arrays ready for the model:
`rna` of shape `(n_cells, n_genes)` and `hic` of shape
`(n_cells, n_chr, 14, 14, 1024)`.

---

## Inference

The `scWeave` class wraps a trained model and exposes three operations:
translate gene expression into chromatin structure, translate chromatin structure
into gene expression, and match cells across the two modalities.

```python
import numpy as np
from scweave import scWeave

model = scWeave.load("scweave.ckpt")   # loads on GPU if available, else CPU

rna = np.load("rna.npy")            # (n_cells, n_genes)
hic = np.load("hic_embeddings.npy") # (n_cells, n_chr, 14, 14, 1024)

hic_pred = model.predict_hic_from_rna(rna)   # (n_cells, n_chr, 224, 224)
rna_pred = model.predict_rna_from_hic(hic)   # (n_cells, n_genes)

similarity, matches = model.match(rna, hic, direction="rna_to_hic")
```

### Outputs

**`predict_hic_from_rna(rna)`** → shape `(n_cells, n_chr, 224, 224)`.
The predicted chromatin structure for each cell: one `224 x 224` *cis* contact
map per chromosome. Values are contact probabilities in `[0, 1]`, where entry
`(a, b)` is the model's confidence that genomic bins *a* and *b* of that
chromosome are in contact.

**`predict_rna_from_hic(hic)`** → shape `(n_cells, n_genes)`.
The predicted gene expression for each cell, in the same normalized,
log-transformed space as the input RNA.

**`match(rna, hic, direction=...)`** → `(similarity, matches)`.
Cells are matched across modalities in scWeave's shared latent space.
The query modality is translated into that space and compared, by cosine
similarity, against the encoded representation of the other (reference) modality.
- `similarity` — shape `(n_query, n_reference)`, the cosine similarity between
  every query cell and every reference cell.
- `matches` — shape `(n_query,)`, the index of the most similar reference cell
  for each query cell.

`direction` selects which modality is the query:
- `"rna_to_hic"` (default) — RNA cells are the query, Hi-C cells the reference;
  `matches[i]` is the Hi-C cell matched to RNA cell `i`.
- `"hic_to_rna"` — Hi-C cells are the query, RNA cells the reference;
  `matches[i]` is the RNA cell matched to Hi-C cell `i`.

---

## Training

The steps below reproduce the Figure 2 (mouse) model. Training scWeave from
scratch has three parts: obtaining the data, building the training splits, and
training the model.

### Preparing datasets from source

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

### Building the training splits

`prepare_dataset` takes the paired datasets (RNA + Hi-C) and writes the on-disk
split layout that training reads. For each dataset it aligns the RNA to the
model's gene set and to the Hi-C cell order, then splits into train / val / test.
Train and validation are merged across datasets; each dataset's test split is
kept separate.

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
> into RAM and gathers each split in RAM before writing it out. Hi-C embeddings
> are large (on the order of 100 GB for a few thousand cells), so this step
> requires a machine with substantial RAM and free disk space.

### Training the model

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

---

## Examples

End-to-end example notebooks will live in [`examples/`](examples):
- [`examples/inference.ipynb`](examples/inference.ipynb) — load a trained model,
  prepare inputs, and run prediction and cross-modal matching.
- [`examples/training.ipynb`](examples/training.ipynb) — prepare a dataset and
  train scWeave from scratch.

Coming soon!

---

## License

scWeave is released under the [Apache License 2.0](LICENSE).
