# Inference

Step-by-step instructions for running a trained scWeave model. A full worked
example on the HiRES brain subset is being added.

scWeave ingests two single-cell modalities, one row per cell: gene expression and
HiCFoundation embeddings of chromatin structure. This guide covers preparing both
inputs, loading a trained model, and running prediction and matching.

## Preparing inputs

### Gene expression (RNA)

`prepare_rna` turns an AnnData of single-cell counts into the array the model
expects. It aligns your cells to the model's gene set (in the model's gene order),
normalizes by library size, and applies a `log1p` transform, exactly as scWeave
was trained.

```python
from scweave import prepare_rna, load_gene_names

gene_names = load_gene_names("mouse")            # or "human"
rna = prepare_rna("expression.h5ad", gene_names) # (n_cells, n_genes)
```

`load_gene_names(species)` returns the Ensembl gene IDs, in the exact order, that
the model was trained on. These lists ship with the package.

Notes:
- `adata` may be an in-memory `AnnData` or a path to an `.h5ad` file.
- `adata.var_names` must be gene identifiers matching `gene_names` (Ensembl gene
  IDs); genes the model expects but missing from your data are filled with zeros.
- `gene_names` also accepts a raw array, or a path to a `.txt` (one gene per line).

### Chromatin structure (Hi-C)

Coming soon: turning raw single-cell contact files into the HiCFoundation
embeddings `(n_cells, n_chr, 14, 14, 1024)` that scWeave consumes.

By the end of this section you have two arrays ready for the model: `rna` of shape
`(n_cells, n_genes)` and `hic` of shape `(n_cells, n_chr, 14, 14, 1024)`.

## Running the model

The `scWeave` class wraps a trained model and exposes three operations: translate
gene expression into chromatin structure, translate chromatin structure into gene
expression, and match cells across the two modalities.

```python
import numpy as np
from scweave import scWeave

model = scWeave.load("scweave_figure2_mouse.ckpt")   # loads on GPU if available, else CPU

rna = np.load("rna.npy")            # (n_cells, n_genes)
hic = np.load("hic_embeddings.npy") # (n_cells, n_chr, 14, 14, 1024)

hic_pred = model.predict_hic_from_rna(rna)   # (n_cells, n_chr, 224, 224)
rna_pred = model.predict_rna_from_hic(hic)   # (n_cells, n_genes)

similarity, matches = model.match(rna, hic, direction="rna_to_hic")
```

### Outputs

**`predict_hic_from_rna(rna)`** → shape `(n_cells, n_chr, 224, 224)`.
The predicted chromatin structure for each cell: one `224 x 224` *cis* contact map
per chromosome. Values are contact probabilities in `[0, 1]`, where entry `(a, b)`
is the model's confidence that genomic bins *a* and *b* of that chromosome are in
contact.

**`predict_rna_from_hic(hic)`** → shape `(n_cells, n_genes)`.
The predicted gene expression for each cell, in the same normalized,
log-transformed space as the input RNA.

**`match(rna, hic, direction=...)`** → `(similarity, matches)`.
Cells are matched across modalities in scWeave's shared latent space. The query
modality is translated into that space and compared, by cosine similarity, against
the encoded representation of the other (reference) modality.
- `similarity` — shape `(n_query, n_reference)`, the cosine similarity between
  every query cell and every reference cell.
- `matches` — shape `(n_query,)`, the index of the most similar reference cell for
  each query cell.

`direction` selects which modality is the query:
- `"rna_to_hic"` (default) — RNA cells are the query, Hi-C cells the reference;
  `matches[i]` is the Hi-C cell matched to RNA cell `i`.
- `"hic_to_rna"` — Hi-C cells are the query, RNA cells the reference;
  `matches[i]` is the RNA cell matched to Hi-C cell `i`.

## Evaluating predictions

scWeave ships minimal metrics for comparing predictions to ground truth:

```python
from scweave.evaluation import spearman_per_cell, hicrep_per_chromosome, auroc_per_cell

scc = spearman_per_cell(rna_true, rna_pred)          # per-cell RNA correlation
hicrep = hicrep_per_chromosome(hic_true, hic_pred)   # per-chromosome pseudobulk HiCRep
auroc = auroc_per_cell(hic_true, hic_pred)           # per-cell contact AUROC
```
