import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_cell_gene(path):
    # Use Scanpy loader to keep 10x feature annotations intact
    return sc.read_10x_h5(path)


def annotate_qc(adata):
    # Flag gene categories used for quality control
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    return adata


def compute_qc_metrics(adata):
    # Compute per-cell QC metrics based on annotated gene categories
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True
    )
    return adata
  

def filter_cells_by_qc(
    adata,
    max_mt_pct=None,
    max_ribo_pct=None,
    max_hb_pct=None,
):
    # Filter cells based on QC percentage thresholds inferred from violin plots
    mask = np.ones(adata.n_obs, dtype=bool)

    if max_mt_pct is not None:
        mask &= adata.obs["pct_counts_mt"] <= max_mt_pct

    if max_ribo_pct is not None:
        mask &= adata.obs["pct_counts_ribo"] <= max_ribo_pct

    if max_hb_pct is not None:
        mask &= adata.obs["pct_counts_hb"] <= max_hb_pct

    return adata[mask].copy()


def filter_cells_genes(adata, min_genes, min_cells):
    # Filter cells and genes using user-defined thresholds
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def detect_doublets(adata):
    # Identify potential doublets using Scrublet (batch-aware)
    sc.pp.scrublet(adata, batch_key="sample")
    return adata


def normalize_log1p(adata):
    # keep raw counts before any normalization
    adata.layers["counts"] = adata.X.copy()

    # normalize each cell to the same total counts
    sc.pp.normalize_total(adata)

    # log(1 + x) transform to stabilize variance
    sc.pp.log1p(adata)

    return adata


def select_hvgs(adata, n_top_genes=2000):
    # identify highly variable genes (does NOT filter yet)
    # HVGs are computed per sample to reduce batch effects
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key="sample"
    )

    return adata


def run_pca(adata, n_comps=50):
    # run PCA on the current gene set (usually HVGs)
    # reduces dimensionality before neighbors / UMAP
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")

    return adata


def run_neighbors_umap(adata, n_neighbors, n_pcs):
    # build kNN graph using PCA space
    # compute UMAP embedding for visualization
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)

    return adata


def run_leiden(adata):
    # cluster cells using Leiden on the neighbor graph
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    return adata


def save_leiden_labels(adata, output_path):
    # save cell-wise Leiden cluster labels to csv
    df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "leiden": adata.obs["leiden"].astype(str).values
    })

    df.to_csv(output_path, index=False)

    return df