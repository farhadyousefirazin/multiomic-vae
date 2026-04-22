import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix    
from pathlib import Path
import matplotlib.pyplot as plt


def load_cell_peak_df(h5_path):
    # Load 10x multiome file and extract cell × peak sparse DataFrame

    h5_path = Path(h5_path)

    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata_peaks = adata[:, adata.var["feature_types"] == "Peaks"]

    return pd.DataFrame.sparse.from_spmatrix(
        adata_peaks.X,
        index=adata_peaks.obs_names,
        columns=adata_peaks.var_names
    )


def filter_valid_chromosomes_df(cell_peak_df):
    # Keep only peaks on chr1–22, chrX, chrY.
        
    valid_chr = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    
    mask = cell_peak_df.columns.str.startswith(tuple(valid_chr))
    
    return cell_peak_df.loc[:, mask]



def ensure_sparse(X):
    # Convert DataFrame to sparse matrix if needed
    if isinstance(X, pd.DataFrame):
        return csr_matrix(X.values)
    return X


def compute_peak_activity(X_atac):
    # Compute per-peak activity across cells
    X = ensure_sparse(X_atac)

    n_cells = X.shape[0]
    peak_counts = X.getnnz(axis=0)
    peak_frac = peak_counts / n_cells

    return peak_counts, peak_frac


def plot_peak_activity(peak_counts, n_cells, figsize=(5, 3), dpi=140):
    # Visualize distribution of peak activity across cells

    plt.figure(figsize=figsize, dpi=dpi)

    plt.hist(
        peak_counts,
        bins=60,
        color="#5A8FD8",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9
    )

    plt.xlim(0, n_cells)
    plt.xticks(np.linspace(0, n_cells, 6).astype(int))

    plt.xlabel("Peak activeness in how many cells?", fontsize=9)
    plt.ylabel("Number of peaks", fontsize=9)
    plt.title("Peak Counts Distribution", fontsize=11)

    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def print_peak_threshold_summary(peak_frac, thresholds=None):
    # Print number of peaks retained under different activity thresholds

    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    for t in thresholds:
        count = (peak_frac >= t).sum()
        print(f"Peaks active in ≥{int(t * 100)}% of cells: {count}")
        

def filter_peaks_by_fraction(X_atac, peak_frac, min_frac):
    # Retain peaks active in at least min_frac fraction of cells

    mask = peak_frac >= min_frac

    if isinstance(X_atac, pd.DataFrame):
        return X_atac.loc[:, mask], mask

    return X_atac[:, mask], mask


def compute_sparsity(X):
    # Compute sparsity level (fraction of zero entries)

    if isinstance(X, pd.DataFrame):
        nnz = (X.values != 0).sum()
        total = X.shape[0] * X.shape[1]
        return 1 - (nnz / total)

    nnz = X.nnz
    total = X.shape[0] * X.shape[1]
    return 1 - (nnz / total)


def log1p_peak_matrix(X):
    # Apply log(1 + x) transformation to peak matrix (dense or sparse)

    if isinstance(X, pd.DataFrame):
        return np.log1p(X)

    X = X.copy()
    X.data = np.log1p(X.data)
    return X
