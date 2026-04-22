import numpy as np
from scipy.sparse import csr_matrix


def save_sparse_matrix(df, save_path):
    # Save DataFrame as sparse matrix while preserving row and column identifiers

    X_sparse = csr_matrix(df.values)

    row_ids = df.index.to_numpy()
    col_ids = df.columns.to_numpy()

    np.savez_compressed(
        save_path,
        data=X_sparse.data,
        indices=X_sparse.indices,
        indptr=X_sparse.indptr,
        shape=X_sparse.shape,
        row_ids=row_ids,
        col_ids=col_ids
    )
