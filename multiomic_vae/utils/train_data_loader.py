import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_npz_as_df(path):
    loaded = np.load(path, allow_pickle=True)

    X_sparse = csr_matrix(
        (loaded["data"], loaded["indices"], loaded["indptr"]),
        shape=loaded["shape"]
    )

    df = pd.DataFrame.sparse.from_spmatrix(
        X_sparse,
        index=loaded["row_ids"],
        columns=loaded["col_ids"]
    )

    return df
