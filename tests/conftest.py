import numpy as np
import pytest
import scipy.sparse as sp

@pytest.fixture
def dummy_interaction_matrix():
    """
    Creates a small 5-user, 10-item sparse CSR matrix for testing.
    """
    rows = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 4])
    cols = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
    data = np.ones(10)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(5, 10))

@pytest.fixture
def raw_csr_data():
    """Generates strictly formatted C-contiguous arrays for the C-API"""
    URM = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 1]
    ], dtype=np.float32)
    
    csr = sp.csr_matrix(URM)
    
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    
    return data, indices, indptr, csr.shape[0], csr.shape[1]
