import pytest
import numpy as np
import scipy.sparse as sp
import ease_core

def test_core_compute_gram_allocation(raw_csr_data):
    """
    Tests that the C-extension correctly allocates and returns 
    a memory-safe NumPy array without segmentation fault
    """
    data, indices, indptr, n_users, n_items = raw_csr_data
    reg_weight = 100.0
    num_threads = 2
    
    # Bypass the Python class and hit the C module directly
    G_inv = ease_core.compute_and_invert_gram(
        data, indices, indptr, n_users, n_items, reg_weight, num_threads
    )
    
    # verifies the C-API PyArray_SimpleNew worked correctly
    assert G_inv is not None
    assert isinstance(G_inv, np.ndarray)
    
    # verifies dimensions and type requested in C
    assert G_inv.shape == (n_items, n_items)
    assert G_inv.dtype == np.float64
    
    # verifies it is a dense, C-contiguous array
    assert G_inv.flags.c_contiguous


def test_predict_top_k_logic():
    """
    Tests the inference C-wrapper
    """
    n_items = 5
    K = 2
    
    # simulates a pre-computed float32 weight matrix (B)
    # item 0 is highly correlated with item 3 and 4
    B = np.array([
        [0.0, 0.1, 0.2, 0.9, 0.8],
        [0.1, 0.0, 0.1, 0.1, 0.1],
        [0.2, 0.1, 0.0, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.0, 0.5],
        [0.8, 0.1, 0.1, 0.5, 0.0]
    ], dtype=np.float32)
    
    # must enforce C-contiguous memory layout before passing to C
    B = np.ascontiguousarray(B)
    
    # user interacted with item 0
    interacted = np.array([0], dtype=np.int32)
    
    top_k_indices = ease_core.predict_top_k(B, interacted, n_items, K)
    
    # verifies C-API array allocation
    assert top_k_indices.shape == (K,)
    assert top_k_indices.dtype == np.int32
    
    # verifies sorting logic (should recommend 3, then 4)
    np.testing.assert_array_equal(top_k_indices, np.array([3, 4], dtype=np.int32))


def test_core_type_enforcement(raw_csr_data):
    """
    Tests that the C "O!" format string correctly rejects invalid types
    and prevents catastrophic memory casts
    """
    data, indices, indptr, n_users, n_items = raw_csr_data
    
    # converts 'data' to a standard Python list instead of a NumPy array
    invalid_data_list = data.tolist()
    
    with pytest.raises(TypeError):
        # C-API should instantly reject this because of "O!"
        ease_core.compute_and_invert_gram(
            invalid_data_list, indices, indptr, n_users, n_items, 100.0, 1
        )

def test_core_math_accuracy(raw_csr_data):
    """
    Proves the C/GSL math engine computes the exact same inverted 
    Gram matrix as a pure 64-bit NumPy implementation
    """
    data, indices, indptr, n_users, n_items = raw_csr_data
    reg_weight = 250.0
    num_threads = 1
    
    # C
    P_c = ease_core.compute_and_invert_gram(
        data, indices, indptr, n_users, n_items, reg_weight, num_threads
    )
    
    # Python
    X = sp.csr_matrix((data, indices, indptr), shape=(n_users, n_items))
    
    G_py = (X.T @ X).toarray()
    G_py += reg_weight * np.eye(n_items)
    P_py = np.linalg.inv(G_py)
    
    # asserting if equal
    np.testing.assert_allclose(P_c, P_py, rtol=1e-5, atol=1e-8)
    
    # sanity check: inverted matrix * original matrix = identity matrix
    identity_check = P_c @ G_py
    np.testing.assert_allclose(identity_check, np.eye(n_items), atol=1e-7)

def test_core_top_k_accuracy():
    """
    Proves the C inference wrapper correctly scores, masks, and sorts 
    recommendations exactly like a pure Python argsort operation.
    """
    n_items = 6
    K = 3
    
    # Mock weight matrix B (items x items)
    B = np.array([
        [0.0,  0.1, 0.8, 0.2, 0.95, 0.7],
        [0.1,  0.0, 0.1, 0.1, 0.1,  0.1],
        [0.8,  0.1, 0.0, 0.3, 0.2,  0.4],
        [0.2,  0.1, 0.3, 0.0, 0.1,  0.1],
        [0.95, 0.1, 0.2, 0.1, 0.0,  0.3],
        [0.7,  0.1, 0.4, 0.1, 0.3,  0.0]
    ], dtype=np.float32)
    B = np.ascontiguousarray(B)
    
    # user interacted with item 0 and 5
    interacted = np.array([0, 5], dtype=np.int32)
    
    # C
    c_preds = ease_core.predict_top_k(B, interacted, n_items, K)
    
    # Python
    user_vector = np.zeros(n_items, dtype=np.float32)
    user_vector[interacted] = 1.0
    
    scores = user_vector @ B
    
    # masking already interacted
    scores[interacted] = -np.inf 
    
    # sorting to get top-K indices
    py_preds = np.argsort(scores)[::-1][:K]
    
    # asserting if they match
    np.testing.assert_array_equal(c_preds, py_preds)
