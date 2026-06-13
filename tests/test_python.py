import pytest
import numpy as np
import scipy.sparse as sp
from ease_recommender import EASE
from unittest.mock import patch

def test_ease_initialization():
    """Test default parameters and attribute initialization"""
    model = EASE(reg_weight=150.0)
    
    assert model.reg_weight == 150.0
    assert model.B is None
    assert model.num_items == 0


def test_fit_memory_and_types(dummy_interaction_matrix):
    """
    Test that the C-extension returns the exact memory layout 
    and data types required for fast inference.
    """
    model = EASE(reg_weight=10.0)
    model.fit(dummy_interaction_matrix)
    
    # checks shape (items x items)
    assert model.B is not None
    assert model.B.shape == (10, 10)
    
    # checks downcasting to float32
    assert model.B.dtype == np.float32
    
    # checks strictly C-contiguous memory layout
    assert model.B.flags.c_contiguous
    
    # checks diagonal is strictly zeroed out
    np.testing.assert_array_equal(np.diag(model.B), np.zeros(10, dtype=np.float32))


def test_fit_handles_non_csr_input(dummy_interaction_matrix):
    """Ensure the model safely converts non-CSR inputs before hitting C"""
    model = EASE()
    coo_matrix = dummy_interaction_matrix.tocoo()
    
    # this should not raise an error, the fit method should auto-convert it
    model.fit(coo_matrix)
    assert model.B is not None


def test_predict_top_k_normal(dummy_interaction_matrix):
    """Test standard Top-K inference behavior"""
    model = EASE()
    model.fit(dummy_interaction_matrix)
    
    history = [0, 2, 4]
    k = 3
    preds = model.predict_new_user(history, k=k)
    
    # checks returned array properties
    assert len(preds) == k
    assert preds.dtype == np.int32
    
    # ensures recommendations do not include items already in the user's history
    for item in preds:
        assert item not in history


def test_predict_fails_before_fit():
    """Ensure a helpful Python error is raised if predicting before training"""
    model = EASE()
    
    with pytest.raises(ValueError, match="model is not fitted yet"):
        model.predict_new_user([1, 2], k=5)


def test_predict_k_exceeds_available(dummy_interaction_matrix):
    """
    Test edge case where requested K is larger than the remaining unseen items,
    the C-layer will crash if this isn't caught, so Python must cap it
    """
    model = EASE()
    model.fit(dummy_interaction_matrix)
    
    history = [0, 1, 2] # 3 items watched out of 10 total
    requested_k = 20    # asking for 20 items (impossible)
    
    preds = model.predict_new_user(history, k=requested_k)
    
    # max possible recommendations should be 7 (10 total - 3 watched)
    assert len(preds) == 7


def test_predict_empty_history(dummy_interaction_matrix):
    """Ensure the model handles brand-new users with zero interactions"""
    model = EASE()
    model.fit(dummy_interaction_matrix)
    
    preds = model.predict_new_user([], k=5)
    
    assert len(preds) == 5
    assert preds.dtype == np.int32

def test_num_threads_initialization():
    """Test that thread counts are properly assigned and fallback to cpu_count"""
    
    model = EASE(num_threads=4)
    assert model.num_threads == 4

    # fallback to os.cpu_count() when passed None
    with patch('os.cpu_count', return_value=10):
        model_auto = EASE(num_threads=None)
        assert model_auto.num_threads == 10
        
    # fallback when os.cpu_count() fails
    with patch('os.cpu_count', return_value=None):
        model_failsafe = EASE(num_threads=None)
        assert model_failsafe.num_threads == 1
