import numpy as np
import scipy.sparse as sp
import ease_core 
import os

class EASE:
    def __init__(self, reg_weight=250.0, num_threads=None):
        """
        Initialize the EASE (Embarrassingly Shallow Autoencoders) model.

        References
        ----------
            Harald Steck. “Embarrassingly Shallow Autoencoders for Sparse Data”. In: 
            The World Wide Web Conference. 2019, pp. 3251-3257

        Parameters
        ----------
        reg_weight : float, optional
            The L2 regularization factor (lambda) to penalize large weights 
            and prevent overfitting. Default is 250.0.
        num_threads : int, optional
            The number of OpenMP threads to use during the parallelized sparse 
            Gram matrix construction. Default is 1.
        """
        self.reg_weight = float(reg_weight)

        if num_threads is None:
            self.num_threads = os.cpu_count() or 1
        else:
            self.num_threads = int(num_threads)

        self.B = None
        self.num_items = 0

    def fit(self, X_train_csr):
        """
        Train the model by computing the Gram matrix directly from sparse CSR, 
        followed by high-precision GSL Cholesky inversion.

        The underlying C engine utilizes cache blocking to maximize L1 cache hits
        and sequential memory prefetching, ensuring high-throughput inference even
        with massive item catalogs.

        Parameters
        ----------
        X_train_csr : (U, I) scipy.sparse.csr_matrix
            The user-item interaction matrix where rows are users and 
            columns are items.

        Returns
        -------
        None
            The method computes and stores the dense weight matrix `self.B` 
            in-place as a float32 contiguous array.
        """
        # ensuring the matrix is strictly in csr format to allow highly efficient 
        # row-wise memory access in the C layer
        if not sp.isspmatrix_csr(X_train_csr):
            X_train_csr = X_train_csr.tocsr()
        
        self.num_items = X_train_csr.shape[1]
        num_users = X_train_csr.shape[0]

        # forcing memory to be C-contiguous and strictly typed, this allows the 
        # C-api wrapper to extract raw memory pointers instantly, using float64 to
        # maintain numerical stability during GSL matrix inversion
        data_fixed = np.ascontiguousarray(X_train_csr.data, dtype=np.float64)
        indices_fixed = np.ascontiguousarray(X_train_csr.indices, dtype=np.int32)
        indptr_fixed = np.ascontiguousarray(X_train_csr.indptr, dtype=np.int32)

        # offloading the O(n^3) gram matrix computation and inversion entirely to 
        # the compiled C/GSL backend
        P = ease_core.compute_and_invert_gram(
            data_fixed, 
            indices_fixed, 
            indptr_fixed,
            num_users, 
            self.num_items, 
            self.reg_weight,
            self.num_threads
        )
        
        # computing the final EASE weight matrix using highly vectorized numpy 
        # operations, this final step is extremely fast and does not require 
        # C-level intervention
        self.B = P / (-np.diag(P))
        np.fill_diagonal(self.B, 0.0)

        # downcasting to float32 to instantly halve the memory footprint of the 
        # dense weight matrix, enforcing c_contiguous layout enables optimal 
        # hardware cache prefetching during inference
        self.B = np.ascontiguousarray(self.B, dtype=np.float32)

    def predict_new_user(self, interacted_item_ids, k=20):
        """
        Predict top-K recommendations for a new user based on their interaction 
        history.

        Parameters
        ----------
        interacted_item_ids : (N,) array_like
            A 1D array or list of item indices the user has already interacted 
            with.
        k : int, optional
            The number of top recommendations to return. Default is 20.

        Returns
        -------
        top_k_indices : (K,) ndarray
            An array of the recommended item indices sorted by score in descending 
            order. 
            Returns an empty array if the requested K is invalid.
        """
        if self.B is None:
            raise ValueError("model is not fitted yet. call 'fit' first.")
            
        # enforcing a contiguous 32-bit integer array to satisfy the strict type 
        # requirements of the C inference wrapper
        interacted_arr = np.ascontiguousarray(interacted_item_ids, dtype=np.int32)
        
        # capping k to the maximum available unseen items to prevent buffer overflow
        # or out-of-bounds memory access in C
        k = min(k, self.num_items - len(interacted_arr))
        if k <= 0:
            return np.array([], dtype=np.int32)

        # bypassing python interpreter overhead by piping the strict numpy memory 
        # blocks directly into the C inference engine
        top_k_indices = ease_core.predict_top_k(
            self.B, 
            interacted_arr, 
            self.num_items, 
            k
        )
        
        return top_k_indices