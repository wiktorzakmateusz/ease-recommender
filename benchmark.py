import time
import argparse
import numpy as np
import scipy.sparse as sp
from ease_recommender import EASE 

class EASE_PurePython:
    """A standard NumPy & Scipy implementation of EASE to act as a baseline."""
    def __init__(self, reg_weight=250.0):
        self.reg_weight = reg_weight
        self.B = None
        self.num_items = 0

    def fit(self, X_train_csr):

        if not sp.isspmatrix_csr(X_train_csr):
            X_train_csr = X_train_csr.tocsr()

        self.num_items = X_train_csr.shape[1]
        
        # gram matrix G = X^T X
        G = X_train_csr.T @ X_train_csr
        
        # added regularization to the diagonal
        G += self.reg_weight * sp.identity(G.shape[0], dtype=np.float32)
        
        # converting to dense and inverting
        G_dense = G.todense()
        P = np.linalg.inv(G_dense)
        
        # weight matrix B
        self.B = P / (-np.diag(P))
        np.fill_diagonal(self.B, 0.0)

    def predict_new_user(self, interacted_item_ids, k=20):

        if self.B is None:
            raise ValueError("model is not fitted yet. call 'fit' first.")
         
        # Create a dense user vector
        user_vector = np.zeros(self.num_items)
        user_vector[interacted_item_ids] = 1.0
        
        # Matrix-vector multiplication
        scores = user_vector @ self.B
        scores = np.asarray(scores).flatten()
        
        # Remove already interacted items
        scores[interacted_item_ids] = -np.inf
        
        # Get top K indices
        return np.argsort(scores)[::-1][:k]

def run_benchmark(n_repeats=5, num_threads=1):

    NUM_USERS = 100000
    NUM_ITEMS = 5000
    DENSITY = 0.005  # 0.5% of the matrix is filled (standard sparsity)
    
    print(f"Generating synthetic sparse dataset...")
    print(f"Users: {NUM_USERS} | Items: {NUM_ITEMS} | Density: {DENSITY*100}%")
    
    # Generates random binary interactions
    X_train = sp.random(NUM_USERS, NUM_ITEMS, density=DENSITY, format='csr', dtype=np.float32, rng=42)
    X_train.data[:] = 1.0
    
    sample_user_interactions = [10, 45, 102, 500, 1200]
    
    # Lists to store the timings for each run
    py_fit_times, py_infer_times = [], []
    c_fit_times, c_infer_times = [], []

    print("-" * 50)
    print(f"Starting Benchmark ({n_repeats} iterations, {num_threads} threads)...")
    
    for i in range(n_repeats):
        print(f"  Running iteration {i + 1}/{n_repeats}...")
        
        # Pure Python EASE
        py_model = EASE_PurePython()
        
        start_time = time.perf_counter()
        py_model.fit(X_train)
        py_fit_times.append(time.perf_counter() - start_time)
        
        start_time = time.perf_counter()
        py_model.predict_new_user(sample_user_interactions)
        py_infer_times.append(time.perf_counter() - start_time)
        
        # C-Optimized EASE
        c_model = EASE(num_threads=num_threads)
        
        start_time = time.perf_counter()
        c_model.fit(X_train)
        c_fit_times.append(time.perf_counter() - start_time)
        
        start_time = time.perf_counter()
        c_model.predict_new_user(sample_user_interactions)
        c_infer_times.append(time.perf_counter() - start_time)

    # Aggregation of results
    py_fit_mean, py_fit_std = np.mean(py_fit_times), np.std(py_fit_times)
    py_infer_mean, py_infer_std = np.mean(py_infer_times), np.std(py_infer_times)
    
    c_fit_mean, c_fit_std = np.mean(c_fit_times), np.std(c_fit_times)
    c_infer_mean, c_infer_std = np.mean(c_infer_times), np.std(c_infer_times)

    print("-" * 50)
    print(f"RESULTS (Averaged over {n_repeats} runs):")
    print("\nFit (Training) Times")
    print(f"Python:       {py_fit_mean:.4f}s +/- {py_fit_std:.4f}s")
    print(f"C-Optimized:  {c_fit_mean:.4f}s +/- {c_fit_std:.4f}s")
    print(f"Speedup:      {py_fit_mean / c_fit_mean:.2f}x faster")

    print("\nInference Times")
    print(f"Python:       {py_infer_mean:.6f}s +/- {py_infer_std:.6f}s")
    print(f"C-Optimized:  {c_infer_mean:.6f}s +/- {c_infer_std:.6f}s")
    print(f"Speedup:      {py_infer_mean / c_infer_mean:.2f}x faster")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the EASE Recommender System.")
    
    # num_threads argument
    parser.add_argument(
        "--num_threads", 
        type=int, 
        default=None, 
        help="Number of OpenMP threads for the C-extension to use."
    )
    
    # repeats argument
    parser.add_argument(
        "--repeats", 
        type=int, 
        default=5, 
        help="Number of times to run the benchmark loop."
    )

    args = parser.parse_args()
    run_benchmark(n_repeats=args.repeats, num_threads=args.num_threads)