
# EASE Recommender with C Extensions

A high-performance implementation of the **Embarrassingly Shallow Autoencoders (EASE)** recommendation algorithm. This package bridges the gap between Python’s ease of use and C’s raw computational power, delivering near-instant recommendations even for large item catalogs.


## Why this implementation?
Standard Python/NumPy implementations of EASE often struggle with the $O(I^3)$ computational complexity of the Gram matrix inversion. This package optimizes the bottleneck by:

* **Parallelization:** Leveraging OpenMP for multi-threaded Gram matrix construction
* **Cache-Blocking:** Using tiled loop structures to maximize L1 cache hits during inference
* **Numerical Stability:** Utilizing the GNU Scientific Library (GSL) for high-precision Cholesky decomposition
* **Memory Efficiency:** Using zero-copy C-API buffers to communicate directly with NumPy

## Install ease_recommender

### System requirements

Requires [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) and `libomp` installed on your system.

#### macOS
```bash
brew install gsl libomp
```

#### linux
```bash
sudo apt-get install libgsl-dev libomp-dev
```

#### windows
```bash
vcpkg install gsl:x64-windows
$env:GSL_ROOT = "C:/vcpkg/installed/x64-windows"
```

### Install from source

Download the source files from GitHub.
```bash
git clone https://github.com/wiktorzakmateusz/ease_recommender.git && cd ease_recommender
```

Run the following command to install:

```bash
pip install -e .
```


## Quick start

To check the functionalities of the package, write the following code:

```python
from ease_recommender import EASE
import numpy as np
import scipy.sparse as sp

# URM of 5 users and 10 movies
URM = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
])

# convert to CSR sparse matrix
X_csr = sp.csr_matrix(URM)

# initialize with your preferred L2 regularization weight
model = EASE(reg_weight=100.0)

# fit on a CSR sparse matrix
model.fit(X_csr)

# get Top-K recommendations for a user
user_history = [2, 7]
recommendations = model.predict_new_user(user_history, k=3)
print(recommendations)
```

then, the output should be:

```none
[1 8 3]
```

## Benchmarking

To check the gain obtained by using the package, use $benchmark.py$ code:

```bash
python benchmark.py --num_threads=10 --repeats=5
```

expected outcome:

```
Generating synthetic sparse dataset...
Users: 100000 | Items: 5000 | Density: 0.5%
--------------------------------------------------
Starting Benchmark (5 iterations, 10 threads)...
  Running iteration 1/5...
  Running iteration 2/5...
  Running iteration 3/5...
  Running iteration 4/5...
  Running iteration 5/5...
--------------------------------------------------
RESULTS (Averaged over 5 runs):

Fit (Training) Times
Python:       1.8773s +/- 0.0224s
C-Optimized:  0.4804s +/- 0.0039s
Speedup:      3.91x faster

Inference Times
Python:       0.017600s +/- 0.000705s
C-Optimized:  0.000030s +/- 0.000001s
Speedup:      581.66x faster
--------------------------------------------------
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.