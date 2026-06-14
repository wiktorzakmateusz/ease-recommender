ease_recommender package
========================

A high-performance implementation of the Embarrassingly Shallow Autoencoders (EASE) recommendation algorithm. This package bridges the gap between Python's ease of use and C's raw computational power, enabling exceptionally fast training on sparse datasets and Top-K inference for large item catalogs.

Why this implementation?
------------------------

Standard SciPy/NumPy implementations of EASE often struggle under the weight of the algorithm's mathematical bottlenecks. This package optimizes the entire pipeline by shifting all heavy computing, specifically the Gram matrix construction, the :math:`O(I^3)` matrix inversion, and the Top-K inference, down to a highly optimized C layer. It achieves this by:

* **Sparse Matrix Multiplication:** Computing the item-item Gram matrix (:math:`X^T X`) directly from Compressed Sparse Row (CSR) arrays, leveraging OpenMP multi-threading to accelerate training.

* **High-Performance Inversion:** Utilizing the GNU Scientific Library (GSL) to perform mathematically robust, in-place Cholesky decomposition and inversion on the symmetric Gram matrix, minimizing RAM overhead.

* **Cache-Aware Inference:** Employing cache-blocked loop structures to maximize L1 hardware cache hits and sequential memory prefetching during Top-K prediction.

* **Zero-Copy Memory:** Using the NumPy C-API to share direct memory pointers between Python and C, eliminating expensive data duplication and casting overhead during runtime.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   getting_started/installation
   getting_started/quickstart
   getting_started/benchmarking

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/ease_model