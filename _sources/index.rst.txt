ease_recommender package
========================

A high-performance implementation of the Embarrassingly Shallow Autoencoders (EASE) recommendation algorithm. This package bridges the gap between Python's ease of use and C's raw computational power, delivering near-instant recommendations even for large item catalogs.

Why this implementation?
------------------------

Standard SciPy/NumPy implementations of EASE often struggle with the :math:`O(I^3)` computational complexity of the Gram matrix inversion. This package optimizes the bottleneck by:

* **Parallelization:** Leveraging OpenMP for multi-threaded Gram matrix construction
* **Cache-Blocking:** Using tiled loop structures to maximize L1 cache hits during inference
* **Numerical Stability:** Utilizing the GNU Scientific Library (GSL) for high-precision Cholesky decomposition
* **Memory Efficiency:** Using zero-copy C-API buffers to communicate directly with NumPy

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