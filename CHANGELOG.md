# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-15
### Added
- Initial release of the `ease_recommender` package.
- Python wrapper class `EASE` for model training and prediction.
- Pure C implementation for Gram matrix calculation from CSR sparse matrices.
- Memory-efficient Min-Heap C implementation for Top-K recommendations.
- `pyproject.toml` and `setup.py` configured for compiling C extensions with NumPy headers.