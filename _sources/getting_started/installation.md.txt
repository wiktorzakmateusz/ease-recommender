# Install ease_recommender

## Prerequisites

This package was developed and tested on **Python 3.11**.

It depends on the following libraries:
* `numpy` (tested on `>=2.4.6`)
* `scipy` (tested on `>=1.17.1`)

*Note: These Python dependencies will be resolved and installed automatically during the `pip install` step. Older versions of NumPy and SciPy are expected to work without issue.*

## System requirements

Requires [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) and `libomp` installed on your system.

### macOS
```bash
brew install gsl libomp
```

### linux
```bash
sudo apt-get install libgsl-dev libomp-dev
```

### windows
```bash
vcpkg install gsl:x64-windows
$env:GSL_ROOT = "C:/vcpkg/installed/x64-windows"
```

## Install from source

Download the source files from GitHub.
```bash
git clone https://github.com/wiktorzakmateusz/ease-recommender.git && cd ease_recommender
```

Run the following command to install:

```bash
pip install -e .
```