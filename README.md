# pyforeca: Forecastable Component Analysis (ForeCA) in Python (Alpha)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⚠️ **Alpha Notice**: This is an **early alpha release** of `pyforeca`.

The API, features, and internals are very likely to change. Functionality is limited and stability is **not guaranteed**. Please do **not** rely on this package for production or long-term reproducibility yet. Use it only for experimentation and feedback.

---

A Python implementation of Forecastable Component Analysis (ForeCA) — a dimension reduction technique for multivariate time series that finds linear combinations with maximum forecastability.  It is sklearn compatible, with the usual `.fit()` and `.transform()` methods.  Can be used as a drop-in replacement of `PCA()` or `FastICA()` for example.


**Note**: `pyforeca` aims to be an sklearn-compatible Python *sibling* to the [**`ForeCA`** R package](https://github.com/gmgeorg/ForeCA).

## What is ForeCA?

Unlike PCA (maximum variance) or ICA (maximum independence), **ForeCA finds components that are maximally forecastable**. This makes it ideal for time series analysis where prediction is often the primary goal.

**Forecastability** is measured using spectral entropy:

* **Low spectral entropy** → High forecastability (predictable patterns)
* **High spectral entropy** → Low forecastability (randomness)

The forecastability measure `Omega` equals 1 minus the normalized spectral entropy of a signal. See Goerg (2013) for details.

## Installation

```bash
poetry add git+https://github.com/gmgeorg/pyforeca.git#main
```

⚠️ Since this is alpha software, breaking changes will occur frequently.

## Quick Start

The code snippet here is a minimum working example to validate that your installation works. For real data examples and tutorials see below.

```python
import numpy as np
from pyforeca.base import ForeCA
from pyforeca.datasets import simulations

latent, mixing_mat, observed = simulations.gen_toy_data(1000)

# Apply ForeCA
foreca = ForeCA(n_components=3, spectrum_method='welch')
# forecastable components found by ForeCA
forecs = foreca.fit_transform(observed)

# View forecastability of components
print("Component spectral entropies (lower = more forecastable):")
for i, omega in enumerate(foreca.omegas_):
    print(f"Component {i+1}: {omega:.4f}")
```

## Features (Alpha)

* `ForeCA` estimator compatible with scikit-learn API (`fit`, `transform`, `fit_transform`)
* Utility functions for univariate and multivariate spectral entropy
* Welch and periodogram spectral estimation options
* Various helper functions for visualization of time series data, biplots (like R), and multivariate/univariate spectral densities.

🚧 Expect incomplete coverage of features from the original R package — many options, controls, and diagnostics are not yet implemented.

## Example Use Cases

For interesting real world data examples see the tutorials & demo notebooks

* simulated data: [`pyforeca-toy-example-data.ipynb`](notebooks/pyforeca-toy-example-demo.ipynb)
* stock market data: [`pyforeca-stock-example.ipynb`](notebooks/pyforeca-stock-example.ipynb)
* weather/climate data: [`pyforeca-weather-example.ipynb`](notebooks/pyforeca-weather-example.ipynb)

See also [SO posts](https://stats.stackexchange.com/search?q=%22foreca%22) for some data & code examples (in R).

⚠️ At this alpha stage, these are illustrative — the implementation is still evolving.

## Algorithm (Simplified)

1. Compute the multivariate spectral density of the input time series.
2. Solve an optimization problem to find linear combinations that minimize spectral entropy.
3. Return components ordered by forecastability (most predictable first).

## References

* Goerg, G. M. (2013). *Forecastable Component Analysis*. Proceedings of the 30th International Conference on Machine Learning (ICML-13). https://proceedings.mlr.press/v28/goerg13.html
* Original R implementation: [ForeCA R package](https://github.com/gmgeorg/ForeCA)

## License

MIT License — see LICENSE file for details.
