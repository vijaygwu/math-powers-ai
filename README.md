# The Math That Powers AI - Code Companion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Interactive Jupyter notebooks and Python implementations accompanying the book **"The Math That Powers AI"** by Dr. Vijay Raghavan.

## About This Repository

This repository provides hands-on code examples for the mathematical concepts covered in Book 1 of the AI Engineering series. Each notebook corresponds to a chapter in the book and includes:

- **Visualizations** of abstract mathematical concepts
- **Interactive widgets** to explore parameters
- **From-scratch implementations** of key algorithms
- **Connections to real ML applications**

## Quick Start with Google Colab

Run any notebook instantly in your browser - no installation required!

| Chapter | Topic | Colab Link |
|---------|-------|------------|
| Ch 1 | Linear Algebra Foundations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch01_linear_algebra.ipynb) |
| Ch 2 | Probability & Statistics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch02_probability.ipynb) |
| Ch 3 | Calculus for ML | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch03_calculus.ipynb) |
| Ch 4 | Information Theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch04_information_theory.ipynb) |
| Ch 5 | Optimization Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch05_optimization.ipynb) |
| Ch 6 | SVD & Matrix Decomposition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch06_svd_compression.ipynb) |
| Ch 7 | Vector Spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch07_vector_spaces.ipynb) |
| Ch 8 | Numerical Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch08_numerical_methods.ipynb) |
| Ch 11 | Gradient Descent Deep Dive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch11_gradient_descent.ipynb) |
| Ch 12 | Advanced Optimizers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/ch12_optimizers.ipynb) |
| Capstone | Build PCA from Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijaygwu/math-powers-ai/blob/main/notebooks/capstone_pca.ipynb) |

## Notebooks Overview

### Part I: Mathematical Foundations

| Notebook | Description | Key Topics |
|----------|-------------|------------|
| `ch01_linear_algebra.ipynb` | Matrix operations and transformations | Eigenvalues, SVD, linear maps, interactive 2D/3D visualizations |
| `ch02_probability.ipynb` | Probability distributions and inference | Bayes theorem, MLE, multivariate Gaussian, interactive distribution explorer |
| `ch03_calculus.ipynb` | Calculus for machine learning | Gradients, chain rule, Jacobians, automatic differentiation concepts |
| `ch04_information_theory.ipynb` | Information-theoretic foundations | Entropy, KL divergence, cross-entropy loss, mutual information |
| `ch05_optimization.ipynb` | Optimization fundamentals | Gradient descent, learning rates, convexity, saddle points, 3D loss landscapes |
| `ch06_svd_compression.ipynb` | SVD applications | Image compression, low-rank approximation, reconstruction error analysis |
| `ch07_vector_spaces.ipynb` | Abstract vector spaces | Basis, projections, Gram-Schmidt, least squares, word embedding analogies |
| `ch08_numerical_methods.ipynb` | Numerical stability | Floating point, catastrophic cancellation, log-sum-exp trick, gradient checking |

### Part II: Optimization & Training (Preview)

| Notebook | Description | Key Topics |
|----------|-------------|------------|
| `ch11_gradient_descent.ipynb` | Gradient descent from scratch | Convergence analysis, momentum, learning rate schedules |
| `ch12_optimizers.ipynb` | Modern optimizers | SGD, Momentum, RMSprop, Adam - side-by-side comparisons |

### Capstone Project

| Notebook | Description | Key Topics |
|----------|-------------|------------|
| `capstone_pca.ipynb` | Build PCA from scratch | Eigendecomposition vs SVD, MNIST visualization, sklearn validation |

## Local Installation

```bash
# Clone the repository
git clone https://github.com/vijaygwu/math-powers-ai.git
cd math-powers-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Project Structure

```
math-powers-ai/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── ch01_linear_algebra.ipynb
│   ├── ch02_probability.ipynb
│   ├── ch03_calculus.ipynb
│   ├── ch04_information_theory.ipynb
│   ├── ch05_optimization.ipynb
│   ├── ch06_svd_compression.ipynb
│   ├── ch07_vector_spaces.ipynb
│   ├── ch08_numerical_methods.ipynb
│   ├── ch11_gradient_descent.ipynb
│   ├── ch12_optimizers.ipynb
│   └── capstone_pca.ipynb
└── src/
    ├── __init__.py
    ├── optimizers.py
    └── visualization.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- Jupyter >= 1.0.0
- ipywidgets >= 7.6.0
- scikit-learn >= 1.0.0 (for capstone project)

## Quick Example

```python
import numpy as np
from src.optimizers import gradient_descent, adam

# Define a simple quadratic function
def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

# Run gradient descent
x_init = np.array([4.0, 3.0])
path = gradient_descent(grad_f, x_init, lr=0.1, n_steps=50)
print(f"Final position: {path[-1]}")  # Should be close to [0, 0]
```

## About the Book

**"The Math That Powers AI"** is Book 1 in *The AI Engineer's Library* series. It covers the essential mathematical foundations for understanding modern machine learning:

- **Linear Algebra**: The language of data and transformations
- **Probability**: Reasoning under uncertainty
- **Calculus**: The engine of learning (backpropagation)
- **Information Theory**: Measuring and minimizing surprise
- **Optimization**: Finding the best parameters
- **Numerical Methods**: Making it work on real computers

## Contributing

Found a bug or have a suggestion? Please open an issue or submit a pull request!

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Dr. Vijay Raghavan

---

**Contact**: vijayrag@gwu.edu
