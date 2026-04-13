# The Math That Powers AI - Code Companion

Interactive Jupyter notebooks and Python implementations accompanying the book **"The Math That Powers AI"** by Vijay Raghavan.

## About This Repository

This repository provides hands-on code examples for the mathematical concepts covered in Book 1 of the AI Engineering series. Each notebook corresponds to a chapter in the book and includes visualizations, interactive elements, and practical implementations.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/math-powers-ai.git
cd math-powers-ai

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Notebooks

| Notebook | Chapter | Description |
|----------|---------|-------------|
| `ch01_linear_algebra.ipynb` | Ch 1: Linear Algebra Foundations | Matrix operations, eigenvalues, SVD, and linear transformations |
| `ch03_calculus.ipynb` | Ch 3: Calculus for Machine Learning | Numerical gradients, chain rule, and gradient descent basics |
| `ch06_svd_compression.ipynb` | Ch 6: SVD Applications | Image compression using rank-k approximations |
| `ch11_gradient_descent.ipynb` | Ch 11: Gradient Descent | From-scratch implementation with learning rate analysis |
| `ch12_optimizers.ipynb` | Ch 12: Optimization Algorithms | SGD, Momentum, RMSprop, and Adam comparisons |

## Quick Start

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
print(f"Final position: {path[-1]}")
```

## Project Structure

```
math-powers-ai/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── ch01_linear_algebra.ipynb
│   ├── ch03_calculus.ipynb
│   ├── ch06_svd_compression.ipynb
│   ├── ch11_gradient_descent.ipynb
│   └── ch12_optimizers.ipynb
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

## License

MIT License

Copyright (c) 2026 Vijay Raghavan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
