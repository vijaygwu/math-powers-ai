"""
Optimization algorithms for gradient-based learning.

This module provides clean implementations of common optimizers used in machine learning,
including vanilla gradient descent, SGD with momentum, RMSprop, and Adam.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


def gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 100
) -> List[np.ndarray]:
    """
    Vanilla gradient descent optimizer.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step (optimization path).

    Example
    -------
    >>> def grad_f(x):
    ...     return np.array([2*x[0], 4*x[1]])
    >>> path = gradient_descent(grad_f, np.array([4.0, 3.0]), lr=0.1, n_steps=50)
    >>> print(f"Final: {path[-1]}")
    """
    x = x_init.copy().astype(float)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        x = x - lr * grad
        path.append(x.copy())

    return path


def sgd(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 100,
    noise_scale: float = 0.1
) -> List[np.ndarray]:
    """
    Stochastic Gradient Descent with simulated noise.

    In practice, the stochasticity comes from mini-batch sampling.
    Here we simulate it by adding noise to the gradient.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    n_steps : int
        Number of optimization steps.
    noise_scale : float
        Scale of Gaussian noise added to gradients.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.
    """
    x = x_init.copy().astype(float)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        # Add stochastic noise to simulate mini-batch variance
        noise = np.random.randn(*grad.shape) * noise_scale
        x = x - lr * (grad + noise)
        path.append(x.copy())

    return path


def momentum(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    n_steps: int = 100
) -> List[np.ndarray]:
    """
    Gradient descent with momentum.

    Momentum accumulates past gradients to accelerate convergence
    and dampen oscillations.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta : float
        Momentum coefficient (typically 0.9).
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        v_t = beta * v_{t-1} + grad
        x_t = x_{t-1} - lr * v_t
    """
    x = x_init.copy().astype(float)
    v = np.zeros_like(x)  # Velocity (momentum term)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        v = beta * v + grad
        x = x - lr * v
        path.append(x.copy())

    return path


def rmsprop(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    n_steps: int = 100
) -> List[np.ndarray]:
    """
    RMSprop optimizer.

    Adapts learning rates based on a running average of squared gradients.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta : float
        Decay rate for squared gradient average (typically 0.9).
    epsilon : float
        Small constant for numerical stability.
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        s_t = beta * s_{t-1} + (1 - beta) * grad^2
        x_t = x_{t-1} - lr * grad / sqrt(s_t + epsilon)
    """
    x = x_init.copy().astype(float)
    s = np.zeros_like(x)  # Running average of squared gradients
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        s = beta * s + (1 - beta) * grad**2
        x = x - lr * grad / (np.sqrt(s) + epsilon)
        path.append(x.copy())

    return path


def adam(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    n_steps: int = 100
) -> List[np.ndarray]:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum with adaptive learning rates. The most widely used
    optimizer in deep learning.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta1 : float
        Decay rate for first moment (momentum).
    beta2 : float
        Decay rate for second moment (squared gradients).
    epsilon : float
        Small constant for numerical stability.
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)  (bias correction)
        v_hat = v_t / (1 - beta2^t)  (bias correction)
        x_t = x_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)

    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    arXiv preprint arXiv:1412.6980.
    """
    x = x_init.copy().astype(float)
    m = np.zeros_like(x)  # First moment (mean of gradients)
    v = np.zeros_like(x)  # Second moment (variance of gradients)
    path = [x.copy()]

    for t in range(1, n_steps + 1):
        grad = grad_fn(x)

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())

    return path


def compare_optimizers(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    n_steps: int = 100,
    optimizers: Optional[List[str]] = None
) -> dict:
    """
    Run multiple optimizers on the same problem for comparison.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    n_steps : int
        Number of optimization steps.
    optimizers : List[str], optional
        List of optimizer names to compare. Defaults to all.

    Returns
    -------
    dict
        Dictionary mapping optimizer names to their optimization paths.
    """
    if optimizers is None:
        optimizers = ['gradient_descent', 'momentum', 'rmsprop', 'adam']

    results = {}

    optimizer_fns = {
        'gradient_descent': lambda: gradient_descent(grad_fn, x_init, lr=0.1, n_steps=n_steps),
        'sgd': lambda: sgd(grad_fn, x_init, lr=0.1, n_steps=n_steps),
        'momentum': lambda: momentum(grad_fn, x_init, lr=0.1, beta=0.9, n_steps=n_steps),
        'rmsprop': lambda: rmsprop(grad_fn, x_init, lr=0.1, n_steps=n_steps),
        'adam': lambda: adam(grad_fn, x_init, lr=0.1, n_steps=n_steps),
    }

    for name in optimizers:
        if name in optimizer_fns:
            results[name] = optimizer_fns[name]()

    return results
