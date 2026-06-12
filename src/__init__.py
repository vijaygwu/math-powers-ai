# Math That Powers AI - Code Companion
# Source modules for optimization and visualization

__version__ = "1.0.0"

from .optimizers import (
    adam,
    compare_optimizers,
    gradient_descent,
    momentum,
    rmsprop,
    sgd,
)
from .visualization import (
    create_comparison_figure,
    plot_convergence,
    plot_gradient_field,
    plot_learning_rate_comparison,
    plot_loss_landscape,
    plot_optimization_path,
)

__all__ = [
    "__version__",
    "adam",
    "compare_optimizers",
    "create_comparison_figure",
    "gradient_descent",
    "momentum",
    "plot_convergence",
    "plot_gradient_field",
    "plot_learning_rate_comparison",
    "plot_loss_landscape",
    "plot_optimization_path",
    "rmsprop",
    "sgd",
]
