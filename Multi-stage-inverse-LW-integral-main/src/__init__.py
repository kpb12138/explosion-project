"""Multi-Stage Inverse Integral Method package"""

from .idt_calculator_nonreac import IDTCalculator, integrate_idt
from .config_utils import load_config, parse_parameter_spaces, validate_config, validate_initial_guess
from .optimization_utils import run_optimization_algorithms
from .plotting_utils import plot_dp_comparison, plot_integral_curves, plot_predicted_idts
from .boundary_utils import check_boundary, log_boundary_header, log_optimization_step
from .residuals_nonreac_integral import objective_1st, objective_hi, objective_total

__version__ = "1.0.0"
__author__ = "Yingtao Wu"
__email__ = "wuyingtao@xjtu.edu.cn"

__all__ = [
    'IDTCalculator',
    'integrate_idt',
    'load_config',
    'parse_parameter_spaces',
    'validate_config',
    'validate_initial_guess',
    'run_optimization_algorithms',
    'plot_dp_comparison',
    'plot_integral_curves',
    'plot_predicted_idts',
    'check_boundary',
    'log_boundary_header',
    'log_optimization_step',
    'objective_1st',
    'objective_hi',
    'objective_total',
]