"""Basic tests for the Multi-Stage Inverse Integral Method package"""

import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.idt_calculator_nonreac import IDTCalculator
from src.config_utils import load_config, parse_parameter_spaces, validate_initial_guess


class TestIDTCalculator:
    """Tests for the IDTCalculator class"""
    
    def test_calculator_creation(self):
        """Test that IDTCalculator can be created successfully"""
        calc = IDTCalculator(
            A1=1e-5, n1=0.5, E1=5000,
            Ah=1e-6, nh=0.6, Eh=6000,
            Teq=300, k=0.5, w=0.1, C0=10, xf=0.5
        )
        assert calc is not None
        assert calc.A1 == 1e-5
        assert calc.n1 == 0.5
    
    def test_idt_calculation(self):
        """Test that IDT calculation produces positive values"""
        calc = IDTCalculator(
            A1=1e-5, n1=0.5, E1=5000,
            Ah=1e-6, nh=0.6, Eh=6000,
            Teq=300, k=0.5, w=0.1, C0=10, xf=0.5
        )
        
        idt = calc.calculate_idt(p=10, T=800)
        assert idt > 0, f"IDT should be positive, got {idt}"
        assert isinstance(idt, float), f"IDT should be float, got {type(idt)}"
    
    def test_idt_1st_calculation(self):
        """Test that first-stage IDT calculation produces positive values"""
        calc = IDTCalculator(
            A1=1e-5, n1=0.5, E1=5000,
            Ah=1e-6, nh=0.6, Eh=6000,
            Teq=300, k=0.5, w=0.1, C0=10, xf=0.5
        )
        
        idt_1st = calc.calculate_idt_1st(p=10, T=800)
        assert idt_1st > 0, f"IDT_1st should be positive, got {idt_1st}"


class TestConfigUtils:
    """Tests for configuration utilities"""
    
    def test_validate_initial_guess(self):
        """Test that initial guess validation works correctly"""
        initial_guess = [1.0, 2.0, 3.0]
        bounds = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
        param_names = ['A', 'B', 'C']
        
        validated = validate_initial_guess(initial_guess, bounds, param_names, 'test_step')
        assert validated == initial_guess, "Valid guess should remain unchanged"
    
    def test_validate_initial_guess_out_of_bounds(self):
        """Test that out-of-bounds initial guess is adjusted"""
        initial_guess = [0.0, 10.0, 3.0]  # First two are out of bounds
        bounds = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
        param_names = ['A', 'B', 'C']
        
        validated = validate_initial_guess(initial_guess, bounds, param_names, 'test_step')
        assert validated[0] == 1.0, "First parameter should be adjusted to midpoint"
        assert validated[1] == 2.0, "Second parameter should be adjusted to midpoint"
        assert validated[2] == 3.0, "Third parameter should remain unchanged"


def test_package_imports():
    """Test that all package modules can be imported"""
    from src.optimization_utils import run_optimization_algorithms
    from src.plotting_utils import plot_dp_comparison, plot_integral_curves
    from src.boundary_utils import check_boundary, log_boundary_header
    from src.residuals_nonreac_integral import objective_1st, objective_hi
    
    assert run_optimization_algorithms is not None
    assert plot_dp_comparison is not None
    assert check_boundary is not None
    assert objective_1st is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])