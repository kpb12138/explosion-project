import numpy as np
from datetime import datetime

def check_boundary(params, bounds, param_names, step_name, log_file):
    """Check if optimized parameters are at boundary """
    boundary_results = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    for param, bound, name in zip(params, bounds, param_names):
        lower, upper = bound 
        status = "OK"
        if np.isclose(param, lower):
            status = "LOWER_BOUND"
            warning_msg = f"Warning: {name} at lower boundary ({param:.6f}) in {step_name}"
            print(warning_msg)
            boundary_results.append((name, param, "lower", (lower, upper)))
        elif np.isclose(param, upper):
            status = "UPPER_BOUND"
            warning_msg = f"Warning: {name} at upper boundary ({param:.6f}) in {step_name}"
            print(warning_msg)
            boundary_results.append((name, param, "upper", (lower, upper)))

    with open(log_file, 'a') as f:
        f.write(f"\n=== Boundary Check Results - {step_name} ===\n")
        f.write(f"Timestamp: {current_time}\n")
        if not boundary_results:
            f.write("All parameters within normal bounds\n")
        else:
            for name, value, bound_type, bounds in boundary_results:
                f.write(f"{name}: {value:.6f} at {bound_type} boundary ({bounds[0]}, {bounds[1]})\n")

    return boundary_results


def log_optimization_step(params, param_names, step_name, objective_value, algorithm, log_file):
    """Log optimization results for each step"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== Optimization Results - {step_name} ===\n")
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Algorithm Used: {algorithm}\n")
        f.write(f"Final Objective Value: {objective_value:.6f}\n")
        f.write("Optimized Parameters:\n")
        for name, value in zip(param_names, params):
            # For A1, E1, Ah and Eh, use scientific notation
            if name in ['A1', 'E1', 'Ah', 'Eh']:
                f.write(f"{name}: {value:.6e}\n")
            else:
                f.write(f"{name}: {value:.6f}\n")


def log_boundary_header(log_file):
    """Create optimization log file with header"""
    with open(log_file, 'w') as f:
        f.write("=== Combustion Model Optimization Log ===\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=====================================================\n\n")
