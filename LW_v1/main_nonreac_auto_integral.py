"""
main_nonreac_auto_integral.py - Single-stage LW optimization
=============================================================
Adapted from version_2026_v3 for single-stage CH3NO2.
Key simplifications:
  - Removed Step 1 (first-stage IDT) - single-stage fuel
  - Step 1 (new): optimize single Arrhenius (A, n, E) using all IDT data
  - Step 2: optimize (Teq, k, w) using Dp data
  - Step 3: all-parameter optimization (8 params total)
  - Parameter count: 8 instead of 11
Usage: python main_nonreac_auto_integral.py
  Configure CH3NO2.json first:
    - single_stage: high-T IDT entries (from experiments)
    - ntc_region: NTC region IDT entries (if applicable)
    - dp_optimization: (T, P, Dp_actual) for pressure-rise fitting
"""
import os
import numpy as np
# Replace skopt.space.Real with local class (avoid skopt dependency)
# The local Real class is defined in config_utils.py (already imported below)
# Import modules (same structure as version_2026_v3)
from config_utils import load_config, parse_parameter_spaces, validate_config, validate_initial_guess, Real
from boundary_utils import check_boundary, log_boundary_header, log_optimization_step
from optimization_utils import run_optimization_algorithms
from plotting_utils import plot_dp_comparison, plot_integral_curves, plot_predicted_idts
from idt_calculator_nonreac import IDTCalculator
from residuals_nonreac_integral import (objective_single, objective_total,
    residuals_vector_total, DeltaP_residuals)
def main():
    # Create results directory
    results_dir = 'results_v2'
    os.makedirs(results_dir, exist_ok=True)
    # Create optimization log file
    optimization_log_file = os.path.join(results_dir, 'optimization.log')
    log_boundary_header(optimization_log_file)
    # Read configuration file
    config = load_config('CH3NO2.json')
    all_files, idts_1st, idts_total, idts_one, idts_ntc = validate_config(config)
    # Parse parameter spaces
    param_spaces = parse_parameter_spaces(config)
    # Read ignition timing data
    t_com = config['timing_parameters']['t_com']  # compression time (ms)
    idts_one_with_tcom = [x + t_com for x in idts_one]
    # Combine all IDT data for Step 3 (all-parameter optimization)
    idts_combined = idts_one + idts_ntc
    idts_combined_with_tcom = [x + t_com for x in idts_combined]
    original_idts_combined = idts_combined.copy()
    # Read initial parameters (8 params for single-stage)
    initial_params = config['initial_parameters']
    A, n, E = initial_params['A'], initial_params['n'], initial_params['E']
    Teq, k, w = initial_params['Teq'], initial_params['k'], initial_params['w']
    C0, xf = initial_params['C0'], initial_params['xf']
    # Load input CSV files
    dfs = []
    import pandas as pd
    for filename, directory in all_files:
        dfs.append(pd.read_csv(os.path.join(directory, filename)))
    # Read adjust factors for Step 3 parameter range narrowing
    step3_factors = config['optimization_settings'].get('adjust_factors', {})
    A_factor = step3_factors.get('A_factor', 0.15)
    n_factor = step3_factors.get('n_factor', 0.15)
    E_factor = step3_factors.get('E_factor', 0.15)
    Teq_factor = step3_factors.get('Teq_factor', 0.02)
    k_factor = step3_factors.get('k_factor', 0.5)
    w_factor = step3_factors.get('w_factor', 0.5)
    # ================================================================
    #  Step 1: Optimize single-stage Arrhenius (A, n, E)
    #  Uses single_stage IDT data from JSON config
    #  Formula: IDT = A * p^n * exp(E/T)
    # ================================================================
    if not idts_one:
        print("Warning: idts_one is empty. Skipping Step 1 optimization.")
        optimized_A, optimized_n, optimized_E = A, n, E
        A_range = (param_spaces['step1'][0].low, param_spaces['step1'][0].high)
        n_range = (param_spaces['step1'][1].low, param_spaces['step1'][1].high)
        E_range = (param_spaces['step1'][2].low, param_spaces['step1'][2].high)
    else:
        print("\nStarting Step 1 optimization: A, n, E (single-stage Arrhenius)")
        # Create log-space objective function
        def obj_step1_log(p_log):
            p_model = log_to_model_space(p_log, [p.name for p in param_spaces['step1']])
            return objective_single(p_model, dfs[:len(idts_one)], idts_one_with_tcom,
                (Teq, k, w, C0, xf), config)
        # Convert initial guess to log space
        initial_guess_step1 = [A, n, E]
        initial_guess_step1_log = model_to_log_space(initial_guess_step1,
            [p.name for p in param_spaces['step1']])
        initial_guess_step1_log = validate_initial_guess(initial_guess_step1_log,
            [(np.log(b.low) if b.name in ['A', 'E'] and b.low > 0 and b.high > 0 else b.low,
              np.log(b.high) if b.name in ['A', 'E'] and b.low > 0 and b.high > 0 else b.high)
             for b in param_spaces['step1']],
            [p.name for p in param_spaces['step1']], 'step1')
        # Create log-space parameter space
        param_spaces_log_step1 = []
        for param in param_spaces['step1']:
            if param.name in ['A', 'E'] and param.low > 0 and param.high > 0:
                param_spaces_log_step1.append(Real(np.log(param.low), np.log(param.high), name=param.name))
            else:
                param_spaces_log_step1.append(param)
        # Run optimization
        optimized_params_step1_log, history_step1, residuals_step1, algorithm_step1 = run_optimization_algorithms(
            obj_step1_log, param_spaces_log_step1, initial_guess_step1_log,
            [p.name for p in param_spaces['step1']], 'step1', config, results_dir)
        # Convert back to model space
        optimized_params_step1 = log_to_model_space(optimized_params_step1_log,
            [p.name for p in param_spaces['step1']])
        optimized_A, optimized_n, optimized_E = optimized_params_step1
        # Log results
        final_objective = residuals_step1[-1] if residuals_step1 else 0.0
        log_optimization_step(optimized_params_step1,
            [p.name for p in param_spaces['step1']],
            "Step 1", final_objective, algorithm_step1, optimization_log_file)
        # Boundary check
        check_boundary(optimized_params_step1,
            [(b.low, b.high) for b in param_spaces['step1']],
            [p.name for p in param_spaces['step1']], "Step 1", optimization_log_file)
        # Narrow ranges for Step 3 (all-parameter)
        A_range = (optimized_A * (1 - A_factor), optimized_A * (1 + A_factor))
        n_range = (optimized_n * (1 + n_factor), optimized_n * (1 - n_factor))
        E_range = (optimized_E * (1 - E_factor), optimized_E * (1 + E_factor))
    # ================================================================
    #  Step 2: Optimize (Teq, k, w) using pressure rise Dp data
    # ================================================================
    dp_data = config['file_specifications']['dp_optimization']
    T_list = np.array(dp_data['initial_temperature'])
    P_list = np.array(dp_data['initial_pressure'])
    dp_actual = np.array(dp_data['D_p_actual'])
    if dp_actual.size == 0:
        print("Warning: dp_actual is empty. Skipping Step 2 optimization.")
        optimized_Teq, optimized_k, optimized_w = Teq, k, w
        Teq_range = (param_spaces['step2'][0].low, param_spaces['step2'][0].high)
        k_range = (param_spaces['step2'][1].low, param_spaces['step2'][1].high)
        w_range = (param_spaces['step2'][2].low, param_spaces['step2'][2].high)
    else:
        print("\nStarting Step 2 optimization: Teq, k, w (pressure rise)")
        if len(T_list) != len(P_list) or len(T_list) != len(dp_actual):
            print(f"Error: Data length mismatch in Step 2: T={len(T_list)}, P={len(P_list)}, Dp={len(dp_actual)}")
            exit(1)
        # Define objective for Step 2
        def obj_step2(p):
            residuals = DeltaP_residuals(p, T_list, P_list, dp_actual)
            return np.mean(residuals ** 2) if np.isfinite(residuals).all() else 1e10
        # Initial guess and validation
        initial_guess_step2 = [Teq, max(k, 1e-6), w]
        initial_guess_step2 = validate_initial_guess(initial_guess_step2,
            [(b.low, b.high) for b in param_spaces['step2']],
            [p.name for p in param_spaces['step2']], 'step2')
        # Run optimization
        optimized_params_step2, history_step2, residuals_step2, algorithm_step2 = run_optimization_algorithms(
            obj_step2, param_spaces['step2'], initial_guess_step2,
            [p.name for p in param_spaces['step2']], 'step2', config, results_dir)
        optimized_Teq, optimized_k, optimized_w = optimized_params_step2
        # Log results
        final_objective = residuals_step2[-1] if residuals_step2 else 0.0
        log_optimization_step(optimized_params_step2,
            [p.name for p in param_spaces['step2']],
            "Step 2", final_objective, algorithm_step2, optimization_log_file)
        # Boundary check
        check_boundary(optimized_params_step2,
            [(b.low, b.high) for b in param_spaces['step2']],
            [p.name for p in param_spaces['step2']], "Step 2", optimization_log_file)
        # Plot Dp comparison
        try:
            D_Tcf = optimized_w * (T_list - optimized_Teq * np.power(P_list, optimized_k))
            Tcf = T_list + 0.5 * (D_Tcf + np.sqrt(np.maximum(D_Tcf ** 2, 0)))
            pcf = Tcf / T_list * P_list
            dp_predicted = pcf - P_list
            plot_dp_comparison(T_list, dp_actual, dp_predicted, results_dir)
        except Exception as e:
            print(f"Failed to plot Dp comparison: {e}")
        # Narrow ranges for Step 3
        Teq_range = (optimized_Teq * (1 - Teq_factor), optimized_Teq * (1 + Teq_factor))
        k_range = (optimized_k * (1 - k_factor), optimized_k * (1 + k_factor))
        w_range = (optimized_w * (1 + w_factor), optimized_w * (1 - w_factor))  # w is negative
    # ================================================================
    #  Step 3: All-parameter optimization (8 parameters)
    #  Narrowed ranges from Steps 1 & 2
    # ================================================================
    C0_low, C0_high = param_spaces['step3'][0].low, param_spaces['step3'][0].high
    xf_low, xf_high = param_spaces['step3'][1].low, param_spaces['step3'][1].high
    print("\nStarting Step 3 optimization: all 8 parameters")
    # Create full parameter space for Step 3
    step3_full_space = [
        Real(A_range[0], A_range[1], name='A'),
        Real(n_range[0], n_range[1], name='n'),
        Real(E_range[0], E_range[1], name='E'),
        Real(Teq_range[0], Teq_range[1], name='Teq'),
        Real(k_range[0], k_range[1], name='k'),
        Real(w_range[0], w_range[1], name='w'),
        Real(C0_low, C0_high, name='C0'),
        Real(xf_low, xf_high, name='xf')]
    # Create log-space objective function
    def obj_step3_log(p_log):
        p_model = log_to_model_space(p_log, [p.name for p in step3_full_space])
        return objective_total(p_model, dfs, original_idts_combined, t_com, config)
    # Create residuals vector function for least_squares
    def residuals_step3_log(p_log):
        p_model = log_to_model_space(p_log, [p.name for p in step3_full_space])
        return residuals_vector_total(p_model, dfs, original_idts_combined, t_com, config)
    # Prepare optimization parameters for Step 3
    initial_guess_step3 = [optimized_A, optimized_n, optimized_E,
                           optimized_Teq, optimized_k, optimized_w, C0, xf]
    param_names_step3 = [s.name for s in step3_full_space]
    bounds_step3 = [(s.low, s.high) for s in step3_full_space]
    # Convert initial guess to log space
    initial_guess_step3_log = model_to_log_space(initial_guess_step3, param_names_step3)
    initial_guess_step3_log = validate_initial_guess(initial_guess_step3_log,
        [(np.log(b[0]) if b[0] > 0 and b[1] > 0 else b[0],
          np.log(b[1]) if b[0] > 0 and b[1] > 0 else b[1])
         if name in ['A', 'E', 'C0'] else b
         for name, b in zip(param_names_step3, bounds_step3)],
        param_names_step3, 'step3')
    # Create log-space parameter space
    param_spaces_log_step3 = []
    for param in step3_full_space:
        if param.name in ['A', 'E', 'C0'] and param.low > 0 and param.high > 0:
            param_spaces_log_step3.append(Real(np.log(param.low), np.log(param.high), name=param.name))
        else:
            param_spaces_log_step3.append(param)
    # Run optimization for Step 3
    optimized_params_step3_log, history_step3, residuals_step3, algorithm_step3 = run_optimization_algorithms(
        obj_step3_log, param_spaces_log_step3, initial_guess_step3_log,
        param_names_step3, 'step3', config, results_dir, residuals_func=residuals_step3_log)
    # Convert back to model space
    optimized_params_step3 = log_to_model_space(optimized_params_step3_log, param_names_step3)
    (final_A, final_n, final_E, final_Teq, final_k, final_w, final_C0, final_xf) = optimized_params_step3
    # Log results
    final_objective = residuals_step3[-1] if residuals_step3 else 0.0
    log_optimization_step(optimized_params_step3, param_names_step3,
        "Step 3", final_objective, algorithm_step3, optimization_log_file)
    # Boundary check for Step 3
    check_boundary(optimized_params_step3, bounds_step3, param_names_step3, "Step 3", optimization_log_file)
    # ================================================================
    #  Generate integral curves and final results
    # ================================================================
    final_calculator = IDTCalculator(
        final_A, final_n, final_E, final_Teq, final_k, final_w, final_C0, final_xf)
    # Plot predicted IDT vs original IDT
    # Note: using plot_predicted_idts from v3 (handles plotting generically)
    # Pass empty idts_1st list since we have no first-stage
    plot_predicted_idts(dfs, [], idts_combined, t_com, final_calculator, results_dir, config, idts_one)
    # Generate integral curves for each experimental trace
    log_data = []
    for filename, directory in all_files:
        plot_integral_curves(os.path.join(directory, filename), final_calculator, results_dir, config, log_data)
    # Save optimized parameters to CSV
    optimized_params = pd.DataFrame({
        'Parameter': ['A', 'n', 'E', 'Teq', 'k', 'w', 'C0', 'xf'],
        'Value': [final_A, final_n, final_E, final_Teq, final_k, final_w, final_C0, final_xf]})
    optimized_params.to_csv(os.path.join(results_dir, 'optimized_parameters.csv'), index=False)
    print("\nAll optimization steps completed! Results saved to:", os.path.abspath(results_dir))
# ============================================================
#  Log-space conversion utilities
# ============================================================
def log_to_model_space(params, param_names):
    """Convert parameters from log space back to model space (exp for A, E, C0)"""
    converted_params = list(params)
    log_params = {'A', 'E', 'C0'}
    for i, name in enumerate(param_names):
        if name in log_params:
            if -700 < params[i] < 700:
                converted_params[i] = np.exp(params[i])
            elif params[i] <= -700:
                converted_params[i] = 1e-300
            else:
                converted_params[i] = 1e300
    return converted_params
def model_to_log_space(params, param_names):
    """Convert parameters from model space to log space (ln for A, E, C0)"""
    converted_params = list(params)
    log_params = {'A', 'E', 'C0'}
    for i, name in enumerate(param_names):
        if name in log_params:
            if params[i] > 0:
                converted_params[i] = np.log(params[i])
            else:
                print(f"Warning: Parameter {name} has non-positive value {params[i]}, using small positive value")
                converted_params[i] = np.log(1e-300)
    return converted_params
if __name__ == "__main__":
    main()
