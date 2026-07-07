import os
import numpy as np
from skopt.space import Real

# Import necessary modules
from src.config_utils import load_config, parse_parameter_spaces, validate_config, validate_initial_guess
from src.boundary_utils import check_boundary, log_boundary_header, log_optimization_step
from src.optimization_utils import run_optimization_algorithms
from src.plotting_utils import plot_dp_comparison, plot_integral_curves, plot_predicted_idts
from src.idt_calculator_nonreac import IDTCalculator
from src.residuals_nonreac_integral import objective_1st, objective_hi, objective_total, residuals_vector_total, DeltaP_residuals

def main():
    # create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # create optimization log file
    optimization_log_file = os.path.join(results_dir, 'optimization.log')
    log_boundary_header(optimization_log_file)
    
    # read configuration file
    config = load_config('Input.json')
    all_files, idts_1st, idts_total, idts_one, idts_ntc = validate_config(config)
    
    # parse parameter spaces
    param_spaces = parse_parameter_spaces(config)
    
    # read ignition timing data
    t_com = config['timing_parameters']['t_com'] # compression time in ms;
    idts_1st_with_tcom = [x + t_com for x in idts_1st]
    idts_one_with_tcom = [x + t_com for x in idts_one] # at least two data should be provided here;
    idts_combined = idts_total + idts_one + idts_ntc
    idts_combined_with_tcom = [x + t_com for x in idts_combined]
    original_idts_combined = idts_combined.copy()
    
    # read initial parameters
    initial_params = config['initial_parameters']
    A1, n1, E1 = initial_params['A1'], initial_params['n1'], initial_params['E1']
    Ah, nh, Eh = initial_params['Ah'], initial_params['nh'], initial_params['Eh']
    Teq, k, w = initial_params['Teq'], initial_params['k'], initial_params['w']
    C0, xf = initial_params['C0'], initial_params['xf']
    
    # load input files
    dfs = []

    import pandas as pd
    for filename, directory in all_files:
        dfs.append(pd.read_csv(os.path.join(directory, filename)))
    
    # read adjust factors
    step4_factors = config['optimization_settings'].get('adjust_factors', {})
    A1_factor = step4_factors.get('A1_factor', 0.001)
    n1_factor = step4_factors.get('n1_factor', 0.001)
    E1_factor = step4_factors.get('E1_factor', 0.001)
    Ah_factor = step4_factors.get('Ah_factor', 0.01)
    nh_factor = step4_factors.get('nh_factor', 0.01)
    Eh_factor = step4_factors.get('Eh_factor', 0.01)
    Teq_factor = step4_factors.get('Teq_factor', 0.02)
    k_factor = step4_factors.get('k_factor', 0.5)
    w_factor = step4_factors.get('w_factor', 0.5)
    
    # Step 1: optimize A1, n1, E1 using first-stage IDTs with Arrhenius behavior
    print("\nStarting Step 1 optimization: A1, n1, E1")
    # check if idts_1st is empty
    if not idts_1st:
        print("Warning: Idt_1st is empty. Skipping Step 1 optimization.")
        optimized_A1, optimized_n1, optimized_E1 = A1, n1, E1
        A1_range = (param_spaces['step1'][0].low, param_spaces['step1'][0].high)
        n1_range = (param_spaces['step1'][1].low, param_spaces['step1'][1].high)
        E1_range = (param_spaces['step1'][2].low, param_spaces['step1'][2].high)
    else:
        original_idts_1st = idts_1st.copy()
        
        # Create a logarithmic space objective function
        def obj_step1_log(p_log):
            # Convert parameters from logarithmic space back to model space
            p_model = log_to_model_space(p_log, [p.name for p in param_spaces['step1']])
            return objective_1st(p_model, dfs[:len(idts_1st)], idts_1st_with_tcom,
                                (Ah, nh, Eh, Teq, k, w, C0, xf), original_idts_1st, config)
        
        # Convert the initial guess to logarithmic space
        initial_guess_step1 = [A1, n1, E1]
        initial_guess_step1_log = model_to_log_space(initial_guess_step1, [p.name for p in param_spaces['step1']])
        initial_guess_step1_log = validate_initial_guess(initial_guess_step1_log,
                                                [(np.log(b.low) if b.name in ['A1', 'E1'] and b.low > 0 and b.high > 0 else b.low,
                                                  np.log(b.high) if b.name in ['A1', 'E1'] and b.low > 0 and b.high > 0 else b.high) 
                                                 for b in param_spaces['step1']],
                                                [p.name for p in param_spaces['step1']],
                                                'step1')
        
        # Create a parameter space in logarithmic space
        param_spaces_log_step1 = []
        for param in param_spaces['step1']:
            if param.name in ['A1', 'E1'] and param.low > 0 and param.high > 0:
                # Use logarithmic space for A1 and E1
                param_spaces_log_step1.append(Real(np.log(param.low), np.log(param.high), name=param.name))
            else:
                # Use original space for n1
                param_spaces_log_step1.append(param)
        
        # Run optimization algorithm
        optimized_params_step1_log, history_step1, residuals_step1, algorithm_step1 = run_optimization_algorithms(
            obj_step1_log, param_spaces_log_step1, initial_guess_step1_log,
            [p.name for p in param_spaces['step1']], 'step1', config, results_dir
        )
        
        # Convert the optimization results back to the model space
        optimized_params_step1 = log_to_model_space(optimized_params_step1_log, [p.name for p in param_spaces['step1']])
        optimized_A1, optimized_n1, optimized_E1 = optimized_params_step1
        
        # Record optimization results - using the final residual value as the target value
        final_objective = residuals_step1[-1] if residuals_step1 else 0.0
        log_optimization_step(optimized_params_step1,
                             [p.name for p in param_spaces['step1']],
                             "Step 1", final_objective, algorithm_step1, optimization_log_file)
        
        # Check the boundary
        check_boundary(optimized_params_step1,
                  [(b.low, b.high) for b in param_spaces['step1']],
                  [p.name for p in param_spaces['step1']],
                  "Step 1", optimization_log_file)

        # For parameters optimized in logarithmic space (A1, E1, Ah, Eh), apply adjustments in log space
        # For parameters optimized in linear space (n1, nh, Teq, k, w), apply adjustments in linear space
        A1_range = (optimized_A1 * np.exp(-A1_factor), optimized_A1 * np.exp(A1_factor))
        n1_range = (optimized_n1 * (1 + n1_factor), optimized_n1 * (1 - n1_factor))
        E1_range = (optimized_E1 * np.exp(-E1_factor), optimized_E1 * np.exp(E1_factor))

    # Step 2: optimize Ah, nh, Eh using high-temperature IDT with Arrhenius behavior
    # check if idts_one is empty
    if not idts_one:
        print("Warning: Idt_one is empty. Skipping Step 2 optimization.")
        optimized_Ah, optimized_nh, optimized_Eh = Ah, nh, Eh
        Ah_range = (param_spaces['step2'][0].low, param_spaces['step2'][0].high)
        nh_range = (param_spaces['step2'][1].low, param_spaces['step2'][1].high)
        Eh_range = (param_spaces['step2'][2].low, param_spaces['step2'][2].high)
    else:
        print("\nStarting Step 2 optimization: Ah, nh, Eh")
        # Create a logarithmic space objective function
        def obj_step2_log(p_log):
            # Convert parameters from logarithmic space back to model space
            p_model = log_to_model_space(p_log, [p.name for p in param_spaces['step2']])
            return objective_hi(p_model, dfs[len(idts_1st):len(idts_1st)+len(idts_one)], idts_one_with_tcom,
                              (optimized_A1, optimized_n1, optimized_E1, Teq, k, w, C0, xf), config)
        
        # Convert the initial guess to logarithmic space
        initial_guess_step2 = [Ah, nh, Eh]
        initial_guess_step2_log = model_to_log_space(initial_guess_step2, [p.name for p in param_spaces['step2']])
        initial_guess_step2_log = validate_initial_guess(initial_guess_step2_log,
                                                     [(np.log(b.low) if b.name in ['Ah', 'Eh'] and b.low > 0 and b.high > 0 else b.low,
                                                       np.log(b.high) if b.name in ['Ah', 'Eh'] and b.low > 0 and b.high > 0 else b.high) 
                                                      for b in param_spaces['step2']],
                                                     [p.name for p in param_spaces['step2']],
                                                     'step2')
        
        # Create a parameter space in logarithmic space
        param_spaces_log_step2 = []
        for param in param_spaces['step2']:
            if param.name in ['Ah', 'Eh'] and param.low > 0 and param.high > 0:
                # Use logarithmic space for Ah and Eh
                param_spaces_log_step2.append(Real(np.log(param.low), np.log(param.high), name=param.name))
            else:
                # Use original space for nh
                param_spaces_log_step2.append(param)
        
        # Run optimization algorithm
        optimized_params_step2_log, history_step2, residuals_step2, algorithm_step2 = run_optimization_algorithms(
            obj_step2_log, param_spaces_log_step2, initial_guess_step2_log,
            [p.name for p in param_spaces['step2']], 'step2', config, results_dir
        )
        
        # Convert the optimization results back to the model space
        optimized_params_step2 = log_to_model_space(optimized_params_step2_log, [p.name for p in param_spaces['step2']])
        optimized_Ah, optimized_nh, optimized_Eh = optimized_params_step2
        
        # Record optimization results
        final_objective = residuals_step2[-1] if residuals_step2 else 0.0
        log_optimization_step(optimized_params_step2,
                             [p.name for p in param_spaces['step2']],
                             "Step 2", final_objective, algorithm_step2, optimization_log_file)
        
        # Check the boundary
        check_boundary(optimized_params_step2,
                      [(b.low, b.high) for b in param_spaces['step2']],
                      [p.name for p in param_spaces['step2']],
                      "Step 2", optimization_log_file)
        
        # For parameters optimized in logarithmic space (Ah, Eh), apply adjustments in log space
        Ah_range = (optimized_Ah * np.exp(-Ah_factor), optimized_Ah * np.exp(Ah_factor))
        nh_range = (optimized_nh * (1 + nh_factor), optimized_nh * (1 - nh_factor))
        Eh_range = (optimized_Eh * np.exp(-Eh_factor), optimized_Eh * np.exp(Eh_factor))

    # Step 3: optimize Teq, k, w using temperature, pressure and correspoding Dp
    dp_data = config['file_specifications']['dp_optimization']
    T_list = np.array(dp_data['initial_temperature'])
    P_list = np.array(dp_data['initial_pressure'])
    dp_actual = np.array(dp_data['D_p_actual'])
    # check if dp_actual is empty
    if dp_actual.size == 0:
        print("Warning: dp_actual is empty. Please check the input file.")
        optimized_Teq, optimized_k, optimized_w = Teq, k, w
        Teq_range = (param_spaces['step3'][0].low, param_spaces['step3'][0].high)
        k_range = (param_spaces['step3'][1].low, param_spaces['step3'][1].high)
        w_range = (param_spaces['step3'][2].low, param_spaces['step3'][2].high)
    else:
        print("\nStarting Step 3 optimization: Teq, k, w")
        # check if T_list, P_list, dp_actual have the same length
        if len(T_list) != len(P_list) or len(T_list) != len(dp_actual):
            print(f"Error: Data length mismatch in Step 3: T={len(T_list)}, P={len(P_list)}, D_p_actual={len(dp_actual)}")
            exit(1)
        
        # define objective function for Step 3
        def obj_step3(p):
            residuals = DeltaP_residuals(p, T_list, P_list, dp_actual)
            return np.mean(residuals ** 2) if np.isfinite(residuals).all() else 1e10
        
        # initial guess and validation
        initial_guess_step3 = [ Teq, max(k, 1e-6), w]  # Ensure k is positive initially
        initial_guess_step3 = validate_initial_guess(initial_guess_step3,
                                                    [(b.low, b.high) for b in param_spaces['step3']],
                                                    [p.name for p in param_spaces['step3']],
                                                    'step3')
        
        # Run optimization algorithm
        optimized_params_step3, history_step3, residuals_step3, algorithm_step3 = run_optimization_algorithms(
            obj_step3, param_spaces['step3'], initial_guess_step3,
            [p.name for p in param_spaces['step3']], 'step3', config, results_dir
        )
        
        optimized_Teq, optimized_k, optimized_w = optimized_params_step3
        
        # Record optimization results
        final_objective = residuals_step3[-1] if residuals_step3 else 0.0
        log_optimization_step(optimized_params_step3,
                             [p.name for p in param_spaces['step3']],
                             "Step 3", final_objective, algorithm_step3, optimization_log_file)
        
        # Check the boundary
        check_boundary(optimized_params_step3,
                      [(b.low, b.high) for b in param_spaces['step3']],
                      [p.name for p in param_spaces['step3']],
                      "Step 3", optimization_log_file)
        
        # plot Dp comparison
        try:
            # calculate Dp predicted with optimized parameters
            D_Tcf = optimized_w * (T_list - optimized_Teq * np.power(P_list, optimized_k))
            Tcf = T_list + 0.5 * (D_Tcf + np.sqrt(np.maximum(D_Tcf ** 2, 0)))
            pcf = Tcf / T_list * P_list
            dp_predicted = pcf - P_list
            plot_dp_comparison(T_list, dp_actual, dp_predicted, results_dir)
        except Exception as e:
            print(f"Failed to plot Dp comparison: {e}")
        
        Teq_range = (optimized_Teq * (1 - Teq_factor), optimized_Teq * (1 + Teq_factor))
        k_range = (optimized_k * (1 - k_factor), optimized_k * (1 + k_factor))
        w_range = (optimized_w * (1 + w_factor), optimized_w * (1 - w_factor))  # w is negative
    
    # Step 4: optimize all parameters
    # Correctly obtain the boundary values of C0 and xf
    C0_low, C0_high = param_spaces['step4'][0].low, param_spaces['step4'][0].high
    xf_low, xf_high = param_spaces['step4'][1].low, param_spaces['step4'][1].high
    # calculate the range for A1, n1, E1, Ah, nh, Eh 
    print("\nStarting Step 4 optimization: all parameters")
    # create the full parameter space for Step 4
    step4_full_space = [
        Real(A1_range[0], A1_range[1], name='A1'),
        Real(n1_range[0], n1_range[1], name='n1'),
        Real(E1_range[0], E1_range[1], name='E1'),
        Real(Ah_range[0], Ah_range[1], name='Ah'),
        Real(nh_range[0], nh_range[1], name='nh'),
        Real(Eh_range[0], Eh_range[1], name='Eh'),
        Real(Teq_range[0], Teq_range[1], name='Teq'),
        Real(k_range[0], k_range[1], name='k'),
        Real(w_range[0], w_range[1], name='w'),
        Real(C0_low, C0_high, name='C0'),
        Real(xf_low, xf_high, name='xf')
    ]

    # Create a logarithmic space objective function
    def obj_step4_log(p_log):
        # Convert parameters from logarithmic space back to model space
        p_model = log_to_model_space(p_log, [p.name for p in step4_full_space])
        return objective_total(p_model, dfs, original_idts_combined, t_com, config)
    
    # Create residuals vector function for least_squares
    def residuals_step4_log(p_log):
        p_model = log_to_model_space(p_log, [p.name for p in step4_full_space])
        return residuals_vector_total(p_model, dfs, original_idts_combined, t_com, config)

    # prepare optimization parameters for Step 4
    initial_guess_step4 = [optimized_A1, optimized_n1, optimized_E1, optimized_Ah, optimized_nh, optimized_Eh, optimized_Teq, optimized_k, optimized_w, C0, xf]
    param_names_step4 = [s.name for s in step4_full_space]
    bounds_step4 = [(s.low, s.high) for s in step4_full_space]

    # Convert the initial guess to logarithmic space
    initial_guess_step4_log = model_to_log_space(initial_guess_step4, param_names_step4)
    initial_guess_step4_log = validate_initial_guess(initial_guess_step4_log,
                                                    [(np.log(b[0]) if b[0] > 0 and b[1] > 0 else b[0],
                                                    np.log(b[1]) if b[0] > 0 and b[1] > 0 else b[1]) 
                                                    if name in ['A1', 'Ah', 'E1', 'Eh', 'C0'] else b 
                                                    for name, b in zip(param_names_step4, bounds_step4)],
                                                    param_names_step4,
                                                    'step4')
    
    # Create a parameter space in logarithmic space
    param_spaces_log_step4 = []
    for param in step4_full_space:
        if param.name in ['A1', 'Ah', 'E1', 'Eh', 'C0'] and param.low > 0 and param.high > 0:
            # Use a logarithmic space for these parameters
            param_spaces_log_step4.append(Real(np.log(param.low), np.log(param.high), name=param.name))
        else:
            # Use original space for other parameters
            param_spaces_log_step4.append(param)
    
    # run optimization algorithms for Step 4
    optimized_params_step4_log, history_step4, residuals_step4, algorithm_step4 = run_optimization_algorithms(
        obj_step4_log, param_spaces_log_step4, initial_guess_step4_log,
        param_names_step4, 'step4', config, results_dir, residuals_func=residuals_step4_log
    )
    
    # Convert the optimization results back to the model space
    optimized_params_step4 = log_to_model_space(optimized_params_step4_log, param_names_step4)
    final_A1, final_n1, final_E1, final_Ah, final_nh, final_Eh, final_Teq, final_k, final_w, final_C0, final_xf = optimized_params_step4
    
    # Record the optimization results
    final_objective = residuals_step4[-1] if residuals_step4 else 0.0
    log_optimization_step(optimized_params_step4,
                            param_names_step4,
                            "Step 4", final_objective, algorithm_step4, optimization_log_file)
    
    # boundary check for Step 4
    check_boundary(optimized_params_step4, bounds_step4, param_names_step4, "Step 4", optimization_log_file)

    # Generate integral curves and final results

    final_calculator = IDTCalculator(
        final_A1, final_n1, final_E1, final_Ah, final_nh, final_Eh,
        final_Teq, final_k, final_w, final_C0, final_xf
    )
    
    # plot the predicted 1st IDT (the time integral_1st = 1 - t_com) with original 1st IDTs using the final calculator;
    # plot the predicted total IDT (the time integral_total = 1 - t_com) with original total IDTs using the final calculator;
    # if idts_1st and idts_combined:
    plot_predicted_idts(dfs, idts_1st, idts_combined, t_com, final_calculator, results_dir, config, idts_one)
    
    log_data = []
    for filename, directory in all_files:
        plot_integral_curves(os.path.join(directory, filename), final_calculator, results_dir, config, log_data)
            
    # Save optimized results
    optimized_params = pd.DataFrame({
        'Parameter': ['A1', 'n1', 'E1', 'Ah', 'nh', 'Eh', 'Teq', 'k', 'w', 'C0', 'xf'],
        'Value': [final_A1, final_n1, final_E1, final_Ah, final_nh, final_Eh,
                 final_Teq, final_k, final_w, final_C0, final_xf]
    })
    optimized_params.to_csv(os.path.join(results_dir, 'optimized_parameters.csv'), index=False)
    print("\nAll optimization steps completed! Results saved to:", os.path.abspath(results_dir))

def log_to_model_space(params, param_names):
    """Convert parameters from the logarithmic space back to the model space"""
    converted_params = params.copy()
    log_params = {'A1', 'Ah', 'E1', 'Eh', 'C0'}
    
    for i, name in enumerate(param_names):
        if name in log_params:
            # Ensure that parameter values are valid to prevent numerical issues.
            if params[i] > -700 and params[i] < 700:  # Avoid exponential overflow
                converted_params[i] = np.exp(params[i])
            else:
                print(f"Warning: Parameter {name} has extreme log value {params[i]}, using boundary value")
                # Apply boundary value handling
                if params[i] <= -700:
                    converted_params[i] = 1e-300  # An extremely small value approaching zero
                else:
                    converted_params[i] = 1e300   # Maximum value
    
    return converted_params


def model_to_log_space(params, param_names):
    """Convert parameters from the model space to the logarithmic space"""
    converted_params = params.copy()
    log_params = {'A1', 'Ah', 'E1', 'Eh', 'C0'}
    
    for i, name in enumerate(param_names):
        if name in log_params:
            # Ensure that the parameter values are positive to avoid log(0) or log(negative numbers).
            if params[i] > 0:
                converted_params[i] = np.log(params[i])
            else:
                print(f"Warning: Parameter {name} has non-positive value {params[i]}, using small positive value")
                converted_params[i] = np.log(1e-300)  # The logarithm using an extremely small positive value close to zero
    
    return converted_params

if __name__ == "__main__":
    main()
