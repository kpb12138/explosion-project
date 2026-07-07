import numpy as np
from skopt import gp_minimize
from scipy.optimize import least_squares
from .plotting_utils import plot_optimization_debug_plots

def genetic_algorithm_optimization(objective_func, bounds, population_size=50, num_generations=100, mutation_rate=0.2):
    """Genetic algorithm optimization"""
    np.random.seed(42)  # Fixing the random population ensures reproducibility.
    population = np.array([[np.random.uniform(low, high) for (low, high) in bounds] for _ in range(population_size)])
    best_fitness = float('inf')
    best_params = None
    best_params_history = []  # Preserve the optimal parameters of each generation.
    best_fitness_history = []  # Preserve the optimal fitness value of each generation

    for gen in range(num_generations):
        if gen % 10 == 0:
            print(f"GA generation {gen}/{num_generations} - Best fitness: {best_fitness:.6f}")

        # Assess fitness
        fitness = np.array([objective_func(ind) for ind in population])
        current_min = np.min(fitness)

        # Elite preservation
        min_idx = np.argmin(fitness)
        current_best_params = population[min_idx].copy()
        
        # Save the optimal parameters and fitness of the current generation
        best_params_history.append(current_best_params)
        best_fitness_history.append(current_min)
        
        if current_min < best_fitness:
            best_fitness = current_min
            best_params = current_best_params.copy()

        # Tournament selection
        selected_indices = []
        for _ in range(population_size):
            candidates = np.random.choice(population_size, size=3, replace=False)
            winner = candidates[np.argmin(fitness[candidates])]
            selected_indices.append(winner)
        selected = population[selected_indices]

        # Simulated Binary Crossover
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = selected[i], selected[i + 1]
            child1, child2 = p1.copy(), p2.copy()
            for j in range(len(bounds)):
                if np.random.rand() < 0.5:
                    beta = np.random.uniform(-0.5, 1.5)
                    child1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    child2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                    # Conduct boundary checks immediately after cross operations.
                    low, high = bounds[j]
                    child1[j] = np.clip(child1[j], low, high)
                    child2[j] = np.clip(child2[j], low, high)
            offspring.extend([child1, child2])

        # Polynomial mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                for j in range(len(bounds)):
                    low, high = bounds[j]
                    delta = np.random.uniform(-0.1, 0.1)
                    offspring[i][j] = np.clip(offspring[i][j] + delta * (high - low), low, high)

        population = np.array(offspring)

    return best_params, best_fitness, best_params_history, best_fitness_history


def run_optimization_algorithms(objective_func, bounds, initial_guess, param_names, step_name, config, results_dir, residuals_func=None):
    """Run optimization algorithms for Steps 1/2/4
    
    Args:
        objective_func: Function returning scalar MSE for GA/GP
        bounds: Parameter bounds
        initial_guess: Initial parameter values
        param_names: Names of parameters
        step_name: Current optimization step name
        config: Configuration dictionary
        results_dir: Directory for saving results
        residuals_func: Optional function returning residuals vector for least_squares.
                       If None, least_squares will use objective_func (less accurate).
    """
    algorithms = config['optimization_settings']['algorithms'][step_name]
    results = []
    n_calls = config['optimization_settings']['n_calls'][step_name]
    tuple_bounds = [(b.low, b.high) for b in bounds]
    
    # Obtain the configurations for the genetic algorithm and the least squares method
    use_genetic = algorithms.get('genetic_algorithm', 0)
    use_least_squares = algorithms.get('least_squares', 0)
    use_gp_minimize = algorithms.get('gp_minimize', 0)
    
    # The initial guess currently used (which may be updated by the results of the genetic algorithm)
    current_initial_guess = initial_guess.copy()
    
    # Results of the genetic algorithm (if executed)
    ga_result = None

    # Genetic Algorithm - Operates Independently or Provides Initial Values for the Least Squares Method
    if use_genetic:
        print(f"Running genetic_algorithm for {step_name} optimization...")
        population_size=config['optimization_settings']['genetic_algorithm']['population_size']
        num_generations=config['optimization_settings']['genetic_algorithm']['num_generations']
        mutation_rate=config['optimization_settings']['genetic_algorithm']['mutation_rate']
        step_results = genetic_algorithm_optimization(
            objective_func, tuple_bounds, population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate
        )
        # The unpacked results include historical data.
        best_params, best_fitness, params_history, fitness_history = step_results
        
        # Calculate the real-time optimal value at each step.
        realtime_best_history = []
        realtime_best_residuals = []
        best_val = float('inf')
        best_params_so_far = None
        
        for params, residual in zip(params_history, fitness_history):
            if residual < best_val:
                best_val = residual
                best_params_so_far = params.copy()
            realtime_best_history.append(best_params_so_far)
            realtime_best_residuals.append(best_val)
        
        ga_result = {
            'algorithm': 'genetic_algorithm',
            'params': best_params,
            'fun': best_fitness,
            'history': realtime_best_history,
            'residuals': realtime_best_residuals
        }
        results.append(ga_result)
        print(f"genetic_algorithm completed with final objective: {best_fitness:.6f}")
        print(f"Using genetic algorithm result as initial guess for least_squares in {step_name} optimization")
        current_initial_guess = best_params.copy()

    # GP minimize - Operate independently
    if use_gp_minimize:
        print(f"Running gp_minimize for {step_name} optimization...")

        def gp_callback(res):
            try:
                current_iter = res.n_calls
            except AttributeError:
                current_iter = len(res.x_iters)
            if current_iter % 5 == 0 and current_iter > 0:
                print(f"GP Iteration {current_iter}/{n_calls} - Current best: {res.fun:.6f}")

        # Ensure x0 is a list, not a numpy array, to avoid comparison issues in skopt
        x0_for_gp = current_initial_guess.tolist() if hasattr(current_initial_guess, 'tolist') else list(current_initial_guess)
        
        step_results = gp_minimize(
            objective_func, tuple_bounds, x0=x0_for_gp, n_calls=n_calls,
            random_state=42, verbose=False, callback=gp_callback
        )
        
        # Calculate the real-time optimal value at each step.
        realtime_best_history = []
        realtime_best_residuals = []
        best_val = float('inf')
        best_params = None
        
        for i, (params, residual) in enumerate(zip(step_results.x_iters, step_results.func_vals)):
            if residual < best_val:
                best_val = residual
                best_params = params.copy()
            realtime_best_history.append(best_params)
            realtime_best_residuals.append(best_val)
        
        results.append({
            'algorithm': 'gp_minimize',
            'params': step_results.x,
            'fun': step_results.fun,
            'history': realtime_best_history,
            'residuals': realtime_best_residuals
        })
        # Update current_initial_guess with the optimal parameters found by gp_minimize.
        current_initial_guess = best_params.copy()
        print(f"gp_minimize completed with final objective: {step_results.fun:.6f}")
        print(f"Updated current_initial_guess with gp_minimize result in {step_name} optimization")

    # Least Squares Method - Results Using Initial Guesses or Genetic Algorithms
    if use_least_squares:
        source = 'gp_minimize' if use_gp_minimize else ('genetic algorithm' if use_genetic else 'config')
        print(f"Running least_squares ... with initial guess from {source}")
        ls_lower = [b.low for b in bounds]
        ls_upper = [b.high for b in bounds]
        
        # Customize a callback to collect iteration history.
        iter_history = [current_initial_guess.copy()]
        residual_history = [objective_func(current_initial_guess)]
        
        def ls_callback(x):
            # The callback of least_squares directly receives the x parameter.
            current_residual = objective_func(x)
            iter_history.append(x.copy())
            residual_history.append(current_residual)
            return False
        
        # Use residuals_func if available, otherwise fall back to objective_func
        if residuals_func is not None:
            ls_func = residuals_func
        else:
            ls_func = lambda p: np.array([objective_func(p)])
        
        step_results = least_squares(
            ls_func, current_initial_guess,
            bounds=(ls_lower, ls_upper), verbose=2, ftol=1e-8, callback=ls_callback
        )
        
        # Calculate the real-time optimal value at each step.
        realtime_best_history = []
        realtime_best_residuals = []
        best_val = float('inf')
        best_params = None
        
        for params, residual in zip(iter_history, residual_history):
            if residual < best_val:
                best_val = residual
                best_params = params.copy()
            realtime_best_history.append(best_params)
            realtime_best_residuals.append(best_val)
        
        # If joint optimization is required, integrate histories
        if use_genetic:
            combined_history = ga_result['history'] + realtime_best_history
            combined_residuals = ga_result['residuals'] + realtime_best_residuals
            
            # Recalculate the merged real-time optimal value.
            merged_realtime_best_history = []
            merged_realtime_best_residuals = []
            best_val = float('inf')
            best_params = None
            
            for params, residual in zip(combined_history, combined_residuals):
                if residual < best_val:
                    best_val = residual
                    best_params = params.copy()
                merged_realtime_best_history.append(best_params)
                merged_realtime_best_residuals.append(best_val)
            
            # Calculate MSE from LS result
            final_mse = objective_func(step_results.x)
            ls_result = {
                'algorithm': 'genetic_algorithm+least_squares',
                'params': step_results.x,
                'fun': final_mse,
                'history': merged_realtime_best_history,
                'residuals': merged_realtime_best_residuals
            }
        else:
            final_mse = objective_func(step_results.x)
            ls_result = {
                'algorithm': 'least_squares',
                'params': step_results.x,
                'fun': final_mse,
                'history': realtime_best_history,
                'residuals': realtime_best_residuals
            }
        
        # Replace or add results
        if use_genetic and 'genetic_algorithm+least_squares' not in [r['algorithm'] for r in results]:
            results.append(ls_result)
        else:
            # Add or update least squares results
            existing_ls = next((i for i, r in enumerate(results) if r['algorithm'] in ['least_squares', 'genetic_algorithm+least_squares']), None)
            if existing_ls is not None:
                results[existing_ls] = ls_result
            else:
                results.append(ls_result)
        
        print(f"least_squares completed with final objective: {final_mse:.6f}")

    if not results:
        print(f"No optimization algorithms selected for {step_name}")
        return initial_guess, [], []

    best_result = min(results, key=lambda x: x['fun'])
    print(f"Best {step_name} result from {best_result['algorithm']}: fun={best_result['fun']:.6f}")
    corrected_params = best_result['params'].copy()
    
    # Conduct boundary checks on the optimization results and apply truncation.
    for i, (low, high) in enumerate(tuple_bounds):
        if corrected_params[i] < low:
            print(f"Warning: Optimized parameter {param_names[i]} ({corrected_params[i]}) is below lower bound ({low}), clipping to lower bound")
            corrected_params[i] = low
        elif corrected_params[i] > high:
            print(f"Warning: Optimized parameter {param_names[i]} ({corrected_params[i]}) is above upper bound ({high}), clipping to upper bound")
            corrected_params[i] = high
    
    plot_optimization_debug_plots(
        best_result['history'], best_result['residuals'], param_names, step_name, results_dir
    )
    return corrected_params, best_result['history'], best_result['residuals'], best_result['algorithm']
