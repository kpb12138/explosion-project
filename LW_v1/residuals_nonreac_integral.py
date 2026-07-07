"""
residuals_nonreac_integral.py - Residuals and objective functions
================================================================
Single-stage CH3NO2 adaptation (based on version_2026_v3).
Changes from v3:
  - Removed objective_1st / calculate_residuals_1st (single-stage fuel)
  - Renamed calculate_residuals_hi to calculate_residuals_single
  - Kept objective_total / residuals_vector_total (Step 4)
  - Kept DeltaP_residuals (Step 3, pressure rise optimization)
Core concept: LW integral criterion integral(1/IDT)*dt = 1 at ignition.
Use log(integral) as the residual; residual -> 0 means integral -> 1.
"""
import numpy as np
from idt_calculator_nonreac import IDTCalculator, integrate_idt
DEFAULT_TIME_COL = 'Time(msec)'
DEFAULT_TEMP_COL = ' Temperature(K)'
DEFAULT_PRES_COL = ' Pressure(bar)'
def _get_column_names(config=None):
    """Read column names from JSON config or use defaults"""
    if config is not None and 'data_columns' in config:
        time_col = config['data_columns'].get('time', DEFAULT_TIME_COL)
        temp_col = config['data_columns'].get('temperature', DEFAULT_TEMP_COL)
        pres_col = config['data_columns'].get('pressure', DEFAULT_PRES_COL)
    else:
        time_col = DEFAULT_TIME_COL
        temp_col = DEFAULT_TEMP_COL
        pres_col = DEFAULT_PRES_COL
    return time_col, temp_col, pres_col
# ============================================================
#  Step 1 (was Step 2 in v3): Single-stage IDT residual
#  Optimizes (A, n, E) for IDT = A * p^n * exp(E/T)
# ============================================================
def calculate_residuals_single(dfs, idts, params, fixed_params, config=None):
    """Calculate LW integral residuals for single-stage IDT.
    For each trace:
      1. Compute instantaneous IDT = A * p^n * exp(E/T) using params
      2. Numerically integrate 1/IDT over time
      3. Take log(integral[labelled_idt]) as residual
    Parameters:
        dfs         : list of experimental trace DataFrames
        idts        : labelled ignition times (with t_com included)
        params      : (A, n, E) - parameters to optimize
        fixed_params: other fixed parameters (Teq, k, w, C0, xf)
        config      : JSON config dictionary
    Returns:
        residuals : array of log(integral) residuals
        weights   : equal-weight array"""
    time_col, temp_col, pres_col = _get_column_names(config)
    A, n, E = params
    Teq, k, w, C0, xf = fixed_params
    residuals = []
    for df, idt_with_tcom in zip(dfs, idts):
        time = df[time_col].values
        temperature = df[temp_col].values
        pressure = df[pres_col].values
        calculator = IDTCalculator(A, n, E, Teq, k, w, C0, xf)
        idt_single = calculator.calculate_idt(pressure, temperature)
        integral_single = integrate_idt(time, idt_single)
        # Find integral at labelled IDT time
        idx = np.argmin(np.abs(time - idt_with_tcom))
        residual = np.log(integral_single[idx])
        residuals.append(residual)
    # Equal weights
    n_pts = len(residuals)
    weights = np.ones(n_pts) / n_pts
    return np.array(residuals), weights
# ============================================================
#  Step 3: Pressure rise DeltaP residual
#  Optimizes (Teq, k, w)
# ============================================================
def DeltaP_residuals(params, T_list, P_list, dp_actual):
    """Compute DeltaP residuals for Step 3 optimization.
    Parameters:
        params    : (Teq, k, w)
        T_list    : initial temperature list
        P_list    : initial pressure list
        dp_actual : measured pressure rise list
    Returns:
        residuals : Dp_predicted - Dp_actual"""
    Teq_opt, k_opt, w_opt = params
    try:
        D_Tcf = w_opt * (T_list - Teq_opt * np.power(P_list, k_opt))
        Tcf = T_list + 0.5 * (D_Tcf + np.sqrt(np.maximum(D_Tcf ** 2, 0.0)))
        pcf = Tcf / T_list * P_list
        D_p_pred = pcf - P_list
        residuals = D_p_pred - dp_actual
        return residuals if np.isfinite(residuals).all() else np.full_like(residuals, np.nan)
    except Exception as e:
        print(f"residuals calculation error: {e}, params: {params}")
        return np.full(len(dp_actual), np.nan)
# ============================================================
#  Step 4: Total IDT residual (all-parameter optimization)
#  Optimizes all 8 params: (A, n, E, Teq, k, w, C0, xf)
# ============================================================
def calculate_residuals_total(dfs, params, original_idts, t_com=None, config=None):
    """Compute total IDT log-residuals for all-parameter optimization.
    For single-stage CH3NO2, total IDT = single-stage IDT.
    Uses LW integral: find time where integral(1/IDT) = 1."""
    A, n, E, Teq, k, w, C0, xf = params
    time_col, temp_col, pres_col = _get_column_names(config)
    residuals_total = []
    for df, idt_original in zip(dfs, original_idts):
        time = df[time_col].values
        temperature = df[temp_col].values
        pressure = df[pres_col].values
        calculator = IDTCalculator(A, n, E, Teq, k, w, C0, xf)
        idt_single = calculator.calculate_idt(pressure, temperature)
        integral_single = integrate_idt(time, idt_single)
        # Find time where integral = 1 -> predicted IDT
        idx = np.argmin(np.abs(integral_single - 1.0))
        predicted_idt_with_tcom = time[idx]
        predicted_idt = predicted_idt_with_tcom - t_com
        # Log residual
        residual = np.log(predicted_idt + 1e-8) - np.log(idt_original + 1e-8)
        residuals_total.append(residual)
    # Equal weights
    n_pts = len(residuals_total)
    weights = np.ones(n_pts) / n_pts
    return np.array(residuals_total), weights
# ============================================================
#  Objective functions (called by optimizer)
# ============================================================
def objective_single(params, dfs, idts, fixed_params, config=None):
    """Single-stage IDT objective (original Step 2). Returns weighted MSE."""
    try:
        residuals, weights = calculate_residuals_single(
            dfs, idts, params, fixed_params, config=config)
        weighted_squared = np.multiply(residuals ** 2, weights)
        obj = np.sum(weighted_squared)
        return obj if np.isfinite(obj) else 1e10
    except Exception as e:
        print(f"Error in objective_single: {e}")
        return 1e10
def objective_total(params, dfs, original_idts, t_com, config=None):
    """All-parameter IDT objective (Step 4). Returns weighted MSE."""
    try:
        residuals, weights = calculate_residuals_total(
            dfs, params, original_idts, t_com, config=config)
        weighted_squared = np.multiply(residuals ** 2, weights)
        obj = np.sum(weighted_squared)
        return obj if np.isfinite(obj) else 1e10
    except Exception as e:
        print(f"Error in objective_total: {e}")
        return 1e10
def residuals_vector_total(params, dfs, original_idts, t_com, config=None):
    """Return weighted residual vector for scipy.optimize.least_squares.
    Guarantees sum(residuals_vector^2) = MSE, consistent with objective_total."""
    try:
        residuals, weights = calculate_residuals_total(
            dfs, params, original_idts, t_com, config=config)
        weighted = residuals * np.sqrt(weights)
        return weighted if np.all(np.isfinite(weighted)) else np.ones(len(residuals)) * 1e5
    except Exception as e:
        print(f"Error in residuals_vector_total: {e}")
        return np.ones(1) * 1e5
