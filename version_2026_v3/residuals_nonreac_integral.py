import numpy as np
from idt_calculator_nonreac import IDTCalculator, integrate_idt

# Default column names
DEFAULT_TIME_COL = 'Time(msec)'
DEFAULT_TEMP_COL = ' Temperature(K)'
DEFAULT_PRES_COL = ' Pressure(bar)'


def DeltaP_residuals(params, T_list, P_list, dp_actual):
    """Calculate residuals of DeltaP for Step 3 optimization"""
    Teq_opt, k_opt, w_opt = params
    try:
        D_Tcf = w_opt * (T_list - Teq_opt * np.power(P_list, k_opt))
        Tcf = T_list + 0.5 * (D_Tcf + np.sqrt(np.maximum(D_Tcf ** 2, 0)))
        pcf = Tcf / T_list * P_list
        D_p_pred = pcf - P_list
        residuals = D_p_pred - dp_actual
        return residuals if np.isfinite(residuals).all() else np.full_like(residuals, np.nan)
    except Exception as e:
        print(f"residuals calculation error: {e}, params: {params}")
        return np.full(len(dp_actual), np.nan)

def _get_column_names(config=None):
    """Obtain Column Name Configuration"""
    if config is not None and 'data_columns' in config:
        time_col = config['data_columns'].get('time', DEFAULT_TIME_COL)
        temp_col = config['data_columns'].get('temperature', DEFAULT_TEMP_COL)
        pres_col = config['data_columns'].get('pressure', DEFAULT_PRES_COL)
    else:
        time_col = DEFAULT_TIME_COL
        temp_col = DEFAULT_TEMP_COL
        pres_col = DEFAULT_PRES_COL
    return time_col, temp_col, pres_col

def calculate_residuals_1st(dfs, idts_1st, params, fixed_params, original_idts, apply_weights=False, config=None):
    # Obtain Column Name Configuration
    time_col, temp_col, pres_col = _get_column_names(config)

    A1, n1, E1 = params
    Ah, nh, Eh, Teq, k, w, C0, xf = fixed_params
    residuals_1st = []
    
    for df, idt_with_tcom in zip(dfs, idts_1st):
        time = df[time_col].values
        temperature = df[temp_col].values
        pressure = df[pres_col].values
        
        calculator = IDTCalculator(A1, n1, E1, Ah, nh, Eh, Teq, k, w, C0, xf)
        idt_1st = calculator.calculate_idt_1st(pressure, temperature)
        integral_1st = integrate_idt(time, idt_1st)
        
        # Find integral when time reach idt_with_tcom
        idx_1st = np.argmin(np.abs(time - idt_with_tcom))
                
        # Calculate the residual of the integral
        residual_1st = np.log(integral_1st[idx_1st])
        
        residuals_1st.append(residual_1st)
    
    if apply_weights:
        # Calculate weights - use the relative magnitude of the logarithmic values of the original IDT as the basis for weighting
        original_idts_array = np.array(original_idts)
        # Avoid taking the logarithm of zero or negative numbers by adding a small constant to ensure non-zero values.
        log_weights = np.log(original_idts_array + 1e-8)
        # Ensure that the values in log_weights are positive, setting any negative values to a small positive number close to zero.
        log_weights = np.where(log_weights < 0, 1e-8, log_weights)
        # Normalize the weights so that the sum of all weights equals 1.
        weights = log_weights / np.sum(log_weights)
    else:
        # Return equal weights, with the sum of all weights equal to 1.
        n = len(residuals_1st)
        weights = np.ones(n)
    
    return np.array(residuals_1st), weights


def calculate_residuals_hi(dfs, idts_one, params, fixed_params, config=None):
    # Obtain Column Name Configuration
    time_col, temp_col, pres_col = _get_column_names(config)

    Ah, nh, Eh = params
    A1, n1, E1, Teq, k, w, C0, xf = fixed_params
    residuals_hi = []
    
    for df, idt_with_tcom in zip(dfs, idts_one):
        time = df[time_col].values
        temperature = df[temp_col].values
        pressure = df[pres_col].values
        
        calculator = IDTCalculator(A1, n1, E1, Ah, nh, Eh, Teq, k, w, C0, xf)
        idt_hi = calculator.calculate_idt_hi(pressure, temperature)
        integral_hi = integrate_idt(time, idt_hi)
        
        # Find integral when time reach idt_with_tcom
        idx_hi = np.argmin(np.abs(time - idt_with_tcom))
               
        # Calculate relative deviation residual
        # residual = integral_hi[idx_hi] - 1.0
        residual = np.log(integral_hi[idx_hi])
        
        residuals_hi.append(residual)
    
    return np.array(residuals_hi)



def calculate_residuals_total(dfs, params, original_idts, t_com=None, apply_weights=False, config=None):
    A1, n1, E1, Ah, nh, Eh, Teq, k, w, C0, xf  = params
    residuals_total = []
    # Obtain Column Name Configuration
    time_col, temp_col, pres_col = _get_column_names(config)
    for df, idt_original in zip(dfs, original_idts):
        time = df[time_col].values
        temperature = df[temp_col].values
        pressure = df[pres_col].values

        calculator = IDTCalculator(A1, n1, E1, Ah, nh, Eh, Teq, k, w, C0, xf)
        idt_1st = calculator.calculate_idt_1st(pressure, temperature)
        integral_1st = integrate_idt(time, idt_1st)
        idt_hi = calculator.calculate_idt_hi(pressure, temperature)
        integral_hi = integrate_idt(time, idt_hi)
        # Find index when integral_1st reaches 1.0
        idx_1st = np.argmin(np.abs(integral_1st - 1.0))
        idt_cf = calculator.calculate_idt_cf(pressure, temperature)
        # Calculate D_Tcf
        D_Tcf = calculator.calculate_DT(pressure, temperature)
        # Define the array Ti
        Ti = np.zeros_like(time)
        # Calculate Tcf
        Tcf = calculator.calculate_Tcf(temperature, D_Tcf)
        #  Calculate pcf
        pcf = calculator.calculate_pcf(pressure, temperature, D_Tcf)
        # Adjust idt_cf
        mask = (time > time[idx_1st]) & (D_Tcf > D_Tcf[idx_1st])
        pcf[mask] = calculator.calculate_pcf(pressure[mask], temperature[mask], D_Tcf[idx_1st])
        Tcf[mask] = calculator.calculate_Tcf(temperature[mask], D_Tcf[idx_1st])
        Ti[mask] = calculator.calculate_Ti(Tcf[mask], temperature[mask])
        idt_cf[mask] = calculator.calculate_idt_hi(pcf[mask], Ti[mask])
        integral_cf = integrate_idt(time, idt_cf)


        # Find the index at total ignition
        try:
            # Ensure that integral_cf and integral_hi are valid values at idx_1st.
            if np.isnan(integral_cf[idx_1st]) or np.isinf(integral_cf[idx_1st]):
                idx_total = -1  # Use the last point as the default value
            elif np.isnan(integral_hi[idx_1st]) or np.isinf(integral_hi[idx_1st]):
                idx_total = -1  # Use the last point as the default value
            else:
                integral_cf_total = 1 + integral_cf[idx_1st] - integral_hi[idx_1st]
                # Find index when integral_cf_total reaches integral_cf_total
                idx_total = np.argmin(np.abs(integral_cf - integral_cf_total))
            
            # Ensure that integral_cf_total is a valid numerical value.
            if np.isnan(integral_cf_total) or np.isinf(integral_cf_total):
                idx_total = -1  # Use the last point as the default value
        except:
            # Use the default value in case of an exception
            idx_total = -1  # Use the last point as the default value
            
        calculated_idt_with_tcom = time[idx_total]
        calculated_idt = calculated_idt_with_tcom - t_com

        # Calculate relative deviation residual
        residual = np.log(calculated_idt + 1e-8) - np.log(idt_original + 1e-8)
        residuals_total.append(residual)
    
    if apply_weights:
        # Calculate weights - use the relative magnitude of the logarithmic values of the original IDT as the basis for weighting.
        original_idts_array = np.array(original_idts)
        # Avoid taking the logarithm of zero or negative numbers by adding a small constant to ensure non-zero values.
        log_weights = np.log(original_idts_array + 1e-8)
        # Ensure that the values in log_weights are positive, and set any negative values to a small positive number close to zero.
        log_weights = np.where(log_weights < 0, 1e-8, log_weights)
        # Normalize the weights so that the sum of all weights equals 1.
        weights = log_weights / np.sum(log_weights)
    else:
        # Return equal weights, with the sum of all weights equal to 1.
        n = len(residuals_total)
        weights = np.ones(n) / n
    
    return np.array(residuals_total), weights

def objective_1st(params, dfs, idts_1st, fixed_params, original_idts, config=None):
    try:
        # Calculate residuals and weights - using the default values of the residuals function
        residuals, weights = calculate_residuals_1st(
            dfs, idts_1st, params, fixed_params, original_idts, config=config
        )
        
        # Use the weighted averaged squared residual as the target value
        weighted_squared_residuals = np.multiply(residuals ** 2, weights)
        objective_value = np.sum(weighted_squared_residuals)
        
        if not np.isfinite(objective_value):
            return 1e10  # Return large value for non-finite results
        return objective_value
    except Exception as e:
        print(f"Error in objective_1st: {e}")
        return 1e10

def objective_hi(params, dfs, idts_one, fixed_params, config=None):
    residuals = calculate_residuals_hi(dfs, idts_one, params, fixed_params, config=config)
    objective_value = np.sum(residuals ** 2)
    return objective_value


def objective_total(params, dfs, original_idts, t_com, config=None):
    try:
        residuals, weights = calculate_residuals_total(
            dfs, params, original_idts, t_com, config=config
        )
        
        weighted_squared_residuals = np.multiply(residuals ** 2, weights)
        objective_value = np.sum(weighted_squared_residuals)

        if not np.isfinite(objective_value):
            return 1e10  # Return large value for non-finite results
        return objective_value
    except Exception as e:
        print(f"Error in objective_total: {e}")
        return 1e10


def residuals_vector_total(params, dfs, original_idts, t_com, config=None):
    """Return weighted residuals vector for least_squares optimization.
    
    The weighted residuals satisfy: sum(weighted_residuals^2) = MSE
    This ensures consistency with objective_total.
    """
    try:
        residuals, weights = calculate_residuals_total(
            dfs, params, original_idts, t_com, config=config
        )
        
        weighted_residuals = residuals * np.sqrt(weights)
        
        if not np.all(np.isfinite(weighted_residuals)):
            return np.ones(len(residuals)) * 1e5
        return weighted_residuals
    except Exception as e:
        print(f"Error in residuals_vector_total: {e}")
        return np.ones(1) * 1e5