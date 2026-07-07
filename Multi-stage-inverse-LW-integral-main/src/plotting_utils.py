import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .idt_calculator_nonreac import integrate_idt

# Default column names
DEFAULT_TIME_COL = 'Time(msec)'
DEFAULT_TEMP_COL = ' Temperature(K)'
DEFAULT_PRES_COL = ' Pressure(bar)'

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

def plot_optimization_debug_plots(params_history, residuals_history, param_names, step_name, results_dir):
    """Generate debugging plots for optimization steps and save real-time optimal results to CSV"""
    debug_dir = os.path.join(results_dir, 'debug_plots', step_name)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Ensure that the history and residuals data are valid.
    if not params_history or not residuals_history:
        print(f"Warning: No valid history or residuals data for {step_name}")
        return
    
    # Convert to a NumPy array for processing
    try:
        params_array = np.array(params_history)
        residuals_array = np.array(residuals_history)
    except Exception as e:
        print(f"Error processing optimization history data for {step_name}: {e}")
        return
    
    # Calculate the optimal parameters and residuals for each iteration.
    # (Assuming that the parameter history already represents the real-time optimal results in the iterative process.)
    iterations = list(range(1, len(residuals_array) + 1))
    
    # Create a DataFrame and save it to a CSV file.
    data = {'Iteration': iterations, 'Residual': residuals_array}
    
    # Add a data column for each parameter.
    if params_array.ndim == 1:
        # In the case of one-dimensional arrays
        data[param_names[0]] = params_array
    else:
        # In the case of a two-dimensional arrays
        for i, name in enumerate(param_names):
            if i < params_array.shape[1]:
                data[name] = params_array[:, i]
    
    # Create a DataFrame and save it to a CSV file.
    df = pd.DataFrame(data)
    csv_path = os.path.join(debug_dir, 'optimization_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"Optimization history saved to {csv_path}")
    
    # Residual history
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, residuals_array, 'o-', color='blue', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title(f'{step_name} Residual vs Iteration')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(debug_dir, 'residual_vs_iteration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot each parameter individually
    for i, name in enumerate(param_names):
        if i < params_array.shape[1]:
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, params_array[:, i], 'o-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel(name)
            plt.title(f'{step_name} {name} vs Iteration')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(debug_dir, f'parameter_{name}_vs_iteration.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Updated debug plots saved for {step_name} in {debug_dir}")

def plot_dp_comparison(T, dp_actual, dp_predicted, results_dir):
    """Plot comparison of predicted vs. actual pressure rise (Dp)"""
    debug_dir = os.path.join(results_dir, 'debug_plots', 'step4')
    os.makedirs(debug_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(T, dp_actual, color='red', marker='o', label='Actual Dp')
    plt.scatter(T, dp_predicted, color='blue', marker='s', label='Predicted Dp')
    plt.plot(T, dp_actual, 'r--', alpha=0.3)
    plt.plot(T, dp_predicted, 'b--', alpha=0.3)
    
    # Add R-squared value
    try:
        correlation_matrix = np.corrcoef(dp_actual, dp_predicted)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Warning: Could not calculate R² value: {e}")
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure Rise (bar)')
    plt.title('Dp Prediction vs. Actual Values (Step 4)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(debug_dir, 'dp_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dp comparison plot saved to {debug_dir}")


def plot_integral_curves(file_path, calculator, results_dir, config, log_data=None):
    """Generate integral curves and save data"""
    integral_dir = os.path.join(results_dir, 'integralFigure')
    os.makedirs(integral_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
    # Obtain Column Name Configuration
    time_col, temp_col, pres_col = _get_column_names(config)

    time = df[time_col].values
    temperature = df[temp_col].values
    pressure = df[pres_col].values
    
    # Calculate IDT components
    idt_1st = calculator.calculate_idt_1st(pressure, temperature)
    idt_cf = calculator.calculate_idt_cf(pressure, temperature)
    idt_hi = calculator.calculate_idt_hi(pressure, temperature)
    idt_total = calculator.calculate_idt(pressure, temperature)
    
    # Calculate integrals
    integral_1st = integrate_idt(time, idt_1st)
    integral_cf = integrate_idt(time, idt_cf)
    integral_hi = integrate_idt(time, idt_hi)
    integral_total = integrate_idt(time, idt_total)
    
    # Save integral data
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_path = os.path.join(integral_dir, f'{base_name}_integral.csv')
    csv_data = pd.DataFrame({
        'Time(msec)': time,
        'Integral_1st': integral_1st,
        'Integral_CF': integral_cf,
        'Integral_HI': integral_hi,
        'Integral_Total': integral_total
    })
    csv_data.to_csv(csv_path, index=False)
    print(f"Integral data saved: {csv_path}")
    
    # Calculate the moment corresponding to Integral Total = 1
    idx_total = np.argmin(np.abs(integral_total - 1))
    time_at_integral_1 = time[idx_total]
    x_max = time_at_integral_1 * 1.2  # Set the maximum value of the horizontal axis to 1.2 times that of the current moment.
    
    # Plotting logic with custom axes limits
    plt.figure(figsize=(12, 8))
    plt.plot(time, integral_1st, label='Integral 1st', alpha=0.7)
    plt.plot(time, integral_cf, label='Integral CF', alpha=0.7)
    plt.plot(time, integral_hi, label='Integral HI', alpha=0.7)
    plt.plot(time, integral_total, label='Integral Total', linewidth=2)
    
    plt.xlabel('Time (msec)')
    plt.ylabel('Integral Value')
    plt.title(f'Integral Curves - {base_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Set the axis ranges: the vertical axis range from 0 to 2, and the horizontal axis range from 0 to the calculated maximum value.
    plt.ylim(0, 2)
    plt.xlim(0, x_max)
    
    plt.savefig(os.path.join(integral_dir, f'{base_name}_integral.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return csv_path

def plot_predicted_idts(dfs, original_idts_1st, original_idts_total, t_com, calculator, results_dir, config, original_idts_one=None):
    # get column names from config
    time_col, temp_col, pres_col = _get_column_names(config)
    
    T_1st = []
    for df in dfs[:len(original_idts_1st)]:
        # get temperature at t_com
        idx = (np.abs(df[time_col] - t_com)).idxmin()
        T_at_tcom = df.loc[idx, temp_col]
        T_1st.append(T_at_tcom)
    
    inv_T_1st = [1000/x for x in T_1st]
    predicted_idts_1st = []
    predicted_idts_total = []

    # Only process dataframes corresponding to original_idts_1st for 1st IDT comparison
    for i, df in enumerate(dfs[:len(original_idts_1st)]):
        time = df[time_col].values
        pressure = df[pres_col].values
        temperature = df[temp_col].values
        idt_1st = calculator.calculate_idt_1st(pressure, temperature)
        integral_1st = integrate_idt(time, idt_1st)
        
        idx_1st = np.argmin(np.abs(integral_1st - 1))
        idt_1st_pred = time[idx_1st] - t_com
        if idt_1st_pred < 0:  
            predicted_idt_1st = np.nan
        else:
            predicted_idt_1st = idt_1st_pred
        predicted_idts_1st.append(predicted_idt_1st)
    
    # Process all dataframes for total IDT comparison
    for df in dfs:
        time = df[time_col].values
        pressure = df[pres_col].values
        temperature = df[temp_col].values
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
        integral_cf_total = 1 + integral_cf[idx_1st] - integral_hi[idx_1st]
        idx_total = np.argmin(np.abs(integral_cf - integral_cf_total))

        idt_total_pred = time[idx_total] - t_com
        if idt_total_pred < 0:  
            predicted_idt_total = np.nan
        else:
            predicted_idt_total = idt_total_pred
        predicted_idts_total.append(predicted_idt_total)
    
    # Process dataframes for high-temperature (step2) IDT comparison if provided
    predicted_idts_one = []
    T_one = []
    if original_idts_one:
        # Get the dataframes corresponding to high-temperature IDTs (step2)
        idts_one_start_idx = len(original_idts_1st) if original_idts_1st else 0
        idts_one_end_idx = idts_one_start_idx + len(original_idts_one)
        
        for df in dfs[idts_one_start_idx:idts_one_end_idx]:
            # Get temperature at t_com
            idx = (np.abs(df[time_col] - t_com)).idxmin()
            T_at_tcom = df.loc[idx, temp_col]
            T_one.append(T_at_tcom)
            
            # Calculate predicted IDT
            time = df[time_col].values
            pressure = df[pres_col].values
            temperature = df[temp_col].values
            idt_one = calculator.calculate_idt_hi(pressure, temperature)
            integral_one = integrate_idt(time, idt_one)
            
            idx_one = np.argmin(np.abs(integral_one - 1))
            idt_one_pred = time[idx_one] - t_com
            if idt_one_pred < 0:  
                predicted_idt_one = np.nan
            else:
                predicted_idt_one = idt_one_pred
            predicted_idts_one.append(predicted_idt_one)
        
        # Calculate inverse temperature for high-temperature IDTs
        inv_T_one = [1000/x for x in T_one]
    
    # Also calculate inv_T_total for total IDT comparison
    T_total = []
    for df in dfs:
        # get temperature at t_com
        idx = (np.abs(df[time_col] - t_com)).idxmin()
        T_at_tcom = df.loc[idx, temp_col]
        T_total.append(T_at_tcom)
    
    inv_T_total = [1000/x for x in T_total]
    
    # Sort the data in order to plot a smooth curve.
    # Sort the 1st IDT data.
    sorted_1st = sorted(zip(inv_T_1st, original_idts_1st, predicted_idts_1st), key=lambda x: x[0])
    sorted_inv_T_1st = [x[0] for x in sorted_1st]
    sorted_original_idts_1st = [x[1] for x in sorted_1st]
    sorted_predicted_idts_1st = [x[2] for x in sorted_1st]
    
    # Sort the Total IDT data.
    sorted_total = sorted(zip(inv_T_total, original_idts_total, predicted_idts_total), key=lambda x: x[0])
    sorted_inv_T_total = [x[0] for x in sorted_total]
    sorted_original_idts_total = [x[1] for x in sorted_total]
    sorted_predicted_idts_total = [x[2] for x in sorted_total]
    
    # Draw a comparative chart
    plt.figure(figsize=(10, 6))
    plt.semilogy(sorted_inv_T_1st, sorted_original_idts_1st, 'ro', label='Original 1st IDTs')
    plt.semilogy(sorted_inv_T_1st, sorted_predicted_idts_1st, 'b-', label='Predicted 1st IDTs')
    plt.xlabel('1000/T (K⁻¹)')
    plt.ylabel('IDT (ms)')
    plt.title('Comparison of Original and Predicted 1st IDTs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, '1st_IDT_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()    
    print(f"First-stage IDT comparison plot saved to {results_dir}")
    
    # Save data to CSV
    csv_data_1st = pd.DataFrame({
        '1000/T (K⁻¹)': sorted_inv_T_1st,
        'Original_1st_IDTs (ms)': sorted_original_idts_1st,
        'Predicted_1st_IDTs (ms)': sorted_predicted_idts_1st
    })
    csv_data_1st.to_csv(os.path.join(results_dir, '1st_IDT_comparison.csv'), index=False)
    print(f"First-stage IDT comparison data saved to {results_dir}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(sorted_inv_T_total, sorted_original_idts_total, 'ro', label='Original Total IDTs')
    plt.semilogy(sorted_inv_T_total, sorted_predicted_idts_total, 'b-', label='Predicted Total IDTs')
    plt.xlabel('1000/T (K⁻¹)')
    plt.ylabel('IDT (ms)')
    plt.title('Comparison of Original and Predicted Total IDTs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'total_IDT_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Total IDT comparison plot saved to {results_dir}")
    
    # Save data to CSV
    csv_data_total = pd.DataFrame({
        '1000/T (K⁻¹)': sorted_inv_T_total,
        'Original_Total_IDTs (ms)': sorted_original_idts_total,
        'Predicted_Total_IDTs (ms)': sorted_predicted_idts_total
    })
    csv_data_total.to_csv(os.path.join(results_dir, 'total_IDT_comparison.csv'), index=False)
    print(f"Total IDT comparison data saved to {results_dir}")
    
    # Plot and save high-temperature (step2) IDT comparison if data is available
    if original_idts_one:
        # Sort data for smooth curve plotting
        sorted_one = sorted(zip(inv_T_one, original_idts_one, predicted_idts_one), key=lambda x: x[0])
        sorted_inv_T_one = [x[0] for x in sorted_one]
        sorted_original_idts_one = [x[1] for x in sorted_one]
        sorted_predicted_idts_one = [x[2] for x in sorted_one]
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.semilogy(sorted_inv_T_one, sorted_original_idts_one, 'ro', label='Original High-Temp IDTs')
        plt.semilogy(sorted_inv_T_one, sorted_predicted_idts_one, 'b-', label='Predicted High-Temp IDTs')
        plt.xlabel('1000/T (K⁻¹)')
        plt.ylabel('IDT (ms)')
        plt.title('Comparison of Original and Predicted High-Temperature IDTs (Step 2)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'hi_IDT_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"High-temperature IDT comparison plot saved to {results_dir}")
        
        # Save data to CSV
        csv_data_one = pd.DataFrame({
            '1000/T (K⁻¹)': sorted_inv_T_one,
            'Original_High_Temp_IDTs (ms)': sorted_original_idts_one,
            'Predicted_High_Temp_IDTs (ms)': sorted_predicted_idts_one
        })
        csv_data_one.to_csv(os.path.join(results_dir, 'hi_IDT_comparison.csv'), index=False)
        print(f"High-temperature IDT comparison data saved to {results_dir}")


    
