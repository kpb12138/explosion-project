import yaml
import csv
import os
import numpy as np
import pandas as pd
from src.idt_calculator_nonreac import IDTCalculator


def load_config(config_file):
    """Load configuration file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set default value derivation logic
    config = _set_default_configs(config)
    
    return config


def _set_default_configs(config):
    """Set default values for configuration file"""
    results_folder = config['results_folder']
    
    # Extract base name from results_folder (remove "_results")
    base_name = results_folder.replace('_results', '')
    
    # Set default path for optimized_params_file
    if not config.get('optimized_params_file') or config['optimized_params_file'].strip() == '':
        config['optimized_params_file'] = f"./{results_folder}/optimized_parameters.csv"
    
    # Set default value for output_dir
    if not config.get('output_dir') or config['output_dir'].strip() == '':
        config['output_dir'] = f"./{base_name}_Constraint"
    
    # Set default value for idt_data_file
    if not config.get('idt_data_file') or config['idt_data_file'].strip() == '':
        config['idt_data_file'] = f"{base_name}_IDT.csv"
    
    return config


def load_constraint_data(config):
    """Load constraint efficiency data file"""
    output_dir = config.get('output_dir', '.')
    output_data_file = config.get('output_data_file', 'constraint_efficiency_data.csv')
    data_file_path = os.path.join(output_dir, output_data_file)
    
    if os.path.exists(data_file_path):
        return pd.read_csv(data_file_path)
    else:
        return None


def load_optimized_params(params_file):
    """Load optimized parameters file"""
    params_df = pd.read_csv(params_file)
    params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
    
    # Convert parameter types to float
    for key in params_dict:
        params_dict[key] = float(params_dict[key])
    
    return params_dict


def normalize_and_filter_data(df, constraint_criteria=0.5):
    """Normalize constraint efficiency and filter top 50% of data"""
    result = {
        'has_total': False,
        'has_1st': False,
        'total_data': None,
        'first_data': None
    }
    
    # Process Constraint Efficiency (Total)
    total_efficiency = df['Constraint Efficiency (Total)']
    if not (total_efficiency.isnull().all() or (total_efficiency == 0).all()):
        result['has_total'] = True
        # Normalize
        max_total = total_efficiency.max()
        df['Normalized Efficiency (Total)'] = total_efficiency / max_total
        # Filter data with normalized efficiency greater than constraint_criteria
        result['total_data'] = df[df['Normalized Efficiency (Total)'] > constraint_criteria]
    
    # Process Constraint Efficiency (1st)
    first_efficiency = df['Constraint Efficiency (1st)']
    if not (first_efficiency.isnull().all() or (first_efficiency == 0).all()):
        result['has_1st'] = True
        # Normalize
        max_first = first_efficiency.max()
        df['Normalized Efficiency (1st)'] = first_efficiency / max_first
        # Filter data with normalized efficiency greater than constraint_criteria
        result['first_data'] = df[df['Normalized Efficiency (1st)'] > constraint_criteria]
    
    return result


def determine_grid_bounds(config, filtered_data):
    """Determine temperature and pressure grid bounds"""
    # If no filtered data, use default values from config
    if not filtered_data['has_total'] and not filtered_data['has_1st']:
        print("Warning: No valid data for constraint efficiency calculation. Using default bounds.")
        return {
            'temp_min': config.get('temp_min', 400.0),
            'temp_max': config.get('temp_max', 900.0),
            'pressure_min': config.get('pressure_min', 10.0),
            'pressure_max': config.get('pressure_max', 20.0)
        }
    
    # Use filtered data to determine bounds
    temps = []
    pressures = []
    
    if filtered_data['has_total'] and filtered_data['total_data'] is not None:
        temps.extend(filtered_data['total_data']['Temperature (K)'].tolist())
        pressures.extend(filtered_data['total_data']['Pressure (bar)'].tolist())
    
    if filtered_data['has_1st'] and filtered_data['first_data'] is not None:
        temps.extend(filtered_data['first_data']['Temperature (K)'].tolist())
        pressures.extend(filtered_data['first_data']['Pressure (bar)'].tolist())
    
    return {
        'temp_min': min(temps),
        'temp_max': max(temps),
        'pressure_min': min(pressures),
        'pressure_max': max(pressures)
    }


def generate_grid_from_filtered_data(filtered_data, config):
    """Generate temperature and pressure grid from filtered data"""
    temps = []
    pressures = []
    
    if filtered_data['has_total'] and filtered_data['total_data'] is not None:
        temps.extend(filtered_data['total_data']['Temperature (K)'].tolist())
        pressures.extend(filtered_data['total_data']['Pressure (bar)'].tolist())
    
    # select conditions only based on total constraint efficiency
    # if filtered_data['has_1st'] and filtered_data['first_data'] is not None:
    #     temps.extend(filtered_data['first_data']['Temperature (K)'].tolist())
    #     pressures.extend(filtered_data['first_data']['Pressure (bar)'].tolist())
    
    if not temps:
        return None
    
    temps = np.array(temps)
    pressures = np.array(pressures)
    
    return {
        'temps': temps,
        'pressures': pressures
    }


def generate_grid(bounds, config):
    """Generate temperature and pressure grid"""
    temp_grid = config.get('temperature_grid', 20)
    pressure_grid = config.get('pressure_grid', 4)
    
    # Generate temperature and pressure grid
    temps = np.linspace(bounds['temp_min'], bounds['temp_max'], temp_grid)
    pressures = np.linspace(bounds['pressure_min'], bounds['pressure_max'], pressure_grid)
    
    # Create grid
    temp_grid, pressure_grid = np.meshgrid(temps, pressures)
    
    return {
        'temps': temp_grid.flatten(),
        'pressures': pressure_grid.flatten()
    }


def calculate_idt(grid, params_dict, has_first_efficiency=False):
    """Calculate IDT using IDTCalculator"""
    # Create IDTCalculator instance
    idt_calculator = IDTCalculator(
        A1=params_dict.get('A1', 0.0),
        n1=params_dict.get('n1', 0.0),
        E1=params_dict.get('E1', 0.0),
        Ah=params_dict.get('Ah', 0.0),
        nh=params_dict.get('nh', 0.0),
        Eh=params_dict.get('Eh', 0.0),
        Teq=params_dict.get('Teq', 0.0),
        k=params_dict.get('k', 0.0),
        w=params_dict.get('w', 0.0),
        C0=params_dict.get('C0', 0.0),
        xf=params_dict.get('xf', 0.0)
    )
    
    results = {
        'temps': grid['temps'],
        'pressures': grid['pressures'],
        'idt_total': [],
        'idt_1st': []
    }
    
    # Calculate IDT for each grid point
    for temp, pressure in zip(grid['temps'], grid['pressures']):
        # Calculate IDT_total
        idt_total = idt_calculator.calculate_idt(pressure, temp)
        results['idt_total'].append(idt_total)
        
        # Calculate IDT_1st
        if has_first_efficiency:
            idt_1st = idt_calculator.calculate_idt_1st(pressure, temp)
            results['idt_1st'].append(idt_1st)
        else:
            results['idt_1st'].append(0.0)
    
    return results


def save_idt_data(results, config):
    """Save IDT data to CSV file"""
    output_dir = config.get('output_dir', '.')
    idt_data_file = config.get('idt_data_file', 'idt_data.csv')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full file path
    file_path = os.path.join(output_dir, idt_data_file)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature (K)', 'Pressure (bar)', 'IDT_total', 'IDT_1st'])
        
        for temp, pressure, idt_total, idt_1st in zip(
            results['temps'],
            results['pressures'],
            results['idt_total'],
            results['idt_1st']
        ):
            writer.writerow([temp, pressure, idt_total, idt_1st])


def main():
    """Main function"""
    # Configuration file path
    config_file = 'ConstraintAnalysis_config.yaml'
    
    # Load configuration
    config = load_config(config_file)
    print("Configuration file loaded successfully")
    
    # Get auto_generate_mode and constraint_criteria
    auto_generate_mode = config.get('auto_generate_mode', 1)
    constraint_criteria = config.get('constraint_criteria', 0.5)
    
    print(f"Auto generate mode: {auto_generate_mode}")
    print(f"Constraint criteria: {constraint_criteria}")
    
    # Load constraint efficiency data
    constraint_df = load_constraint_data(config)
    print("Constraint efficiency data loaded successfully")
    
    # Determine execution logic based on auto_generate_mode
    if auto_generate_mode == 1:
        # Mode 1: Only generate IDT for temperature and pressure points where normalized Constraint Efficiency > constraint_criteria
        if constraint_df is None:
            print("Error: No constraint data available for auto_generate_mode=1")
            return
        
        # Normalize and filter data
        filtered_data = normalize_and_filter_data(constraint_df, constraint_criteria)
        
        # Generate grid from filtered data
        grid = generate_grid_from_filtered_data(filtered_data, config)
        
        if grid is None:
            print("Error: No valid data points found after filtering")
            return
        
        print(f"Generated grid: {len(grid['temps'])} points (from filtered data)")
    else:
        # Mode 0: Execute full IDT generation logic
        # Determine grid bounds
        if constraint_df is not None:
            # Normalize and filter data
            filtered_data = normalize_and_filter_data(constraint_df, constraint_criteria)
            # Determine grid bounds
            bounds = determine_grid_bounds(config, filtered_data)
        else:
            # Use default values from config
            bounds = {
                'temp_min': config.get('temp_min', 400.0),
                'temp_max': config.get('temp_max', 900.0),
                'pressure_min': config.get('pressure_min', 10.0),
                'pressure_max': config.get('pressure_max', 20.0)
            }
        
        print(f"Grid bounds: Temperature {bounds['temp_min']}-{bounds['temp_max']} K, Pressure {bounds['pressure_min']}-{bounds['pressure_max']} bar")
        
        # Generate grid
        grid = generate_grid(bounds, config)
        print(f"Generated grid: {len(grid['temps'])} points")
    
    # Load optimized parameters
    optimized_params_file = config.get('optimized_params_file', 'optimized_parameters.csv')
    params_dict = load_optimized_params(optimized_params_file)
    print("Optimized parameters loaded successfully")
    
    # Check if IDT_1st calculation is needed
    has_first_efficiency = False
    if constraint_df is not None:
        first_efficiency = constraint_df['Constraint Efficiency (1st)']
        if not (first_efficiency.isnull().all() or (first_efficiency == 0).all()):
            has_first_efficiency = True
    
    # Calculate IDT
    idt_results = calculate_idt(grid, params_dict, has_first_efficiency)
    print("IDT calculation completed")
    
    # Save results
    save_idt_data(idt_results, config)
    output_dir = config.get('output_dir', '.')
    idt_data_file = config.get('idt_data_file', 'idt_data.csv')
    print(f"Results saved to {os.path.join(output_dir, idt_data_file)}")


if __name__ == "__main__":
    main()
