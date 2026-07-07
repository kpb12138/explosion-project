import json
import os

# Replacement for skopt.space.Real (avoid skopt dependency)
class Real:
    def __init__(self, low, high, name=''):
        self.low = float(low)
        self.high = float(high)
        self.name = name


def load_config(config_path):
    """Load configuration parameters from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {config_path}")
        exit(1)


def parse_parameter_spaces(config):
    """Parse parameter spaces from JSON"""
    spaces = {}

    for step, params in config['parameter_spaces'].items():
        space = []
        for param in params:
            if param['type'] == 'real':
                low, high = param['low'], param['high']
                space.append(Real(low, high, name=param['name']))
        spaces[step] = space

    return spaces


def validate_config(config):
    """Validate configuration parameters and file-IDT correspondence"""
    all_files = []
    all_idts_1st = []
    all_idts_total = []
    all_idts_one = []
    all_idts_ntc = []

    # Process two-stage files
    two_stage_dir = config['file_specifications']['two_stage']['directory']
    two_stage_files = config['file_specifications']['two_stage']['files']
    for file_info in two_stage_files:
        filename = file_info['name']
        file_path = os.path.join(two_stage_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: Two-stage file not found - {file_path}")
            exit(1)
        all_files.append((filename, two_stage_dir))
        all_idts_1st.append(file_info['idt_1st'])
        all_idts_total.append(file_info['idt_total'])

    # Process single-stage files
    single_stage_dir = config['file_specifications']['single_stage']['directory']
    single_stage_files = config['file_specifications']['single_stage']['files']
    for file_info in single_stage_files:
        filename = file_info['name']
        file_path = os.path.join(single_stage_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: Single-stage file not found - {file_path}")
            exit(1)
        all_files.append((filename, single_stage_dir))
        all_idts_one.append(file_info['idt_one'])

    # Process NTC files
    ntc_dir = config['file_specifications']['ntc_region']['directory']
    ntc_files = config['file_specifications']['ntc_region']['files']
    for file_info in ntc_files:
        filename = file_info['name']
        file_path = os.path.join(ntc_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: NTC file not found - {file_path}")
            exit(1)
        all_files.append((filename, ntc_dir))
        all_idts_ntc.append(file_info['idt_ntc'])

    print(f"Configuration validation passed: {len(all_files)} files with matching IDT values")
    return all_files, all_idts_1st, all_idts_total, all_idts_one, all_idts_ntc


def validate_initial_guess(initial_guess, bounds, param_names, step_name):
    """Validate that initial guesses are within bounds"""
    validated_guess = initial_guess.copy()

    for i, (guess, bound, name) in enumerate(zip(initial_guess, bounds, param_names)):
        lower, upper = bound[0], bound[1]

        if not (lower <= guess <= upper):
            print(f"Warning: Initial guess for {name} ({guess}) is outside bounds [{lower}, {upper}] in {step_name}")
            adjusted_guess = (lower + upper) / 2
            print(f"Adjusted {name} initial guess to midpoint: {adjusted_guess}")
            validated_guess[i] = adjusted_guess

    return validated_guess
