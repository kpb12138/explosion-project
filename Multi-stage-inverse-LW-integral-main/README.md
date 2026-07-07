# Multi-Stage Inverse Integral Method

A Python implementation of the Multi-Stage Inverse Integral Method for deriving ignition delay correlation from transient autoignition experimental measurements.

## 📖 Reference

If you use this code in your research, please cite our paper:

> Yingtao Wu, Zhonghao Zhao, Yuxin Fang, Pengzhi Wang, Song Cheng, Chenglong Tang, Zuohua Huang, Henry Curran.  
> "A Multi-Stage Inverse Integral Method for Deriving Ignition Delay Correlation from Transient Autoignition Experimental Measurements."  
> (submitted)

## 📋 Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## 🔬 Introduction

This code implements a multi-stage inverse integral method for deriving ignition delay time (IDT) correlations from transient autoignition experimental measurements. The method optimizes correlation parameters to match experimentally measured thermal (pressure and temperature) profiles and ignition timing.

### Key Features

- Multi-stage optimization strategy for correlation parameters
- Support for first-stage and total ignition delay times
- Pressure-rise correlation for non-reactive simulations
- Constraint efficiency analysis for data validation
- Visualization tools for optimization results

## 💾 Installation

### Prerequisites

- Python 3.8+
- Git

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

Run the main optimization script:

```bash
python main_nonreac_auto_integral.py
```

This will:
1. Load the configuration from `Input.json`
2. Perform a four-stage optimization process
3. Save results to a `results/` directory

### Complete Analysis Pipeline

On Windows, you can run the constraint efficiency analysis and IDT correlation generation using the batch script:

```bash
run_analysis.bat
```

This will:
1. Run constraint efficiency analysis (`dataConstraintAnalysis.py`)
2. Generate IDT correlation (`correlation_IDT_gen.py`)

**Note:** This is a separate workflow from the main optimization. Ensure `ConstraintAnalysis_config.yaml` is properly configured with the correct `results_folder` path before running.

### Configuration

The `Input.json` file contains all necessary parameters:

| Section | Description |
|---------|-------------|
| `file_specifications` | Paths to input CSV files |
| `timing_parameters` | Timing parameters (e.g., compression time) |
| `initial_parameters` | Initial guess for kinetic parameters |
| `parameter_spaces` | Bounds for each optimization step |
| `optimization_settings` | Optimization algorithm settings |

### Example Input Files

Example input files are provided in the `examples/` directory:
- `Input.json` - Example configuration
- Input CSV files containing pressure and temperature data

## 📁 Project Structure

```
Inverse_LW/
├── main_nonreac_auto_integral.py    # Main optimization script
├── correlation_IDT_gen.py           # IDT correlation generator
├── dataConstraintAnalysis.py        # Constraint efficiency analysis
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── config_utils.py              # Configuration utilities
│   ├── idt_calculator_nonreac.py    # IDT calculator class
│   ├── optimization_utils.py        # Optimization algorithms
│   ├── plotting_utils.py            # Plotting utilities
│   ├── boundary_utils.py            # Boundary checking utilities
│   ├── residuals_nonreac_integral.py# Residual calculation functions
│   └── Constrain_plot_utils.py      # Constraint plotting utilities
├── examples/                        # Example input files
│   ├── Input.json                   # Example configuration
│   └── UoG_F10_20bar_inputfile/     # Example input CSV files
├── tests/                           # Test files
│   └── test_basic.py                # Basic unit tests
├── requirements.txt                 # Dependencies
├── LICENSE                          # MIT License
├── README.md                        # This file
├── ConstraintAnalysis_config.yaml   # Constraint analysis configuration
└── run_analysis.bat                 # Batch script for analysis
```

## ⚙️ Optimization Stages

The optimization process consists of four stages:

1. **Stage 1**: Optimize first-stage Arrhenius parameters (A1, n1, E1)
2. **Stage 2**: Optimize high-temperature Arrhenius parameters (Ah, nh, Eh)
3. **Stage 3**: Optimize pressure-rise correlation parameters (Teq, k, w)
4. **Stage 4**: Full optimization of all parameters

## 📊 Output

The script generates:
- `optimized_parameters.csv` - Final optimized parameters
- `optimization.log` - Optimization log file
- Various plots showing IDT comparisons and convergence

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please contact:

- Yingtao Wu  
  Email: wuyingtao@xjtu.edu.cn  
  Xi'an Jiaotong University

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Steps to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request