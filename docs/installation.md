# Installation Guide

This guide provides detailed instructions for installing the Modular Backtesting System in various environments.

## Prerequisites

Before installing the Modular Backtesting System, ensure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for installation from source)

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The easiest way to install the Modular Backtesting System is via pip:

```bash
pip install backtester
```

This will install the latest stable release of the package along with all its dependencies.

<div class="tip">
To install a specific version, use:

```bash
pip install backtester==1.0.0
```
</div>

### Method 2: Install from Source

For the latest development version or if you want to contribute to the project:

```bash
git clone https://github.com/yourusername/backtester.git
cd backtester
pip install -e .
```

<div class="key-concept">
The <code>-e</code> flag installs the package in "editable" mode, which means that changes to the source code will be immediately reflected without needing to reinstall the package.
</div>

### Method 3: Install with Development Dependencies

If you plan to contribute to the project or run tests, you should install the development dependencies:

```bash
git clone https://github.com/yourusername/backtester.git
cd backtester
pip install -e ".[dev]"
```

This will install additional packages required for development, such as pytest, flake8, and black.

## Verifying the Installation

To verify that the installation was successful, you can run the following command:

```bash
python -c "import backtester; print(backtester.__version__)"
```

This should print the version number of the installed package.

## Installing in a Virtual Environment

It's recommended to install the Modular Backtesting System in a virtual environment to avoid conflicts with other packages.

### Using venv

```bash
# Create a virtual environment
python -m venv backtester-env

# Activate the virtual environment
# On Windows
backtester-env\Scripts\activate
# On macOS/Linux
source backtester-env/bin/activate

# Install the package
pip install backtester
```

### Using conda

```bash
# Create a conda environment
conda create -n backtester-env python=3.8

# Activate the environment
conda activate backtester-env

# Install the package
pip install backtester
```

## Troubleshooting

### Common Issues

#### Missing Dependencies

If you encounter errors related to missing dependencies, try installing them manually:

```bash
pip install -r requirements.txt
```

#### Permission Errors

If you encounter permission errors when installing the package, try using the `--user` flag:

```bash
pip install --user backtester
```

#### Conflicts with Existing Packages

If you encounter conflicts with existing packages, try installing the package in a virtual environment as described above.

### Getting Help

If you encounter any issues during installation, please:

1. Check the [GitHub Issues](https://github.com/yourusername/backtester/issues) to see if the issue has already been reported
2. If not, create a new issue with details about the error and your environment

## Next Steps

Now that you have installed the Modular Backtesting System, you can:

1. Follow the [Quick Start Guide](quickstart.md) to learn how to use the system
2. Explore the [Examples](examples/basic.md) to see more complex use cases
3. Read the [Architecture](architecture.md) documentation to understand the system design 