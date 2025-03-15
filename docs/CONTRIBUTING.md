# Contributing to the Modular Backtesting System

Thank you for considering contributing to the Modular Backtesting System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to the project:

1. **Reporting Bugs**: If you find a bug, please create an issue with a detailed description of the problem, steps to reproduce, and your environment.
2. **Suggesting Enhancements**: If you have ideas for new features or improvements, please create an issue with a detailed description of your suggestion.
3. **Contributing Code**: If you want to contribute code, please follow the process below.

## Development Process

1. **Fork the Repository**: Create a fork of the repository on GitHub.
2. **Clone Your Fork**: Clone your fork to your local machine.
3. **Create a Branch**: Create a branch for your changes.
4. **Make Your Changes**: Make your changes to the codebase.
5. **Run Tests**: Make sure all tests pass.
6. **Submit a Pull Request**: Submit a pull request to the main repository.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/backtester.git
   cd backtester
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Coding Standards

We follow PEP 8 for Python code style. Please ensure your code adheres to these standards.

- Use 4 spaces for indentation
- Use descriptive variable names
- Write docstrings for all functions, classes, and modules
- Keep lines under 100 characters
- Use type hints where appropriate

We use the following tools to enforce coding standards:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting

You can run these tools with:

```bash
black backtester tests
isort backtester tests
flake8 backtester tests
```

## Testing

All new code should include tests. We use pytest for testing.

- Unit tests should be placed in the `tests/unit` directory
- Integration tests should be placed in the `tests/integration` directory

## Documentation

All new code should include documentation:

- Docstrings for all functions, classes, and modules
- Updates to relevant documentation files in the `docs` directory

## Pull Request Process

1. Ensure your code follows the coding standards
2. Ensure all tests pass
3. Update documentation if necessary
4. Submit a pull request with a clear description of the changes

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

## Questions

If you have any questions, please create an issue or contact the maintainers. 