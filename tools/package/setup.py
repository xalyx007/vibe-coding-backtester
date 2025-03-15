#!/usr/bin/env python
"""
Setup script for the Modular Backtesting System.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open(os.path.join('backtester', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Read long description from README.md
with open('README.md', 'r') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Development dependencies
dev_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'black>=23.0.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'sphinx>=6.0.0',
    'sphinx-rtd-theme>=1.2.0',
    'twine>=4.0.0',
    'wheel>=0.40.0',
    'tox>=4.0.0',
]

setup(
    name="backtester",
    version=version,
    description="A modular backtesting system for trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/backtester",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "backtester=backtester.cli:main",
        ],
    },
) 