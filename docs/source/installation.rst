Installation
============

Prerequisites
------------

Before installing the Modular Backtesting System, ensure you have the following prerequisites:

* Python 3.8 or higher
* pip (Python package installer)
* Git (optional, for development installation)

Installation Methods
------------------

There are several ways to install the Modular Backtesting System:

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~

The simplest way to install the Modular Backtesting System is from PyPI using pip:

.. code-block:: bash

    pip install backtester

This will install the latest stable release of the package and all its dependencies.

From Source
~~~~~~~~~~

To install the latest development version from source:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/yourusername/backtester.git
       cd backtester

2. Install the package in development mode:

   .. code-block:: bash

       pip install -e .

   This will install the package in development mode, allowing you to modify the code and see the changes immediately.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

If you plan to contribute to the Modular Backtesting System, you should install the development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This will install the package in development mode along with additional dependencies for testing, linting, and documentation.

Verifying Installation
--------------------

To verify that the installation was successful, you can run:

.. code-block:: bash

    backtester --version

This should display the version number of the installed package.

Dependencies
-----------

The Modular Backtesting System depends on the following Python packages:

* numpy: For numerical computations
* pandas: For data manipulation and analysis
* matplotlib: For visualization
* scipy: For scientific computing
* pyyaml: For configuration file parsing

These dependencies will be automatically installed when you install the package using pip. 