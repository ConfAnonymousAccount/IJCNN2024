# Long Covid analysis and study of the vaccination impact
Description


![Scheme](docs/images/Covid19.jpg)

*   [1 Introduction](#introduction)
*   [2 Usage example](#usage-example)
*   [3 Installation](#installation)
    *   [3.1 Setup a Virtualenv (optional)](#setup-a-virtualenv-optional)
    *   [3.2 Install from source](#install-from-source)
*   [4 Getting Started](#getting-started)
*   [5 Documentation](#documentation)
*   [6 Contribution](#contribution)
*   [7 License information](#license-information)

## Introduction
A brief introduction to the study and if there will be any platform

## Usage example
If there is an usage example

## Installation
To be able to run the experiments in this repository, the users should install the last lips package from its github repository. The following steps show how to install this package and its dependencies from source.

### Requirements
- Python >= 3.6

### Setup a Virtualenv (optional)
#### Create a virtual environment

```commandline
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv
```

#### Enter virtual environment
```commandline
source venv/bin/activate
```

### Install from source
```commandline
git clone https://git.irt-systemx.fr/xplo-covid/misc.git
cd misc
pip3 install -U .
cd ..
```

### To contribute
```commandline
pip3 install -e .[recommended]
```

# Getting Started
Some Jupyter notebook are provided as tutorials for this package. They are located in the
[getting_started](getting_started) directories.

# Documentation
The documentation is accessible from here.

To generate locally the documentation:
```commandline
pip install sphinx
pip install sphinx-rtd-theme
cd docs
make clean
make html
```

# Contribution
* Supplementary features could be requested using github issues.
* Other contributions are welcomed and can be integrated using pull requests.
