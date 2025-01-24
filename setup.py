import setuptools
from setuptools import setup


__version__ = '0.0.1'

pkgs = {
    "required": [
        "numpy",
        "scikit_learn",
        "tqdm",
        "matplotlib",
        "scipy",
        "six",
        "pathlib",
        "numba",
    ],
    "extras": {
        "recommended": [
            "pandas",
            "jupyter",
            "tensorflow==2.8.0",
            "torch",
        ],
        "docs": [
            "numpydoc>=0.9.2",
            "sphinx>=2.4.4",
            "sphinx-rtd-theme>=0.4.3",
            "sphinxcontrib-trio>=1.1.0",
            "autodocsumm>=0.1.13",
            "gym>=0.17.2"
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "pytest-metadata",
            "ipykernel",
            "pylint",
            "pylint-exit",
            "jupytext"
        ]
    }
}

pkgs["extras"]["test"] += pkgs["extras"]["recommended"]
pkgs["extras"]["test"] += pkgs["extras"]["docs"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='package',
      version=__version__,
      description='LongCovid : Analysis of vaccination impact',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='LongCovid, COVID19, Vaccination',
      author='Milad Leyli-abadi',
      author_email='milad.leyli-abadi@irt-systemx.fr',
      url="https://git.irt-systemx.fr/xplo-covid/misc.git",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini"],
            },
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)
