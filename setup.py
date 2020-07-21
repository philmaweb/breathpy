import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="breathpy",
    version="0.8.4",
    description="Breath analysis in python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/philmaweb/breathpy",
    author="Philipp Weber",
    author_email="pweber@imada.sdu.dk",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "graphviz>=0.13.2",
        "ipdb",
        "matplotlib>=3.2.1",
        "matplotlib-venn>=0.11.5",
        "numpy>=1.18.1",
        "pandas>=1.0.3",
        "psutil>=3.4.2",
        "pyopenms==2.4.0",
        "pywavelets>=1.1.1",
        "scikit-image>=0.16.2",
        "scikit-learn>=0.22.0,<0.23.0",
        "scipy>=1.4.1",
        "seaborn>=0.10.0",
        "statannot>=0.2.3",
        "statsmodels>=0.11.1",
        "xlrd>=1.2.0",
    ],
)
