# gneiss

[![Build Status](https://travis-ci.org/biocore/gneiss.png?branch=master)](https://travis-ci.org/biocore/gneiss)
[![Coverage Status](https://coveralls.io/repos/biocore/gneiss/badge.svg)](https://coveralls.io/r/biocore/gneiss)
[![Gitter](https://badges.gitter.im/biocore/gneiss.svg)](https://gitter.im/biocore/gneiss?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Canonically pronouced *nice*


gneiss is a compositional statistics and visualization toolbox.  See [here](https://biocore.github.io/gneiss/) for API documentation.
 
Note that gneiss is not compatible with python 2, and is compatible with Python 3.4 or later.
gneiss is currently in alpha.  We are actively developing it, and __backward-incompatible interface changes may arise__.

# Installation

To install this package, it is recommended to use conda.  First make sure that the appropriate channels are configured.

```
conda config --add channels https://conda.anaconda.org/bioconda
conda config --add channels https://conda.anaconda.org/biocore
conda config --add channels https://conda.anaconda.org/qiime2
conda config --add channels https://conda.anaconda.org/qiime2/label/r2017.6
```

Then gneiss can be installed in a conda environment as follows
```
conda create -n gneiss_env gneiss
```
To install the most up to date version of gneiss, run the following command

```
pip install git+https://github.com/biocore/gneiss.git
```

# Examples

IPython notebooks demonstrating some of the modules in gneiss can be found as follows

* [What are balances](https://github.com/biocore/gneiss/blob/master/ipynb/balance_trees.ipynb)
* [Linear regression on balances in the 88 soils](https://github.com/biocore/gneiss/blob/master/ipynb/88soils/88soils.ipynb)
* [Linear mixed effects models on balances in a CF study](https://github.com/biocore/gneiss/blob/master/ipynb/cfstudy/cfstudy.ipynb)


If you use this software package in your own publications, please cite it at
```
Morton JT, Sanders J, Quinn RA, McDonald D, Gonzalez A, Vázquez-Baeza Y, 
Navas-Molina JA, Song SJ, Metcalf JL, Hyde ER, Lladser M, Dorrestein PC, 
Knight R. 2017. Balance trees reveal microbial niche differentiation. 
mSystems 2:e00162-16. https://doi.org/10.1128/mSystems.00162-16.
```
