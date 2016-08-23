# gneiss

[![Build Status](https://travis-ci.org/biocore/gneiss.png?branch=master)](https://travis-ci.org/biocore/gneiss)
[![Coverage Status](https://coveralls.io/repos/biocore/gneiss/badge.svg)](https://coveralls.io/r/biocore/gneiss)

Canonically pronouced *nice*


 compositional statistics and visualization toolbox
+gneiss is a compositional statistics and visualization toolbox.  
 
Note that gneiss is not compatible with python 2, and is compatible with Python 3.4 or later.
gneiss is currently in alpha.  We are actively developing it, and __backward-incompatible interface changes can and will arise__.

# Installation

To install this package, it is recommended to use conda as follows

```
conda install -c biocore gneiss
```

You can also install Gneiss via pip

```
pip install gneiss
```

To run through the tutorials, you'll need a few more packages, namely `seaborn`, `biom-format` and `h5py`.
These packages can be installed with conda as follows
```
source activate gneiss
conda install seaborn h5py
pip install biom-format
```

# Examples

IPython notebooks demonstrating some of the modules in gneiss can be found as follows

* [What are balances](https://github.com/biocore/gneiss/blob/master/ipynb/balance_trees.ipynb)
* [Linear regression on balances in the 88 soils](https://github.com/biocore/gneiss/blob/master/ipynb/88soils.ipynb)

