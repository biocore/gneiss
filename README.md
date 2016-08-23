# gneiss

[![Build Status](https://travis-ci.org/biocore/gneiss.png?branch=master)](https://travis-ci.org/biocore/gneiss)
[![Coverage Status](https://coveralls.io/repos/biocore/gneiss/badge.svg)](https://coveralls.io/r/biocore/gneiss)

Canonically pronouced *nice*


gneiss is a compositional statistics and visualization toolbox.  
 
Note that gneiss is not compatible with python 2, and is compatible with Python 3.4 or later.
gneiss is currently in alpha.  We are actively developing it, and __backward-incompatible interface changes may arise__.

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
