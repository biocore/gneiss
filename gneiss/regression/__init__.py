"""
Regression functions (:mod:`gneiss.regression`)
===============================================

.. currentmodule:: gneiss.regression

This module contains functions that can convert proportions
to balances for regression analysis

Functions
---------

.. autosummary::
   :toctree: generated/

   ols
   mixedlm

Classes
-------
.. autosummary::
   :toctree: generated/

   RegressionResults

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._summary import RegressionResults
from ._regression import ols, mixedlm

__all__ = ["ols", "mixedlm", "RegressionResults"]
