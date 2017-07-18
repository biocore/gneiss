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

   OLSModel
   LMEModel

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ._ols import ols, OLSModel
from ._mixedlm import mixedlm, LMEModel


__all__ = ["ols", "OLSModel", "mixedlm", "LMEModel"]
