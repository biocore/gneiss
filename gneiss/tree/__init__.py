"""
Tree building functions (:mod:`gneiss.tree`)
============================================

.. currentmodule:: gneiss.correlation

This module contains functions to build trees.


Functions
---------

.. autosummary::
   :toctree: generated/

       hcpba
       supgma
"""

from ._pba import proportional_linkage, gradient_linkage


__all__ = ['proportional_linkage', 'gradient_linkage']
