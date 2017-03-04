"""
Plotting functions (:mod:`gneiss.plot`)
===============================================

.. currentmodule:: gneiss.plot

This module contains plotting functionality

Functions
---------

.. autosummary::
   :toctree: generated/

   heatmap
   diamondtree
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._heatmap import heatmap
from ._blobtree import diamondtree
from ._radial import radialplot
from ._plot import ols_summary, lme_summary, dendrogram_heatmap


__all__ = ["heatmap", "diamondtree", "radialplot",
           "ols_summary", "lme_summary", "dendrogram_heatmap"]
