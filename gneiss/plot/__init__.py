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
   radialplot
   balance_boxplot
   balance_barplots
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from ._heatmap import heatmap
from ._radial import radialplot
from ._decompose import balance_boxplot, balance_barplots, proportion_plot


__all__ = ["heatmap", "radialplot", "balance_boxplot",
           "balance_barplots", "proportion_plot"]
