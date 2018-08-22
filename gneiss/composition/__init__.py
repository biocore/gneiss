"""
Composition functions (:mod:`gneiss.composition`)
===============================================

.. currentmodule:: gneiss.composition

This module contains compositional functions

Functions
---------

.. autosummary::
   :toctree: generated/

   variation_matrix

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ._composition import ilr_hierarchical, ilr_phylogenetic
from ._variance import variation_matrix


__all__ = ["ilr_hierarchical", "ilr_phylogenetic", "variation_matrix"]
