"""
Clustering functions (:mod:`gneiss.cluster`)
============================================

.. currentmodule:: gneiss.cluster

This module contains functions to build hierarchical clusterings.


Functions
---------

.. autosummary::
   :toctree: generated/

       proportional_linkage
       gradient_linkage

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the GPLv3 License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ._pba import proportional_linkage, gradient_linkage
from ._cluster import proportional_clustering, gradient_clustering


__all__ = ['proportional_linkage', 'gradient_linkage',
           'proportional_clustering', 'gradient_clustering']
