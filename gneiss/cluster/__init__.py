"""
Clustering functions (:mod:`gneiss.cluster`)
============================================

.. currentmodule:: gneiss.cluster

This module contains functions to build hierarchical clusterings.


Functions
---------

.. autosummary::
   :toctree: generated/

       correlation_linkage
       gradient_linkage
       rank_linkage
       random_linkage

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from ._pba import (correlation_linkage, gradient_linkage,
                   rank_linkage, random_linkage)


__all__ = ['correlation_linkage', 'gradient_linkage',
           'rank_linkage', 'random_linkage']
