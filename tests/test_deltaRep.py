
"""
Created on Mon Oct  7 11:16:56 2024

@author: kabo1917

Learning how to test piece of simplex assimilate, simple class transport
"""

import numpy as np
import pytest
from simplex_assimilate import delta

def test_delta():
    a = np.array([[0, 0.981, 0.010, 0.002, 0.006],
             [0, 0.981, 0.010, 0.002, 0.006]])
    v = np.array([[0, 0.956, 0.014, 0.005, 0.028],
              [0, 0.956, 0.014, 0.005, 0.028]])
    h = np.nan_to_num(v/a,0)

    h_bnd = np.array(np.array([[0.00000               ,0.645],
                   [0.645  ,1.391],
                   [1.391  ,2.470],
                   [2.470   ,4.567],
                   [4.567   ,9.334]]))
    d = delta.deltaRep(a, v, h_bnd)
    [aicen, vicen] = delta.invDeltaRep(d, h_bnd)
    result = np.array([np.max(aicen-a), np.max(vicen-v)])
    assert result.all() == 0


def test_delta_neg_vol():
    a = np.array([[0, 0.981, 0.010, 0.002, 0.006],
             [0, 0.981, 0.010, 0.002, 0.006]])
    v = np.array([[0, 0.956, 0.014, 0.005, 0.28],
              [0, 0.956, 0.014, 0.005, 0.28]])
    h = np.nan_to_num(v/a,0)

    h_bnd = np.array(np.array([[0.00000               ,0.645],
                   [0.645  ,1.391],
                   [1.391  ,2.470],
                   [2.470   ,4.567],
                   [4.567   ,9.334]]))
    d = delta.deltaRep(a, v, h_bnd)
    [aicen, vicen] = delta.invDeltaRep(d, h_bnd)
    result = np.array([np.max(aicen-a), np.max(vicen-v)])
    assert result.all() == 0