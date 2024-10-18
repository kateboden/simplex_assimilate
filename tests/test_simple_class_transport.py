#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:18:52 2024

@author: kabo1917

Testing Simple Class Transport
"""

import pytest
import numpy as np
from simplex_assimilate import simple_class_transport as sct

"""Test Scale Factor"""
def test_scale_factor_equal():
    """Test that if a0prior = a0posterior then the scale factor is 1"""
    assert sct.get_scale_factor(0.1345, 0.1345) == 1
    
def test_scale_factor_increase():
    """Test that if a0prior < a0posterior the scale factor is correct"""
    a0 = 0.132467897
    a1 = 0.432344567
    s = (1-a1)/(1-a0)
    assert sct.get_scale_factor(a0, a1) == s
    
def test_scale_factor_decrease():
    """Test that if a0prior > a0posterior the scale factor is correct"""
    a0 = 0.432467897
    a1 = 0.231744567
    s = (1-a1)/(1-a0)
    assert sct.get_scale_factor(a0, a1) == s
