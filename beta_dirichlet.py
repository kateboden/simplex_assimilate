#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:21:25 2024
Playing with beta and dirichlet distributions

@author: kabo1917
"""

from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
import numpy as np

# Plot the beta distribution pdf and cdf for various values of a and b
a = 1e11
b = 1e11

x = np.linspace(0.01,0.99,100)
betaPDF = beta.pdf(x,a,b)
betaCDF = beta.cdf(x,a,b)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(x,betaPDF)
ax2.plot(x,betaCDF)
ax1.set_title('alpha = ' + str(a) + ' beta = ' + str(b))

# Visualize dirichlet distribution
