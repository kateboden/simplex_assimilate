#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:35 2024
Goal: Test the functions within simple_class_transport to make sure it is doing what we expect

@author: kabo1917
"""

import numpy as np
import sys
import simplex_assimilate as sa
from simplex_assimilate import dirichlet 
import scipy.stats as ss
from matplotlib import pyplot as plt

sys.path.append('/Users/kabo1917/opt/simplex_assimilate_kb/simplex_assimilate')
import simple_class_transport

"""
TEST transport_pipeline 
input: aicen_extended, x_0
output: transport aicen_extended

Approach: give transport pipline 10 ensemble members each with 5 variables to represent the 5 ice categories
The sum of the variables will be 1
"""
# Generate random data from a dirichlet distribution
alpha = np.array([5,50,.5,.1,.1])
aicen = ss.dirichlet.rvs(alpha, size = 10, random_state = 1)

fig, ax = plt.subplots(3,5, sharey = True)
ax[0,0].plot(aicen[:,0],'*-')
ax[0,0].set_title('Cat 1')
ax[0,1].plot(aicen[:,1],'*-')
ax[0,1].set_title('Cat 2')
ax[0,2].plot(aicen[:,2],'*-')
ax[0,2].set_title('Cat 3')
ax[0,3].plot(aicen[:,3],'*-')
ax[0,3].set_title('Cat 4')
ax[0,4].plot(aicen[:,4],'*-')
ax[0,4].set_title('Cat 4')





"""

TEST dirichlet.mle: look at error in fitted parameters vs expected parameters

1. Pick alpha
2. Generate samples from dir(alpha)
3. Find alpha using dirichlet.mle(samples)
4. Compare 1 and 3

"""

sampleSize = np.linspace(2,1000,100)
alphaTrue = np.array([.10,1,50])

error = np.zeros((sampleSize.shape[0],alphaTrue.shape[0]))

for i in range(sampleSize.shape[0]):
    samples   = ss.dirichlet.rvs(alphaTrue, size = int(sampleSize[i]), random_state = 1)
    alphaTest = dirichlet.mle(samples)
    error[i,:] = np.abs(alphaTrue-alphaTest)

rel_error = error/alphaTrue

fig, ax = plt.subplots()
ax.plot(sampleSize,rel_error[:,0], label = 'alpha = '+ str(alphaTrue[0]))
ax.plot(sampleSize,rel_error[:,1], label = 'alpha = '+ str(alphaTrue[1]))
ax.plot(sampleSize,rel_error[:,2], label = 'alpha = ' + str(alphaTrue[2]))
ax.axvline(x = 79, c = 'red')
ax.legend()
ax.set_xlabel('Sample Size')
ax.set_ylabel('Relative Error')
ax.set_title('Method = Fixed Point')

"""

TEST unif_dirichlet_samples

1. Pick alpha
2. Create random samples from a dirichlet distribution 
3. Transform dirichlet r.v to uniform r.v

Not testing unif_dirichlet_mixed_samples because the only difference is that the mixed version first separates
by class and then calls unif_dirichlet_samples on each class

"""

alpha = np.array([.1,1,10])
samples = ss.dirichlet.rvs(alpha, size = 100, random_state = 1)

# Plot the beta distributions from the dirichlet distribution
fig, axs = plt.subplots(1,3, sharey = True)
fig.suptitle('Initial dirichlet r.v')
axs[0].hist(samples[:,0])
axs[0].set_title('Alpha = ' + str(alpha[0]))
axs[1].hist(samples[:,1])
axs[1].set_title('Alpha = ' + str(alpha[1]))
axs[2].hist(samples[:,2])
axs[2].set_title('Alpha = ' + str(alpha[2]))

# create uniform samples
u = simple_class_transport.unif_dirichlet_samples(samples, alpha)

# Plot uniform samples

fig, axs = plt.subplots(1,3, sharey = True)
axs[0].hist(u[:,0])
axs[0].set_title('Alpha = ' + str(alpha[0]))
axs[1].hist(u[:,1])
axs[1].set_title('Alpha = ' + str(alpha[1]))
axs[2].hist(u[:,2])
axs[2].set_title('Alpha = ' + str(alpha[2]))
fig.suptitle('New transformed uniform r.v')



"""

Comparing sample histogram to transport pipeline histogram for cycle 1 in free_2ste


"""
# Look at all variables a_0 to a_5
fig, axes = plt.subplots(2,6, sharey = True, sharex = True)
axes[0,0].hist(samples[:,0])
axes[0, 0].set_title("a_0")
axes[0,1].hist(samples[:,1])
axes[0, 1].set_title("a_1")
axes[0,2].hist(samples[:,2])
axes[0, 2].set_title("a_2")
axes[0,3].hist(samples[:,3])
axes[0, 3].set_title("a_3")
axes[0,4].hist(samples[:,4])
axes[0, 4].set_title("a_4")
axes[0,5].hist(samples[:,5])
axes[0, 5].set_title("a_5")
axes[0,0].set_ylabel('Prior Frequency')

axes[1,0].hist(X[:,0])
axes[1,1].hist(X[:,1])
axes[1,2].hist(X[:,2])
axes[1,3].hist(X[:,3])
axes[1,4].hist(X[:,4])
axes[1,5].hist(X[:,5])
axes[1,0].set_ylabel('Posterior Frequency')

# Look just at a_0 and a_1
fig, axes = plt.subplots(2,2, sharey = True, sharex = True)
axes[0,0].hist(samples[:,0])
axes[0, 0].set_title("a_0")
axes[0,1].hist(samples[:,1])
axes[0, 1].set_title("a_1")
axes[0,0].set_ylabel('Prior Frequency')

axes[1,0].hist(X[:,0])
axes[1,1].hist(X[:,1])
axes[1,0].set_ylabel('Posterior Frequency')

# Add in the uniform step
fig, axes = plt.subplots(3,6, sharey = True, sharex = True)
axes[0,0].hist(samples[:,0])
axes[0, 0].set_title("a_0")
axes[0,1].hist(samples[:,1])
axes[0, 1].set_title("a_1")
axes[0,2].hist(samples[:,2])
axes[0, 2].set_title("a_2")
axes[0,3].hist(samples[:,3])
axes[0, 3].set_title("a_3")
axes[0,4].hist(samples[:,4])
axes[0, 4].set_title("a_4")
axes[0,5].hist(samples[:,5])
axes[0, 5].set_title("a_5")
axes[0,0].set_ylabel('Prior')

axes[1,0].hist(U[:,0])
axes[1,1].hist(U[:,1])
axes[1,2].hist(U[:,2])
axes[1,3].hist(U[:,3])
axes[1,4].hist(U[:,4])
axes[1,5].hist(U[:,5])
axes[1,0].set_ylabel('Uniform Prior')

axes[2,0].hist(X[:,0])
axes[2,1].hist(X[:,1])
axes[2,2].hist(X[:,2])
axes[2,3].hist(X[:,3])
axes[2,4].hist(X[:,4])
axes[2,5].hist(X[:,5])
axes[2,0].set_ylabel('Posterior')






