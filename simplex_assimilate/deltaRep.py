
"""
Created on Mon Oct  7 11:21:17 2024

@author: kabo1917
"""
import numpy as np

def deltaRep(a,v,h):
    [row, col] = np.shape(a)
    aicen_extended = np.zeros((row, col*2+1))
    aicen_extended[:,0] = 1-np.sum(a, axis = 1)
    for i in range(col):
        aicen_extended[:,2*i+1] = (a[:,i]*h[i,1] - v[:,i]) / (h[i,1]-h[i,0])
        aicen_extended[:,2*i+2] = (v[:,i] - a[:,i]*h[i,0]) / (h[i,1]-h[i,0])
    assert (aicen_extended >= 0).all(), 'area cannot be negative'
    return aicen_extended

#  Convert from an extended vector back to area/volume
def invDeltaRep(delta,h_bnd):
    [row, col] = np.shape(delta)
    aicen_ens_posterior = np.zeros((row, (col-1)/2))
    vicen_ens_posterior = np.zeros((row, (col-1)/2))
    for i in range(5):
        aicen_ens_posterior[:,i] = delta[:,2*i+1] + delta[:,2*i+2]
        vicen_ens_posterior[:,i] = h_bnd[i,0]*delta[:,2*i+1] + h_bnd[i,1]*delta[:,2*i+2]
    return [aicen_ens_posterior, vicen_ens_posterior]
    