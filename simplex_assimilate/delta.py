
"""
Created on Mon Oct  7 11:21:17 2024

@author: kabo1917
"""
import numpy as np


def check_v(v,a, h_max):
    h = np.nan_to_num(v/a, 0)
    if np.max(h) < h_max:
        return v
    else:
        idx = np.where(h > h_max)
        v[idx] = a[idx]*h_max
        print('Change to volume to enforce non-negative values')
        return v

def deltaRep(a,v,h):
    [row, col] = np.shape(a)
    aicen_extended = np.zeros((row, col*2+1))
    aicen_extended[:,0] = 1-np.sum(a, axis = 1)
    v = check_v(v,a,h[4,1])
    for i in range(col):
        aicen_extended[:,2*i+1] = (a[:,i]*h[i,1] - v[:,i]) / (h[i,1]-h[i,0])
        aicen_extended[:,2*i+2] = (v[:,i] - a[:,i]*h[i,0]) / (h[i,1]-h[i,0])
    assert (aicen_extended >= 0).all(), 'values cannot be negative'
    return aicen_extended

#  Convert from an extended vector back to area/volume
def invDeltaRep(delta,h_bnd):
    [row, col] = np.shape(delta)
    aicen_ens_posterior = np.zeros((row, int((col-1)/2)))
    vicen_ens_posterior = np.zeros((row, int((col-1)/2)))
    for i in range(5):
        aicen_ens_posterior[:,i] = delta[:,2*i+1] + delta[:,2*i+2]
        vicen_ens_posterior[:,i] = h_bnd[i,0]*delta[:,2*i+1] + h_bnd[i,1]*delta[:,2*i+2]
    return [aicen_ens_posterior, vicen_ens_posterior]

def deltaRep2(a,h, h_bnd):
    [row, col] = np.shape(a)
    aicen_extended = np.zeros((row, col*2+1))
    aicen_extended[:,0] = 1-np.sum(a, axis = 1)
    for i in range(col):
        aicen_extended[:,2*i+1] = a[:,i]* (h_bnd[i,1] - h[:,i]) / (h_bnd[i,1]-h_bnd[i,0])
        aicen_extended[:,2*i+2] = a[:,i]* (h[:,i] - h_bnd[i,0]) / (h_bnd[i,1]-h_bnd[i,0])
    assert (aicen_extended >= 0).all(), 'area cannot be negative'
    if (aicen_extended < 0).any():
        idx = np.where(aicen_extended < 0)
        print(idx)
        print(aicen_extended[idx])
        assert 'area cannot be negative'
    return aicen_extended