import simplex_assimilate as sa
# import ice_simplex_assimilate as ice
import numpy as np
import warnings

# warnings.simplefilter('error', RuntimeWarning)

with open('samples.bin','rb') as f, open('open.txt','r') as g:
    open_frac = float(g.read())
    samples = np.frombuffer(f.read())
    samples = samples.reshape(-1, 11)

'''
# bounds that define thickness categories
h_bnd = np.array([0,
                  0.64450721681942580,
                  1.3914334975763036,
                  2.4701793819598885,
                  4.5672879188504911,
                  9.3338418158681744])
h_bnd = ice.deltize.HeightBounds(h_bnd)

with open('thousand.h5','rb') as f:
    aicen_ens_prior, vicen_ens_prior = f["forecast/aicen"][1035], f["forecast/vicen"][1035]
    raw_ensemble = ice.deltize.build_raw_ensemble(aicen_ens_prior, vicen_ens_prior)
    samples = ice.deltize.process_ensemble(raw_ensemble, h_bnd)
    # convert delta-representation back to area and volume matrices
    # post_raw_ensemble = ice.deltize.post_process_ensemble(post_samples, h_bnd)
    # aicen_ens_posterior, vicen_ens_posterior = ice.deltize.raw_ensemble_to_matrices(post_raw_ensemble)
'''

x_0 = np.ones(len(samples)) * open_frac
x_post = sa.transport.transport_pipeline(samples, x_0)
pass
