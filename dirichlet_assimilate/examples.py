from . import shared_classes
import numpy as np

# raw samples
h_bnd = shared_classes.HeightBounds([0., 1., 2., 4.])

area = np.array([[0.20,0.65,0.1],
                 [0.17,0.66,0.11],
                 [0.10,0.65,0.15],
                 [0.30,0.40,0.00],
                 ])
volume = np.array([[0.08, 1., 0.31],
                   [0.08, 1., 0.31],
                   [0.08, 1., 0.31],
                   [0.15, 0.5, 0.0],
                ])
snow = np.array([[0.08, 1., 0.31],
                 [0.08, 1., 0.31],
                 [0.08, 1., 0.31],
                 [0.,   0., 0.00],
                ])

raw_ensemble = shared_classes.RawEnsemble(samples = [shared_classes.RawSample(area=a, volume=v, snow=s) for a,v,s in zip(area, volume, snow)])
raw_sample = raw_ensemble.samples[0]

# processed samples
ensemble = shared_classes.Ensemble(samples=[shared_classes.Sample([0.05 , 0.12 , 0.08 , 0.3  , 0.35 , 0.045, 0.055]),
                                            shared_classes.Sample([0.06 , 0.09 , 0.08 , 0.32 , 0.34 , 0.065, 0.045]),
                                            shared_classes.Sample([0.1  , 0.02 , 0.08 , 0.3  , 0.35 , 0.145, 0.005]),
                                            shared_classes.Sample([0.3  , 0.15 , 0.15 , 0.3  , 0.1  , 0.   , 0.   ]),
                                            ])
sample = ensemble.samples[0]
