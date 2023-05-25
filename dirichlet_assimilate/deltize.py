import numpy as np
from dataclasses import dataclass
from typing import List
from .shared_classes import Sample, Ensemble

# PROCESS TO DELTIZED FORM
def process_sample(raw_sample: RawSample, h_bnd: HeightBounds, threshold=1e-7) -> Sample:
    x = []
    for i, interval in enumerate(h_bnd.intervals):
        M = np.array([[1., 1.,],
                      interval  ])
        x += list(np.linalg.inv(M) @ np.array([raw_sample.area[i], raw_sample.volume[i]]))
    x.insert(0, 1-sum(x))  # first component is open water. How much isn't covered in ice?
    x = np.array(x)
    x[x < threshold] = 0.  # set small measurements to exactly zero
    x /= x.sum()  # renormalize to one
    assert np.isclose(x.sum(), 1., atol=1e-10)
    return Sample(x)

def process_ensemble(raw_ensemble: RawEnsemble, h_bnd: HeightBounds, threshold=1e-7) -> Ensemble:
    return Ensemble(samples=[process_sample(raw_sample, h_bnd, threshold) for raw_sample in raw_ensemble.samples])

# CONVERT BACK TO RAW FORM
def post_process_sample(sample: Sample, h_bnd: HeightBounds) -> RawSample:
    l, r = sample[1::2], sample[2::2]  # delta size on left and right of each interval
    a = l + r
    v = h_bnd[:-1]*l + h_bnd[1:]*r
    return RawSample(area=np.array(a), volume=np.array(v), snow=np.zeros_like(a))

def post_process_ensemble(ensemble: Ensemble, h_bnd: HeightBounds) -> RawEnsemble:
    raw_samples = []
    for sample in ensemble.samples:
        raw_samples.append(post_process_sample(sample, h_bnd=h_bnd))
    return RawEnsemble(samples=raw_samples)
