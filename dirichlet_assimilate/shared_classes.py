from dataclasses import dataclass
import numpy as np
from typing import List

import scipy.stats


class Sample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def sample_class(self):
        return SampleClass(self > 0)

    def threshold(self, threshold):
        self[self < threshold] = 0

class SampleClass(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=bool).view(cls)

@dataclass
class Ensemble:
    samples: List[Sample]
    def __post_init__(self):
        self.sample_classes = [SampleClass(c) for c in {tuple(s.sample_class) for s in self.samples}]
        self.sample_classes = sorted(self.sample_classes, key = lambda x: list(x))
        self.class_ensembles = [ClassEnsemble(samples=[s for s in self.samples if np.all(s.sample_class==c)]) for c in self.sample_classes]

class ClassEnsemble(Ensemble):
    def __post_init__(self):
        self.sample_class = self.samples[0].sample_class

class UniformSample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)


@dataclass
class ClassDirichlet:
    alpha: np.ndarray
    sample_class: SampleClass

    def __post_init__(self):
        assert np.any(self.sample_class), f'Sample class must have at least one true component.'
        assert len(self.alpha) == np.count_nonzero(self.sample_class),\
            f'alpha ({self.alpha}) does not match number of true components in sample class ({self.sample_class}).'

    @property
    def full_alpha(self):
        a = np.zeros_like(self.sample_class, dtype=float)
        a[self.sample_class] = self.alpha
        return a

    @property
    def full_mean_sample(self):
        return Sample(self.full_alpha / self.full_alpha.sum())

    def to_beta(self):
        # If we are only interested the first component, we might want to simplify the dirichlet to a beta
        alpha = self.alpha[0], self.alpha[1:].sum()
        sample_class = np.array([self.sample_class[0], np.any(self.sample_class[1:])])
        return ClassDirichlet(alpha=alpha, sample_class=sample_class)

    @staticmethod
    def unif():
        return scipy.stats.uniform(0, 1).rvs()

    def marg_cdf(self, x0) -> float:
        # marginal cdf of the first component

        # if x0 is a zero component then the cdf is
        #    1 if x is positive
        #    0 if x is negative
        #    ? if x is 0 (we map to a uniform random variable)
        if not self.sample_class[0]:
            return 1 if x0>0 else self.unif()
        # if all the remaining components are zero then the cdf is
        #    0 if x is less than 1
        #    ? if x = 1 (we map to a uniform random var)
        #    1 if x > 1
        elif self.sample_class[0] and not np.any(self.sample_class[1:]):
            return 0 if not np.isclose(x0, 1.) else self.unif()
        else:
            return scipy.stats.beta(self.alpha[0], self.alpha[1:].sum()).cdf(x0)

@dataclass
class MixedDirichlet:
    mixing_rates: np.ndarray
    dirichlets: List[ClassDirichlet]

    def marg_cdf(self, x0) -> float:
        return np.array([cd.marg_cdf(x0) for cd in self.dirichlets]) @ self.mixing_rates

@dataclass
class UniformEnsemble:
    samples: List[UniformSample]

@dataclass
class RawSample:
    area: np.ndarray
    volume: np.ndarray
    snow: np.ndarray = None
    def __post_init__(self):
        if self.snow is None:
            self.snow = np.zeros_like(self.area)
        if not len(self.area)==len(self.volume)==len(self.snow):
            raise ValueError('Area, Volume, and Snow vectors must have the same length.')

    def threshold(self, a=None, v=None, s=None):
        ''' Set area, volume, or snow to zero if below the given threshold '''
        if a:
            self.area[self.area < a]     = 0.
        if v:
            self.volume[self.volume < v] = 0.
        if s:
            self.snow[self.snow < s]     = 0.

@dataclass
class RawEnsemble:
    samples: List[RawSample]

class HeightBounds(np.ndarray):
    min_interval = 1e-7
    max_interval = 20.0

    def __new__(cls, input_array):
        a = np.asarray(input_array).view(cls)
        assert np.all(a[1:] - a[:-1] >= cls.min_interval), f'Height bounds must be provided in sorted order'\
                                                           f'and spaced by more than {cls.min_interval}: {a}'
        assert a[0] == 0, f'Lowest height bound should be 0.0, not {a}'
        assert a.ndim == 1, f'Height bounds must be a vector, but ndim={a.ndim}'
        return a

    @property
    def intervals(self):
        return zip(self[:-1], self[1:])

    @classmethod
    def from_interval_widths(cls, intervals: np.ndarray):
        assert np.all(intervals >= cls.min_interval)
        a = np.cumsum(intervals)
        a = np.insert(a, 0, 0)
        return HeightBounds(a)

@dataclass
class Observation:
    n: int
    r: int

