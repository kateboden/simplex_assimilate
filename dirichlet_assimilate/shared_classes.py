from dataclasses import dataclass
import numpy as np
from typing import List

class Sample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def sample_class(self):
        return SampleClass(self > 0)

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

    @property
    def full_alpha(self):
        a = np.zeros_like(self.sample_class, dtype=float)
        a[self.sample_class] = self.alpha
        return a

    @property
    def full_mean_sample(self):
        return Sample(self.full_alpha / self.full_alpha.sum())

@dataclass
class MixedDirichlet:
    mixing_rates: np.ndarray
    dirichlets: List[ClassDirichlet]

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

@dataclass
class RawEnsemble:
    samples: List[RawSample]

class HeightBounds(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def intervals(self):
        return zip(self[:-1], self[1:])

@dataclass
class Observation:
    n: int
    r: int

