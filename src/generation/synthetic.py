from typing import Tuple
from .base.generator import BaseGenerator
import numpy as np

from math import pi, radians
import random


def synthetic_vector(size: int, mlow: float, mhigh: float, ang: float, rev: bool = False) -> np.ndarray:
    ang = radians(ang)
    base = np.arange(size) % 2
    if rev:
        base = np.roll(base, 1)
    tn = np.tan(ang)
    ref = np.arange(size)
    res = (base * (mhigh - mlow)) + mlow + (ref * tn) - ((size-1) * tn / 2)
    res[res < 0] = 0
    return res


class Generator(BaseGenerator):
    """
    Synthetic data generator
    """

    def __init__(self, n: int, s: int, seed: float = None) -> None:
        self.num = n
        self.size = s
        self.generator = np.random.default_rng(seed)
        self.lmin, self.lmax = 1,5 # low level
        self.gmin, self.gmax = 1,5 # gap (betwen low level and gigh level)
        self.amin, self.amax = -15,15 # angle

    def get_estimated_cloud_load(self) -> float:
        return (3 + 3) * self.size * self.num

    def generate_cloud_load_vectors(self) -> np.ndarray:
        low = self.generator.uniform(self.lmin, self.lmax, size=self.num)
        gap = self.generator.uniform(self.gmin, self.gmax, size=self.num)
        ang = self.generator.uniform(self.amin, self.amax, size=self.num)
        result = []
        for i in range(self.num):
            result.append(synthetic_vector(self.size, low[i], low[i]+gap[i], ang[i], i%2 == 0))

        return np.array(result)
