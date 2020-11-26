from typing import Tuple
from .base.generator import BaseGenerator
import numpy as np

from numba import njit

from time import time

import math

class Generator(BaseGenerator):
    """
    This is generator utilizes gamma distribution to generate load correlated vectors with gamma distribution
    ref: https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    def __init__(self, n: int, s: int, cor: float, scals_range: Tuple[float, float], shape: float = 2.0, seed: float = None) -> None:
        self.num = n
        self.size = s
        self.generator = np.random.default_rng(seed)
        self.r_start, self.r_end = scals_range
        self.shape = shape

        cor = cor if cor < 1 else cor - 10**-8
        # self.cov_mat = [[1.0 if i == j else cor for j in range(n)] for i in range(n)]
        self.cov_mat = generate_cov_mat2(cor, n)

        # self.trans_mat = np.linalg.cholesky(self.cov_mat)
        self.trans_mat = do_cholesky(np.array(self.cov_mat))

        self.scales_vector = self.generator.uniform(self.r_start, self.r_end, size=self.num)

        self.means_vector = self.scales_vector * shape
        self.stds_vector = self.scales_vector * math.sqrt(shape)

    def get_estimated_cloud_load(self) -> float:
        return (self.r_start + self.r_end)/2 * self.shape * self.num * self.size


    def generate_cloud_load_vectors(self) -> np.ndarray:
        mat = self.generator.gamma(scale=self.scales_vector, shape=self.shape, size=(self.size, self.num))

        # normalize
        # means = np.mean(mat, axis=0)
        # stds = np.std(mat, axis=0)
        # mat = (mat - means)/stds
        mat = (mat - self.means_vector)/self.stds_vector

        # transform
        # mat = np.dot(self.trans_mat, mat.T).T
        mat = do_dot(self.trans_mat, mat)

        # denormalize
        # mat = (mat * stds + means)
        mat = (mat * self.stds_vector + self.means_vector)
        mat[mat < 0] = 0

        return mat.T


@njit
def do_dot(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.dot(m1, m2.T).T
    
@njit
def do_cholesky(m: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(m)

@njit(parallel=True)
def gamma_generator(scales: np.ndarray, shape: float, s: int, n: int) -> np.ndarray:
    res = np.zeros((n, s))
    for i in range(scales.size):
        for j in range(s):
            res[i][j] = np.random.gamma(shape=shape, scale=scales[i])
    return res

@njit
def generate_cov_mat(cor: float, n: int) -> np.ndarray: 
    return np.array([[1.0 if i == j else cor for j in range(n)] for i in range(n)])

@njit(parallel=True)
def generate_cov_mat2(cor: float, n: int) -> np.ndarray: 
    res = np.zeros((n,n)) + cor
    for i in range(n):
        res[i][i] = 1.0
    return res

if __name__ == '__main__':
    num = 1000
    size = 100
    cor = 1.0
    generator = Generator(num, size, cor, (2.0, 4.0))

    estimated = generator.get_estimated_cloud_load()
    assert estimated == num * size * 3 * 2 # (mean of means range) (defult shape)

    vectors = generator.generate_cloud_load_vectors()
    assert vectors.shape == (num, size)

    su = np.sum(vectors)
    assert estimated * 0.90 < su < estimated*1.1, "Estimated was: {}, actual was: {}".format(estimated, su)