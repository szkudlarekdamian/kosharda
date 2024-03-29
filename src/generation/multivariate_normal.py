import numpy as np
import matplotlib.pyplot as plt
import math

from typing import Tuple

from .base.generator import BaseGenerator

from numba import njit

from time import time


class Generator(BaseGenerator):
    """
    This generator utilizes multivariate normal distribution to generate random correlated vectors with normal distribution
    """
    
    def __init__(self, n: int, s: int, cor: float, means_range: Tuple[float, float], stds_range: Tuple[float, float], seed: float = None) -> None:
        """
        TODO
        :param n: int, number of vectors
        :param s: int, number of elements in each vector
        :param cor: float correlation coefficient
        :param means_range: range of randomize
        :param stds_range:
        """
        self.num = n
        self.size = s
        self.generator = np.random.default_rng(seed)

        self.m_start, self.m_end = means_range
        s_start, s_end = stds_range
        self.means_vector = self.generator.uniform(self.m_start, self.m_end, n)
        self.stds_vector = self.generator.uniform(s_start, s_end, n)

        # self.cov_mat = [[0 for _ in range(n)] for _ in range(n)]
        # self.cov_mat = np.zeros((n,n))
        # variances_vec = self.stds_vector ** 2
        # for i, cv in enumerate(variances_vec):  # column
        #     for j, rv in enumerate(variances_vec):  # row
        #         # na przekątnej wariancje
        #         # pozostałych komórkach kowariancja (korelacja zdenormalizowana)
        #         self.cov_mat[i][j] = self.cov_mat[j][i] = cv if i == j else cor * math.sqrt(cv)*math.sqrt(rv)

        self.cov_mat = cov_mat(self.stds_vector, cor)
        

    def get_estimated_cloud_load(self) -> float:
        return (self.m_start + self.m_end)/2 * self.num * self.size

    def generate_cloud_load_vectors(self) -> np.ndarray:
        result = self.generator.multivariate_normal(self.means_vector, self.cov_mat, self.size).T
        result[result < 0] = 0
        return result


class Generator2(BaseGenerator):

    def __init__(self, n: int, s: int, cor: float, means_range: Tuple[float, float], stds_range: Tuple[float, float], seed: float = None) -> None:
        """
        TODO
        :param n: int, number of vectors
        :param s: int, number of elements in each vector
        :param cor: float correlation coefficient
        :param means_range: range of randomize
        :param stds_range:
        """
        self.num = n
        self.size = s
        self.generator = np.random.default_rng(seed)

        self.m_start, self.m_end = means_range
        s_start, s_end = stds_range
        self.means_vector = self.generator.uniform(self.m_start, self.m_end, n)
        self.stds_vector = self.generator.uniform(s_start, s_end, n)

        cor = cor if cor < 1 else cor - 10**-8
        self.cov_mat = cov_mat(self.stds_vector, cor)

        self.trans_mat = do_cholesky(np.array(self.cov_mat))
        

    def get_estimated_cloud_load(self) -> float:
        return (self.m_start + self.m_end)/2 * self.num * self.size

    def generate_cloud_load_vectors(self) -> np.ndarray:

        mat = self.generator.normal(np.zeros(self.num), np.ones(self.num), size=(self.size, self.num))

        mat = do_dot(self.trans_mat, mat)
        mat = mat + self.means_vector

        mat[mat < 0] = 0

        return mat.T


@njit
def do_dot(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.dot(m1, m2.T).T
    
@njit
def do_cholesky(m: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(m)

@njit(parallel=True)
def cov_mat(stds: np.ndarray, cor: float) -> np.ndarray:
    n = stds.size
    cov_mat = np.dot(stds.reshape((n,1)), stds.reshape((n,1)).T) * cor
    vars = stds ** 2
    for i in range(n):
        cov_mat[i][i] = vars[i]
    return cov_mat

def random_corelated_vectors(means_vec, variances_vec, cor=1, size=100):
    """
    Base version of random correlated vectors generator function
    """
    assert len(means_vec) == len(variances_vec)
    cov = [[0 for _ in range(len(variances_vec))] for _ in range(
        len(variances_vec))]  # zerowanie macierzy kowariancji
    # budowa macierzy kowariancji
    for i, cv in enumerate(variances_vec):  # column
        for j, rv in enumerate(variances_vec):  # row
            # na przekątnej wariancje
            # pozostałych komórkach kowariancja (korelacja zdenormalizowana)
            cov[i][j] = cov[j][i] = cv if i == j else cor * \
                math.sqrt(cv)*math.sqrt(rv)
    # rozkład
    # python goes brrrrr
    return np.random.multivariate_normal(means_vec, cov, size).T


if __name__ == '__main__':
    num = 1000
    size = 100
    cor = 1.0
    generator = Generator2(num, size, cor, (4.0, 4.0), (1.0, 1.0))

    estimated = generator.get_estimated_cloud_load()
    assert estimated == num * size * 4 # (mean of means range)

    st = time()
    vectors = generator.generate_cloud_load_vectors()
    en = time()
    assert vectors.shape == (num, size)
    print(en-st)

    su = np.sum(vectors)
    assert estimated * 0.90 < su < estimated*1.1, "Estimated was: {}, actual was: {}".format(estimated, su)