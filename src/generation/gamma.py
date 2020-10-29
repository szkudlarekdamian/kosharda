from typing import Tuple
from .base.generator import BaseGenerator
import numpy as np

class Generator(BaseGenerator):
    """
    This is generator utilizes gamma distribution to generate load correlated vectors with gamma distribution
    ref: https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    def __init__(self, n: int, s: int, cor: float, scals_range: Tuple[float, float], shape: float = 2.0) -> None:
        # TODO
        self.num = n
        self.size = s
        self.generator = np.random.default_rng()
        self.r_start, self.r_end = scals_range
        self.shape = shape
        
        cor = cor if cor < 1 else cor - 10**-8
        self.cov_mat = [[1.0 if i == j else cor for j in range(n)] for i in range(n)]
        self.trans_mat = np.linalg.cholesky(self.cov_mat)
        
        self.scales_vector = self.generator.uniform(self.r_start, self.r_end, size=self.num)
        

    def get_estimated_cloud_load(self) -> float:
        return (self.r_start + self.r_end)/2 * self.shape * self.num * self.size

    def generate_cloud_load_vectors(self) -> np.ndarray:
        mat = self.generator.gamma(self.scales_vector, self.shape, size=(self.size, self.num))

        # normalize
        means = np.mean(mat, axis=0)
        stds = np.std(mat, axis=0)
        mat = (mat - means)/stds

        # transform
        mat = np.dot(self.trans_mat, mat.T).T

        # denormalize
        mat = (mat * stds + means)
        mat[mat < 0] = 0

        return mat.T



if __name__ == '__main__':
    num = 2
    size = 1000
    cor = 1.0
    generator = Generator(num, size, cor, (2.0, 4.0))

    estimated = generator.get_estimated_cloud_load()
    assert estimated == num * size * 3 * 2 # (mean of means range) (defult shape)

    vectors = generator.generate_cloud_load_vectors()
    assert vectors.shape == (num, size)

    su = np.sum(vectors)
    assert estimated * 0.90 < su < estimated*1.1, "Estimated was: {}, actual was: {}".format(estimated, su)