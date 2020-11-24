import pandas as pd
import numpy as np
import src.generation.gamma as ga
import src.generation.multivariate_normal as mn
from src.generation.base.generator import BaseGenerator
from tqdm import tqdm

"""
Bare metal test (no itermidate kernel)
"""

if __name__ == '__main__':

    # params
    N = 1000
    F = N
    size=100

    def gamma_wrapper(cor: float) -> BaseGenerator:
        return ga.Generator(F, size, cor, (2,4))

    def mn_wrapper(cor: float) -> BaseGenerator:
        return mn.Generator(F, size, cor, (4,6), (1,1))

    def calc(wrapper, cor: float):
        generator = wrapper(cor)
        vectors = generator.generate_cloud_load_vectors()
        cm = np.corrcoef(vectors)
        # return cm[cm != 1]
        return np.mean(cm[cm != 1]), np.sum(vectors)

    cor_rng = np.arange(0, 1.01, 0.05)

    repeats = 100
    ga_res = []

    pbar = tqdm(total=repeats * cor_rng.size)

    for cor in cor_rng:
        for _ in range(repeats):
            r_ga, l_ga= calc(gamma_wrapper, cor)

            ga_res.append((cor, r_ga, l_ga))
            pbar.update(1)

    pbar.close()

    # ------------

    repeats = 20
    mn_res = []

    pbar = tqdm(total=repeats * cor_rng.size)

    for cor in cor_rng:
        for _ in range(repeats):
            r_mn, l_mn= calc(mn_wrapper, cor)

            ga_res.append((cor, r_mn, l_mn))
            pbar.update(1)

    pbar.close()